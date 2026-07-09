#!/usr/bin/env python3
"""
Kronos Two-Soup Ensemble Training — for RunPod GPU.
Trains 5 models on 1d data + 5 models on 1h data.
Each soup averaged independently. Bot loads both at inference.

Usage:
  python3 runpod_training/train_soup.py

Output:
  outputs/soup_1d/ — averaged 1d model + individual models
  outputs/soup_1h/ — averaged 1h model + individual models
  GitHub Release with both soups + individual models
"""
import os, sys, json, time, math, random, pickle, subprocess, tarfile, shutil, warnings
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

warnings.filterwarnings('ignore')

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'finetune'))

from model.kronos import Kronos, KronosTokenizer
from indicators import compute_all_indicators, get_indicator_feature_names


# ── Config ──
ALL_PAIRS = [
    # ── Forex —─
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',
    'AUDJPY=X', 'CHFJPY=X', 'EURAUD=X', 'EURCHF=X', 'GBPCHF=X',
    'AUDCAD=X', 'AUDCHF=X', 'AUDNZD=X', 'CADJPY=X', 'NZDJPY=X',
    'GBPAUD=X', 'GBPNZD=X', 'NZDCAD=X', 'EURCAD=X', 'EURNZD=X',
    'GBPCAD=X',
    # ── Crypto —─
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD',
    'DOGE-USD', 'MATIC-USD',
]

TIMEFRAMES = {
    '1d': {'period': '5y', 'years': 5},
    '1h': {'period': '2y', 'years': 2},
}

SEEDS = [42, 43, 44, 45, 46]
N_TRAIN_MAX = 6000     # max windows per pair per timeframe
N_VAL_MAX = 1000
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
GRAD_ACCUM_STEPS = 4
EMA_DECAY = 0.995
CONTEXT_LEN = 64
PRED_LEN = 16

MODEL_CFG = {
    'd_in': None,  # detected at runtime
    'd_model': 512, 'n_heads': 8, 'ff_dim': 1024,
    'n_layers': 6, 'n_enc_layers': 3, 'n_dec_layers': 3,
    'ffn_dropout_p': 0.2, 'attn_dropout_p': 0.0, 'resid_dropout_p': 0.2,
    'token_dropout_p': 0.0, 'learn_te': True,
    's1_bits': 10, 's2_bits': 10,
    'beta': 0.05, 'gamma0': 1.0, 'gamma': 1.1, 'zeta': 0.05, 'group_size': 4,
}

DATA_DIR = os.path.join(PROJECT_ROOT, 'runpod_training', 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(DATA_DIR, exist_ok=True)

D_IN = None
FEATURE_NAMES = None


# ═══════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════

def download_data(tf):
    """Download data for all pairs at given timeframe."""
    cfg = TIMEFRAMES[tf]
    label = f"{len(ALL_PAIRS)} pairs @ {tf}"
    print(f"\n{'='*60}")
    print(f"  📥 Downloading {label} ({cfg['period']})")
    print(f"{'='*60}")

    all_data = {}
    for pair in ALL_PAIRS:
        safe = pair.replace('=', '_')
        cache = os.path.join(DATA_DIR, f'{safe}_{tf}.pkl')
        if os.path.exists(cache):
            all_data[pair] = pd.read_pickle(cache)
            continue

        try:
            df = yf.download(pair, period=cfg['period'], interval=tf, progress=False)
            if df.empty or len(df) < 100:
                print(f"  ⚠️  {pair} — {len(df)} rows, skip")
                continue
            # Flatten MultiIndex columns
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.to_pickle(cache)
            all_data[pair] = df
        except Exception as e:
            print(f"  ❌ {pair} — {e}")

    print(f"  ✅ Got {len(all_data)}/{len(ALL_PAIRS)} pairs")
    return all_data


def make_windows(raw_data, tf):
    """Convert raw OHLCV data into training windows with indicators."""
    global D_IN, FEATURE_NAMES

    all_features, all_labels, all_weights = [], [], []

    for pair, df in raw_data.items():
        if len(df) < CONTEXT_LEN + PRED_LEN + 20:
            continue

        # Normalize column names to lowercase and rename for indicators
        df.columns = [c.lower() for c in df.columns]
        if 'adj close' in df.columns:
            df = df.drop(columns=['adj close'])
        if 'volume' in df.columns:
            df = df.rename(columns={'volume': 'vol'})
        if 'close' not in df.columns:
            continue

        df_ = compute_all_indicators(df)
        feat_names = get_indicator_feature_names()
        if FEATURE_NAMES is None:
            FEATURE_NAMES = feat_names
            D_IN = len(feat_names)

        arr = df_[feat_names].values
        close = df_['Close'].values

        for i in range(len(arr) - CONTEXT_LEN - PRED_LEN):
            fut_close = close[i + CONTEXT_LEN : i + CONTEXT_LEN + PRED_LEN]
            cur_close = close[i + CONTEXT_LEN - 1]
            profit = 1.0 if fut_close[-1] > cur_close else 0.0

            ctx = arr[i : i + CONTEXT_LEN].flatten()
            tgt = arr[i + CONTEXT_LEN : i + CONTEXT_LEN + PRED_LEN]

            all_features.append(ctx)
            all_labels.append({
                'target': torch.from_numpy(tgt).float(),
                'profit': torch.tensor(profit).float(),
                'direction': torch.tensor(profit).float(),
            })
            all_weights.append(1.0)

    print(f"  Windows: {len(all_features)}")
    return all_features, all_labels, all_weights


class FxDataset(Dataset):
    def __init__(self, features, labels, weights, augment=False):
        self.features = features
        self.labels = labels
        self.weights = weights
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = np.array(self.features[idx], dtype=np.float32)
        if self.augment:
            feat += np.random.normal(0, 0.001, feat.shape)
        feat = torch.from_numpy(feat).float()
        return feat, self.labels[idx], torch.tensor(self.weights[idx]).float()


# ═══════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════

def profit_loss(logits, profit_labels):
    pred = logits[:, 0].sigmoid()
    bce = nn.functional.binary_cross_entropy(pred, profit_labels)
    conf = -torch.log(pred * profit_labels + (1 - pred) * (1 - profit_labels) + 1e-8)
    spread = torch.exp(-torch.abs(pred - 0.5) / 0.0005)
    return bce + 0.1 * conf.mean() + 0.05 * spread.mean()


def train_one(seed, loader, val_loader, tokenizer, tf):
    print(f"\n{'='*60}")
    print(f"  🎲 Training ({tf}) — seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = Kronos(
        s1_bits=MODEL_CFG['s1_bits'], s2_bits=MODEL_CFG['s2_bits'],
        n_layers=MODEL_CFG['n_layers'], d_model=MODEL_CFG['d_model'],
        n_heads=MODEL_CFG['n_heads'], ff_dim=MODEL_CFG['ff_dim'],
        ffn_dropout_p=MODEL_CFG['ffn_dropout_p'],
        attn_dropout_p=MODEL_CFG['attn_dropout_p'],
        resid_dropout_p=MODEL_CFG['resid_dropout_p'],
        token_dropout_p=MODEL_CFG['token_dropout_p'],
        learn_te=MODEL_CFG['learn_te'],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = CosineAnnealingWarmRestarts(opt, T_0=len(loader) * 4, T_mult=1, eta_min=1e-7)

    # EMA shadow
    ema = OrderedDict()
    for n, p in model.named_parameters():
        if p.requires_grad:
            ema[n] = p.data.clone().detach()

    best_loss = float('inf')
    out = os.path.join(OUTPUT_DIR, f'soup_{tf}', f'seed_{seed}')
    os.makedirs(out, exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        opt.zero_grad()

        for bi, (x, lbls, w) in enumerate(loader):
            x = x.to(device)
            w = w.to(device)
            pl = torch.stack([l['profit'] for l in lbls]).to(device)
            dl = torch.stack([l['direction'] for l in lbls]).to(device)

            logits = model(x)
            ploss = profit_loss(logits, pl)

            d_logits = logits[:, 1] if logits.size(1) > 1 else logits[:, 0]
            dloss = nn.functional.binary_cross_entropy(d_logits.sigmoid(), dl)

            closs = torch.tensor(0.0).to(device)
            if logits.size(0) > 4:
                z = logits[:, 0].detach()
                pm = (pl == pl.unsqueeze(1)).float()
                sim = torch.mm(z.unsqueeze(1), z.unsqueeze(0))
                pos_sim = (sim * pm).sum(1) / (pm.sum(1) + 1e-8)
                neg_sim = (sim * (1 - pm)).sum(1) / ((1 - pm).sum(1) + 1e-8)
                closs = torch.relu(neg_sim - pos_sim + 0.1).mean()

            loss = (ploss * 1.0 + dloss * 0.3 + closs * 0.15) / GRAD_ACCUM_STEPS
            loss.backward()

            if (bi + 1) % GRAD_ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                for n, p in model.named_parameters():
                    if p.requires_grad and n in ema:
                        ema[n] = EMA_DECAY * ema[n] + (1 - EMA_DECAY) * p.data

            loss_sum += loss.item() * GRAD_ACCUM_STEPS

        avg = loss_sum / len(loader)
        vloss = validate(model, val_loader, device)
        print(f"  E{ep:2d} — train: {avg:.4f} | val: {vloss:.4f}")

        if vloss < best_loss:
            best_loss = vloss
            safetensors_save(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                os.path.join(out, 'model.safetensors')
            )
            safetensors_save(
                {k: v.contiguous() for k, v in ema.items()},
                os.path.join(out, 'model_ema.safetensors')
            )

    print(f"  ✅ seed={seed} best val: {best_loss:.4f}")
    return out


def validate(model, loader, device):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for x, lbls, _ in loader:
            x = x.to(device)
            pl = torch.stack([l['profit'] for l in lbls]).to(device)
            logits = model(x)
            tot += profit_loss(logits, pl).item()
    return tot / len(loader)


# ═══════════════════════════════════════════
#  MODEL SOUP
# ═══════════════════════════════════════════

def average_soup(seed_dirs, tf):
    print(f"\n{'='*60}")
    print(f"  🥣 Averaging {len(seed_dirs)} models → soup_{tf}")
    print(f"{'='*60}")

    sds = [safetensors_load(os.path.join(d, 'model.safetensors')) for d in seed_dirs]
    avg = OrderedDict((k, torch.stack([sd[k].float() for sd in sds]).mean(0)) for k in sds[0])

    soup_dir = os.path.join(OUTPUT_DIR, f'soup_{tf}', 'model')
    os.makedirs(soup_dir, exist_ok=True)
    safetensors_save(avg, os.path.join(soup_dir, 'model.safetensors'))

    # EMA soup
    emas = []
    for d in seed_dirs:
        p = os.path.join(d, 'model_ema.safetensors')
        if os.path.exists(p):
            emas.append(safetensors_load(p))
    if emas:
        avg_ema = OrderedDict((k, torch.stack([e[k].float() for e in emas]).mean(0)) for k in emas[0])
        safetensors_save(avg_ema, os.path.join(soup_dir, 'model_ema.safetensors'))

    # config
    cfg = {
        'd_in': D_IN,
        'd_model': MODEL_CFG['d_model'],
        'n_heads': MODEL_CFG['n_heads'],
        'n_layers': MODEL_CFG['n_layers'],
        'trained_on': f'{tf} — {len(ALL_PAIRS)} pairs',
    }
    with open(os.path.join(soup_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"  ✅ soup_{tf} saved")
    return soup_dir


# ═══════════════════════════════════════════
#  RELEASE
# ═══════════════════════════════════════════

def upload_release(soup_dir_1d, soup_dir_1h):
    print(f"\n{'='*60}")
    print(f"  🚀 Uploading to GitHub Release")
    print(f"{'='*60}")

    tag = f"v2-ensemble-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    arch = os.path.join(OUTPUT_DIR, f'kronos-ensemble-{tag}.tar.gz')

    with tarfile.open(arch, 'w:gz') as tar:
        tar.add(soup_dir_1d, arcname='soup_1d')
        tar.add(soup_dir_1h, arcname='soup_1h')

    sz = os.path.getsize(arch)
    print(f"  Archive: {sz/1e6:.1f}MB")

    notes = (
        f'Two-soup ensemble: '
        f'{len(SEEDS)} models on 1d (5yr, {len(ALL_PAIRS)} pairs) '
        f'+ {len(SEEDS)} models on 1h (2yr, {len(ALL_PAIRS)} pairs). '
        f'Profit loss + EMA + grad accumulation + cosine annealing. '
        f'Seeds={SEEDS}.'
    )

    result = subprocess.run([
        'gh', 'release', 'create', tag,
        '--title', f'Kronos v2 Ensemble - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '--notes', notes,
        arch
    ], capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print(f"  ✅ https://github.com/Zapders321/Kronos/releases/tag/{tag}")
        os.remove(arch)
        return True
    else:
        print(f"  ❌ {result.stderr[:500]}")
        return False


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def train_timeframe(tf, tokenizer):
    """Full pipeline for one timeframe: download → windows → N seeds → soup."""
    print(f"\n{'='*70}")
    print(f"  📊 TIMEFRAME: {tf}  ({TIMEFRAMES[tf]['period']})")
    print(f"{'='*70}")

    raw = download_data(tf)
    if len(raw) < 3:
        print(f"  ❌ Not enough data for {tf}")
        return None

    feats, lbls, ws = make_windows(raw, tf)
    n = len(feats)
    n_tr = int(n * 0.85)

    tr_feats, tr_lbls, tr_ws = feats[:n_tr], lbls[:n_tr], ws[:n_tr]
    va_feats, va_lbls, va_ws = feats[n_tr:], lbls[n_tr:], ws[n_tr:]

    print(f"\n  Train: {len(tr_feats)} windows | Val: {len(va_feats)} windows")

    tr_ds = FxDataset(tr_feats, tr_lbls, tr_ws, augment=True)
    va_ds = FxDataset(va_feats, va_lbls, va_ws)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    seed_dirs = []
    for seed in SEEDS:
        sd = train_one(seed, tr_ld, va_ld, tokenizer, tf)
        seed_dirs.append(sd)

    soup_dir = average_soup(seed_dirs, tf)
    return soup_dir


def main():
    print(f"\n{'='*70}")
    print(f"  KRONOS TWO-SOUP ENSEMBLE")
    print(f"  {datetime.now().isoformat()}")
    print(f"  GPU: {'✅ CUDA' if torch.cuda.is_available() else '❌ CPU'}")
    if torch.cuda.is_available():
        print(f"       {torch.cuda.get_device_name(0)}")
        print(f"       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB VRAM")
    print(f"  Pairs: {len(ALL_PAIRS)} (FX + crypto)")
    print(f"  Timeframes: 1d (5yr) + 1h (2yr)")
    print(f"  Seeds: {SEEDS} (5 per timeframe = 10 total)")
    print(f"  Epochs: {EPOCHS}")
    print(f"{'='*70}")

    t0 = time.time()

    # ── Model config ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_in_val = D_IN or 45

    tokenizer = KronosTokenizer(
        d_in=d_in_val,
        d_model=MODEL_CFG['d_model'], n_heads=MODEL_CFG['n_heads'],
        ff_dim=MODEL_CFG['ff_dim'],
        n_enc_layers=MODEL_CFG['n_enc_layers'],
        n_dec_layers=MODEL_CFG['n_dec_layers'],
        ffn_dropout_p=MODEL_CFG['ffn_dropout_p'],
        attn_dropout_p=MODEL_CFG['attn_dropout_p'],
        resid_dropout_p=MODEL_CFG['resid_dropout_p'],
        s1_bits=MODEL_CFG['s1_bits'], s2_bits=MODEL_CFG['s2_bits'],
        beta=MODEL_CFG['beta'], gamma0=MODEL_CFG['gamma0'],
        gamma=MODEL_CFG['gamma'], zeta=MODEL_CFG['zeta'],
        group_size=MODEL_CFG['group_size'],
    ).to(device).eval()

    MODEL_CFG['d_in'] = d_in_val

    soup_1d = train_timeframe('1d', tokenizer)
    if soup_1d is None:
        print("❌ 1d training failed")
        sys.exit(1)

    soup_1h = train_timeframe('1h', tokenizer)
    if soup_1h is None:
        print("⚠️ 1h training failed — uploading 1d only")
        soup_1h = soup_1d  # fallback: upload just 1d as tar

    upload_release(soup_1d, soup_1h)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ✅ DONE! {datetime.now().isoformat()}")
    print(f"  Time: {elapsed/60:.0f} min ({elapsed/3600:.1f} hrs)")
    print(f"  10 models → 2 soups → 1 release")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()