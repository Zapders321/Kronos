#!/usr/bin/env python3
"""
Kronos Model Soup + Real Data Training — for RunPod GPU.
Trains 5 models with different seeds on real 1d FX data, averages their weights.

Usage:
  python3 runpod_training/train_soup.py

Output:
  model_soup_output/ — averaged model weights
  Creates GitHub Release with the soup model + individual models
"""
import os, sys, json, time, math, random, pickle, subprocess, tarfile, shutil
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'finetune'))

from model.kronos import Kronos, KronosTokenizer
from indicators import compute_all_indicators, get_indicator_feature_names

# ── Config ──
ALL_PAIRS = [
    # ── Forex (via Yahoo Finance) ──
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',
    'AUDJPY=X', 'CHFJPY=X', 'EURAUD=X', 'EURCHF=X', 'GBPCHF=X',
    'AUDCAD=X', 'AUDCHF=X', 'AUDNZD=X', 'CADJPY=X', 'NZDJPY=X',
    'GBPAUD=X', 'GBPNZD=X', 'NZDCAD=X', 'EURCAD=X', 'EURNZD=X',
    'GBPCAD=X',
    # ── Crypto (via Yahoo Finance) ──
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD',
    'DOGE-USD', 'MATIC-USD',
]
YEARS_BACK = 5
TRAIN_SPLIT = 0.85
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
GRAD_ACCUM_STEPS = 4
EMA_DECAY = 0.995
CONTEXT_LEN = 64
PRED_LEN = 16
N_TRAIN = 50000  # windows per pair total
N_VAL = 5000
D_IN = 45  # will be detected from data
FEATURE_NAMES = None

DATA_DIR = os.path.join(PROJECT_ROOT, 'runpod_training', 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'model_soup_output')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════

def download_real_data():
    """Download YEARS of 1d data for all FX pairs from yfinance."""
    print(f"\n{'='*60}")
    print(f"  📥 Downloading {len(ALL_PAIRS)} FX + crypto pairs, {YEARS_BACK} years of 1d data")
    print(f"{'='*60}")

    all_data = {}
    for pair in ALL_PAIRS:
        cache_path = os.path.join(DATA_DIR, f'{pair.replace("=", "_")}.pkl')
        if os.path.exists(cache_path):
            print(f"  📦 Cached: {pair}")
            all_data[pair] = pd.read_pickle(cache_path)
            continue

        print(f"  📡 {pair}...")
        try:
            df = yf.download(pair, period=f'{YEARS_BACK}y', interval='1d', progress=False)
            if df.empty or len(df) < 100:
                print(f"     ⚠️  Only {len(df)} rows, skipping")
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.to_pickle(cache_path)
            all_data[pair] = df
            print(f"     ✅ {len(df)} rows")
        except Exception as e:
            print(f"     ❌ {e}")

    print(f"\n  ✅ Downloaded {len(all_data)} pairs")
    return all_data


def prepare_dataset(all_data):
    """Convert raw data to training windows with indicators."""
    print(f"\n{'='*60}")
    print(f"  🔧 Processing {len(all_data)} pairs into windows")
    print(f"{'='*60}")

    global D_IN, FEATURE_NAMES

    features_list = []
    labels_list = []
    weights_list = []
    pair_names = []

    for pair, df in all_data.items():
        # Skip too-small datasets
        if len(df) < CONTEXT_LEN + PRED_LEN + 10:
            continue

        # Compute indicators
        df_with_indicators = compute_all_indicators(df)
        feat_names = get_indicator_feature_names()
        
        if FEATURE_NAMES is None:
            FEATURE_NAMES = feat_names
            D_IN = len(feat_names)

        # Create windows
        data = df_with_indicators[feat_names].values
        close = df_with_indicators['Close'].values
        dates = df_with_indicators.index.values

        for i in range(len(data) - CONTEXT_LEN - PRED_LEN):
            ctx = data[i : i + CONTEXT_LEN]
            tgt = data[i + CONTEXT_LEN : i + CONTEXT_LEN + PRED_LEN]
            fut_close = close[i + CONTEXT_LEN : i + CONTEXT_LEN + PRED_LEN]
            cur_close = close[i + CONTEXT_LEN - 1]

            # Profit label: direction of close price change over prediction horizon
            profit = 1.0 if fut_close[-1] > cur_close else 0.0

            features_list.append(ctx.flatten())
            labels_list.append({
                'target': torch.from_numpy(tgt).float(),
                'profit': torch.tensor(profit).float(),
                'direction': torch.tensor(1.0 if fut_close[-1] > cur_close else 0.0).float(),
            })
            weights_list.append(1.0)
            pair_names.append(pair)

    print(f"  Total windows: {len(features_list)}")
    return features_list, labels_list, weights_list, pair_names


class RealFxDataset(Dataset):
    def __init__(self, features, labels, weights=None, augment=False):
        self.features = features
        self.labels = labels
        self.weights = weights if weights is not None else [1.0] * len(features)
        self.augment = augment
        self.n = len(features)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.augment:
            # Add noise for augmentation
            feat = np.array(self.features[idx], dtype=np.float32)
            noise = np.random.normal(0, 0.001, feat.shape)
            feat = feat + noise
            feat = torch.from_numpy(feat).float()
        else:
            feat = torch.from_numpy(np.array(self.features[idx], dtype=np.float32)).float()

        lbl = self.labels[idx]
        weight = torch.tensor(self.weights[idx]).float()
        return feat, lbl, weight


# ═══════════════════════════════════════════
#  PROFIT LOSS
# ═══════════════════════════════════════════

def compute_profit_loss(logits, profit_labels, close_prices=None, margin=0.0005):
    """Profit-based loss: penalize direction + magnitude error."""
    batch_size = logits.size(0)
    pred_profit = logits[:, 0].sigmoid()
    
    # Binary profit direction loss
    profit_loss = nn.functional.binary_cross_entropy(pred_profit, profit_labels)
    
    # Spread penalty — closer to decision boundary = worse for small moves
    confidence_penalty = -torch.log(pred_profit * profit_labels + (1 - pred_profit) * (1 - profit_labels) + 1e-8)
    spread_penalty = torch.exp(-torch.abs(pred_profit - 0.5) / margin)
    
    return profit_loss + 0.1 * confidence_penalty.mean() + 0.05 * spread_penalty.mean()


# ═══════════════════════════════════════════
#  MODEL SOUP TRAINING
# ═══════════════════════════════════════════

def train_single_model(seed, train_loader, val_loader, model_cfg, tokenizer):
    """Train one model with given seed."""
    print(f"\n{'='*60}")
    print(f"  🎲 Training Model (Seed={seed})")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Create model
    model = Kronos(tokenizer=tokenizer, **model_cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    total_batches = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_batches * 4, T_mult=1, eta_min=1e-7
    )

    # EMA shadow weights
    ema_weights = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema_weights[name] = param.data.clone().detach()

    best_val_loss = float('inf')
    best_model_dir = os.path.join(OUTPUT_DIR, f'model_seed_{seed}')
    os.makedirs(best_model_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (features, labels, weights) in enumerate(train_loader):
            features = features.to(device)
            weights = weights.to(device)

            profit_labels = torch.stack([l['profit'] for l in labels]).to(device)
            direction_labels = torch.stack([l['direction'] for l in labels]).to(device)

            logits = model(features)

            # Profit loss
            p_loss = compute_profit_loss(logits, profit_labels)

            # Direction loss
            d_logits = logits[:, 1] if logits.size(1) > 1 else logits[:, 0]
            direction_loss = nn.functional.binary_cross_entropy(
                d_logits.sigmoid(), direction_labels
            )

            # Contrastive loss (pairwise similarity)
            c_loss = torch.tensor(0.0).to(device)
            if logits.size(0) > 4:
                z = logits[:, 0].detach()
                pos_mask = (profit_labels == profit_labels.unsqueeze(1)).float()
                sim = torch.mm(z.unsqueeze(1), z.unsqueeze(0))
                pos_sim = (sim * pos_mask).sum(1) / (pos_mask.sum(1) + 1e-8)
                neg_sim = (sim * (1 - pos_mask)).sum(1) / ((1 - pos_mask).sum(1) + 1e-8)
                c_loss = torch.relu(neg_sim - pos_sim + 0.1).mean()

            loss = (p_loss * 1.0 + direction_loss * 0.3 + c_loss * 0.15)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update EMA
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in ema_weights:
                            ema_weights[name] = EMA_DECAY * ema_weights[name] + (1 - EMA_DECAY) * param.data

            total_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch:2d}/{EPOCHS} — train_loss: {avg_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader, device)
        print(f"                  val_loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            safetensors_save(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                os.path.join(best_model_dir, 'model.safetensors')
            )
            # Save EMA weights
            safetensors_save(
                {k: v.contiguous() for k, v in ema_weights.items()},
                os.path.join(best_model_dir, 'model_ema.safetensors')
            )
            print(f"  💾 New best model saved (val_loss: {val_loss:.4f})")

    print(f"  ✅ Model seed={seed} done. Best val_loss: {best_val_loss:.4f}")
    return best_model_dir


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels, weights in val_loader:
            features = features.to(device)
            weights = weights.to(device)
            profit_labels = torch.stack([l['profit'] for l in labels]).to(device)

            logits = model(features)
            loss = compute_profit_loss(logits, profit_labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def average_models(model_seed_dirs, tokenizer):
    """Average weights from all trained models (model soup)."""
    print(f"\n{'='*60}")
    print(f"  🥣 Model Soup: averaging {len(model_seed_dirs)} models")
    print(f"{'='*60}")

    # Load all state dicts
    all_state_dicts = []
    for d in model_seed_dirs:
        path = os.path.join(d, 'model.safetensors')
        sd = safetensors_load(path)
        all_state_dicts.append(sd)
        print(f"  Loaded: {path}")

    # Average
    averaged = OrderedDict()
    for key in all_state_dicts[0].keys():
        averaged[key] = torch.stack([sd[key].float() for sd in all_state_dicts]).mean(dim=0)

    # Save soup model
    soup_dir = os.path.join(OUTPUT_DIR, 'model_soup')
    os.makedirs(soup_dir, exist_ok=True)
    safetensors_save(averaged, os.path.join(soup_dir, 'model.safetensors'))

    # Also average EMA weights
    all_ema = []
    for d in model_seed_dirs:
        ema_path = os.path.join(d, 'model_ema.safetensors')
        if os.path.exists(ema_path):
            all_ema.append(safetensors_load(ema_path))

    if all_ema:
        averaged_ema = OrderedDict()
        for key in all_ema[0].keys():
            averaged_ema[key] = torch.stack([sd[key].float() for sd in all_ema]).mean(dim=0)
        safetensors_save(averaged_ema, os.path.join(soup_dir, 'model_ema.safetensors'))

    # Save config
    model = Kronos(tokenizer=tokenizer)
    config = model.config.to_dict() if hasattr(model, 'config') else {}
    with open(os.path.join(soup_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  ✅ Model soup saved to {soup_dir}")
    return soup_dir


def upload_release(soup_dir):
    """Pack and upload as GitHub Release."""
    print(f"\n{'='*60}")
    print(f"  🚀 Uploading model to GitHub Release")
    print(f"{'='*60}")

    tag = f"v2-soup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    archive = os.path.join(OUTPUT_DIR, f'kronos-soup-{tag}.tar.gz')

    print(f"  Packing: {soup_dir} → {archive}")
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(soup_dir, arcname='best_model')

    size = os.path.getsize(archive)
    print(f"  Archive: {size/1e6:.1f}MB")

    print(f"  Creating release: {tag}")
    result = subprocess.run([
        'gh', 'release', 'create', tag,
        '--title', f'Kronos Model Soup - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '--notes', f'Model soup of {len(SEEDS)} models trained on {YEARS_BACK}y real FX data. Epochs={EPOCHS}, seeds={SEEDS}, profit loss + EMA + grad accumulation + cosine annealing.',
        archive
    ], capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print(f"  ✅ https://github.com/Zapders321/Kronos/releases/tag/{tag}")
        os.remove(archive)
        return True
    else:
        print(f"  ❌ {result.stderr[:500]}")
        return False


def main():
    print(f"\n{'='*60}")
    print(f"  KRONOS MODEL SOUP TRAINING")
    print(f"  {datetime.now().isoformat()}")
    print(f"  GPU: {'✅ CUDA' if torch.cuda.is_available() else '❌ CPU'}")
    if torch.cuda.is_available():
        print(f"        {torch.cuda.get_device_name(0)}")
        print(f"        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB VRAM")
    print(f"  Pairs: {len(ALL_PAIRS)}")
    print(f"  Years: {YEARS_BACK}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Epoch: {EPOCHS}")
    print(f"{'='*60}")

    # Step 1: Download real data
    all_data = download_real_data()
    if len(all_data) < 3:
        print("❌ Not enough data downloaded")
        sys.exit(1)

    # Step 2: Prepare windows
    features, labels, weights, pair_names = prepare_dataset(all_data)
    n_total = len(features)
    n_train = int(n_total * TRAIN_SPLIT)

    train_features = features[:n_train]
    train_labels = labels[:n_train]
    train_weights = weights[:n_train]
    val_features = features[n_train:]
    val_labels = labels[n_train:]
    val_weights = weights[n_train:]

    print(f"  Train: {len(train_features)} windows")
    print(f"  Val:   {len(val_features)} windows")

    train_dataset = RealFxDataset(train_features, train_labels, train_weights, augment=True)
    val_dataset = RealFxDataset(val_features, val_labels, val_weights)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Step 3: Create base model config
    print(f"\n  D_IN detected: {D_IN}")
    print(f"  Features: {FEATURE_NAMES[:5]}... ({len(FEATURE_NAMES)} total)")

    tokenizer = KronosTokenizer()
    model_cfg = {
        'd_in': D_IN,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
    }

    # Step 4: Train N models with different seeds
    model_seed_dirs = []
    for seed in SEEDS:
        model_dir = train_single_model(seed, train_loader, val_loader, model_cfg, tokenizer)
        model_seed_dirs.append(model_dir)

    # Step 5: Model soup (average weights)
    soup_dir = average_models(model_seed_dirs, tokenizer)

    # Step 6: Upload as GitHub Release
    upload_release(soup_dir)

    print(f"\n{'='*60}")
    print(f"  ✅ DONE! {datetime.now().isoformat()}")
    print(f"  Trained {len(SEEDS)} models, souped & released")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()