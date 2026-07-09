#!/usr/bin/env python3
"""
Kronos RunPod Ensemble — self-contained, no shell scripts.
Run: python3 runpod_ensemble.py
"""
import os, sys, json, time, pickle, subprocess, tarfile, shutil, random
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from safetensors.torch import save_file, load_file

# ── Config ──
ALL_PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',
    'AUDJPY=X', 'CHFJPY=X', 'EURAUD=X', 'EURCHF=X', 'GBPCHF=X',
    'AUDCAD=X', 'AUDCHF=X', 'AUDNZD=X', 'CADJPY=X', 'NZDJPY=X',
    'GBPAUD=X', 'GBPNZD=X', 'NZDCAD=X', 'EURCAD=X', 'EURNZD=X',
    'GBPCAD=X',
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD',
    'DOGE-USD', 'MATIC-USD',
]
SEEDS = [42, 43, 44, 45, 46]
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'finetune', 'data', 'processed_datasets')
os.environ['OMP_NUM_THREADS'] = '1'


def setup():
    """Create config, patch feedback_train, install deps if needed."""
    print("=== Setup ===")
    os.makedirs(os.path.join(ROOT, 'bot', 'models', 'kronos_live'), exist_ok=True)
    cfg = {
        'd_model': 512, 'n_heads': 8, 'ff_dim': 1024, 'n_layers': 6,
        'n_enc_layers': 3, 'n_dec_layers': 3,
        'ffn_dropout_p': 0.2, 'attn_dropout_p': 0.0, 'resid_dropout_p': 0.2,
        'token_dropout_p': 0.0, 'learn_te': True,
        's1_bits': 10, 's2_bits': 10,
        'beta': 0.05, 'gamma0': 1.0, 'gamma': 1.1, 'zeta': 0.05, 'group_size': 4,
        'd_in': 38,
    }
    json.dump(cfg, open(os.path.join(ROOT, 'bot', 'models', 'kronos_live', 'config.json'), 'w'), indent=2)
    print("  ✅ config.json created")


def download_data(tf, period):
    """Download yfinance data, compute indicators, save as .pkl."""
    print(f"\n=== Downloading {tf} ({period}) ===")
    os.makedirs(DATA_DIR, exist_ok=True)

    sys.path.insert(0, os.path.join(ROOT, 'finetune'))
    from indicators import compute_all_indicators

    all_dfs = {}
    for pair in ALL_PAIRS:
        try:
            df = yf.download(pair, period=period, interval=tf, progress=False)
            if df.empty or len(df) < 100:
                continue
            df.columns = [c.lower() for c in df.columns]
            if 'adj close' in df.columns:
                df = df.drop(columns=['adj close'])
            if 'volume' in df.columns:
                df = df.rename(columns={'volume': 'vol'})
            df = compute_all_indicators(df)
            all_dfs[pair] = df
            print(f"  ✅ {pair}: {len(df)} rows")
        except Exception as e:
            print(f"  ⚠️ {pair}: {e}")

    n = len(all_dfs)
    train = {k: v.iloc[:int(len(v)*0.85)] for k, v in all_dfs.items()}
    val = {k: v.iloc[int(len(v)*0.85):] for k, v in all_dfs.items()}
    pickle.dump(train, open(os.path.join(DATA_DIR, 'train_data.pkl'), 'wb'))
    pickle.dump(val, open(os.path.join(DATA_DIR, 'val_data.pkl'), 'wb'))
    print(f"  ✅ {n} pairs saved ({sum(len(v) for v in train.values())} train rows)")
    return n > 0


def train_seed(seed):
    """Run feedback_train.py with given seed."""
    print(f"\n=== Training seed {seed} ===")
    sys.path.insert(0, os.path.join(ROOT, 'finetune'))
    import feedback_train as ft

    # Override params
    ft.SEED = seed
    ft.EPOCHS = 12
    ft.GRAD_ACCUM_STEPS = 4
    ft.EMA_DECAY = 0.995
    ft.PROFIT_LOSS_WEIGHT = 1.0
    ft.COSINE_RESTART_EPOCHS = 4
    ft.DIRECTION_LOSS_WEIGHT = 0.3
    ft.CONTRASTIVE_LOSS_WEIGHT = 0.15
    ft.USE_CONTRASTIVE_LOSS = True
    ft.USE_SYNTHETIC_PRETRAIN = False
    ft.WEIGHTED_SAMPLING = True
    ft.BATCH_SIZE = 32
    ft.LEARNING_RATE = 5e-5

    # Monkey-patch fetch_fresh_data to skip
    ft.fetch_fresh_data = lambda: True

    return ft.train()


def average_soup(seed_dirs, tf):
    """Average weights from all seeds."""
    print(f"\n=== Averaging {tf} soup ===")
    sds = [load_file(os.path.join(d, 'model.safetensors')) for d in seed_dirs]
    avg = OrderedDict((k, torch.stack([sd[k].float() for sd in sds]).mean(0)) for k in sds[0])

    soup_dir = os.path.join(ROOT, 'outputs', f'soup_{tf}')
    os.makedirs(soup_dir, exist_ok=True)
    save_file(avg, os.path.join(soup_dir, 'model.safetensors'))
    print(f"  ✅ soup_{tf} saved ({len(seed_dirs)} models averaged)")
    return soup_dir


def upload_release(soup_1d, soup_1h):
    print(f"\n=== Uploading Release ===")
    tag = f"v2-ensemble-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    arch = os.path.join(ROOT, 'outputs', f'kronos-{tag}.tar.gz')
    with tarfile.open(arch, 'w:gz') as tar:
        tar.add(soup_1d, arcname='soup_1d')
        tar.add(soup_1h, arcname='soup_1h')
    sz = os.path.getsize(arch) / 1e6

    result = subprocess.run([
        'gh', 'release', 'create', tag,
        '--title', f'Kronos v2 - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '--notes', f'Two-soup: 1d+1h, {len(SEEDS)} seeds each, profit loss+EMA+grad accum',
        arch
    ], capture_output=True, text=True, cwd=ROOT)

    if result.returncode == 0:
        print(f"  ✅ {sz:.0f}MB → https://github.com/Zapders321/Kronos/releases/tag/{tag}")
        os.remove(arch)
    else:
        print(f"  ❌ {result.stderr[:300]}")


def main():
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  KRONOS ENSEMBLE — {datetime.now().isoformat()}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Pairs: {len(ALL_PAIRS)} | Seeds: {SEEDS}")
    print(f"{'='*60}")

    setup()

    # ── 1d ──
    download_data('1d', '5y')
    seed_dirs = []
    for seed in SEEDS:
        print(f"\n--- 1d Seed {seed} ---")
        # Reset random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        train_seed(seed)
        seed_dirs.append(os.path.join(ROOT, 'outputs', 'kronos_base_finetuned', 'checkpoints', 'best_model'))
    soup_1d = average_soup(seed_dirs, '1d')

    # ── 1h ──
    download_data('1h', '2y')
    seed_dirs = []
    for seed in SEEDS:
        print(f"\n--- 1h Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        train_seed(seed)
        seed_dirs.append(os.path.join(ROOT, 'outputs', 'kronos_base_finetuned', 'checkpoints', 'best_model'))
    soup_1h = average_soup(seed_dirs, '1h')

    # ── Release ──
    upload_release(soup_1d, soup_1h)

    print(f"\n{'='*60}")
    print(f"  ✅ DONE! {(time.time()-t0)/3600:.1f}h")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()