#!/usr/bin/env python3
"""
Feedback loop — next-gen Kronos training with synthetic data + contrastive learning.

Flow:
  1. Fetch real market data for all pairs/timeframes
  2. Generate synthetic market data (GBM+GARCH+regimes) for pre-training
  3. Optional: curriculum learning (synthetic first, then real)
  4. Train with: prediction loss + direction loss + contrastive (NT-Xent) loss
  5. Track confidence per bucket
  6. Save model + tokenizer + config

Run:  python3 feedback_train.py

Config knobs at top of file:
  USE_SYNTHETIC_PRETRAIN   — generate synthetic data for pre-training
  USE_CONTRASTIVE_LOSS     — add NT-Xent contrastive loss
  CURRICULUM_SYNTHETIC     — epochs of pure synthetic before mixing real
  WEIGHTED_SAMPLING        — weight higher timeframes more
"""
import os, sys, json, pickle, time, random, subprocess, csv
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file as safetensors_save

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.kronos import KronosTokenizer, Kronos
from finetune.indicators import compute_all_indicators

# ============================================================
# CONFIG
# ============================================================
FINETUNE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(FINETUNE_DIR, '..')
BOT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'kronos_base_finetuned', 'checkpoints', 'best_model')
DATASET_PATH = os.path.join(FINETUNE_DIR, 'data', 'processed_datasets')
CONFIDENCE_LOG = os.path.join(PROJECT_ROOT, 'bot', 'confidence_log.csv')

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
LOOKBACK = 90
PREDICT_LEN = 10
CLIP_VAL = 5.0
N_TRAIN = 2000
N_VAL = 500
DIRECTION_LOSS_WEIGHT = 0.3
CONTRASTIVE_LOSS_WEIGHT = 0.15   # weight for NT-Xent contrastive loss
CONTRASTIVE_TEMPERATURE = 0.15

# ── Heavy-hitter toggles ──
USE_SYNTHETIC_PRETRAIN = True    # generate synthetic data for pre-training
SYNTHETIC_CANDLES = 2000000      # total synthetic candles to generate (2M)
CURRICULUM_SYNTHETIC = 1         # 1 epoch synthetic first, then real
USE_CONTRASTIVE_LOSS = True     # add NT-Xent contrastive loss
WEIGHTED_SAMPLING = True         # sample higher timeframes more
TIMEFRAME_WEIGHTS = {            # how much to weight each timeframe
    '5m':  1, '15m': 1, '30m': 1,
    '1h':  2, '4h':  3, '1d':  5,
}

# ── New heavy-hitters for overnight run ──
LABEL_SMOOTHING = 0.1            # label smoothing epsilon (0 = off)
FEATURE_DROPOUT = 0.1            # randomly drop 10% of input features per batch
EPOCHS = 12                      # overnight: 12 epochs ~ 9 hours

# ── Kronos training features ──
GRAD_ACCUM_STEPS = 4            # accumulate gradients over N steps before optimizer.step()
EMA_DECAY = 0.995               # exponential moving average decay for model weights
PROFIT_LOSS_WEIGHT = 1.0        # weight for profit-based loss component (0 = disabled)
COSINE_RESTART_EPOCHS = 4       # cosine annealing restarts every N epochs

# ============================================================
# DATA: Timeframe-aware weighting
# ============================================================
def _extract_tf(name):
    """Extract timeframe from key like 'EUR/USD_5m'."""
    return name.split('_')[-1] if '_' in name else '1h'


class FxDataset(Dataset):
    """
    Loads pre-computed data with ALL features (base + indicators).
    Auto-detects feature columns. Supports weighted sampling by timeframe.
    """
    def __init__(self, data_path=None, raw_data=None, n_samples=2000,
                 weight_by_tf=False, tf_weights=None, label=''):
        assert data_path or raw_data is not None, "Need data_path or raw_data"

        if data_path:
            with open(data_path, 'rb') as f:
                raw = pickle.load(f)
        else:
            raw = raw_data

        self.window = LOOKBACK + PREDICT_LEN + 1
        self.tfeats = ['minute', 'hour', 'weekday', 'day', 'month']
        self.exclude_cols = {'datetime'} | set(self.tfeats)
        self.samples = []
        self.data = {}
        self.weights = []

        for sym, df in raw.items():
            df = df.reset_index(drop=True)
            dt = pd.to_datetime(df['datetime'])
            d = pd.DataFrame()

            all_cols = set(df.columns.tolist()) - self.exclude_cols
            self.feats = sorted(all_cols)

            for f in self.feats:
                d[f] = df[f].values.astype(np.float32)
            d['minute'] = dt.dt.minute.values
            d['hour'] = dt.dt.hour.values
            d['weekday'] = dt.dt.weekday.values
            d['day'] = dt.dt.day.values
            d['month'] = dt.dt.month.values

            self.data[sym] = d
            tf = _extract_tf(sym)
            tf_w = tf_weights.get(tf, 1) if tf_weights else 1

            for _ in range(len(d) - self.window + 1):
                self.samples.append((sym, len(self.samples)))
                self.weights.append(tf_w)

        self.n = min(n_samples, len(self.samples))
        self.rng = random.Random(42)
        self._weights = np.array(self.weights, dtype=np.float32)
        n_feats = len(self.feats)
        print(f"  [{label}] {len(self.data)} symbols, {len(self.samples)} windows, "
              f"{self.n}/epoch, {n_feats} features"
              + (" (weighted)" if weight_by_tf else ""))

    def set_epoch_seed(self, epoch):
        self.rng.seed(42 + epoch)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        idx = self.rng.choices(range(len(self.samples)), weights=self._weights, k=1)[0]
        sym, _ = self.samples[idx]
        start = self.rng.randint(0, len(self.data[sym]) - self.window)
        win = self.data[sym].iloc[start:start + self.window]
        x = win[self.feats].values.astype(np.float32)
        xs = win[self.tfeats].values.astype(np.float32)
        past = x[:LOOKBACK]
        m, s = past.mean(0), past.std(0)
        x = np.clip((x - m) / (s + 1e-5), -CLIP_VAL, CLIP_VAL)
        return torch.from_numpy(x), torch.from_numpy(xs)


# ============================================================
# SYNTHETIC DATA PIPELINE
# ============================================================
def generate_synthetic_for_training(n_candles=500000):
    """Generate synthetic data, compute indicators, return same dict format as prepare_data."""
    from finetune.synthetic_data import generate_multi_regime_dataset

    print(f"\n[{datetime.now().isoformat()}] 🧬 Generating synthetic market data...")
    t0 = time.time()

    df = generate_multi_regime_dataset(total_candles=n_candles, pair_type='forex')

    # Compute indicators exactly like prepare_data does
    print(f"  Computing indicators on synthetic data...")
    df = compute_all_indicators(df)
    df = df.dropna()
    elapsed = time.time() - t0
    print(f"  Synthetic data ready: {len(df):,} candles ({elapsed:.1f}s)")

    # Return in same format as prepare_data: {key: df}
    return {'SYNTHETIC_5m': df}


def build_synthetic_dataset(n_candles=500000, weight_by_tf=False, tf_weights=None):
    """Generate synthetic data and return FxDataset."""
    raw = generate_synthetic_for_training(n_candles)
    ds = FxDataset(
        raw_data=raw, n_samples=N_TRAIN,
        weight_by_tf=weight_by_tf, tf_weights=tf_weights,
        label='SYNTH',
    )
    return ds


# ============================================================
# LOSS FUNCTIONS
# ============================================================
def compute_direction_loss(logits, out0, out1, s1_bits=10):
    """
    Penalize wrong directional predictions.
    Differentiable: uses soft probabilities, not hard 0/1.
    """
    probs = torch.softmax(logits[0], dim=-1)
    half = 2 ** (s1_bits - 1)
    pos_prob = probs[..., half:].sum(dim=-1)  # differentiable prob of upward move
    actual_dir = (out0 > half).float()
    dir_loss = nn.functional.binary_cross_entropy(pos_prob, actual_dir, reduction='mean')
    return dir_loss


def compute_contrastive_loss(model, x, xs, tokenizer, temperature=0.15):
    """
    NT-Xent contrastive loss.
    Augments input → encodes both → pulls positive pairs together.
    """
    from finetune.contrastive import augment_ohlcv, nt_xent_loss

    with torch.no_grad():
        t0t_orig, _ = tokenizer.encode(x, half=True)

    x_aug = augment_ohlcv(x)
    with torch.no_grad():
        t0t_aug, _ = tokenizer.encode(x_aug, half=True)

    # Concatenate for single forward pass
    t0t = torch.cat([t0t_orig, t0t_aug], dim=0)
    xs_double = torch.cat([xs, xs], dim=0)
    logits = model(t0t[:, :-1], t0t[:, :-1], xs_double[:, :-1, :])

    # Pool encoder output to get embeddings
    pooled = logits[0].mean(dim=1)
    return nt_xent_loss(pooled, temperature)


def compute_label_smoothed_loss(logits, targets, vocab_size, smoothing=0.1):
    """
    Cross-entropy with label smoothing.
    Reduces overfitting by softening the target distribution.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    with torch.no_grad():
        # Smoothed targets: (1-smoothing) * one_hot + smoothing / vocab_size
        smooth_targets = torch.full_like(log_probs, smoothing / (vocab_size - 1))
        smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - smoothing)
    # KLD between log-probs and smoothed targets
    loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
    return loss


def compute_profit_loss(logits, out0, out1, s1_bits=10, half_divisor=False):
    """
    Differentiable profit-based loss.

    Uses tokenized price change direction to compute an approximate return:
    return = (out0 - half) / half   (fraction of midpoint)

    The model learns to maximize expected profit by taking long/short positions.
    """
    half = 2 ** (s1_bits - 1)
    pred_probs = torch.softmax(logits[0], dim=-1)
    pos_prob = pred_probs[..., half:].sum(dim=-1)  # differentiable prob of upward move
    action = 2 * pos_prob - 1  # +1 = long, -1 = short

    # Actual return: (token_value - midpoint) / midpoint
    if half_divisor:
        batch_returns = (out0.float() - half) / half
    else:
        batch_returns = (out0 > half).float() * 2 - 1

    profit = (action * batch_returns).mean()
    return -profit  # minimize negative profit = maximize profit


def apply_feature_dropout(x, dropout_rate=0.1):
    """Randomly drop feature dimensions. Forces model to learn from ALL indicators."""
    if dropout_rate <= 0:
        return x
    mask = torch.rand(x.shape[-1], device=x.device) > dropout_rate
    # Keep the proportion of alive features consistent
    scale = 1.0 / (1.0 - dropout_rate)
    return x * mask.float() * scale


# ============================================================
# CONFIDENCE TRACKING
# ============================================================
def track_confidence(logits, out0, s1_bits=10):
    """Track prediction confidence vs correctness."""
    results = []
    probs = torch.softmax(logits[0], dim=-1)
    for b in range(probs.shape[0]):
        for t in range(probs.shape[1]):
            token_probs = probs[b, t]
            confidence = token_probs.max().item()
            correct = 1 if token_probs.argmax().item() == out0[b, t].item() else 0
            results.append((confidence, correct))
    return results


def print_confidence(epoch_confidence):
    """Print accuracy by confidence bucket."""
    arr = np.array(epoch_confidence)
    if len(arr) == 0:
        return "no data"
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0)]
    parts = []
    for lo, hi in buckets:
        mask = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        sub = arr[mask]
        if len(sub) > 0:
            parts.append(f"{lo:.0f}-{hi:.0f}:{sub[:,1].mean()*100:.0f}%")
    parts.append(f"overall:{arr[:,1].mean()*100:.0f}%")
    return " | ".join(parts)


# ============================================================
# MAIN TRAINING
# ============================================================
def fetch_fresh_data():
    print(f"[{datetime.now().isoformat()}] 📡 Fetching real market data...")
    prepare_script = os.path.join(FINETUNE_DIR, 'prepare_data.py')
    result = subprocess.run([sys.executable, prepare_script],
                            capture_output=True, text=True, cwd=FINETUNE_DIR)
    if result.returncode != 0:
        print("❌ Data fetch failed:")
        print(result.stderr[-500:])
        return False
    print("✅ Data fetch complete")
    return True


def train():
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now().isoformat()}] 🏋️ Training on {device}")

    # ── Load model config ──
    with open(os.path.join(BOT_MODEL_DIR, 'config.json')) as f:
        model_cfg = json.load(f)

    # ── Detect D_IN from real data ──
    train_path = os.path.join(DATASET_PATH, 'train_data.pkl')
    val_path = os.path.join(DATASET_PATH, 'val_data.pkl')
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("❌ Dataset files not found. Run prepare_data.py first.")
        return False

    with open(train_path, 'rb') as f:
        sample_raw = pickle.load(f)
    feat_cols = sorted(set(sample_raw[list(sample_raw.keys())[0]].columns) - {'datetime', 'minute', 'hour', 'weekday', 'day', 'month'})
    D_IN = len(feat_cols)
    model_cfg['d_in'] = D_IN
    s1_bits = model_cfg.get('s1_bits', 10)
    print(f"  D_IN={D_IN} ({D_IN - 6} indicators + 6 base)")

    # ── Init tokenizer ──
    tokenizer = KronosTokenizer(
        d_in=D_IN, d_model=model_cfg.get('d_model', 832),
        n_heads=model_cfg.get('n_heads', 16), ff_dim=model_cfg.get('ff_dim', 2048),
        n_enc_layers=model_cfg.get('n_enc_layers', model_cfg.get('n_layers', 12)//2),
        n_dec_layers=model_cfg.get('n_dec_layers', model_cfg.get('n_layers', 12)//2),
        ffn_dropout_p=model_cfg.get('ffn_dropout_p', 0.2),
        attn_dropout_p=model_cfg.get('attn_dropout_p', 0.0),
        resid_dropout_p=model_cfg.get('resid_dropout_p', 0.2),
        s1_bits=s1_bits, s2_bits=model_cfg.get('s2_bits', 10),
        beta=model_cfg.get('beta', 0.05), gamma0=model_cfg.get('gamma0', 1.0),
        gamma=model_cfg.get('gamma', 1.1), zeta=model_cfg.get('zeta', 0.05),
        group_size=model_cfg.get('group_size', 4)
    ).to(device).eval()

    # ── Init model ──
    model = Kronos(
        s1_bits=s1_bits, s2_bits=model_cfg.get('s2_bits', 10),
        n_layers=model_cfg.get('n_layers', 12), d_model=model_cfg.get('d_model', 832),
        n_heads=model_cfg.get('n_heads', 16), ff_dim=model_cfg.get('ff_dim', 2048),
        ffn_dropout_p=model_cfg.get('ffn_dropout_p', 0.2),
        attn_dropout_p=model_cfg.get('attn_dropout_p', 0.0),
        resid_dropout_p=model_cfg.get('resid_dropout_p', 0.2),
        token_dropout_p=model_cfg.get('token_dropout_p', 0.0),
        learn_te=model_cfg.get('learn_te', True)
    )

    # ── Load any matching weights ──
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(BOT_MODEL_DIR, 'model.safetensors'))
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).train()
    loaded = len(filtered)
    total = len(model_state)
    missing = total - loaded
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params ({loaded}/{total} keys, {missing} new)")

    # ── Build datasets ──
    tf_weights = TIMEFRAME_WEIGHTS if WEIGHTED_SAMPLING else None

    # Real data loaders
    train_ds_real = FxDataset(data_path=train_path, n_samples=N_TRAIN,
                              weight_by_tf=WEIGHTED_SAMPLING, tf_weights=tf_weights, label='REAL')
    val_ds = FxDataset(data_path=val_path, n_samples=N_VAL, label='VAL')
    train_loader_real = DataLoader(train_ds_real, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Synthetic data loader (if enabled)
    train_loader_synth = None
    if USE_SYNTHETIC_PRETRAIN:
        synth_ds = build_synthetic_dataset(SYNTHETIC_CANDLES, WEIGHTED_SAMPLING, tf_weights)
        train_loader_synth = DataLoader(synth_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    total_batches = len(train_loader_real)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=total_batches * COSINE_RESTART_EPOCHS, T_mult=1, eta_min=1e-7)

    # ── EMA weights ──
    ema_model = None
    if EMA_DECAY > 0:
        ema_model = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        print(f"  EMA decay: {EMA_DECAY} (maintaining shadow weights)")

    # ── Features enabled ──
    features = []
    if USE_SYNTHETIC_PRETRAIN:
        features.append(f"synthetic({SYNTHETIC_CANDLES//1000}k)")
    if USE_CONTRASTIVE_LOSS:
        features.append("contrastive")
    if DIRECTION_LOSS_WEIGHT > 0:
        features.append("direction")
    if WEIGHTED_SAMPLING:
        features.append("weighted")
    if LABEL_SMOOTHING > 0:
        features.append(f"label_smooth({LABEL_SMOOTHING})")
    if FEATURE_DROPOUT > 0:
        features.append(f"feat_dropout({FEATURE_DROPOUT})")
    if GRAD_ACCUM_STEPS > 1:
        features.append(f"grad_accum({GRAD_ACCUM_STEPS})")
    if PROFIT_LOSS_WEIGHT > 0:
        features.append(f"profit_loss({PROFIT_LOSS_WEIGHT})")

    print(f"  Features: {' + '.join(features)}")
    print(f"  Training: {total_batches} batches/epoch × {EPOCHS} epochs")

    # ── Training loop ──
    confidence_data = []
    best_val_loss = float('inf')
    optimizer.zero_grad()  # ensure clean gradient state before accumulation

    for epoch in range(EPOCHS):
        train_ds_real.set_epoch_seed(epoch)
        model.train()
        total_loss = 0
        total_dir = 0
        total_contra = 0
        total_profit = 0
        n_batches = 0
        accum_step = 0
        t0 = time.time()

        # Determine data source for this epoch
        if USE_SYNTHETIC_PRETRAIN and epoch < CURRICULUM_SYNTHETIC and train_loader_synth:
            loader = train_loader_synth
            data_src = 'SYNTH'
        else:
            loader = train_loader_real
            data_src = 'REAL'

        for x, xs in loader:
            x, xs = x.to(device), xs.to(device)
            
            # Apply feature dropout (forces model to use ALL indicators)
            x = apply_feature_dropout(x, FEATURE_DROPOUT)

            with torch.no_grad():
                t0t, t1t = tokenizer.encode(x, half=True)

            inp0, inp1 = t0t[:, :-1], t1t[:, :-1]
            out0, out1 = t0t[:, 1:], t1t[:, 1:]
            logits = model(inp0, inp1, xs[:, :-1, :])

            # ── Main prediction loss (with label smoothing) ──
            if LABEL_SMOOTHING > 0:
                vocab_s1 = 2 ** s1_bits
                main_loss = compute_label_smoothed_loss(
                    logits[0].reshape(-1, vocab_s1),
                    out0.reshape(-1),
                    vocab_s1, smoothing=LABEL_SMOOTHING
                )
            else:
                main_loss, _, _ = model.head.compute_loss(logits[0], logits[1], out0, out1)
            loss = main_loss

            # ── Direction loss ──
            if DIRECTION_LOSS_WEIGHT > 0:
                dl = compute_direction_loss(logits, out0, out1, s1_bits)
                loss = loss + DIRECTION_LOSS_WEIGHT * dl
                total_dir += dl.item()

            # ── Contrastive loss (runs on real data too) ──
            if USE_CONTRASTIVE_LOSS:
                cl = compute_contrastive_loss(model, x, xs, tokenizer, CONTRASTIVE_TEMPERATURE)
                loss = loss + CONTRASTIVE_LOSS_WEIGHT * cl
                total_contra += cl.item()

            # ── Profit-based loss ──
            if PROFIT_LOSS_WEIGHT > 0:
                pl = compute_profit_loss(logits, out0, out1, s1_bits, half_divisor=True)
                loss = loss + PROFIT_LOSS_WEIGHT * pl
                total_profit += pl.item()

            # ── Gradient accumulation ──
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += main_loss.item()
            n_batches += 1

            if n_batches % 25 == 0:
                elapsed = time.time() - t0
                print(f"    E{epoch+1}/{EPOCHS} [{data_src}] {n_batches}/{total_batches} | "
                      f"loss:{total_loss/n_batches:.4f} | {elapsed:.0f}s")

        # Flush any remaining accumulated gradients
        if accum_step % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train = total_loss / n_batches

        # ── Update EMA weights ──
        if ema_model is not None:
            with torch.no_grad():
                current = model.state_dict()
                for k in ema_model:
                    ema_model[k] = ema_model[k].to(current[k].device)
                    ema_model[k] = EMA_DECAY * ema_model[k] + (1 - EMA_DECAY) * current[k]
                    ema_model[k] = ema_model[k].cpu()

        # ── Validation ──
        model.eval()
        val_loss = 0
        v_batches = 0
        epoch_conf = []

        with torch.no_grad():
            for x, xs in val_loader:
                x, xs = x.to(device), xs.to(device)
                t0t, t1t = tokenizer.encode(x, half=True)
                inp0, inp1 = t0t[:, :-1], t1t[:, :-1]
                out0, out1 = t0t[:, 1:], t1t[:, 1:]
                logits = model(inp0, inp1, xs[:, :-1, :])
                # Use same loss function as training for consistent metrics
                if LABEL_SMOOTHING > 0:
                    vocab_s1 = 2 ** s1_bits
                    vloss = compute_label_smoothed_loss(
                        logits[0].reshape(-1, vocab_s1),
                        out0.reshape(-1),
                        vocab_s1, smoothing=LABEL_SMOOTHING
                    )
                else:
                    vloss, _, _ = model.head.compute_loss(logits[0], logits[1], out0, out1)
                val_loss += vloss.item()
                v_batches += 1
                epoch_conf.extend(track_confidence(logits, out0, s1_bits))

        avg_val = val_loss / v_batches

        # Summary
        dir_str = f" dir:{total_dir/n_batches:.4f}" if DIRECTION_LOSS_WEIGHT > 0 else ""
        contra_str = f" cont:{total_contra/n_batches:.4f}" if USE_CONTRASTIVE_LOSS and total_contra > 0 else ""
        profit_str = f" profit:{total_profit/n_batches:.4f}" if PROFIT_LOSS_WEIGHT > 0 and total_profit > 0 else ""
        conf_str = print_confidence(epoch_conf)
        print(f"  E{epoch+1}/{EPOCHS} | {data_src} | train:{avg_train:.6f}{dir_str}{contra_str}{profit_str} | "
              f"val:{avg_val:.6f} | lr:{scheduler.get_last_lr()[0]:.2e}")
        print(f"  Conf: {conf_str}")

        confidence_data.extend(epoch_conf)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            print(f"  🏆 New best (val_loss: {avg_val:.6f})")

    # ── Save ──
    print(f"[{datetime.now().isoformat()}] 💾 Saving...")
    os.makedirs(BOT_MODEL_DIR, exist_ok=True)
    safetensors_save({k: v.cpu() for k, v in model.state_dict().items()},
                     os.path.join(BOT_MODEL_DIR, 'model.safetensors'))
    safetensors_save({k: v.cpu() for k, v in tokenizer.state_dict().items()},
                     os.path.join(BOT_MODEL_DIR, 'tokenizer.safetensors'))
    with open(os.path.join(BOT_MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(model_cfg, f, indent=2)

    # ── Save EMA weights ──
    if ema_model is not None:
        ema_path = os.path.join(BOT_MODEL_DIR, 'model_ema.safetensors')
        safetensors_save(ema_model, ema_path)
        print(f"  💾 EMA weights saved to {ema_path}")

    # Confidence CSV
    if confidence_data:
        with open(CONFIDENCE_LOG, 'w') as f:
            w = csv.writer(f)
            w.writerow(['confidence', 'correct'])
            for c, ok in confidence_data:
                w.writerow([f'{c:.4f}', ok])

    print(f"  ✅ Model + tokenizer saved to {BOT_MODEL_DIR}")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print(f"  KRONOS — {datetime.now().isoformat()}")
    print("=" * 60)
    print(f"  Synthetic: {USE_SYNTHETIC_PRETRAIN} ({SYNTHETIC_CANDLES//1000}k)")
    print(f"  Contrastive: {USE_CONTRASTIVE_LOSS}")
    print(f"  Direction loss: {DIRECTION_LOSS_WEIGHT}")
    print(f"  Weighted sampling: {WEIGHTED_SAMPLING}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Feature dropout: {FEATURE_DROPOUT}")
    print(f"  Epochs: {EPOCHS}")
    print("=" * 60)

    if not fetch_fresh_data():
        print("❌ Aborted: data fetch failed")
        sys.exit(1)
    if not train():
        print("❌ Aborted: training failed")
        sys.exit(1)

    print("✅ Training complete — copy model + tokenizer to bot directory")