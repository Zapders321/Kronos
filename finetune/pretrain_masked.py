#!/usr/bin/env python3
"""
Self-supervised masked pre-training for Kronos.

Randomly masks 15% of timesteps in each input window and trains the model
to reconstruct the original features at masked positions using MSE loss.

Flow:
  1. Generate synthetic multi-regime data (or load real data)
  2. Create windows with random masking
  3. Train the model to reconstruct masked positions
  4. Save pre-trained weights

Usage:
  python3 finetune/pretrain_masked.py

Output:
  pretrain_outputs/pretrained_model/  — pre-trained model + tokenizer
"""
import os, sys, json, time, math, random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.kronos import KronosTokenizer, Kronos
from finetune.indicators import compute_all_indicators
from finetune.feedback_train import (
    FxDataset, generate_synthetic_for_training
)

# ── Prevent sleep on MacBook ──
os.system("caffeinate -d -i -m -u -t 86400 &>/dev/null & disown")

# ── MPS stability ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ============================================================
# CONFIG
# ============================================================
FINETUNE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(FINETUNE_DIR, '..')
BOT_PRETRAIN_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'pretrained_model', 'checkpoints')
DATASET_PATH = os.path.join(FINETUNE_DIR, 'data', 'processed_datasets')

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
LOOKBACK = 90
PREDICT_LEN = 10
WINDOW = LOOKBACK + PREDICT_LEN + 1
CLIP_VAL = 5.0
N_TRAIN = 3000
N_VAL = 500
EPOCHS = 8
MASK_RATIO = 0.15           # fraction of timesteps to mask
MASK_REPLACE_VALUE = 0.0    # value to replace masked positions with
GRAD_ACCUM_STEPS = 4
SYNTHETIC_CANDLES = 2000000  # total synthetic candles for pre-training


# ============================================================
# MASKED DATASET
# ============================================================
class MaskedFxDataset(Dataset):
    """
    Wraps an existing FxDataset-style dataset but applies random masking.
    Returns (masked_x, xs, original_x) where masked_x has 15% of time steps
    replaced with zeros.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.mask_ratio = MASK_RATIO
        self.mask_value = MASK_REPLACE_VALUE

    def set_epoch_seed(self, epoch):
        self.base.set_epoch_seed(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, xs = self.base[idx]
        # Create mask: randomly zero out MASK_RATIO of time steps
        seq_len = x.shape[0]
        n_mask = max(1, int(seq_len * self.mask_ratio))
        mask_indices = torch.randperm(seq_len)[:n_mask]
        masked_x = x.clone()
        masked_x[mask_indices] = self.mask_value
        return masked_x, xs, x  # return original as target


# ============================================================
# RECONSTRUCTION LOSS (not used — we use model.head.compute_loss directly)
# ============================================================ 


# ============================================================
# DATA LOADING
# ============================================================
def load_or_generate_data():
    """Try to load real data first; fall back to synthetic."""
    train_path = os.path.join(DATASET_PATH, 'train_data.pkl')
    val_path = os.path.join(DATASET_PATH, 'val_data.pkl')

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"  Loading real data from {DATASET_PATH}")
        train_ds = FxDataset(data_path=train_path, n_samples=N_TRAIN, label='PRETRAIN_TRAIN (real)')
        val_ds = FxDataset(data_path=val_path, n_samples=N_VAL, label='PRETRAIN_VAL (real)')
    else:
        print(f"  Real data not found. Generating {SYNTHETIC_CANDLES:,} synthetic candles...")
        from finetune.feedback_train import build_synthetic_dataset, TIMEFRAME_WEIGHTS, WEIGHTED_SAMPLING
        train_ds = build_synthetic_dataset(
            SYNTHETIC_CANDLES,
            weight_by_tf=False,  # single timeframe for synthetic
            tf_weights=None,
        )
        # Generate a separate val set
        val_synth = generate_synthetic_for_training(n_candles=200000)
        val_ds_raw = FxDataset(raw_data=val_synth, n_samples=N_VAL, label='PRETRAIN_VAL (synth)')
        val_ds = val_ds_raw

    return train_ds, val_ds


# ============================================================
# PRETRAINING
# ============================================================
def pretrain():
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now().isoformat()}] 🏋️ Pre-training on {device}")

    # ── Load model config from base bot model ──
    base_model_dir = os.path.join(PROJECT_ROOT, 'outputs', 'kronos_base_finetuned', 'checkpoints', 'best_model')
    with open(os.path.join(base_model_dir, 'config.json')) as f:
        model_cfg = json.load(f)

    # ── Detect D_IN ──
    train_ds, val_ds = load_or_generate_data()
    D_IN = len(train_ds.feats)
    model_cfg['d_in'] = D_IN
    s1_bits = model_cfg.get('s1_bits', 10)
    print(f"  D_IN={D_IN}")

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

    # ── Load any matching base weights ──
    base_ckpt = os.path.join(base_model_dir, 'model.safetensors')
    if os.path.exists(base_ckpt):
        state_dict = safetensors_load(base_ckpt)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        model = model.to(device).train()
        loaded = len(filtered)
        total = len(model_state)
        print(f"  Loaded {loaded}/{total} pretrained keys")
    else:
        model = model.to(device).train()

    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Build data loaders with masking ──
    train_masked = MaskedFxDataset(train_ds)
    val_masked = MaskedFxDataset(val_ds)
    train_loader = DataLoader(train_masked, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_masked, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total_batches = len(train_loader)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=total_batches * 2, T_mult=1, eta_min=1e-7
    )

    print(f"  Training: {total_batches} batches/epoch × {EPOCHS} epochs (mask_ratio={MASK_RATIO})")

    # ── Training loop ──
    best_val_loss = float('inf')
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        train_ds.set_epoch_seed(epoch)
        model.train()
        total_loss = 0
        n_batches = 0
        accum_step = 0
        t0 = time.time()

        for masked_x, xs, orig_x in train_loader:
            masked_x = masked_x.to(device)
            xs = xs.to(device)
            orig_x = orig_x.to(device)

            with torch.no_grad():
                # Encode original (clean) features to get target tokens
                t0t_orig, t1t_orig = tokenizer.encode(orig_x, half=True)
                # Encode masked features for input
                t0t_masked, t1t_masked = tokenizer.encode(masked_x, half=True)

            inp0 = t0t_masked[:, :-1]
            inp1 = t1t_masked[:, :-1]
            out0 = t0t_orig[:, 1:]  # Target: original tokens (what was masked)
            out1 = t1t_orig[:, 1:]

            logits = model(inp0, inp1, xs[:, :-1, :])

            # Standard cross-entropy loss — same as fine-tuning, but on original tokens
            vocab_s1 = 2 ** s1_bits
            loss, _, _ = model.head.compute_loss(logits[0], logits[1], out0, out1)

            # ── Gradient accumulation ──
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS  # undo division for logging
            n_batches += 1

            if n_batches % 25 == 0:
                elapsed = time.time() - t0
                print(f"    E{epoch+1}/{EPOCHS} {n_batches}/{total_batches} | "
                      f"loss:{total_loss/n_batches:.4f} | {elapsed:.0f}s")

        # Flush remaining gradients
        if accum_step % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train = total_loss / n_batches

        # ── Validation ──
        model.eval()
        val_loss = 0
        v_batches = 0
        with torch.no_grad():
            for masked_x, xs, orig_x in val_loader:
                masked_x = masked_x.to(device)
                xs = xs.to(device)
                orig_x = orig_x.to(device)

                t0t_orig, t1t_orig = tokenizer.encode(orig_x, half=True)
                t0t_masked, t1t_masked = tokenizer.encode(masked_x, half=True)

                inp0 = t0t_masked[:, :-1]
                inp1 = t1t_masked[:, :-1]
                out0 = t0t_orig[:, 1:]
                out1 = t1t_orig[:, 1:]

                logits = model(inp0, inp1, xs[:, :-1, :])
                vloss, _, _ = model.head.compute_loss(logits[0], logits[1], out0, out1)
                val_loss += vloss.item()
                v_batches += 1

        avg_val = val_loss / v_batches

        print(f"  E{epoch+1}/{EPOCHS} | train:{avg_train:.6f} | val:{avg_val:.6f} | "
              f"lr:{scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            print(f"  🏆 New best (val_loss: {avg_val:.6f})")
            # Save checkpoint
            os.makedirs(BOT_PRETRAIN_DIR, exist_ok=True)
            safetensors_save({k: v.cpu() for k, v in model.state_dict().items()},
                             os.path.join(BOT_PRETRAIN_DIR, 'model.safetensors'))
            safetensors_save({k: v.cpu() for k, v in tokenizer.state_dict().items()},
                             os.path.join(BOT_PRETRAIN_DIR, 'tokenizer.safetensors'))
            with open(os.path.join(BOT_PRETRAIN_DIR, 'config.json'), 'w') as f:
                json.dump(model_cfg, f, indent=2)
            print(f"  💾 Saved checkpoint to {BOT_PRETRAIN_DIR}")

    print(f"[{datetime.now().isoformat()}] ✅ Pre-training complete")
    print(f"  Best val_loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {BOT_PRETRAIN_DIR}")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print(f"  KRONOS MASKED PRE-TRAINING — {datetime.now().isoformat()}")
    print("=" * 60)
    print(f"  Mask ratio: {MASK_RATIO} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"  Synthetic candles: {SYNTHETIC_CANDLES:,}")
    print(f"  Grad accum: {GRAD_ACCUM_STEPS}")
    print("=" * 60)

    if not pretrain():
        print("❌ Pre-training failed")
        sys.exit(1)

    print("✅ Done")