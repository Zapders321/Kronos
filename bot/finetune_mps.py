#!/usr/bin/env python3
"""
Kronos Predictor Fine-Tune — Single-Device (MPS/CPU)
Loads prepared pickle data (forex/crypto), fine-tunes Kronos-small predictor,
saves best model checkpoint.
"""
import os, sys, time, json, pickle, random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer

# ── Config ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / 'finetune' / 'data' / 'processed_datasets'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'models' / 'finetune_forex_crypto'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
LOG_DIR = OUTPUT_DIR / 'logs'

PRETRAINED_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
PRETRAINED_PREDICTOR = "NeoQuasar/Kronos-small"

# Training hyperparams
EPOCHS = 10
BATCH_SIZE = 16           # smaller for MPS memory
LEARNING_RATE = 4e-5
LOOKBACK = 90
PREDICT_WINDOW = 10
MAX_CONTEXT = 512
CLIP_VAL = 5.0
LOG_INTERVAL = 50
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_WEIGHT_DECAY = 0.1
SEED = 42
NUM_WORKERS = 0  # MPS doesn't support pin_memory, use 0 workers

# ── Dataset ─────────────────────────────────────────────
class PickleKlineDataset(Dataset):
    """Loads multi-symbol pickle data from prepare_data.py."""

    def __init__(self, data_path, data_type='train', lookback=LOOKBACK,
                 pred_window=PREDICT_WINDOW, clip=CLIP_VAL, seed=SEED):
        self.lookback = lookback
        self.pred_window = pred_window
        self.window = lookback + pred_window + 1
        self.clip_val = clip
        self.seed = seed
        self.py_rng = random.Random(seed)

        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        pkl_path = data_path / f'{data_type}_data.pkl'
        print(f"  Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        # Build index of (symbol, start_idx) pairs
        self.indices = []
        self.symbol_data = {}
        for symbol, df in raw_data.items():
            df = df.reset_index(drop=True)
            # Generate time features
            dt = df['datetime']
            df['minute'] = dt.dt.minute
            df['hour'] = dt.dt.hour
            df['weekday'] = dt.dt.weekday
            df['day'] = dt.dt.day
            df['month'] = dt.dt.month
            self.symbol_data[symbol] = df

            series_len = len(df)
            num_samples = series_len - self.window + 1
            if num_samples > 0:
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        self.n_samples = len(self.indices)
        print(f"  {data_type.upper()}: {len(self.symbol_data)} symbols, {self.n_samples} samples")

    def set_epoch_seed(self, epoch):
        self.py_rng.seed(self.seed + epoch)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random sample from the pool
        rand_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[rand_idx]

        df = self.symbol_data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Normalize on lookback window only (no future leakage)
        past = x[:self.lookback]
        x_mean = np.mean(past, axis=0)
        x_std = np.std(past, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip_val, self.clip_val)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)


# ── Training ────────────────────────────────────────────
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"KRONOS PREDICTOR FINE-TUNE")
    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # ── Load models ──
    print("Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(PRETRAINED_TOKENIZER).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False  # freeze tokenizer

    print("Loading predictor...")
    model = Kronos.from_pretrained(PRETRAINED_PREDICTOR).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Data ──
    print("\nLoading datasets...")
    train_ds = PickleKlineDataset(DATA_DIR, 'train')
    val_ds = PickleKlineDataset(DATA_DIR, 'val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
    print(f"  Train: {len(train_loader)} batches/epoch | Val: {len(val_loader)} batches")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader), epochs=EPOCHS,
        pct_start=0.03, div_factor=10,
    )

    # ── Training loop ──
    best_val_loss = float('inf')
    start_time = time.time()
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_ds.set_epoch_seed(epoch * 10000)
        val_ds.set_epoch_seed(0)

        epoch_train_loss = 0.0
        train_batches = 0

        for batch_x, batch_x_stamp in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Tokenize
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs/targets (autoregressive: predict next token)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            train_batches += 1
            global_step += 1

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(f"  [Epoch {epoch+1}/{EPOCHS} Step {global_step}] "
                      f"Loss: {loss.item():.4f} | S1: {s1_loss.item():.4f} | S2: {s2_loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed/60:.1f}m elapsed")

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                val_loss += loss.item()
                val_batches += 1

        avg_train = epoch_train_loss / max(train_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        print(f"  Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        print(f"  Epoch: {epoch_time:.1f}s | Total: {total_time/60:.1f}m")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = CHECKPOINT_DIR / 'best_model'
            model.save_pretrained(str(save_path))
            print(f"  ✅ Best model saved! (val_loss: {best_val_loss:.4f})")

        # Save epoch checkpoint
        ckpt_path = CHECKPOINT_DIR / f'epoch_{epoch+1}'
        model.save_pretrained(str(ckpt_path))

        print()

    # ── Summary ──
    total_time = time.time() - start_time
    summary = {
        'device': str(device),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'best_val_loss': best_val_loss,
        'total_time_min': round(total_time / 60, 2),
        'completed_at': datetime.now().isoformat(),
        'pretrained_predictor': PRETRAINED_PREDICTOR,
        'pretrained_tokenizer': PRETRAINED_TOKENIZER,
        'best_model_path': str(CHECKPOINT_DIR / 'best_model'),
    }
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Model saved: {CHECKPOINT_DIR / 'best_model'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()