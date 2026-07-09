#!/usr/bin/env python3
"""
Fine-tune Kronos predictor on forex/crypto data.
Runs on MPS (Apple Silicon) without DDP or Comet ML.
Usage: python3 finetune_predictor.py
"""
import os, sys, pickle, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.kronos import KronosTokenizer, Kronos

os.environ['COMET_MODE'] = 'DISABLED'

# ── Config ──
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 4e-5
LOOKBACK = 90
PREDICT_LEN = 10
CLIP = 5.0
N_TRAIN = 2000
N_VAL = 500
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/processed_datasets')
SAVE_PATH = os.path.join(os.path.dirname(__file__), 'outputs/models/finetune_predictor_demo')


class FxDataset(Dataset):
    """Dataset for forex/crypto data from our pickle format."""
    def __init__(self, data_path, n_samples=2000):
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        self.window = LOOKBACK + PREDICT_LEN + 1
        self.feats = ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.tfeats = ['minute', 'hour', 'weekday', 'day', 'month']
        self.samples = []
        self.data = {}
        for sym, df in raw.items():
            df = df.reset_index(drop=True)
            dt = pd.to_datetime(df['datetime'])
            d = pd.DataFrame()
            for f in self.feats:
                d[f] = df[f].values
            d['minute'] = dt.dt.minute.values
            d['hour'] = dt.dt.hour.values
            d['weekday'] = dt.dt.weekday.values
            d['day'] = dt.dt.day.values
            d['month'] = dt.dt.month.values
            self.data[sym] = d
            for i in range(len(d) - self.window + 1):
                self.samples.append((sym, i))
        self.n = min(n_samples, len(self.samples))
        self.rng = random.Random(42)
        print(f"  {len(self.data)} symbols, {len(self.samples)} windows, {self.n}/epoch")

    def set_epoch_seed(self, epoch):
        self.rng.seed(42 + epoch)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        idx = self.rng.randint(0, len(self.samples) - 1)
        sym, start = self.samples[idx]
        win = self.data[sym].iloc[start:start + self.window]
        x = win[self.feats].values.astype(np.float32)
        xs = win[self.tfeats].values.astype(np.float32)
        past = x[:LOOKBACK]
        m, s = past.mean(0), past.std(0)
        x = np.clip((x - m) / (s + 1e-5), -CLIP, CLIP)
        return torch.from_numpy(x), torch.from_numpy(xs)


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer (frozen)
    print("Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    tokenizer.eval().to(device)
    print(f"  Params: {sum(p.numel() for p in tokenizer.parameters()):,}")

    # Load model (Kronos-small for faster training)
    print("Loading model (Kronos-small)...")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Datasets
    print("\nLoading data...")
    train_ds = FxDataset(os.path.join(DATASET_PATH, 'train_data.pkl'), n_samples=N_TRAIN)
    val_ds = FxDataset(os.path.join(DATASET_PATH, 'val_data.pkl'), n_samples=N_VAL)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LEARNING_RATE,
                                                 steps_per_epoch=len(train_loader), epochs=EPOCHS,
                                                 pct_start=0.03, div_factor=10)

    os.makedirs(SAVE_PATH, exist_ok=True)
    best_val = float('inf')
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"  Train samples: {N_TRAIN}, Val samples: {N_VAL}")
    print(f"  Save path: {SAVE_PATH}")
    print(f"{'='*60}")

    for ep in range(EPOCHS):
        model.train()
        train_ds.set_epoch_seed(ep)
        ep_loss = 0.0
        t0 = time.time()

        for bi, (bx, bxs) in enumerate(train_loader):
            bx = bx.to(device, non_blocking=True)
            bxs = bxs.to(device, non_blocking=True)

            with torch.no_grad():
                t0t, t1t = tokenizer.encode(bx, half=True)

            inp0, inp1 = t0t[:, :-1], t1t[:, :-1]
            out0, out1 = t0t[:, 1:], t1t[:, 1:]

            logits = model(inp0, inp1, bxs[:, :-1, :])
            loss, s1, s2 = model.head.compute_loss(logits[0], logits[1], out0, out1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            opt.step()
            sched.step()

            ep_loss += loss.item()
            global_step += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_cnt = 0
        with torch.no_grad():
            for bx, bxs in val_loader:
                bx = bx.to(device, non_blocking=True)
                bxs = bxs.to(device, non_blocking=True)
                t0t, t1t = tokenizer.encode(bx, half=True)
                inp0, inp1 = t0t[:, :-1], t1t[:, :-1]
                out0, out1 = t0t[:, 1:], t1t[:, 1:]
                logits = model(inp0, inp1, bxs[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], out0, out1)
                val_loss += loss.item()
                val_cnt += 1

        avg_val = val_loss / val_cnt
        avg_train = ep_loss / len(train_loader)
        elapsed = time.time() - t0

        print(f"E{ep+1:2d}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}  ({elapsed:.1f}s)")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'best_model.pt'))
            print(f"  ✓ Saved (val_loss={avg_val:.4f})")

    print(f"\n{'='*60}")
    print(f"Training complete! Best val_loss: {best_val:.4f}")
    print(f"Model saved to: {SAVE_PATH}/best_model.pt")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()