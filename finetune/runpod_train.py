#!/usr/bin/env python3
"""
RunPod training script for Kronos with technical indicators.
Trains on a single GPU (no DDP). Handles tokenizer embed resizing.

Usage:
    python3 runpod_train.py --model base --epochs 10 --batch 32
"""
import os, sys, pickle, time, random, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.kronos import KronosTokenizer, Kronos
from finetune.indicators import get_indicator_feature_names, get_full_feature_list

os.environ['COMET_MODE'] = 'DISABLED'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# ── Args ──
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='base', choices=['small', 'base'])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=4e-5)
parser.add_argument('--lookback', type=int, default=90)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--n_train', type=int, default=5000)
parser.add_argument('--n_val', type=int, default=1000)
parser.add_argument('--data_dir', type=str, default='/workspace/Kronos/finetune/data/processed_datasets')
parser.add_argument('--output_dir', type=str, default='/workspace/Kronos/finetune/outputs/models/kronos_indicator_finetuned')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURES = get_full_feature_list()  # base + indicators
N_FEATURES = len(FEATURES)
TIMEFEATS = ['minute', 'hour', 'weekday', 'day', 'month']
TIMEFEAT_DIM = len(TIMEFEATS)

print(f"Device: {DEVICE}")
print(f"Features: {N_FEATURES} ({len(FEATURES)})")
print(f"  Base: open, high, low, close, vol, amt")
print(f"  Indicators: {len(get_indicator_feature_names())}")
print(f"Model: Kronos-{args.model}")
print(f"Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
print(f"Data: {args.data_dir}")
print(f"Output: {args.output_dir}")


# ── Dataset ──
class FxIndicatorDataset(Dataset):
    def __init__(self, data_path, n_samples=5000, lookback=90, pred_len=10):
        print(f"Loading {data_path}...")
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)

        self.lookback = lookback
        self.window = lookback + pred_len + 1
        self.samples = []
        self.data = {}

        for sym, df in raw.items():
            df = df.reset_index(drop=True)
            dt = pd.to_datetime(df['datetime'])

            d = pd.DataFrame()
            for f in FEATURES:
                d[f] = df[f].values.astype(np.float32)

            # Time features
            d['minute'] = dt.dt.minute.values.astype(np.float32)
            d['hour'] = dt.dt.hour.values.astype(np.float32)
            d['weekday'] = dt.dt.weekday.values.astype(np.float32)
            d['day'] = dt.dt.day.values.astype(np.float32)
            d['month'] = dt.dt.month.values.astype(np.float32)

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

        x = win[FEATURES].values.astype(np.float32)
        xs = win[TIMEFEATS].values.astype(np.float32)

        # Normalize on lookback window only (no future leakage)
        past = x[:self.lookback]
        m, s = past.mean(0), past.std(0)
        x = np.clip((x - m) / (s + 1e-5), -5.0, 5.0)

        return torch.from_numpy(x), torch.from_numpy(xs)


# ── Model with resized tokenizer ──
def load_model_with_resized_tokenizer(model_size: str):
    """Load tokenizer and resize embed layer for N_FEATURES inputs."""
    model_id = f"NeoQuasar/Kronos-{model_size}"

    print(f"Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(f"NeoQuasar/Kronos-Tokenizer-{model_size}")
    tokenizer.eval().to(DEVICE)

    # Resize embed layer to accept indicator-expanded features
    old_embed = tokenizer.embed
    old_weight = old_embed.weight.data  # shape (256, 6)

    new_embed = nn.Linear(N_FEATURES, old_embed.out_features, bias=old_embed.bias is not None)
    with torch.no_grad():
        # Copy old weights for first 6 features (OHLCV + amt)
        new_embed.weight[:, :6] = old_weight
        # Initialize new indicator weights small
        nn.init.xavier_uniform_(new_embed.weight[:, 6:], gain=0.01)
        if old_embed.bias is not None:
            new_embed.bias = old_embed.bias

    tokenizer.embed = new_embed.to(DEVICE)
    tokenizer.d_in = N_FEATURES
    print(f"  Resized embed: in_features={old_embed.in_features} -> {N_FEATURES}")

    print(f"Loading Kronos-{model_size}...")
    model = Kronos.from_pretrained(model_id)
    model.to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Also resize predictor if it uses the tokenizer output dimension
    total = sum(p.numel() for p in tokenizer.parameters()) + sum(p.numel() for p in model.parameters())
    print(f"  Total params (tokenizer + model): {total:,}")

    return tokenizer, model


# ── Training ──
def main():
    print(f"\n{'='*70}")
    print(f"KRONOS INDICATOR FINE-TUNE (RunPod)")
    print(f"{'='*70}")

    # Data
    print(f"\n--- Loading data ---")
    train_ds = FxIndicatorDataset(
        os.path.join(args.data_dir, 'train_data.pkl'),
        n_samples=args.n_train,
        lookback=args.lookback,
        pred_len=args.pred_len,
    )
    val_ds = FxIndicatorDataset(
        os.path.join(args.data_dir, 'val_data.pkl'),
        n_samples=args.n_val,
        lookback=args.lookback,
        pred_len=args.pred_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    print(f"\n--- Loading model ---")
    tokenizer, model = load_model_with_resized_tokenizer(args.model)

    # Optimizer
    opt = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    # Warmup + cosine schedule
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        steps_per_epoch=steps_per_epoch, epochs=args.epochs,
        pct_start=0.05, div_factor=10,
    )

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val = float('inf')
    global_step = 0
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Training: {args.epochs} epochs, {args.n_train} train/{args.n_val} val/epoch")
    print(f"{'='*70}")

    for ep in range(args.epochs):
        model.train()
        tokenizer.eval()  # Keep tokenizer in eval mode (but embed is trainable)
        train_ds.set_epoch_seed(ep)
        ep_loss = 0.0
        t0 = time.time()

        for bi, (bx, bxs) in enumerate(train_loader):
            bx = bx.to(DEVICE, non_blocking=True)
            bxs = bxs.to(DEVICE, non_blocking=True)

            # Tokenize: uses resized embed with all features
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

            if (bi + 1) % 200 == 0:
                elapsed = time.time() - t0
                lr_now = sched.get_last_lr()[0]
                print(f"  Ep {ep+1} step {bi+1}: loss={loss.item():.4f} lr={lr_now:.2e} ({elapsed:.0f}s)")

        # Validation
        model.eval()
        val_loss = 0.0
        val_cnt = 0
        with torch.no_grad():
            for bx, bxs in val_loader:
                bx = bx.to(DEVICE, non_blocking=True)
                bxs = bxs.to(DEVICE, non_blocking=True)
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
        total_elapsed = time.time() - start_time

        print(f"\nEpoch {ep+1}/{args.epochs}:")
        print(f"  train loss: {avg_train:.4f}  |  val loss: {avg_val:.4f}")
        print(f"  epoch time: {elapsed:.0f}s  |  total: {total_elapsed:.0f}s")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'tokenizer_state_dict': tokenizer.state_dict(),
                'val_loss': avg_val,
                'features': FEATURES,
                'args': vars(args),
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (val_loss={avg_val:.4f})")
            cfg = {
                'd_model': model.config.d_model if hasattr(model, 'config') else model.d_model if hasattr(model, 'd_model') else 0,
                'n_layers': model.config.n_layers if hasattr(model, 'config') else model.n_layers if hasattr(model, 'n_layers') else 0,
                'n_features': N_FEATURES,
                'features': FEATURES,
                'time_features': TIMEFEATS,
            }
            with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
                json.dump(cfg, f)

        # Log progress
        with open(os.path.join(args.output_dir, 'training_log.json'), 'a') as f:
            f.write(json.dumps({
                'epoch': ep + 1,
                'train_loss': round(avg_train, 4),
                'val_loss': round(avg_val, 4),
                'epoch_seconds': round(elapsed),
                'total_seconds': round(total_elapsed),
            }) + '\n')

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val_loss: {best_val:.4f}")
    print(f"  Total time: {time.time() - start_time:.0f}s")
    print(f"  Model: {checkpoint_dir}/best_model.pt")
    print(f"  Features ({N_FEATURES}): {', '.join(FEATURES)}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()