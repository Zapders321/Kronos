#!/usr/bin/env python3
"""
Backtest: Compare vanilla Kronos-base vs fine-tuned Kronos-small
on the same validation data to see which predicts better.
Uses MPS (Apple Silicon) for faster inference.
"""
import sys, time, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor

DATA_DIR = Path(__file__).resolve().parent.parent / 'finetune' / 'data' / 'processed_datasets'
N_TESTS = 10  # test windows per symbol (keep low for CPU speed)

# Fine-tuned Kronos-small path
FINETUNED_SMALL = str(Path(__file__).resolve().parent.parent / 'outputs' / 'models' / 'finetune_forex_crypto' / 'checkpoints' / 'best_model')

print("=" * 60)
print("KRONOS MODEL COMPARISON BACKTEST")
print("=" * 60)

# ── Load both models ──
print("\n📦 Loading models...")
print("  Model A: Kronos-base (vanilla, 102M params)")
print("  Model B: Kronos-small (fine-tuned on forex/crypto, 25M params)\n")

t0 = time.time()
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

model_base = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor_base = KronosPredictor(model_base, tokenizer, max_context=512)
print(f"  ✓ Model A loaded in {time.time()-t0:.1f}s")

t0 = time.time()
model_small = Kronos.from_pretrained(FINETUNED_SMALL)
predictor_small = KronosPredictor(model_small, tokenizer, max_context=512)
print(f"  ✓ Model B loaded in {time.time()-t0:.1f}s")

# ── Load validation data ──
print("\n📊 Loading validation data...")
with open(DATA_DIR / 'val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)

print(f"  {sum(len(v) for v in val_data.values()):,} rows across {len(val_data)} symbols\n")

# ── Run comparison ──
overall_base_correct = 0
overall_base_total = 0
overall_small_correct = 0
overall_small_total = 0

results = {}
for symbol, df in val_data.items():
    df = df.reset_index(drop=True)
    n = len(df)
    
    lookback = 400
    pred_len = 20
    
    if n < lookback + pred_len:
        continue
    
    rng = random.Random(42)
    base_correct = 0
    small_correct = 0
    total = 0
    base_times = []
    small_times = []
    
    for test_idx in range(N_TESTS):
        max_start = n - lookback - pred_len
        start = rng.randint(0, max_start)
        
        x_df = df.iloc[start:start + lookback][['open', 'high', 'low', 'close', 'vol', 'amt']]
        y_true = df.iloc[start + lookback:start + lookback + pred_len]['close'].values
        x_ts = pd.Series(df.iloc[start:start + lookback]['datetime'])
        y_ts = pd.Series(df.iloc[start + lookback:start + lookback + pred_len]['datetime'])
        
        actual_dir = 1 if (y_true[-1] - y_true[0]) > 0 else -1
        
        # Model A: Kronos-base
        t1 = time.time()
        pred_base = predictor_base.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=5,
        )
        base_times.append(time.time() - t1)
        base_dir = 1 if (pred_base['close'].values[-1] - y_true[0]) > 0 else -1
        if base_dir == actual_dir: base_correct += 1
        total += 1
        
        # Model B: Fine-tuned Kronos-small
        t2 = time.time()
        pred_small = predictor_small.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=5,
        )
        small_times.append(time.time() - t2)
        small_dir = 1 if (pred_small['close'].values[-1] - y_true[0]) > 0 else -1
        if small_dir == actual_dir: small_correct += 1
    
    base_acc = base_correct / total * 100
    small_acc = small_correct / total * 100
    base_avg_ms = np.mean(base_times) * 1000
    small_avg_ms = np.mean(small_times) * 1000
    
    results[symbol] = {
        'base_acc': base_acc, 'small_acc': small_acc,
        'base_time': base_avg_ms, 'small_time': small_avg_ms,
    }
    
    winner = "Base" if base_acc > small_acc else "Small" if small_acc > base_acc else "Tie"
    print(f"  {symbol:<10}  Base: {base_acc:5.1f}%  Small: {small_acc:5.1f}%  → {winner} wins")
    
    overall_base_correct += base_correct
    overall_base_total += total
    overall_small_correct += small_correct
    overall_small_total += total

# ── Final Results ──
print(f"\n{'='*60}")
print("  OVERALL RESULTS")
print(f"{'='*60}")
base_all = overall_base_correct / overall_base_total * 100
small_all = overall_small_correct / overall_small_total * 100
print(f"  Kronos-base (vanilla):         {base_all:.1f}% direction accuracy")
print(f"  Kronos-small (fine-tuned):     {small_all:.1f}% direction accuracy")

if base_all > small_all:
    print(f"\n  ✅ Kronos-base wins by {base_all - small_all:.1f}%")
    print(f"     Better to stick with what the bot already uses")
else:
    print(f"\n  ✅ Kronos-small wins by {small_all - base_all:.1f}%")
    print(f"     Worth switching the bot to use this model")

print(f"\n  Speed per prediction:")
print(f"  Kronos-base:     {np.mean([r['base_time'] for r in results.values()]):.0f}ms")
print(f"  Kronos-small:    {np.mean([r['small_time'] for r in results.values()]):.0f}ms")
print(f"{'='*60}")