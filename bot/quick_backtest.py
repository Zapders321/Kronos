#!/usr/bin/env python3
"""Quick backtest: test Kronos-base prediction accuracy on validation data."""
import sys, time, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor

DATA_DIR = Path(__file__).resolve().parent.parent / 'finetune' / 'data' / 'processed_datasets'
N_TESTS = 20  # how many random windows to test per symbol

print("Loading model...")
t0 = time.time()
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)
print(f"  Loaded in {time.time()-t0:.1f}s")

print("Loading validation data...")
with open(DATA_DIR / 'val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)

results = {}
for symbol, df in val_data.items():
    df = df.reset_index(drop=True)
    n = len(df)
    print(f"\n{symbol}: {n} rows")
    
    lookback = 400
    pred_len = 20
    
    if n < lookback + pred_len:
        print(f"  SKIP: too few rows")
        continue
    
    correct_directions = 0
    total_tests = 0
    test_results = []
    
    rng = random.Random(42)
    
    for _ in range(N_TESTS):
        max_start = n - lookback - pred_len
        start = rng.randint(0, max_start)
        
        x_df = df.iloc[start:start + lookback][['open', 'high', 'low', 'close', 'vol', 'amt']]
        y_true_df = df.iloc[start + lookback:start + lookback + pred_len]
        
        x_ts = pd.Series(df.iloc[start:start + lookback]['datetime'])
        y_ts = pd.Series(df.iloc[start + lookback:start + lookback + pred_len]['datetime'])
        
        t1 = time.time()
        pred = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=5,
        )
        inf = time.time() - t1
        
        pred_move = pred['close'].values[-1] - y_true_df['close'].values[0]
        actual_move = y_true_df['close'].values[-1] - y_true_df['close'].values[0]
        
        pred_dir = 1 if pred_move > 0 else -1
        actual_dir = 1 if actual_move > 0 else -1
        
        # Per-step direction accuracy
        pred_steps = np.sign(np.diff(pred['close'].values, prepend=y_true_df['close'].values[0]))
        actual_steps = np.sign(np.diff(y_true_df['close'].values, prepend=y_true_df['close'].values[0]))
        step_acc = (pred_steps[1:] == actual_steps[1:]).mean()
        
        correct = pred_dir == actual_dir
        if correct:
            correct_directions += 1
        total_tests += 1
        
        test_results.append({
            'correct': correct,
            'pred_dir': pred_dir,
            'actual_dir': actual_dir,
            'step_acc': step_acc,
            'pred_move_pct': pred_move / y_true_df['close'].values[0] * 100,
            'actual_move_pct': actual_move / y_true_df['close'].values[0] * 100,
            'inf_time': inf,
        })
    
    dir_acc = correct_directions / total_tests * 100
    avg_step_acc = np.mean([r['step_acc'] for r in test_results]) * 100
    
    results[symbol] = {
        'dir_acc': dir_acc,
        'avg_step_acc': avg_step_acc,
        'n_tests': total_tests,
        'avg_inf_time': np.mean([r['inf_time'] for r in test_results]),
    }
    
    print(f"  Direction accuracy: {dir_acc:.0f}% ({correct_directions}/{total_tests})")
    print(f"  Step accuracy: {avg_step_acc:.0f}%")
    print(f"  Avg inference: {results[symbol]['avg_inf_time']:.1f}s")

print(f"\n{'='*60}")
print("BACKTEST SUMMARY")
print(f"{'='*60}")
print(f"{'Symbol':<12} {'Dir Acc':>10} {'Step Acc':>10} {'Tests':>8} {'Inf/s':>8}")
print(f"{'-'*50}")

total_correct = 0
total_tests = 0
for sym, r in sorted(results.items()):
    total_correct += r['dir_acc'] * r['n_tests'] / 100
    total_tests += r['n_tests']
    print(f"{sym:<12} {r['dir_acc']:>9.1f}% {r['avg_step_acc']:>9.1f}% {r['n_tests']:>8} {r['avg_inf_time']:>7.1f}s")

overall = total_correct / total_tests * 100 if total_tests else 0
print(f"{'-'*50}")
print(f"{'OVERALL':<12} {overall:>9.1f}%")
print(f"{'='*60}")