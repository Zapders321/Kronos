#!/usr/bin/env python3
"""
Backtest using the bot's own data pipeline (engine functions).
Runs predictions on recent data for all pairs/timeframes.
"""
import sys, time, json, random
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Import bot's engine functions
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bot'))
from engine import pull_data, run_prediction, get_direction, get_pred_cfg, get_predictor, TIMEFRAMES, ALL_PAIRS

# Disable engine logging to stdout
import logging
logging.getLogger('engine').setLevel(logging.CRITICAL)

N_TESTS = 20


def main():
    print("=" * 60)
    print("  KRONOS BACKTEST — Bot Engine Pipeline")
    print("=" * 60)

    print("Loading model...")
    predictor = get_predictor()
    print(f"  Model loaded")

    results = {}
    for pair_name, symbol in ALL_PAIRS.items():
        print(f"\n[{pair_name}] ({symbol})")
        pair_results = {}

        pred_cfg = get_pred_cfg(pair_name)

        for tf_name, tf_cfg in TIMEFRAMES.items():
            interval = tf_cfg['interval']
            period = tf_cfg['period']
            lookback = tf_cfg['lookback']
            pred_len = tf_cfg['pred_len']

            df = pull_data(symbol, interval, period)
            if df is None or len(df) < lookback + pred_len + 5:
                print(f"  {tf_name:4s}: insufficient data ({len(df) if df is not None else 0} rows)")
                continue

            rng = random.Random(42)
            correct = 0
            total = 0
            inf_times = []
            step_accs = []

            for _ in range(N_TESTS):
                max_start = len(df) - lookback - pred_len
                start = rng.randint(0, max_start)

                x_df = df.iloc[start:start + lookback]
                y_true_df = df.iloc[start + lookback:start + lookback + pred_len]

                t1 = time.time()
                try:
                    x_df_input = x_df.reset_index(drop=True)
                    x_ts = pd.Series(x_df.index)
                    y_ts = pd.Series(y_true_df.index)
                    pred = predictor.predict(
                        df=x_df_input, x_timestamp=x_ts, y_timestamp=y_ts,
                        pred_len=pred_len, T=pred_cfg['temperature'],
                        top_p=pred_cfg['top_p'], sample_count=pred_cfg['sample_count'],
                    )
                except Exception as e:
                    continue
                inf = time.time() - t1

                pred_dir = get_direction(pred, y_true_df, pred_cfg)
                pred_move = pred['close'].values[-1] - y_true_df['close'].values[0]
                actual_move = y_true_df['close'].values[-1] - y_true_df['close'].values[0]
                actual_dir = 'UP' if actual_move > 0 else 'DOWN'

                if pred_dir == actual_dir:
                    correct += 1
                total += 1
                inf_times.append(inf)

                pred_steps = np.sign(np.diff(pred['close'].values, prepend=y_true_df['close'].values[0]))
                actual_steps = np.sign(np.diff(y_true_df['close'].values, prepend=y_true_df['close'].values[0]))
                step_accs.append((pred_steps[1:] == actual_steps[1:]).mean())

            dir_acc = correct / total * 100 if total > 0 else 0
            avg_step = np.mean(step_accs) * 100 if step_accs else 0
            avg_inf = np.mean(inf_times) if inf_times else 0

            pair_results[tf_name] = {
                'dir_acc': dir_acc, 'step_acc': avg_step,
                'n_tests': total, 'avg_inf': avg_inf,
            }
            print(f"  {tf_name:4s}: {dir_acc:5.0f}% dir ({correct}/{total})  {avg_step:4.0f}% step  {avg_inf:.1f}s avg")

        results[pair_name] = pair_results

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Pair':<12} {'TF':<5} {'Dir Acc':>8} {'Step Acc':>8} {'Tests':>6} {'Inf/s':>6}")
    print(f"{'-'*50}")
    all_dirs = []
    for pair_name in sorted(results):
        for tf_name in sorted(results[pair_name]):
            r = results[pair_name][tf_name]
            print(f"{pair_name:<12} {tf_name:<5} {r['dir_acc']:>7.0f}% {r['step_acc']:>7.0f}% {r['n_tests']:>6} {r['avg_inf']:>5.1f}s")
            all_dirs.append(r['dir_acc'])

    if all_dirs:
        print(f"{'-'*50}")
        print(f"{'AVERAGE':<12} {'':<5} {np.mean(all_dirs):>7.0f}%")
        print(f"{'MEDIAN':<12} {'':<5} {np.median(all_dirs):>7.0f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()