#!/usr/bin/env python3
"""
Simple RunPod data prep: downloads yfinance data, saves as .pkl.
Then runs feedback_train.py directly.
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'finetune'))

from indicators import compute_all_indicators, get_indicator_feature_names

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
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed_datasets')
os.makedirs(DATA_DIR, exist_ok=True)


def download(tf, period):
    """Download all pairs, compute indicators, return merged dict of DataFrames."""
    all_dfs = {}
    for pair in ALL_PAIRS:
        try:
            df = yf.download(pair, period=period, interval=tf, progress=False)
            if df.empty or len(df) < 100:
                continue
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            for drop_col in ['adj close']:
                if drop_col in df.columns:
                    df = df.drop(columns=[drop_col])
            if 'volume' in df.columns:
                df = df.rename(columns={'volume': 'vol'})
            # Compute indicators
            df = compute_all_indicators(df)
            all_dfs[pair] = df
        except Exception as e:
            print(f"  ⚠️ {pair}: {e}")
    return all_dfs


def make_pkl(all_dfs, split=0.85):
    """Convert to feedback_train format: dict of {pair_id: DataFrame}."""
    train_data = {}
    val_data = {}
    for pair, df in all_dfs.items():
        n = len(df)
        train_data[pair] = df.iloc[:int(n*split)]
        val_data[pair] = df.iloc[int(n*split):]
    return train_data, val_data


def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else '1d'
    period = sys.argv[2] if len(sys.argv) > 2 else '5y'

    print(f"Downloading {len(ALL_PAIRS)} pairs @ {tf} ({period})...")
    all_dfs = download(tf, period)
    print(f"Got {len(all_dfs)} pairs, saving...")

    train_data, val_data = make_pkl(all_dfs)
    with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(DATA_DIR, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)

    print(f"Train: {sum(len(v) for v in train_data.values())} rows")
    print(f"Val:   {sum(len(v) for v in val_data.values())} rows")
    print(f"Saved to {DATA_DIR}")


if __name__ == '__main__':
    main()