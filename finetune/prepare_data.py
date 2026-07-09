#!/usr/bin/env python3
"""
Prepare Forex & Crypto training data for Kronos fine-tuning.
Fetches ALL timeframes (5m, 15m, 30m, 1h, 4h, 1d) at MAXIMUM history.

Data sources:
  - OANDA (free API): 7 forex pairs — YEARS of 5m/15m/30m data
  - yfinance (free):   2 crypto pairs — limited intraday history

To use OANDA:
  1. Create free demo account: https://www.oanda.com/demo-account/
  2. Get API key: https://www.oanda.com/account/demo/api-access/
  3. Set OANDA_API_KEY env var or paste key when prompted
"""
import os, sys, pickle, time, json, urllib.request, urllib.error
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from indicators import compute_all_indicators, get_indicator_feature_names


# ── All timeframes the bot trades on ──
TIMEFRAMES = [
    ('5m',  'M5'),
    ('15m', 'M15'),
    ('30m', 'M30'),
    ('1h',  'H1'),
    ('4h',  'H4'),
    ('1d',  'D'),
    # 2h not supported by Yahoo or OANDA
]

# Forex via OANDA, crypto via yfinance
PAIRS = {
    'EUR/USD': ('EUR_USD', 'forex'),
    'GBP/USD': ('GBP_USD', 'forex'),
    'USD/JPY': ('USD_JPY', 'forex'),
    'AUD/USD': ('AUD_USD', 'forex'),
    'USD/CAD': ('USD_CAD', 'forex'),
    'USD/CHF': ('USD_CHF', 'forex'),
    'NZD/USD': ('NZD_USD', 'forex'),
    'BTC/USD': ('BTC-USD', 'crypto'),
    'ETH/USD': ('ETH-USD', 'crypto'),
}

OANDA_COUNT = 2500  # candles per request (max 5000)
OANDA_ENVIRONMENTS = {
    'practice': 'https://api-fxpractice.oanda.com',
    'live': 'https://api-fxtrade.oanda.com',
}


def get_oanda_key():
    """Get OANDA API key from env or prompt."""
    key = os.environ.get('OANDA_API_KEY')
    if key:
        return key, 'live'
    key = os.environ.get('OANDA_API_KEY_PRACTICE')
    if key:
        return key, 'practice'
    # For Cursor/PC use, it'll come from env var
    print("  ⚠️  No OANDA_API_KEY set. Falling back to yfinance (60 days max).")
    return None, None


def fetch_oanda(instrument, granularity, api_key, env):
    """Fetch ALL historical candles from OANDA."""
    base = OANDA_ENVIRONMENTS[env]
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    all_candles = []
    from_time = None

    while True:
        url = f'{base}/v3/instruments/{instrument}/candles?granularity={granularity}&count={OANDA_COUNT}&price=MBA'
        if from_time:
            url += f'&to={from_time}'

        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
        except Exception as e:
            print(f"  [OANDA] Error: {e}")
            break

        candles = data.get('candles', [])

        # OANDA returns newest-first; we need oldest-first
        for c in reversed(candles):
            if c.get('complete', False):
                ts = c['time'].replace('Z', '+00:00')
                all_candles.append({
                    'datetime': pd.Timestamp(ts),
                    'open': float(c['mid']['o']),
                    'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']),
                    'close': float(c['mid']['c']),
                    'vol': float(c.get('volume', 0)),
                })

        if len(candles) < OANDA_COUNT:
            break  # no more data

        from_time = candles[0]['time']  # oldest candle in this batch

        time.sleep(0.2)  # rate limit

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles)
    df['amt'] = df['close'] * df['vol'].clip(lower=1)
    df = df.dropna()
    return df


def fetch_yfinance(ticker, tf_name):
    """Fetch from yfinance with appropriate period."""
    period_map = {
        '5m':  '60d',
        '15m': '60d',
        '30m': '60d',
        '1h':  '2y',
        '4h':  '2y',
        '1d':  'max',
    }
    period = period_map.get(tf_name, '60d')
    df = yf.download(ticker, period=period, interval=tf_name, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    col_map = {'Datetime': 'datetime', 'Date': 'datetime',
               'Open': 'open', 'High': 'high', 'Low': 'low',
               'Close': 'close', 'Volume': 'vol'}
    df.rename(columns=col_map, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['amt'] = df['close'] * df['vol'].clip(lower=1)
    return df[['datetime', 'open', 'high', 'low', 'close', 'vol', 'amt']].dropna()


def process_df(df):
    """Normalize and compute indicators."""
    print(f"  Computing indicators ({len(get_indicator_feature_names())} features)...")
    df = compute_all_indicators(df)
    df = df.dropna()
    return df


def split_train_val(df, name):
    """Split by date if range covers both eras, else positional 85/15."""
    dt_col = df['datetime']
    if hasattr(dt_col.dtype, 'tz') and dt_col.dtype.tz is not None:
        t_start, t_end = pd.Timestamp("2020-01-01", tz='UTC'), pd.Timestamp("2024-12-31", tz='UTC')
        v_start, v_end = pd.Timestamp("2025-01-01", tz='UTC'), pd.Timestamp("2025-12-31", tz='UTC')
    else:
        t_start, t_end = pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")
        v_start, v_end = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")

    train_part = df[(dt_col >= t_start) & (dt_col <= t_end)]
    val_part = df[(dt_col >= v_start) & (dt_col <= v_end)]

    if len(train_part) < 50 and len(df) > 200:
        split = int(len(df) * 0.85)
        train_part = df.iloc[:split]
        val_part = df.iloc[split:]
        print(f"    {name}: positional split ({len(train_part)} train / {len(val_part)} val)")
    else:
        print(f"    {name}: {len(train_part)} train / {len(val_part)} val")

    return train_part if len(train_part) > 50 else None, val_part if len(val_part) > 20 else None


def main():
    config = Config()
    output_dir = config.dataset_path
    os.makedirs(output_dir, exist_ok=True)

    # OANDA setup
    oanda_key, oanda_env = get_oanda_key()

    train_data, val_data = {}, {}

    for pair_name, (ticker_or_instr, kind) in PAIRS.items():
        for tf_name, oanda_gran in TIMEFRAMES:
            key = f"{pair_name}_{tf_name}"
            print(f"\n[{key}]", end='')

            df = None

            # Forex via OANDA (years of intraday data)
            if kind == 'forex' and oanda_key:
                print(f" (OANDA {ticker_or_instr} {oanda_gran})", end=' ')
                df = fetch_oanda(ticker_or_instr, oanda_gran, oanda_key, oanda_env)
                if df is not None:
                    print(f"-> {len(df)} rows", end='')

            # Fallback to yfinance
            if df is None:
                source = 'OANDA' if kind == 'forex' else 'yfinance'
                print(f" ({source} {ticker_or_instr} {tf_name})", end=' ')
                if kind == 'crypto':
                    df = fetch_yfinance(ticker_or_instr, tf_name)
                else:
                    df = fetch_yfinance(ticker_or_instr.replace('_', ''), tf_name)
                if df is not None:
                    print(f"-> {len(df)} rows", end='')

            if df is None:
                print(" -> No data")
                continue

            print()
            df = process_df(df)
            print(f"  {len(df)} rows ({df['datetime'].min().date()} -> {df['datetime'].max().date()})")
            time.sleep(0.3)

            train_part, val_part = split_train_val(df, key)
            if train_part is not None:
                train_data[key] = train_part
            if val_part is not None:
                val_data[key] = val_part

    # ── Save ──
    train_path = f"{output_dir}/train_data.pkl"
    val_path = f"{output_dir}/val_data.pkl"
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)

    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    for key in sorted(train_data):
        tc = len(train_data[key])
        vc = len(val_data.get(key, []))
        feats = len(train_data[key].columns) - 1
        print(f"  {key:20s} -> {tc:5d} train / {vc:5d} val ({feats} feats)")
    print(f"{'='*60}")
    print(f"Total: {len(train_data)} entries train, {len(val_data)} entries val")
    print(f"Features per entry: {len(get_indicator_feature_names())} indicators + 6 base = {len(get_indicator_feature_names()) + 6} total")
    print(f"\nOANDA key found: {'YES' if oanda_key else 'NO'}")
    if not oanda_key:
        print("Tip: Set OANDA_API_KEY env var for years of free intraday forex data!")


if __name__ == '__main__':
    main()