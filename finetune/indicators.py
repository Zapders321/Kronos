"""
Technical indicator computation module for Kronos.
Computes common TA indicators using pandas/numpy (no TA-Lib needed).

All functions take a DataFrame with at minimum columns:
  open, high, low, close, vol
and return a DataFrame with the indicator columns appended.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _ma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted moving average (linear weights)."""
    weights = np.arange(1, period + 1)
    def _calc(w):
        if len(w) < period:
            return np.nan
        return np.dot(w, weights) / weights.sum()
    return series.rolling(window=period, min_periods=period).apply(_calc, raw=True)


# ── Momentum / Trend ──────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = 14, col: str = 'close') -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = 'close') -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    ema_fast = _ema(df[col], fast)
    ema_slow = _ema(df[col], slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_line - signal_line
    return df


def bb(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = 'close') -> pd.DataFrame:
    """Bollinger Bands: upper, lower, width, %B, bandwidth."""
    mid = _ma(df[col], period)
    std = df[col].rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    df['bb_upper'] = upper
    df['bb_mid'] = mid
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / (mid + 1e-10)
    df['bb_pctb'] = (df[col] - lower) / (upper - lower + 1e-10)
    return df


def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = _ema(tr, period)
    df['atr_pct'] = df['atr'] / (close + 1e-10) * 100  # ATR as % of price
    return df


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, col: str = 'close') -> pd.DataFrame:
    """Stochastic Oscillator %K and %D."""
    low_min = df['low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()
    stoch_k = 100 * (df[col] - low_min) / (high_max - low_min + 1e-10)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = _ma(stoch_k, d_period)
    return df


def rate_of_change(df: pd.DataFrame, period: int = 10, col: str = 'close') -> pd.DataFrame:
    """Rate of Change / Momentum."""
    df['roc'] = df[col].pct_change(periods=period) * 100
    df['mom'] = df[col] - df[col].shift(period)
    return df


def sma_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21, col: str = 'close') -> pd.DataFrame:
    """SMA crossover signals."""
    sma_fast = _ma(df[col], fast)
    sma_slow = _ma(df[col], slow)
    df[f'sma_{fast}'] = sma_fast
    df[f'sma_{slow}'] = sma_slow
    df['sma_dist'] = (sma_fast - sma_slow) / (sma_slow + 1e-10) * 100  # % distance
    return df


def ema_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, col: str = 'close') -> pd.DataFrame:
    """EMA crossover signals."""
    ema_fast = _ema(df[col], fast)
    ema_slow = _ema(df[col], slow)
    df[f'ema_{fast}'] = ema_fast
    df[f'ema_{slow}'] = ema_slow
    df['ema_dist'] = (ema_fast - ema_slow) / (ema_slow + 1e-10) * 100
    return df


# ── Volume ──────────────────────────────────────────

def volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based indicators."""
    close, vol = df['close'], df['vol']

    # Volume ratio (current vs 20-period avg)
    vol_ma20 = _ma(vol, 20)
    df['vol_ratio'] = vol / (vol_ma20 + 1e-10)

    # On-Balance Volume
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df['obv'] = obv
    df['obv_sma'] = _ma(obv, 20)

    # Volume Price Trend
    vpt = (close.diff() / close.shift(1) * vol).fillna(0).cumsum()
    df['vpt'] = vpt

    # Money Flow Index-like (simplified)
    typical_price = (df['high'] + df['low'] + close) / 3
    money_flow = typical_price * vol
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_mf_sum = _ema(positive_flow, 14)
    neg_mf_sum = _ema(negative_flow, 14)
    mfi_ratio = pos_mf_sum / (neg_mf_sum + 1e-10)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))

    return df


# ── Volatility & Price Action ──────────────────────

def volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility measures beyond ATR."""
    close = df['close']

    # Historical volatility (20-day standard deviation of log returns)
    log_ret = np.log(close / close.shift(1))
    df['hist_vol'] = log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100

    # Price range ratio
    df['range_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-10) * 100

    # Gap (open vs previous close)
    df['gap_pct'] = (df['open'] - close.shift(1)) / (close.shift(1) + 1e-10) * 100

    # High-Low ratio
    df['hl_ratio'] = df['high'] / (df['low'] + 1e-10)

    return df


# ── Price Patterns (simplified) ────────────────────

def price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Simple candlestick & price pattern features."""
    open_, high, low, close = df['open'], df['high'], df['low'], df['close']

    # Body and shadow sizes
    df['body'] = (close - open_).abs()
    df['upper_shadow'] = high - pd.concat([open_, close], axis=1).max(axis=1)
    df['lower_shadow'] = pd.concat([open_, close], axis=1).min(axis=1) - low
    df['body_ratio'] = df['body'] / (close + 1e-10) * 100
    df['upper_shadow_ratio'] = df['upper_shadow'] / (close + 1e-10) * 100
    df['lower_shadow_ratio'] = df['lower_shadow'] / (close + 1e-10) * 100

    # Direction (bullish/bearish candle)
    df['bullish'] = (close > open_).astype(float)

    # Consecutive up/down candles
    df['consecutive_up'] = (df['bullish'] * (df['bullish'] == df['bullish'].shift(1).fillna(0))).cumsum()
    df['consecutive_down'] = ((1 - df['bullish']) * ((1 - df['bullish']) == (1 - df['bullish']).shift(1).fillna(0))).cumsum()

    return df


# ── Composite ──────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on an OHLCV DataFrame.
    Input must have columns: open, high, low, close, vol
    Returns the DataFrame with indicator columns added.
    """
    df = df.copy()

    # Ensure numeric
    for c in ['open', 'high', 'low', 'close', 'vol']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Order by date
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').reset_index(drop=True)

    # Momentum / Trend
    df = rsi(df)
    df = macd(df)
    df = bb(df)
    df = atr(df)
    df = stochastic(df)
    df = rate_of_change(df)
    df = sma_cross(df)
    df = ema_cross(df)

    # Volume
    df = volume_indicators(df)

    # Volatility
    df = volatility_indicators(df)

    # Patterns
    df = price_patterns(df)

    return df


# ── Feature name helpers ────────────────────────────

def get_indicator_feature_names() -> list[str]:
    """Return the list of all indicator column names added by compute_all_indicators."""
    return [
        # RSI
        'rsi',
        # MACD
        'macd', 'macd_signal', 'macd_hist',
        # Bollinger Bands
        'bb_upper', 'bb_mid', 'bb_lower', 'bb_width', 'bb_pctb',
        # ATR
        'atr', 'atr_pct',
        # Stochastic
        'stoch_k', 'stoch_d',
        # ROC / Momentum
        'roc', 'mom',
        # SMA cross
        'sma_9', 'sma_21', 'sma_dist',
        # EMA cross
        'ema_12', 'ema_26', 'ema_dist',
        # Volume
        'vol_ratio', 'obv', 'obv_sma', 'vpt', 'mfi',
        # Volatility
        'hist_vol', 'range_ratio', 'gap_pct', 'hl_ratio',
        # Patterns
        'body', 'upper_shadow', 'lower_shadow', 'body_ratio',
        'upper_shadow_ratio', 'lower_shadow_ratio',
        'bullish', 'consecutive_up', 'consecutive_down',
    ]


def get_full_feature_list() -> list[str]:
    """Return the complete feature list (OHLCV + amount + indicators)."""
    base = ['open', 'high', 'low', 'close', 'vol', 'amt']
    return base + get_indicator_feature_names()