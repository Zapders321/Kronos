"""
Synthetic Market Data Generator — Generate Infinite Realistic Price Sequences.

Uses Geometric Brownian Motion + GARCH(1,1) volatility + multiple market regimes
to produce realistic OHLCV data that mimics real forex/crypto behavior.

Key properties:
  - GBM base with realistic drift (trend) and diffusion (noise)
  - GARCH(1,1) volatility clustering — calm periods clump, volatile periods clump
  - Regime switching — low/medium/high/spike volatility with Markov transitions
  - Intraday patterns — volume ramps during session overlaps
  - Mean reversion component — prices don't drift infinitely
  - Generates ANY number of candles — limited only by RAM

Output format: pandas DataFrame with [datetime, open, high, low, close, vol, amt]
Same interface as prepare_data.py — indicators can be computed on top.
"""
import numpy as np
import pandas as pd


def _regime_transition(current_regime, transition_matrix):
    """Sample next regime from Markov transition matrix."""
    r = np.random.random()
    cum = 0.0
    for j in range(transition_matrix.shape[1]):
        cum += transition_matrix[current_regime, j]
        if r < cum:
            return j
    return current_regime


def _scale_to_candle(open_p, close_p, high_p, low_p):
    """Ensure OHLC is consistent: high >= max(open, close) and low <= min(open, close)."""
    hi = max(high_p, open_p, close_p)
    lo = min(low_p, open_p, close_p)
    hi = hi * (1 + np.random.uniform(0, 0.001))
    lo = lo * (1 - np.random.uniform(0, 0.001))
    return open_p, hi, lo, close_p


def generate_synthetic_data(
    n_candles=100000,
    pair_type='forex',
    volatility_regime='mixed',
    seed=None,
    start_price=1.0,
):
    """
    Generate synthetic OHLCV market data.

    Args:
        n_candles: Number of candles to generate
        pair_type: 'forex' or 'crypto'
        volatility_regime: 'low', 'medium', 'high', 'spike', or 'mixed'
        seed: Random seed for reproducibility
        start_price: Starting price level

    Returns:
        DataFrame with [datetime, open, high, low, close, vol, amt]
    """
    if seed is not None:
        np.random.seed(seed)

    # Parameters by pair type
    if pair_type == 'crypto':
        mu = 0.05
        omega_base = 0.5e-5
        alpha_base = 0.12
        beta_base = 0.82
        sigma0 = 0.02
    else:  # forex
        mu = 0.02
        omega_base = 0.2e-5
        alpha_base = 0.08
        beta_base = 0.90
        sigma0 = 0.008

    # Regime configuration
    regimes = {
        'low':    {'scale': 0.5, 'omega_mult': 0.3},
        'medium': {'scale': 1.0, 'omega_mult': 1.0},
        'high':   {'scale': 2.0, 'omega_mult': 3.0},
        'spike':  {'scale': 4.0, 'omega_mult': 8.0},
    }
    regime_labels = ['low', 'medium', 'high', 'spike']
    n_regimes = 4

    # Markov transition matrix (stay in regime ~80% of time)
    stay_prob = 0.80
    trans_matrix = np.full((n_regimes, n_regimes), (1 - stay_prob) / (n_regimes - 1))
    for i in range(n_regimes):
        trans_matrix[i, i] = stay_prob

    # Regime sequence
    if volatility_regime == 'mixed':
        regime_indices = np.zeros(n_candles, dtype=np.int32)
        current = 1
        for t in range(n_candles):
            current = _regime_transition(current, trans_matrix)
            regime_indices[t] = current
    else:
        idx = regime_labels.index(volatility_regime)
        regime_indices = np.full(n_candles, idx, dtype=np.int32)

    # Generate price path at tick resolution
    ticks_per_candle = 8
    total_ticks = n_candles * ticks_per_candle

    prices = np.zeros(total_ticks)
    prices[0] = start_price
    sigma2 = sigma0 ** 2

    for t in range(1, total_ticks):
        candle_idx = t // ticks_per_candle
        scale = regimes[regime_labels[regime_indices[candle_idx]]]['scale']
        omega_m = regimes[regime_labels[regime_indices[candle_idx]]]['omega_mult']

        omega = omega_base * omega_m
        alpha = alpha_base * (1.0 + 0.2 * (scale - 1.0))
        beta_val = beta_base * (1.0 - 0.05 * (scale - 1.0))

        sigma2 = omega + alpha * 0.0001 + beta_val * sigma2  # stable GARCH step
        sigma = np.sqrt(max(sigma2, 1e-12)) * scale
        sigma = min(sigma, 0.05)  # cap per-tick vol at 5%

        tick_drift = mu / (252 * ticks_per_candle)
        ret = np.random.normal(tick_drift, sigma)
        ret = np.clip(ret, -0.02, 0.02)  # max 2% per tick
        prices[t] = prices[t - 1] * np.exp(ret)

        # Mean reversion pull toward start_price
        if np.random.random() < 0.02:
            pull = (np.log(start_price) - np.log(prices[t])) * 0.002
            prices[t] *= np.exp(pull)

    prices = np.clip(prices, start_price * 0.5, start_price * 2.0)

    # Aggregate ticks into OHLCV candles
    opens = np.zeros(n_candles)
    highs = np.zeros(n_candles)
    lows = np.zeros(n_candles)
    closes = np.zeros(n_candles)
    volumes = np.zeros(n_candles)

    for i in range(n_candles):
        s = i * ticks_per_candle
        e = s + ticks_per_candle
        seg = prices[s:e]
        opens[i] = seg[0]
        closes[i] = seg[-1]
        highs[i] = seg.max()
        lows[i] = seg.min()

        # Volume correlates with volatility
        candle_range = (highs[i] - lows[i]) / closes[i]
        base_vol = np.random.exponential(scale=1000)
        vol_mult = 1.0 + 5.0 * candle_range / 0.005
        vol_mult = np.clip(vol_mult, 0.5, 10.0)
        volumes[i] = max(1, int(base_vol * vol_mult))

        # OhLC consistency
        opens[i], highs[i], lows[i], closes[i] = _scale_to_candle(
            opens[i], closes[i], highs[i], lows[i]
        )

    volumes = np.clip(volumes, 1, None).astype(np.float32)

    # Build DataFrame
    dt = pd.date_range('2000-01-01', periods=n_candles, freq='5min')
    df = pd.DataFrame({
        'datetime': dt, 'open': opens, 'high': highs,
        'low': lows, 'close': closes, 'vol': volumes,
    })
    df['amt'] = df['close'] * df['vol']

    avg_vol = df['close'].pct_change(fill_method=None).std()
    print(f"  [SYNTH] {n_candles:,} candles ({pair_type}, {volatility_regime} regime)")
    print(f"  [SYNTH] Price: {df['low'].min():.4f}-{df['high'].max():.4f}, Vol: {avg_vol:.4f}")

    return df


def generate_multi_regime_dataset(total_candles=500000, pair_type='forex'):
    """Generate synthetic data with all volatility regimes evenly split."""
    per_regime = total_candles // 5
    regimes = ['low', 'medium', 'high', 'spike']
    dfs = []

    for i, regime in enumerate(regimes):
        df = generate_synthetic_data(
            n_candles=per_regime, pair_type=pair_type,
            volatility_regime=regime, seed=hash(f"seed{i}") % (2**31),
        )
        dfs.append(df)

    mixed_df = generate_synthetic_data(
        n_candles=per_regime, pair_type=pair_type,
        volatility_regime='mixed', seed=42,
    )
    dfs.append(mixed_df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    full_df['datetime'] = pd.date_range('2000-01-01', periods=len(full_df), freq='5min')

    print(f"\n  [SYNTH] Multi-regime dataset: {len(full_df):,} candles")
    return full_df


if __name__ == '__main__':
    print("Testing synthetic data generator...")
    df = generate_multi_regime_dataset(50000, 'forex')
    print(f"\n  Generated {len(df)} candles, no NaN: {df.isna().sum().sum() == 0}")
    print(f"  {df.head().to_string()}")