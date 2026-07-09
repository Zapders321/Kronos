#!/usr/bin/env python3
"""
Kronos AutoTrader v2 — Multi-Pair Paper Trading Bot
- Multiple forex pairs (EUR/USD, GBP/USD, USD/JPY)
- Multi-timeframe Kronos predictions per pair
- Confluence-based entry (all timeframes agree)
- Independent position management per pair
- Live dashboard with charts at http://localhost:8081
"""
import sys, os, time, json, threading, logging
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pandas as pd
import yfinance as yf
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('kronos-bot')

# ── Config ──────────────────────────────────────────────
CFG_PATH = Path(__file__).resolve().parent / 'config.json'
with open(CFG_PATH) as f:
    CFG = json.load(f)

PAIRS = CFG['pairs']['forex']
CRYPTO_PAIRS = CFG['pairs']['crypto']
ALL_PAIRS = {**PAIRS, **CRYPTO_PAIRS}
TIMEFRAMES = CFG['timeframes']
LOOP_SECONDS = CFG['loop']['interval_seconds']
INITIAL_BALANCE = CFG['risk']['initial_balance']
RISK_PER_TRADE = CFG['risk']['risk_per_trade']
STOP_LOSS_PIPS = CFG['risk']['forex']['stop_loss_pips']
TAKE_PROFIT_PIPS = CFG['risk']['forex']['take_profit_pips']
LOT_SIZE = CFG['risk']['forex']['lot_size']
CRYPTO_SL_PCT = CFG['risk']['crypto']['stop_loss_pct']
CRYPTO_TP_PCT = CFG['risk']['crypto']['take_profit_pct']

def get_pred_cfg(pair_name):
    return CFG['prediction']['crypto'] if pair_name in CRYPTO_PAIRS else CFG['prediction']['forex']

HTTP_PORT = CFG['dashboard']['port']
STATE_FILE = Path(__file__).resolve().parent / 'state.json'

# ── State ───────────────────────────────────────────────
def load_state():
    """Load state from disk, or return fresh defaults."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            log.info(f"Loaded state: ${saved.get('balance', 0):.2f}, {saved.get('total_trades', 0)} trades")
            # Ensure missing keys exist (backwards compat with old state files)
            for key in ['total_trades', 'winning_trades', 'price_history', 'current_prices', 'unrealized_pnl']:
                saved.setdefault(key, 0 if key in ('total_trades', 'winning_trades') else {})
            return saved
        except Exception as e:
            log.warning(f"Failed to load state: {e}, starting fresh")
    return {
        'balance': INITIAL_BALANCE,
        'starting_balance': INITIAL_BALANCE,
        'positions': {},
        'signals': {},
        'trade_history': [],
        'last_update': None,
        'status': 'starting',
        'error': None,
        'total_trades': 0,
        'winning_trades': 0,
    }


def save_state():
    """Atomically persist state to disk so crash mid-write doesn't corrupt."""
    try:
        with state_lock:
            tmp = str(STATE_FILE) + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(state, f, default=str, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(STATE_FILE))
    except Exception as e:
        log.error(f"Failed to save state: {e}")


state = load_state()
state_lock = threading.RLock()  # reentrant — allows close_position inside open_position

# Initialize chart/live-pnl fields in state if missing
if 'price_history' not in state:
    state['price_history'] = {}
if 'current_prices' not in state:
    state['current_prices'] = {}
if 'unrealized_pnl' not in state:
    state['unrealized_pnl'] = {}
if 'signals' not in state:
    state['signals'] = {}
if 'last_update' not in state:
    state['last_update'] = None

BOT_DIR = Path(__file__).resolve().parent


# ── Model (lazy singleton) ──────────────────────────────
_model = _tokenizer = _predictor = None

def get_predictor():
    global _model, _tokenizer, _predictor
    if _predictor is None:
        log.info("Loading fine-tuned Kronos-base model...")
        _tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        _model = Kronos.from_pretrained(str(Path(__file__).resolve().parent.parent / 'outputs' / 'kronos_base_finetuned' / 'checkpoints' / 'best_model'))
        _predictor = KronosPredictor(_model, _tokenizer, max_context=CFG['prediction']['forex']['max_context'])
        log.info("Model loaded.")
    return _predictor


def pull_data(symbol, interval, period, max_retries=CFG['loop']['data_retries']):
    for attempt in range(max_retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty:
                return None
            df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df['amount'] = 0
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 2
                log.warning(f"  Retry {attempt+1}/{max_retries} for {symbol} {interval}: {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"  [{symbol}] data fetch failed after {max_retries} attempts: {e}")
                return None


def run_prediction(predictor, df, lookback, pred_len, pred_cfg):
    x_df_raw = df.iloc[-lookback - pred_len : -pred_len].copy()
    y_true = df.iloc[-pred_len:].copy()
    x_df_input = x_df_raw.reset_index(drop=True)
    x_ts = pd.Series(x_df_raw.index)
    y_ts = pd.Series(y_true.index)
    t0 = time.time()
    pred = predictor.predict(
        df=x_df_input, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=pred_len, T=pred_cfg['temperature'], top_p=pred_cfg['top_p'], sample_count=pred_cfg['sample_count'],
    )
    return pred, y_true, time.time() - t0


def get_direction(pred_df, y_true, pred_cfg):
    pred_final = pred_df['close'].values[-1]
    actual_last = y_true['close'].values[0]
    move = pred_final - actual_last
    pct = abs(move) / actual_last * 100
    if pct < pred_cfg['min_move_pct']:
        return 'FLAT'
    return 'UP' if move > 0 else 'DOWN'


def _get_pip_value(symbol):
    return 0.01 if 'JPY' in symbol else 0.0001


def _pnl_to_usd(pair_name, raw_pnl):
    """Convert raw P&L to USD. Crypto pairs are already USD-denominated."""
    if pair_name in CRYPTO_PAIRS:
        return raw_pnl  # crypto already in USD
    quote = pair_name.split('/')[1]
    if quote == 'USD':
        return raw_pnl
    # Need USD/quote rate to convert quote → USD
    usd_pair = f'USD/{quote}'
    usd_price = state['current_prices'].get(usd_pair)
    if usd_price:
        return raw_pnl / usd_price
    # Try inverse: quote/USD
    inv_pair = f'{quote}/USD'
    inv_price = state['current_prices'].get(inv_pair)
    if inv_price:
        return raw_pnl * inv_price
    return raw_pnl  # fallback if no conversion rate available


def close_position(symbol, exit_price, exit_reason):
    with state_lock:
        pos = state['positions'].get(symbol)
        if pos is None:
            return
        is_crypto = pos.get('asset_type') == 'crypto'
        if is_crypto:
            raw_pnl = (exit_price - pos['entry_price']) * pos['size']  # size in units
        else:
            raw_pnl = (exit_price - pos['entry_price']) * pos['size'] * LOT_SIZE  # size in lots
        if pos['type'] == 'SHORT':
            raw_pnl = -raw_pnl
        pnl = _pnl_to_usd(symbol, raw_pnl)
        state['balance'] += pnl
        state['total_trades'] += 1
        if pnl > 0:
            state['winning_trades'] += 1
        trade = {
            'symbol': symbol,
            'type': pos['type'],
            'entry': round(pos['entry_price'], 6),
            'exit': round(exit_price, 6),
            'size': round(pos['size'], 4),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl / (state['balance'] - pnl) * 100, 2) if state['balance'] != pnl else 0,
            'reason': exit_reason,
            'entry_time': pos['timestamp'],
            'exit_time': datetime.now(timezone.utc).isoformat(),
        }
        state['trade_history'].append(trade)
        del state['positions'][symbol]
        log.info(f"CLOSED {symbol} {pos['type']} | P&L: ${pnl:.2f} | {exit_reason}")
        save_state()  # persist immediately so trade isn't lost on crash


def open_position(symbol, direction, entry_price):
    with state_lock:
        if symbol in state['positions']:
            close_position(symbol, entry_price, 'signal_reversal')
        risk_amount = state['balance'] * RISK_PER_TRADE
        is_crypto = symbol in CRYPTO_PAIRS
        if is_crypto:
            # Crypto: percentage-based SL/TP, size in coin units
            sl = entry_price * (1 - CRYPTO_SL_PCT) if direction == 'UP' else entry_price * (1 + CRYPTO_SL_PCT)
            tp = entry_price * (1 + CRYPTO_TP_PCT) if direction == 'UP' else entry_price * (1 - CRYPTO_TP_PCT)
            size = risk_amount / (entry_price * CRYPTO_SL_PCT)
            state['positions'][symbol] = {
                'type': 'LONG' if direction == 'UP' else 'SHORT',
                'entry_price': entry_price,
                'size': round(size, 6),
                'sl': round(sl, 2),
                'tp': round(tp, 2),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'asset_type': 'crypto',
            }
        else:
            # Forex: pip-based SL/TP, size in lots
            pip_value = _get_pip_value(symbol)
            sl_distance = STOP_LOSS_PIPS * pip_value
            size = risk_amount / (sl_distance * LOT_SIZE)
            sl = entry_price - sl_distance if direction == 'UP' else entry_price + sl_distance
            tp = entry_price + (TAKE_PROFIT_PIPS * pip_value) if direction == 'UP' else entry_price - (TAKE_PROFIT_PIPS * pip_value)
            state['positions'][symbol] = {
                'type': 'LONG' if direction == 'UP' else 'SHORT',
                'entry_price': entry_price,
                'size': round(size, 4),
                'sl': round(sl, 6),
                'tp': round(tp, 6),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        log.info(f"OPENED {symbol} {state['positions'][symbol]['type']} @ {entry_price:.6f}")


def check_stops(symbol, current_price):
    with state_lock:
        pos = state['positions'].get(symbol)
        if pos is None:
            return

        # Normal stop check
        if pos['type'] == 'LONG':
            if current_price <= pos['sl']:
                close_position(symbol, pos['sl'], 'stop_loss')
            elif current_price >= pos['tp']:
                close_position(symbol, pos['tp'], 'take_profit')
        else:
            if current_price >= pos['sl']:
                close_position(symbol, pos['sl'], 'stop_loss')
            elif current_price <= pos['tp']:
                close_position(symbol, pos['tp'], 'take_profit')


def calc_unrealized_pnl():
    """Calculate unrealized P&L for all open positions."""
    with state_lock:
        unrealized = {}
        total_unrealized = 0.0
        for pair_name, pos in state['positions'].items():
            cp = state['current_prices'].get(pair_name)
            if cp is None:
                continue
            is_crypto = pos.get('asset_type') == 'crypto'
            if pos['type'] == 'LONG':
                raw_pnl = (cp - pos['entry_price']) * pos['size']
            else:
                raw_pnl = (pos['entry_price'] - cp) * pos['size']
            if not is_crypto:
                raw_pnl *= LOT_SIZE
            upnl = _pnl_to_usd(pair_name, raw_pnl)
            unrealized[pair_name] = round(upnl, 2)
            total_unrealized += upnl
        unrealized['total'] = round(total_unrealized, 2)
        state['unrealized_pnl'] = unrealized


# ── Main Loop ───────────────────────────────────────────
def trading_loop():
    predictor = get_predictor()
    with state_lock:
        state['status'] = 'running'

    while True:
        try:
            now = datetime.now(timezone.utc)
            log.info(f"--- Cycle @ {now.strftime('%H:%M:%S')} ---")

            for pair_name, symbol in ALL_PAIRS.items():
                try:
                    log.info(f"  [{pair_name}]")
                    pred_cfg = get_pred_cfg(pair_name)
                    timeframe_results = {}
                    directions = {}
                    current_price = None

                    # Run predictions on all timeframes
                    for tf_name, tf_cfg in TIMEFRAMES.items():
                        df = pull_data(symbol, tf_cfg['interval'], tf_cfg['period'])
                        if df is None or len(df) < tf_cfg['lookback']:
                            log.warning(f"    {tf_name}: insufficient data")
                            continue
                        pred, y_true, inf_time = run_prediction(predictor, df, tf_cfg['lookback'], tf_cfg['pred_len'], pred_cfg)
                        direction = get_direction(pred, y_true, pred_cfg)
                        directions[tf_name] = direction
                        timeframe_results[tf_name] = {
                            'pred_close': pred['close'].values,
                            'actual_close': y_true['close'].values,
                            'inf_time': inf_time,
                        }
                        log.info(f"    {tf_name}: {direction} ({inf_time:.1f}s)")

                    # Small delay between pairs to avoid rate limiting
                    time.sleep(CFG['loop']['delay_between_pairs'])

                    # Get current price from 1m data and store price history for charts
                    df_1m = pull_data(symbol, '1m', '1d')
                    if df_1m is not None and len(df_1m) > 0:
                        current_price = float(df_1m['close'].iloc[-1])
                        # Resample to 5m candles for charting (last ~200 candles)
                        chart_df = df_1m.copy()
                        chart_df_5m = chart_df.resample('5min').agg({
                            'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min'
                        }).dropna().tail(200)
                        price_history = []
                        for idx, row in chart_df_5m.iterrows():
                            price_history.append({
                                't': idx.isoformat(),
                                'o': round(float(row['open']), 6),
                                'h': round(float(row['high']), 6),
                                'l': round(float(row['low']), 6),
                                'c': round(float(row['close']), 6),
                            })
                        with state_lock:
                            state['price_history'][pair_name] = price_history
                            state['current_prices'][pair_name] = round(current_price, 6)
                    elif '5m' in timeframe_results:
                        current_price = float(timeframe_results['5m']['actual_close'][-1])

                    # Build signal — weighted voting
                    # Longer timeframes have more weight in the decision
                    TIMEFRAME_WEIGHTS = {tf: cfg.get('weight', 1) for tf, cfg in TIMEFRAMES.items()}
                    up_weight = sum(TIMEFRAME_WEIGHTS.get(tf, 1) for tf, d in directions.items() if d == 'UP')
                    down_weight = sum(TIMEFRAME_WEIGHTS.get(tf, 1) for tf, d in directions.items() if d == 'DOWN')
                    total_weight = sum(TIMEFRAME_WEIGHTS.values())
                    threshold = total_weight // 2 + 1  # strict majority
                    consensus = 'UP' if up_weight >= threshold else 'DOWN' if down_weight >= threshold else 'MIXED'
                    confluence = consensus in ('UP', 'DOWN')

                    # Correlation filter: if correlated pairs disagree, skip
                    # Groups: pairs sharing the same base/quote currency
                    CORRELATION_GROUPS = {
                        'EUR': ['EUR/USD', 'EUR/GBP', 'EUR/JPY', 'EUR/AUD', 'EUR/CHF', 'EUR/CAD', 'EUR/NZD'],
                        'USD': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CAD', 'USD/CHF', 'AUD/USD', 'NZD/USD'],
                        'GBP': ['GBP/USD', 'EUR/GBP', 'GBP/JPY', 'GBP/CHF', 'GBP/AUD'],
                        'JPY': ['USD/JPY', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY'],
                        'AUD': ['AUD/USD', 'EUR/AUD', 'AUD/JPY', 'AUD/CAD', 'AUD/NZD', 'GBP/AUD'],
                        'NZD': ['NZD/USD', 'EUR/NZD', 'NZD/CAD', 'AUD/NZD'],
                        'CAD': ['USD/CAD', 'EUR/CAD', 'AUD/CAD', 'NZD/CAD'],
                        'CHF': ['USD/CHF', 'EUR/CHF', 'GBP/CHF'],
                    }
                    # Find which groups this pair belongs to
                    pair_groups = [g for g, members in CORRELATION_GROUPS.items() if pair_name in members]
                    # Check if any correlated pair in the same group has a conflicting signal
                    correlation_conflict = False
                    if confluence:
                        for group in pair_groups:
                            for other_pair, other_signal in state.get('signals', {}).items():
                                if other_pair == pair_name:
                                    continue
                                if other_pair in CORRELATION_GROUPS[group]:
                                    other_consensus = other_signal.get('consensus')
                                    if other_consensus in ('UP', 'DOWN') and other_consensus != consensus:
                                        correlation_conflict = True
                                        log.info(f"    {pair_name}: correlation conflict with {other_pair} ({other_consensus} != {consensus})")
                                        break
                            if correlation_conflict:
                                break

                    tfs = {}
                    for tf_name, direction in directions.items():
                        r = timeframe_results[tf_name]
                        move_pct = (r['pred_close'][-1] - r['actual_close'][0]) / r['actual_close'][0] * 100
                        tfs[tf_name] = {'direction': direction, 'move_pct': round(move_pct, 4), 'inf_time': round(r['inf_time'], 1)}

                    signal = {
                        'timestamp': now.isoformat(),
                        'directions': tfs,
                        'confluence': confluence,
                        'consensus': consensus,
                    }

                    with state_lock:
                        state['signals'][pair_name] = signal
                        state['last_update'] = now.isoformat()
                        state['error'] = None

                    # Check stops
                    if current_price:
                        check_stops(pair_name, current_price)

                    # Check correlation filter: skip if correlated pairs disagree
                    if correlation_conflict:
                        log.info(f"    {pair_name}: skipped (correlation conflict)")
                        continue

                    # Check confluence for entry
                    if confluence and current_price:
                        with state_lock:
                            current_pos = state['positions'].get(pair_name)
                            no_position = current_pos is None
                            different_direction = current_pos and (
                                (consensus == 'UP' and current_pos['type'] == 'SHORT') or
                                (consensus == 'DOWN' and current_pos['type'] == 'LONG')
                            )
                        if no_position or different_direction:
                            open_position(pair_name, consensus, current_price)
                            log.info(f"    🟢 {pair_name} SIGNAL: {consensus}")
                        else:
                            log.info(f"    {pair_name}: holding {current_pos['type']}")
                    elif not confluence:
                        with state_lock:
                            current_pos = state['positions'].get(pair_name)
                        if current_pos and current_price:
                            close_position(pair_name, current_price, 'mixed_signal')
                            log.info(f"    {pair_name}: closed position — market mixed ({consensus})")
                        else:
                            log.info(f"    {pair_name}: no confluence ({consensus})")
                except Exception as e:
                    log.warning(f"    [{pair_name}] skipped due to error: {e}")
                    continue

            # Calculate unrealized P&L
            calc_unrealized_pnl()

            with state_lock:
                state['status'] = 'idle'

            save_state()  # end-of-cycle save (belt + suspenders)

        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)
            with state_lock:
                state['error'] = str(e)
                state['status'] = 'error'

        log.info(f"Sleeping {LOOP_SECONDS}s...")
        time.sleep(LOOP_SECONDS)


# ── Dashboard ───────────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kronos AutoTrader — Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
  .header h1 { font-size: 1.5em; color: #00d4aa; }
  .status { padding: 4px 12px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
  .status.running, .status.idle { background: #00d4aa22; color: #00d4aa; }
  .status.error { background: #ff444422; color: #ff4444; }
  .status.starting { background: #ffaa0022; color: #ffaa00; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px; }
  .card { background: #14141f; border: 1px solid #222; border-radius: 12px; padding: 16px; }
  .card h3 { font-size: 0.85em; color: #888; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 1.8em; font-weight: bold; }
  .card .sub { font-size: 0.8em; color: #888; margin-top: 4px; }
  .positive { color: #00d4aa; }
  .negative { color: #ff4444; }
  .pnl-row { margin-top: 8px; padding-top: 8px; border-top: 1px solid #222; font-size: 0.82em; }
  .pnl-row span { vertical-align: middle; }
  .tab-bar { display: flex; gap: 0; margin-bottom: 16px; border-bottom: 2px solid #222; }
.tab-btn { padding: 8px 24px; background: none; border: none; color: #666; cursor: pointer; font-size: 0.9em; font-weight: bold; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; }
.tab-btn:hover { color: #aaa; }
.tab-btn.active { color: #00d4aa; border-bottom-color: #00d4aa; }
.pair-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; margin-bottom: 20px; }
  .pair-card { background: #14141f; border: 1px solid #222; border-radius: 12px; padding: 14px; }
  .pair-card .pair-name { font-size: 1.1em; font-weight: bold; margin-bottom: 6px; }
  .pair-card .signal-badge { display: inline-block; padding: 3px 10px; border-radius: 8px; font-size: 0.8em; font-weight: bold; }
  .signal-badge.UP { background: #00d4aa22; color: #00d4aa; }
  .signal-badge.DOWN { background: #ff444422; color: #ff4444; }
  .signal-badge.MIXED { background: #ffaa0022; color: #ffaa00; }
  .signal-badge.FLAT { background: #88888822; color: #888; }
  .pair-card .chart-wrap { height: 120px; margin: 6px 0; position: relative; }
  .pair-card .chart-wrap canvas { width: 100% !important; height: 100% !important; }
  .pair-card .tf-row { display: flex; gap: 6px; margin-top: 6px; font-size: 0.78em; }
  .pair-card .tf-item { flex:1; background: #1a1a2e; border-radius: 6px; padding: 5px; text-align: center; }
  .pair-card .tf-dir { font-weight: bold; }
  .pair-card .tf-dir.UP { color: #00d4aa; }
  .pair-card .tf-dir.DOWN { color: #ff4444; }
  .pair-card .tf-dir.FLAT { color: #888; }
  .pair-card .position-info { margin-top: 6px; font-size: 0.78em; color: #888; line-height: 1.4; }
  .pair-card .position-info .pos-type { font-weight: bold; }
  .pair-card .position-info .LONG { color: #00d4aa; }
  .pair-card .position-info .SHORT { color: #ff4444; }
  .pair-card .pair-pnl { margin-top: 2px; font-size: 0.82em; font-weight: bold; }
  .trades-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  .trades-table th { text-align: left; padding: 8px; color: #888; border-bottom: 1px solid #222; }
  .trades-table td { padding: 8px; border-bottom: 1px solid #1a1a2e; }
  .pnl-pos { color: #00d4aa; }
  .pnl-neg { color: #ff4444; }
  .error-banner { background: #ff444422; border: 1px solid #ff4444; border-radius: 8px; padding: 12px; margin-bottom: 16px; color: #ff4444; font-size: 0.85em; display: none; }
  .update-time { font-size: 0.75em; color: #555; text-align: right; margin-top: 16px; }
  @media (max-width: 1200px) { .pair-cards { grid-template-columns: repeat(3, 1fr); } }
  @media (max-width: 900px) { .pair-cards { grid-template-columns: 1fr; } .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="header">
  <h1>⚡ Kronos AutoTrader</h1>
  <span id="status" class="status idle">idle</span>
</div>
<div id="error-banner" class="error-banner"></div>

<!-- Tab bar -->
<div class="tab-bar">
  <button class="tab-btn active" data-tab="forex" onclick="switchTab('forex')">💱 Forex (20)</button>
  <button class="tab-btn" data-tab="crypto" onclick="switchTab('crypto')">₿ Crypto (2)</button>
</div>

<!-- Summary cards -->
<div class="grid">
  <div class="card">
    <h3>💰 Balance</h3>
    <div class="value" id="balance">$10,000.00</div>
    <div class="sub" id="pnl-total"></div>
    <div class="pnl-row">
      <span style="color:#666;font-size:0.75em">Realized:</span>
      <span id="realized-pnl">$0.00</span>
      <span style="margin:0 8px;color:#333">|</span>
      <span style="color:#666;font-size:0.75em">Unrealized:</span>
      <span id="unrealized-pnl">$0.00</span>
      <span style="margin:0 8px;color:#333">|</span>
      <span id="live-balance" style="font-weight:bold"></span>
    </div>
    <div id="live-positions-pnl" style="margin-top:4px;font-size:0.78em"></div>
  </div>
  <div class="card">
    <h3>📊 Stats</h3>
    <div class="value" id="winrate">--</div>
    <div class="sub" id="trade-count">0 trades</div>
  </div>
  <div class="card">
    <h3>📈 Active Positions</h3>
    <div class="value" id="active-count">0</div>
    <div class="sub" id="active-pairs"></div>
  </div>
</div>

<!-- Pair cards -->
<div class="pair-cards" id="pair-cards"></div>

<!-- Trade history -->
<div class="card" style="margin-bottom:20px">
  <h3>📋 Trade History</h3>
  <table class="trades-table">
    <thead><tr><th>Time</th><th>Pair</th><th>Type</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead>
    <tbody id="trades-body"><tr><td colspan="7" style="color:#555">No trades yet</td></tr></tbody>
  </table>
</div>
<div class="update-time" id="update-time"></div>

<script>
const API = '/state';
const FOREX_PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/AUD', 'EUR/CHF', 'GBP/CHF', 'AUD/CAD', 'NZD/CAD', 'EUR/NZD', 'GBP/AUD', 'AUD/NZD', 'EUR/CAD'];
const CRYPTO_LIST = ['BTC/USD', 'ETH/USD'];
let activeTab = 'forex';
const charts = {};

function makeChart(canvasId, prices) {
  if (!prices || prices.length < 2) return null;
  const closes = prices.map(p => p.c);
  const labels = prices.map((p, i) =>
    i % Math.max(1, Math.floor(prices.length / 6)) === 0 ? p.t.substring(11, 16) : '');
  const isUp = closes[closes.length - 1] >= closes[0];
  const lineColor = isUp ? '#00d4aa' : '#ff4444';
  const fillColor = isUp ? 'rgba(0,212,170,0.08)' : 'rgba(255,68,68,0.08)';

  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        data: closes,
        borderColor: lineColor,
        borderWidth: 1.5,
        backgroundColor: fillColor,
        fill: true,
        pointRadius: 0,
        tension: 0.2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: true, mode: 'index', intersect: false } },
      scales: { x: { display: false }, y: { display: false } },
      interaction: { mode: 'index', intersect: false },
    }
  });
}

function updateChart(pair, prices) {
  if (!prices || prices.length < 2) return;
  const key = 'chart-' + pair.replace('/', '_');
  if (charts[key]) { charts[key].destroy(); }
  charts[key] = makeChart(key, prices);
}

async function refresh() {
  try {
    const r = await fetch(API);
    const s = await r.json();

    document.getElementById('status').textContent = s.status;
    document.getElementById('status').className = 'status ' + s.status;
    document.getElementById('update-time').textContent = 'Last update: ' + (s.last_update || '--');

    const errBanner = document.getElementById('error-banner');
    if (s.error) { errBanner.style.display = 'block'; errBanner.textContent = '\u26a0\ufe0f ' + s.error; }
    else { errBanner.style.display = 'none'; }

    // ── Balance with live P&L ──
    const bal = s.balance;
    const unrealized = (s.unrealized_pnl || {}).total || 0;
    const realizedPnl = bal - s.starting_balance;
    const liveBalance = bal + unrealized;
    const liveTotalPnl = liveBalance - s.starting_balance;

    document.getElementById('balance').textContent = '$' + bal.toFixed(2);

    const rPn = document.getElementById('realized-pnl');
    rPn.textContent = (realizedPnl >= 0 ? '+' : '') + '$' + realizedPnl.toFixed(2);
    rPn.className = realizedPnl >= 0 ? 'positive' : 'negative';

    const uPn = document.getElementById('unrealized-pnl');
    uPn.textContent = (unrealized >= 0 ? '+' : '') + '$' + unrealized.toFixed(2);
    uPn.className = unrealized >= 0 ? 'positive' : 'negative';

    const lb = document.getElementById('live-balance');
    lb.textContent = 'Live: $' + liveBalance.toFixed(2) + ' (' + (liveTotalPnl >= 0 ? '+' : '') + ((liveTotalPnl / s.starting_balance) * 100).toFixed(2) + '%)';
    lb.className = liveTotalPnl >= 0 ? 'positive' : 'negative';

    const pnlEl = document.getElementById('pnl-total');
    pnlEl.textContent = (liveTotalPnl >= 0 ? '+' : '') + '$' + liveTotalPnl.toFixed(2) + ' total P&L';
    pnlEl.className = 'sub ' + (liveTotalPnl >= 0 ? 'positive' : 'negative');

    // Per-position live P&L breakdown
    const livePosPnl = document.getElementById('live-positions-pnl');
    const posEntries = Object.entries(s.unrealized_pnl || {}).filter(function(e) { return e[0] !== 'total'; });
    if (posEntries.length > 0) {
      livePosPnl.innerHTML = posEntries.map(function(e) {
        var pair = e[0], upnl = e[1];
        return '<span style="color:#666">' + pair + ':</span> ' +
          '<span class="' + (upnl >= 0 ? 'positive' : 'negative') + '">' +
          (upnl >= 0 ? '+' : '') + '$' + upnl.toFixed(2) + '</span>';
      }).join('&nbsp;&nbsp;');
    } else { livePosPnl.innerHTML = ''; }

    // Stats
    const wr = s.total_trades > 0 ? ((s.winning_trades / s.total_trades) * 100).toFixed(0) : '--';
    document.getElementById('winrate').textContent = wr + '%';
    document.getElementById('trade-count').textContent = s.total_trades + ' trades (' + s.winning_trades + ' wins)';

    // Active positions
    const posCount = Object.keys(s.positions || {}).length;
    document.getElementById('active-count').textContent = posCount;
    document.getElementById('active-pairs').textContent = posCount > 0 ? Object.keys(s.positions).join(', ') : 'none';

    // ── Pair cards with charts ──
    const activePairs = activeTab === 'forex' ? FOREX_PAIRS : CRYPTO_LIST;
    const cardsDiv = document.getElementById('pair-cards');
    cardsDiv.innerHTML = activePairs.map(function(pair) {
      const sig = (s.signals || {})[pair] || {};
      const pos = (s.positions || {})[pair];
      const dirs = sig.directions || {};
      const con = sig.confluence;
      const consensus = sig.consensus || '--';
      const pairUpnl = (s.unrealized_pnl || {})[pair];
      const chartId = 'chart-' + pair.replace('/', '_');

      let badgeClass = consensus, badgeText = consensus;

      // Price chart canvas
      let chartHTML = '<div class="chart-wrap"><canvas id="' + chartId + '"></canvas></div>';

      // Live P&L for active positions
      let pnlHTML = '';
      if (pos && pairUpnl !== undefined) {
        const cls = pairUpnl >= 0 ? 'positive' : 'negative';
        pnlHTML = '<div class="pair-pnl ' + cls + '">P&L: ' + (pairUpnl >= 0 ? '+' : '') + '$' + pairUpnl.toFixed(2) + '</div>';
      }

      // Position details
      let posHTML = '';
      if (pos) {
        posHTML = '<div class="position-info"><span class="pos-type ' + pos.type + '">' + pos.type + '</span> @ ' + pos.entry_price +
          '<br>SL: ' + pos.sl + ' | TP: ' + pos.tp + '<br>Size: ' + pos.size + '</div>';
      }

      // Timeframe signals
      let tfHTML = '';
      if (Object.keys(dirs).length > 0) {
        tfHTML = '<div class="tf-row">' + Object.entries(dirs).map(function(e) {
          var tf = e[0], d = e[1];
          return '<div class="tf-item"><div style="font-size:0.7em;color:#666">' + tf + '</div>' +
            '<div class="tf-dir ' + d.direction + '">' +
            (d.direction === 'UP' ? '\u25b2' : d.direction === 'DOWN' ? '\u25bc' : '\u2014') + '</div>' +
            '<div style="font-size:0.65em;color:#666">' + (d.move_pct >= 0 ? '+' : '') + d.move_pct.toFixed(3) + '%</div></div>';
        }).join('') + '</div>';
      } else {
        tfHTML = '<div style="color:#555;font-size:0.8em;margin-top:6px">Waiting for data...</div>';
      }

      let conHTML = '';
      if (con) {
        conHTML = '<div style="margin-top:4px;font-size:0.75em;color:#00d4aa">\u2705 CONFLUENCE \u2014 ' + consensus + '</div>';
      } else if (consensus === 'MIXED') {
        conHTML = '<div style="margin-top:4px;font-size:0.75em;color:#ffaa00">\u26a0\ufe0f Mixed \u2014 no trade</div>';
      } else if (consensus === 'FLAT') {
        conHTML = '<div style="margin-top:4px;font-size:0.75em;color:#888">\u2014 Flat market</div>';
      }

      return '<div class="pair-card">' +
        '<div class="pair-name">' + pair + ' <span class="signal-badge ' + badgeClass + '">' + badgeText + '</span></div>' +
        chartHTML + pnlHTML + tfHTML + conHTML + posHTML + '</div>';
    }).join('');

    // Update charts after DOM is built
    activePairs.forEach(function(pair) {
      updateChart(pair, (s.price_history || {})[pair]);
    });

    // Trade history table
    const tbody = document.getElementById('trades-body');
    if (s.trade_history && s.trade_history.length > 0) {
      tbody.innerHTML = s.trade_history.slice().reverse().slice(0, 20).map(function(t) {
        return '<tr>' +
          '<td>' + (t.exit_time || '').substring(11, 19) + '</td>' +
          '<td>' + (t.symbol || '') + '</td>' +
          '<td style="color:' + (t.type === 'LONG' ? '#00d4aa' : '#ff4444') + '">' + t.type + '</td>' +
          '<td>' + t.entry + '</td>' +
          '<td>' + t.exit + '</td>' +
          '<td class="' + (t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg') + '">$' + t.pnl.toFixed(2) + '</td>' +
          '<td style="color:#888;font-size:0.8em">' + t.reason + '</td>' +
          '</tr>';
      }).join('');
    } else {
      tbody.innerHTML = '<tr><td colspan="7" style="color:#555">No trades yet</td></tr>';
    }
  } catch(e) { console.error(e); }
}
  function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(function(btn) {
      btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    refresh();
  }
  refresh();
  setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/state':
            calc_unrealized_pnl()
            with state_lock:
                data = json.dumps(state, default=str)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data.encode())
        elif self.path in ('/', '/dashboard'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *a): pass


def start_dashboard():
    """Start dashboard on configured port; fall back to next available if port is taken."""
    port = HTTP_PORT
    while True:
        try:
            server = HTTPServer(('0.0.0.0', port), DashboardHandler)
            log.info(f"Dashboard: http://localhost:{port}")
            server.serve_forever()
            return
        except OSError as e:
            if 'Address already in use' in str(e) and port < HTTP_PORT + 10:
                port += 1
                continue
            log.warning(f"Dashboard not available: {e}")
            return


if __name__ == '__main__':
    log.info("=" * 60)
    log.info("KRONOS AUTOTRADER v2 — Multi-Pair Paper Trading")
    log.info(f"Pairs: {', '.join(PAIRS.keys())}")
    log.info(f"Risk: {RISK_PER_TRADE*100}% per trade | SL: {STOP_LOSS_PIPS} pips | TP: {TAKE_PROFIT_PIPS} pips")
    log.info(f"Balance: ${INITIAL_BALANCE}")
    log.info(f"Loop: {LOOP_SECONDS}s | Dashboard: http://localhost:{HTTP_PORT}")
    log.info("=" * 60)

    threading.Thread(target=start_dashboard, daemon=True).start()
    trading_loop()
