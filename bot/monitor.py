#!/usr/bin/env python3
"""
Kronos Trade Monitor — run via cron to detect and report trade events.
Tracks: entries, SL/TP hits, signal reversals.
Outputs alert text when changes detected.
"""
import json, urllib.request, os
from datetime import datetime, timezone

STATE_URL = 'http://localhost:8081/state'
TRACK_FILE = os.path.expanduser('~/.openclaw/workspace/repos/kronos/bot/.monitor_state.json')

def fetch_state():
    try:
        with urllib.request.urlopen(STATE_URL, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[monitor] fetch error: {e}")
        return None

def load_track():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE) as f:
            return json.load(f)
    return {'positions': {}, 'trade_count': 0, 'last_update': None}

def save_track(track):
    with open(TRACK_FILE, 'w') as f:
        json.dump(track, f, default=str)

def main():
    s = fetch_state()
    if s is None:
        return

    t = load_track()
    alerts = []

    old_positions = t.get('positions', {})
    old_trade_count = t.get('trade_count', 0)
    new_positions = s.get('positions', {})
    new_trade_count = s.get('total_trades', 0)
    new_history = s.get('trade_history', [])
    unrealized = s.get('unrealized_pnl', {})

    # Detect new trades opened
    for pair in new_positions:
        if pair not in old_positions:
            pos = new_positions[pair]
            alerts.append(f'🟢 NEW TRADE: {pair} {pos["type"]} @ {pos["entry_price"]:.5f} | SL: {pos["sl"]:.5f} | TP: {pos["tp"]:.5f}')

    # Detect closed trades
    if new_trade_count > old_trade_count:
        new_trades = new_history[-(new_trade_count - old_trade_count):]
        for trade in new_trades:
            emoji = '✅' if trade['pnl'] > 0 else '🔴'
            alerts.append(f'{emoji} CLOSED: {trade["symbol"]} {trade["type"]} | Entry: {trade["entry"]:.5f} → Exit: {trade["exit"]:.5f} | P&L: ${trade["pnl"]:.2f} | {trade["reason"]}')

    # Detect significant unrealized P&L movement on open positions (> $1 change)
    old_upnl = t.get('unrealized_pnl', {})
    new_upnl = unrealized
    for pair in new_positions:
        old_val = old_upnl.get(pair, 0)
        new_val = new_upnl.get(pair, 0)
        if abs(new_val - old_val) > 1.0:
            direction = '📈' if new_val > old_val else '📉'
            alerts.append(f'{direction} {pair} P&L moved: ${old_val:.2f} → ${new_val:.2f}')

    # Print alerts
    if alerts:
        total = unrealized.get('total', 0)
        bal = s['balance']
        live = bal + total
        header = f'⚡ Kronos @ {datetime.now(timezone.utc).strftime("%H:%M")} UTC | Live: ${live:.2f} | {len(new_positions)} pos | {new_trade_count} trades'
        print(header)
        for a in alerts:
            print(a)

    # Update track file
    t['positions'] = {k: {'type': v['type'], 'entry_price': v['entry_price']} for k, v in new_positions.items()}
    t['trade_count'] = new_trade_count
    t['unrealized_pnl'] = {k: v for k, v in unrealized.items()}
    t['last_update'] = s.get('last_update')
    save_track(t)

if __name__ == '__main__':
    main()
