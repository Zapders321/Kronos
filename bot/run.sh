#!/bin/bash
# Kronos Engine Runner — auto-restarts on crash
cd "$(dirname "$0")"
VENV="../.venv/bin/python3"
LOG="engine.log"

echo "🔄 Kronos Engine Runner started at $(date)"
echo "PID: $$"
echo "Dashboard: http://localhost:8081/"
echo "---"

while true; do
    echo "[$(date)] Starting engine..."
    $VENV engine.py >> "$LOG" 2>&1
    EXIT_CODE=$?
    echo "[$(date)] Engine exited with code $EXIT_CODE — restarting in 3s..." >> "$LOG"
    sleep 3
done