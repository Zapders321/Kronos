#!/bin/bash
# ─────────────────────────────────────────────────────────
# KRONOS Deploy to RunPod — Run this on your Mac.
# Usage: bash deploy_to_runpod.sh <pod-ip> [model] [epochs] [batch]
#
# Example:
#   bash deploy_to_runpod.sh 194.26.xxx.xxx base 10 32
# ─────────────────────────────────────────────────────────
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pod-ip> [model] [epochs] [batch]"
    echo ""
    echo "  pod-ip  — your RunPod pod IP (shown in pod dashboard)"
    echo "  model   — 'small' or 'base' (default: base)"
    echo "  epochs  — number of epochs (default: 10)"
    echo "  batch   — batch size (default: 32)"
    echo ""
    echo "Example:"
    echo "  $0 194.26.123.45 base 10 32"
    exit 1
fi

POD_IP="$1"
MODEL="${2:-base}"
EPOCHS="${3:-10}"
BATCH="${4:-32}"
REPO_DIR="$HOME/.openclaw/workspace/repos/kronos"

echo "========================================"
echo " KRONOS — Deploy to RunPod"
echo " Pod:     $POD_IP"
echo " Model:   $MODEL"
echo " Epochs:  $EPOCHS"
echo " Batch:   $BATCH"
echo " Repo:    $REPO_DIR"
echo "========================================"
echo ""

# ── 1. Rsync repo to pod ──
echo ">>> Uploading repo to pod..."
rsync -avz --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
    --exclude '*.pyc' --exclude '.DS_Store' \
    "$REPO_DIR/" "root@$POD_IP:/workspace/Kronos/"

echo ""

# ── 2. Run training on pod ──
echo ">>> Starting training on pod..."
echo "    (This will take hours. It runs in a screen session so it keeps going if you disconnect.)"
echo ""

ssh -t "root@$POD_IP" "
    cd /workspace/Kronos
    chmod +x runpod_training/runpod_train_indicators.sh

    # Launch in a screen session so it survives SSH disconnect
    screen -dmS kronos_train bash runpod_training/runpod_train_indicators.sh $MODEL $EPOCHS $BATCH

    echo ''
    echo 'Training launched in screen session "kronos_train".'
    echo 'To check progress:  ssh root@$POD_IP \"screen -r kronos_train\"'
    echo 'To detach:          Ctrl+A, D'
    echo 'To see logs later:  tail -f /workspace/Kronos/finetune/outputs/logs/training.log'
    echo ''
    echo 'Waiting 30s to confirm it started...'
    sleep 30
    screen -ls
    echo ''
    echo 'Recent log output:'
    tail -20 /workspace/Kronos/finetune/outputs/logs/training.log 2>/dev/null || echo '(log not yet created)'
" || true

echo ""
echo "========================================"
echo " DEPLOYED!"
echo ""
echo "To check progress:"
echo "  ssh root@$POD_IP 'screen -r kronos_train'"
echo ""
echo "To check latest log:"
echo "  ssh root@$POD_IP 'tail -f /workspace/Kronos/finetune/outputs/logs/training.log'"
echo ""
echo "To download model when done:"
echo "  rsync -avz root@$POD_IP:/workspace/Kronos/finetune/outputs/models/kronos_indicator_finetuned/ ./downloaded_model/"
echo "========================================"