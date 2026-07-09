#!/bin/bash
# ─────────────────────────────────────────────────────────────
# KRONOS RunPod — simple setup, runpod_ensemble.py does the rest
# ─────────────────────────────────────────────────────────────
set -e

echo "========================================"
echo "  KRONOS Setup — $(date)"
echo "========================================"

# Install deps
apt-get update -qq && apt-get install -y -qq git python3-pip 2>/dev/null
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q pandas numpy yfinance safetensors transformers matplotlib tqdm scipy einops

# Install gh CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg 2>/dev/null | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo 'deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main' > /etc/apt/sources.list.d/github-cli.list
apt-get update -qq && apt-get install -y -qq gh 2>/dev/null

# Clone repo
cd /workspace
[ -d "Kronos" ] && rm -rf Kronos
git clone --branch clean-main https://github.com/Zapders321/Kronos.git
cd Kronos

echo "========================================"
echo "  ✅ Setup done — launching ensemble..."
echo "========================================"

# Run the Python ensemble script (direct, no screen)
nohup python3 -u runpod_training/runpod_ensemble.py > /workspace/training.log 2>&1 &

echo ""
echo "✅ Training launched in background (PID $!)"
echo "📊 tail -f /workspace/training.log"