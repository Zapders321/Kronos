#!/bin/bash
# ─────────────────────────────────────────────────────────────
# KRONOS – One-shot RunPod Setup (Model Soup Edition)
# Paste into RunPod web terminal.
# Downloads real 5yr 1d data, trains 5 model soup, uploads release.
# ─────────────────────────────────────────────────────────────
set -e

echo "========================================"
echo "  KRONOS Model Soup Setup — $(date)"
echo "========================================"

# ── Config ──
REPO_URL="https://github.com/Zapders321/Kronos"
BRANCH="master"

# ── 1. Install system deps ──
echo ">>> Installing system packages..."
apt-get update -qq
apt-get install -y -qq git python3-pip screen htop nvtop 2>/dev/null

# ── 2. Install Python deps ──
echo ">>> Installing Python packages..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q pandas numpy yfinance safetensors transformers matplotlib tqdm scipy

# ── 3. Clone repo ──
echo ">>> Cloning Kronos..."
cd /workspace
if [ -d "Kronos" ]; then
    cd Kronos && git pull
else
    git clone --branch "$BRANCH" "$REPO_URL"
    cd Kronos
fi

# ── 4. Install gh CLI for releases ──
echo ">>> Installing GitHub CLI..."
type gh >/dev/null 2>&1 || {
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
        dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
        tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    apt-get update -qq && apt-get install -y -qq gh
}
echo ">>> gh version: $(gh --version 2>/dev/null | head -1)"

# ── 5. Authenticate gh for release upload ──
echo ">>> gh auth status:"
gh auth status 2>&1 || echo "(gh not authenticated — release upload may fail)"

echo "========================================"
echo "  Setup complete!"
echo "  Launching model soup training..."
echo "  Steps:"
echo "    1. Download 5yr 1d data for 27 FX pairs"
echo "    2. Prepare windows + indicators"
echo "    3. Train 5 models (seeds 42-46) with profit loss + EMA"
echo "    4. Model soup: average all weights"
echo "    5. Upload as GitHub Release"
echo "========================================"

# ── 6. Launch training in screen session ──
screen -dmS kronos_soup bash -c "
    cd /workspace/Kronos
    python3 -u runpod_training/train_soup.py 2>&1 | tee /workspace/training.log
    echo ''
    echo '========== TRAINING COMPLETE ==========' >> /workspace/training.log
    date >> /workspace/training.log
"

echo ""
echo "✅ Training launched in screen session 'kronos_soup'"
echo ""
echo "📊 To check progress:"
echo "   screen -r kronos_soup"
echo "   (Ctrl+A, D to detach)"
echo ""
echo "📋 To see latest logs:"
echo "   tail -50 /workspace/training.log"
echo ""
echo "📦 To watch live:"
echo "   tail -f /workspace/training.log"
echo ""

# Give it 60s to confirm startup
sleep 60
echo "--- Screen sessions ---"
screen -ls 2>/dev/null || echo "(no screens found)"

echo "--- Recent log ---"
tail -30 /workspace/training.log 2>/dev/null || echo "(log not yet created)"