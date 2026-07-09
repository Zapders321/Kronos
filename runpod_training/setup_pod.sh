#!/bin/bash
# ─────────────────────────────────────────────────────────────
# KRONOS RunPod — uses existing feedback_train.py correctly
# ─────────────────────────────────────────────────────────────
set -e

echo "========================================"
echo "  KRONOS Ensemble — $(date)"
echo "========================================"

REPO_URL="https://github.com/Zapders321/Kronos"
BRANCH="clean-main"

# ── 1. Install deps ──
echo ">>> Installing packages..."
apt-get update -qq && apt-get install -y -qq git python3-pip screen 2>/dev/null
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q pandas numpy yfinance safetensors transformers matplotlib tqdm scipy einops

# ── 2. Clone repo ──
echo ">>> Cloning Kronos..."
cd /workspace
[ -d "Kronos" ] && cd Kronos && git pull || { git clone --branch "$BRANCH" "$REPO_URL" && cd Kronos; }

# ── 3. Create config ──
mkdir -p bot/models/kronos_live
python3 -c "
import json
json.dump({
    'd_model': 512, 'n_heads': 8, 'ff_dim': 1024, 'n_layers': 6,
    'n_enc_layers': 3, 'n_dec_layers': 3,
    'ffn_dropout_p': 0.2, 'attn_dropout_p': 0.0, 'resid_dropout_p': 0.2,
    'token_dropout_p': 0.0, 'learn_te': True,
    's1_bits': 10, 's2_bits': 10,
    'beta': 0.05, 'gamma0': 1.0, 'gamma': 1.1, 'zeta': 0.05, 'group_size': 4,
    'd_in': 38,
}, open('bot/models/kronos_live/config.json', 'w'), indent=2)
"

# ── 4. Patch feedback_train to skip data fetch ──
echo ">>> Patching feedback_train for RunPod..."
sed -i 's/if not fetch_fresh_data():/if False: # skip fetch on RunPod/' finetune/feedback_train.py
sed -i 's/^SEED = 42/SEED = 42/' finetune/feedback_train.py  # no-op, just to have it

# Set training params
sed -i 's/^EPOCHS = .*/EPOCHS = 12/' finetune/feedback_train.py
sed -i 's/^GRAD_ACCUM_STEPS = .*/GRAD_ACCUM_STEPS = 4/' finetune/feedback_train.py
sed -i 's/^EMA_DECAY = .*/EMA_DECAY = 0.995/' finetune/feedback_train.py
sed -i 's/^PROFIT_LOSS_WEIGHT = .*/PROFIT_LOSS_WEIGHT = 1.0/' finetune/feedback_train.py
sed -i 's/^COSINE_RESTART_EPOCHS = .*/COSINE_RESTART_EPOCHS = 4/' finetune/feedback_train.py
sed -i 's/^DIRECTION_LOSS_WEIGHT = .*/DIRECTION_LOSS_WEIGHT = 0.3/' finetune/feedback_train.py
sed -i 's/^CONTRASTIVE_LOSS_WEIGHT = .*/CONTRASTIVE_LOSS_WEIGHT = 0.15/' finetune/feedback_train.py
sed -i 's/^USE_CONTRASTIVE_LOSS = .*/USE_CONTRASTIVE_LOSS = True/' finetune/feedback_train.py
sed -i 's/^USE_SYNTHETIC_PRETRAIN = .*/USE_SYNTHETIC_PRETRAIN = False/' finetune/feedback_train.py
sed -i 's/^WEIGHTED_SAMPLING = .*/WEIGHTED_SAMPLING = True/' finetune/feedback_train.py
sed -i 's/^BATCH_SIZE = .*/BATCH_SIZE = 32/' finetune/feedback_train.py
sed -i 's/^LEARNING_RATE = .*/LEARNING_RATE = 5e-5/' finetune/feedback_train.py

# Install gh CLI
type gh >/dev/null 2>&1 || { curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg 2>/dev/null | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg; echo 'deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main' | tee /etc/apt/sources.list.d/github-cli.list >/dev/null; apt-get update -qq && apt-get install -y -qq gh; }

echo "========================================"
echo "  Setup complete!"
echo "========================================"

# ── 5. Launch training ──
screen -dmS kronos_soup bash -c "
    cd /workspace/Kronos

    echo '=== 1d DATA (5yr) ==='
    python3 -u runpod_training/prep_data.py 1d 5y 2>&1
    mkdir -p /workspace/models/soup_1d

    for seed in 42 43 44 45 46; do
        echo \"--- 1d Seed \$seed ---\"
        sed -i 's/^SEED = .*/SEED = '\$seed'/' finetune/feedback_train.py
        python3 -u finetune/feedback_train.py 2>&1
        cp outputs/kronos_base_finetuned/checkpoints/best_model/model.safetensors /workspace/models/soup_1d/seed_\${seed}.safetensors
        cp outputs/kronos_base_finetuned/checkpoints/best_model/model_ema.safetensors /workspace/models/soup_1d/seed_\${seed}_ema.safetensors 2>/dev/null || true
    done

    echo '=== AVERAGE 1d ==='
    python3 -c \"
import torch, os
from collections import OrderedDict
from safetensors.torch import save_file, load_file
sds = [load_file(f'/workspace/models/soup_1d/seed_{s}.safetensors') for s in [42,43,44,45,46]]
avg = OrderedDict((k, torch.stack([sd[k].float() for sd in sds]).mean(0)) for k in sds[0])
os.makedirs('outputs/soup_1d', exist_ok=True)
save_file(avg, 'outputs/soup_1d/model.safetensors')
print('✅ 1d soup')
\"

    echo '=== 1h DATA (2yr) ==='
    python3 -u runpod_training/prep_data.py 1h 2y 2>&1
    mkdir -p /workspace/models/soup_1h

    for seed in 42 43 44 45 46; do
        echo \"--- 1h Seed \$seed ---\"
        sed -i 's/^SEED = .*/SEED = '\$seed'/' finetune/feedback_train.py
        python3 -u finetune/feedback_train.py 2>&1
        cp outputs/kronos_base_finetuned/checkpoints/best_model/model.safetensors /workspace/models/soup_1h/seed_\${seed}.safetensors
        cp outputs/kronos_base_finetuned/checkpoints/best_model/model_ema.safetensors /workspace/models/soup_1h/seed_\${seed}_ema.safetensors 2>/dev/null || true
    done

    echo '=== AVERAGE 1h ==='
    python3 -c \"
import torch, os
from collections import OrderedDict
from safetensors.torch import save_file, load_file
sds = [load_file(f'/workspace/models/soup_1h/seed_{s}.safetensors') for s in [42,43,44,45,46]]
avg = OrderedDict((k, torch.stack([sd[k].float() for sd in sds]).mean(0)) for k in sds[0])
os.makedirs('outputs/soup_1h', exist_ok=True)
save_file(avg, 'outputs/soup_1h/model.safetensors')
print('✅ 1h soup')
\"

    echo '=== UPLOAD RELEASE ==='
    TAG=\"v2-ensemble-\$(date +%Y%m%d-%H%M%S)\"
    tar -czf /workspace/kronos-\${TAG}.tar.gz -C outputs soup_1d soup_1h
    gh release create \$TAG --title \"Kronos v2 - \$(date)\" --notes 'Two-soup: 1d+1h, 5 seeds each, profit loss+EMA+grad accum' /workspace/kronos-\${TAG}.tar.gz 2>&1 || echo 'Release upload failed'

    echo '========== DONE =========='
    date
" 2>&1 | tee /workspace/training.log

echo ""
echo "✅ Training in screen 'kronos_soup'"
echo "📊 tail -f /workspace/training.log"