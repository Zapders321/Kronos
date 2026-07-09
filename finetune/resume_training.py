#!/usr/bin/env python3
"""
Resume Kronos training E11-E12 from best_model checkpoint (E10).
Uses correct direction loss (soft probs, not hard binary).
"""
import os, sys, time, json, csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file as safetensors_save

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.kronos import KronosTokenizer, Kronos

FINETUNE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(FINETUNE_DIR, '..')
BOT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'kronos_base_finetuned', 'checkpoints', 'best_model')
DATASET_PATH = os.path.join(FINETUNE_DIR, 'data', 'processed_datasets')
CONFIDENCE_LOG = os.path.join(PROJECT_ROOT, 'bot', 'confidence_log.csv')

BATCH_SIZE = 16
CONTINUE_EPOCHS = 2
FEATURE_DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
DIRECTION_LOSS_WEIGHT = 0.3
CONTRASTIVE_LOSS_WEIGHT = 0.15
CONTRASTIVE_TEMPERATURE = 0.15
USE_CONTRASTIVE_LOSS = True
N_VAL = 500
N_TRAIN = 2000
WEIGHTED_SAMPLING = True
TIMEFRAME_WEIGHTS = {'5m':1,'15m':1,'30m':1,'1h':2,'4h':3,'1d':5}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
os.environ["OMP_NUM_THREADS"] = "1"

# ── Import correct loss functions from original training code ──
sys.path.insert(0, FINETUNE_DIR)
from feedback_train import compute_direction_loss, compute_label_smoothed_loss, track_confidence, print_confidence
from feedback_train import apply_feature_dropout, FxDataset
from finetune.contrastive import augment_ohlcv, nt_xent_loss

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_contrastive_loss(model, x, xs, tokenizer, temperature=0.15):
    bs = x.shape[0]
    if bs < 2: return torch.tensor(0.0, device=x.device, requires_grad=True)
    with torch.no_grad():
        t0t_orig, _ = tokenizer.encode(x, half=True)
    x_aug = augment_ohlcv(x)
    with torch.no_grad():
        t0t_aug, _ = tokenizer.encode(x_aug, half=True)
    t0t = torch.cat([t0t_orig, t0t_aug], dim=0)
    xs_double = torch.cat([xs, xs], dim=0)
    logits = model(t0t[:, :-1], t0t[:, :-1], xs_double[:, :-1, :])
    return nt_xent_loss(logits[0].mean(dim=1), temperature)

# ── Sleep protection ──
os.system("caffeinate -d -i -m -u -t 86400 &>/dev/null & disown")
log("✅ Sleep prevention active (caffeinate)")

# ── Load checkpoint ──
if not os.path.exists(BOT_MODEL_DIR):
    sys.exit(f"❌ No checkpoint at {BOT_MODEL_DIR}")
log(f"Loading checkpoint from {BOT_MODEL_DIR}...")
t_start = time.time()

with open(os.path.join(BOT_MODEL_DIR, 'config.json')) as f:
    cfg = json.load(f)
s1_bits = cfg.get('s1_bits', 10)

# Tokenizer
tokenizer = KronosTokenizer(
    d_in=cfg.get('d_in', 11), d_model=cfg.get('d_model', 832),
    n_heads=cfg.get('n_heads', 16), ff_dim=cfg.get('ff_dim', 2048),
    n_enc_layers=cfg.get('n_layers', 12)//2, n_dec_layers=cfg.get('n_layers', 12)//2,
    ffn_dropout_p=cfg.get('ffn_dropout_p', 0.2), attn_dropout_p=cfg.get('attn_dropout_p', 0.0),
    resid_dropout_p=cfg.get('resid_dropout_p', 0.2),
    s1_bits=cfg.get('s1_bits', 10), s2_bits=cfg.get('s2_bits', 10),
    beta=cfg.get('beta', 0.05), gamma0=cfg.get('gamma0', 1.0),
    gamma=cfg.get('gamma', 1.1), zeta=cfg.get('zeta', 0.05), group_size=cfg.get('group_size', 4),
)
tokenizer.load_state_dict(load_file(os.path.join(BOT_MODEL_DIR, 'tokenizer.safetensors')))
tokenizer = tokenizer.eval().to(device)
for p in tokenizer.parameters(): p.requires_grad = False

# Model
model = Kronos(
    s1_bits=cfg.get('s1_bits', 10), s2_bits=cfg.get('s2_bits', 10),
    n_layers=cfg.get('n_layers', 12), d_model=cfg.get('d_model', 832),
    n_heads=cfg.get('n_heads', 16), ff_dim=cfg.get('ff_dim', 2048),
    ffn_dropout_p=cfg.get('ffn_dropout_p', 0.2), attn_dropout_p=cfg.get('attn_dropout_p', 0.0),
    resid_dropout_p=cfg.get('resid_dropout_p', 0.2), token_dropout_p=cfg.get('token_dropout_p', 0.0),
    learn_te=cfg.get('learn_te', True),
)
model.load_state_dict(load_file(os.path.join(BOT_MODEL_DIR, 'model.safetensors')))
model = model.train().to(device)
log(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

# ── Load datasets ──
train_ds = FxDataset(data_path=os.path.join(DATASET_PATH, 'train_data.pkl'),
                     n_samples=N_TRAIN, weight_by_tf=WEIGHTED_SAMPLING,
                     tf_weights=TIMEFRAME_WEIGHTS, label='REAL')
val_ds = FxDataset(data_path=os.path.join(DATASET_PATH, 'val_data.pkl'),
                   n_samples=N_VAL, label='VAL')

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
total_batches = len(train_loader)

log(f"  Train: {len(train_ds)} windows ({total_batches} batches) | Val: {len(val_ds)} windows")

# ── Optimizer: correct LR continuation ──
# Original: OneCycleLR(max_lr=2e-5, total_steps=1500, pct_start=0.03, div_factor=10)
# At step 1250 (E10 end): LR ≈ 1.43e-6
RESUME_LR = 1.43e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=RESUME_LR, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=RESUME_LR, total_steps=CONTINUE_EPOCHS * total_batches,
    pct_start=0.05, div_factor=1.0, final_div_factor=100)

log(f"Training {CONTINUE_EPOCHS} epochs, LR={RESUME_LR:.2e} → {RESUME_LR/100:.2e}")
est = total_batches * CONTINUE_EPOCHS * 1.5 / 3600
log(f"Estimated: {est:.1f}h")

# ── Training loop ──
best_val = float('inf')
confidence_data = []

for local_ep in range(CONTINUE_EPOCHS):
    epoch = 11 + local_ep
    train_ds.set_epoch_seed(epoch)
    model.train()
    total_loss = 0.0; total_dir = 0.0; total_contra = 0.0
    n_batches = 0; t0_ep = time.time()

    for x, xs in train_loader:
        x, xs = x.to(device), xs.to(device)
        x = apply_feature_dropout(x, FEATURE_DROPOUT)
        optimizer.zero_grad()

        with torch.no_grad():
            t0t, t1t = tokenizer.encode(x, half=True)

        inp0, inp1 = t0t[:, :-1], t1t[:, :-1]
        out0, out1 = t0t[:, 1:], t1t[:, 1:]
        logits = model(inp0, inp1, xs[:, :-1, :])

        # ── Main loss (label smoothed) ──
        vocab_s1 = 2 ** s1_bits
        main_loss = compute_label_smoothed_loss(
            logits[0].reshape(-1, vocab_s1), out0.reshape(-1), vocab_s1, LABEL_SMOOTHING)
        loss = main_loss

        # ── Direction loss (correct soft-prob version) ──
        if DIRECTION_LOSS_WEIGHT > 0:
            dl = compute_direction_loss(logits, out0, out1, s1_bits)
            loss += DIRECTION_LOSS_WEIGHT * dl; total_dir += dl.item()

        # ── Contrastive loss ──
        if USE_CONTRASTIVE_LOSS:
            cl = compute_contrastive_loss(model, x, xs, tokenizer, CONTRASTIVE_TEMPERATURE)
            loss += CONTRASTIVE_LOSS_WEIGHT * cl; total_contra += cl.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step(); scheduler.step()

        total_loss += main_loss.item(); n_batches += 1

        if n_batches % 25 == 0:
            cur_lr = scheduler.get_last_lr()[0]
            log(f"E{epoch+1}/12 [REAL] {n_batches}/{total_batches} | "
                f"loss:{total_loss/n_batches:.4f} | dir:{total_dir/n_batches:.4f} | "
                f"lr:{cur_lr:.2e} | {time.time()-t0_ep:.0f}s")

        if n_batches % 500 == 0:
            safetensors_save({k: v.cpu() for k,v in model.state_dict().items()},
                             os.path.join(BOT_MODEL_DIR, 'model_resume_temp.safetensors'))

    # ── Validation ──
    model.eval()
    val_loss = 0.0; v_batches = 0; epoch_conf = []
    with torch.no_grad():
        for x, xs in val_loader:
            x, xs = x.to(device), xs.to(device)
            t0t, t1t = tokenizer.encode(x, half=True)
            vlogits = model(t0t[:, :-1], t1t[:, :-1], xs[:, :-1, :])
            l, _, _ = model.head.compute_loss(vlogits[0], vlogits[1], t0t[:, 1:], t1t[:, 1:])
            val_loss += l.item(); v_batches += 1
            epoch_conf.extend(track_confidence(vlogits, t0t[:, 1:], s1_bits))

    avg_train = total_loss / n_batches
    avg_val = val_loss / v_batches
    cur_lr = scheduler.get_last_lr()[0]
    conf_str = print_confidence(epoch_conf)

    log(f"E{epoch+1}/12 | REAL | train:{avg_train:.6f} dir:{total_dir/n_batches:.4f} "
        f"cont:{total_contra/n_batches:.4f} | val:{avg_val:.6f} | lr:{cur_lr:.2e}")
    log(f"Conf: {conf_str}")
    confidence_data.extend(epoch_conf)

    if avg_val < best_val:
        best_val = avg_val
        log(f"🏆 New best (val_loss: {avg_val:.6f})")

# ── Save ──
log("💾 Saving final model...")
os.makedirs(BOT_MODEL_DIR, exist_ok=True)
safetensors_save({k:v.cpu() for k,v in model.state_dict().items()},
                 os.path.join(BOT_MODEL_DIR, 'model.safetensors'))
safetensors_save({k:v.cpu() for k,v in tokenizer.state_dict().items()},
                 os.path.join(BOT_MODEL_DIR, 'tokenizer.safetensors'))
with open(os.path.join(BOT_MODEL_DIR, 'config.json'), 'w') as f:
    json.dump(cfg, f, indent=2)

total_sec = int(time.time() - t_start)
with open(os.path.join(BOT_MODEL_DIR, '..', 'training_info.json'), 'w') as f:
    json.dump({
        'best_val_loss': best_val if best_val != float('inf') else 0,
        'time_seconds': total_sec, 'epochs': 12
    }, f, indent=2)

if confidence_data:
    with open(CONFIDENCE_LOG, 'w') as f:
        w = csv.writer(f); w.writerow(['confidence','correct'])
        for c, ok in confidence_data: w.writerow([f'{c:.4f}', ok])

log(f"✅ Complete! {total_sec}s, best val_loss: {best_val}")