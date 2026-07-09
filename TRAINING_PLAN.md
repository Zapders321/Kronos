# KRONOS TRAINING PLAN v2 — Fri 2026-07-10 23:00

Run on **MacBook Air M4** with sleep prevention. Estimated: ~20 hours.

---

## New Features (building now)

### 1. Self-Supervised Masked Pre-Training (Phase 1)

**What:** Download ALL yfinance 1d data (years). Randomly mask 15% of timesteps per window. Train model to reconstruct them — like BERT for time series.

**Why:** Model learns deep price structure before any trading objective.

**File:** `finetune/pretrain_masked.py` → saves to `outputs/pretrained/`

**Time:** ~3-4 hours on M4

### 2. Profit-Based Loss

**What:** Differentiable loss that rewards profitable directional bets, not just accurate predictions.

```
pos_prob = softmax prob above midpoint
action = 2 × pos_prob - 1  (+1 long, -1 short)
profit_loss = -mean(action × actual_return)
```

Combined: `loss = main_loss + 0.3*direction_loss + 0.15*profit_loss + 0.15*contrastive_loss`

### 3. Gradient Accumulation

**What:** Accumulate gradients over 4 steps before optimizer update. Effective batch = 16 × 4 = 64. Stabler gradients, fewer deadlocks.

### 4. EMA Weights

**What:** `ema = 0.995 × ema + 0.005 × weights` every step. EMA weights always slightly better than raw weights at inference.

### 5. Cosine Annealing with Restarts

**What:** LR restarts every 4 epochs. Escapes bad minima, finds better solutions. Proven ~5% better than OneCycleLR for long runs.

---

## Pipeline (23:00 start)

```
23:00 — 03:00  Phase 1: Masked pre-training (1d data, all pairs)
03:00 — 19:00  Phase 2: Fine-tune 12 epochs
                ├─ Load pre-trained weights
                ├─ Synthetic epoch 1
                ├─ Real epochs 2-12
                ├─ Profit + direction + contrastive losses
                ├─ Gradient accumulation (batch 64)
                ├─ EMA weights
                ├─ Cosine annealing restarts (every 4 epochs)
                └─ Save best + EMA to bot model dir
```

**Estimated finish:** ~19:00 Friday

## Skipped (needs RTX 5080)
- Model soup (3-run weight averaging)

## What To Do
- Nothing. Cron fires at 23:00.
- Check tomorrow afternoon — results will be announced to Telegram.