#!/usr/bin/env python3
"""
Kronos Training v2 — Two-Phase Training Pipeline

Phase 1: Self-supervised masked pre-training (3-4 hours)
  - Loads synthetic data, applies 15% random masking
  - Trains model to reconstruct masked positions
  - Saves pre-trained weights

Phase 2: Fine-tuning with all bells and whistles (overnight, ~8-9 hours)
  - Loads pre-trained weights from Phase 1
  - Full fine-tuning: profit loss + gradient accumulation + EMA + cosine annealing
  - Saves best model + EMA weights
  - Deploys to bot model directory

Usage:
  python3 finetune/train_v2.py [--phase1-only] [--phase2-only] [--skip-data-fetch]

Environment:
  Designed for MacBook M4 with MPS.
  Automatically disables sleep and configures MPS stability flags.

Output:
  Phase 1: outputs/pretrained_model/checkpoints/
  Phase 2: outputs/kronos_base_finetuned/checkpoints/best_model/ (+ EMA weights)
"""
import os, sys, json, time, argparse
from datetime import datetime

# ── Prevent sleep on MacBook ──
os.system("caffeinate -d -i -m -u -t 86400 &>/dev/null & disown")

# ── MPS stability ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

FINETUNE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(FINETUNE_DIR, '..')

sys.path.insert(0, FINETUNE_DIR)
sys.path.insert(0, PROJECT_ROOT)


def phase1_pretrain():
    """Run masked pre-training (Phase 1)."""
    print("\n" + "=" * 60)
    print("  🔮 PHASE 1: Masked Pre-Training")
    print("=" * 60)

    from pretrain_masked import pretrain as run_pretrain
    return run_pretrain()


def phase2_finetune():
    """Run fine-tuning with enhanced training features (Phase 2)."""
    print("\n" + "=" * 60)
    print("  🚀 PHASE 2: Fine-Tuning (profit loss + grad accum + EMA + cosine)")
    print("=" * 60)

    # ── Import and override config for Phase 2 ──
    import feedback_train as ft

    # Apply Phase 2 overrides
    ft.GRAD_ACCUM_STEPS = 4
    ft.EMA_DECAY = 0.995
    ft.PROFIT_LOSS_WEIGHT = 1.0
    ft.COSINE_RESTART_EPOCHS = 4
    ft.LABEL_SMOOTHING = 0.1
    ft.FEATURE_DROPOUT = 0.1
    ft.EPOCHS = 12
    ft.USE_CONTRASTIVE_LOSS = True
    ft.USE_SYNTHETIC_PRETRAIN = True
    ft.SYNTHETIC_CANDLES = 2000000
    ft.CURRICULUM_SYNTHETIC = 1
    ft.WEIGHTED_SAMPLING = True
    ft.DIRECTION_LOSS_WEIGHT = 0.3
    ft.CONTRASTIVE_LOSS_WEIGHT = 0.15

    # ── Deploy pre-trained weights if available ──
    pretrained_dir = os.path.join(PROJECT_ROOT, 'outputs', 'pretrained_model', 'checkpoints')
    pretrained_model_path = os.path.join(pretrained_dir, 'model.safetensors')
    if os.path.exists(pretrained_model_path):
        print(f"\n  📦 Found pre-trained weights at {pretrained_model_path}")
        print(f"  Will load them before fine-tuning (see feedback_train.py init logic)")
    else:
        print(f"\n  ⚠️  No pre-trained weights found at {pretrained_model_path}")
        print(f"  Proceeding with base model weights only")

    return ft.train()


def release_upload():
    """Pack best_model and upload as GitHub Release asset."""
    print("\n" + "=" * 60)
    print("  🚀 Uploading model to GitHub Release")
    print("=" * 60)

    import subprocess, tarfile

    model_dir = os.path.join(PROJECT_ROOT, 'outputs', 'kronos_base_finetuned', 'checkpoints', 'best_model')
    if not os.path.exists(model_dir):
        print(f"  ❌ No model at {model_dir}")
        return False

    # Pack into tar.gz
    tag = f"v2-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    archive_name = f"kronos-best-model-{tag}.tar.gz"
    archive_path = os.path.join(PROJECT_ROOT, 'outputs', archive_name)

    print(f"  📦 Packing {model_dir} → {archive_name}...")
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(model_dir, arcname='best_model')

    file_size = os.path.getsize(archive_path)
    print(f"  📦 Archive: {file_size/1e6:.1f}MB")

    # Create GitHub release
    print(f"  🏷️  Creating release: {tag}")
    result = subprocess.run([
        'gh', 'release', 'create', tag,
        '--title', f"Kronos v2 Model - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        '--notes', 'Trained with: masked pre-training + profit loss + gradient accumulation + EMA + cosine annealing',
        archive_path
    ], capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print(f"  ✅ Release created: https://github.com/Zapders321/Kronos/releases/tag/{tag}")
        # Clean up local archive
        os.remove(archive_path)
        print(f"  🧹 Cleaned up local archive")
        return True
    else:
        print(f"  ❌ Release failed: {result.stderr}")
        return False


def deploy():
    """Copy best model + EMA weights to bot directory."""
    print("\n" + "=" * 60)
    print("  📦 Deploying to bot model directory")
    print("=" * 60)

    import shutil

    src = os.path.join(PROJECT_ROOT, 'outputs', 'kronos_base_finetuned', 'checkpoints', 'best_model')
    dst = os.path.join(PROJECT_ROOT, 'bot', 'models', 'kronos_live')

    if not os.path.exists(src):
        print(f"  ❌ Source not found: {src}")
        return False

    os.makedirs(dst, exist_ok=True)

    for fname in ['model.safetensors', 'tokenizer.safetensors', 'config.json', 'model_ema.safetensors']:
        src_path = os.path.join(src, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dst, fname))
            print(f"  ✅ Copied {fname}")

    # Also copy confidence log
    conf_src = os.path.join(PROJECT_ROOT, 'bot', 'confidence_log.csv')
    if os.path.exists(conf_src):
        shutil.copy2(conf_src, os.path.join(dst, 'confidence_log.csv'))
        print(f"  ✅ Copied confidence_log.csv")

    print(f"  📍 Deployed to {dst}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Kronos Training v2 — Two-Phase Pipeline')
    parser.add_argument('--phase1-only', action='store_true', help='Run only masked pre-training')
    parser.add_argument('--phase2-only', action='store_true', help='Run only fine-tuning')
    parser.add_argument('--skip-data-fetch', action='store_true', help='Skip data fetch for Phase 2')
    parser.add_argument('--deploy', action='store_true', help='Deploy models after training')
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 60)
    print(f"  KRONOS TRAIN V2  {datetime.now().isoformat()}")
    print(f"  MacBook M4 — MPS enabled")
    print(f"  Flags: phase1={not args.phase2_only}, phase2={not args.phase1_only}, deploy={args.deploy}")
    print("=" * 60)

    # ── Phase 1: Masked Pre-Training ──
    if not args.phase2_only:
        if not phase1_pretrain():
            print("❌ Phase 1 failed — aborting")
            sys.exit(1)
        elapsed = time.time() - t_start
        print(f"\n  ⏱️  Phase 1 completed in {elapsed/60:.1f} minutes")

    # ── Phase 2: Fine-Tuning ──
    if not args.phase1_only:
        if not args.skip_data_fetch:
            from feedback_train import fetch_fresh_data
            if not fetch_fresh_data():
                print("❌ Data fetch failed")
                sys.exit(1)

        if not phase2_finetune():
            print("❌ Phase 2 failed — aborting")
            sys.exit(1)
        elapsed = time.time() - t_start
        print(f"\n  ⏱️  Phase 2 completed in {elapsed/60:.1f} minutes")

    # ── Deploy ──
    if args.deploy:
        deploy()
        release_upload()

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  ✅ TRAINING COMPLETE  {datetime.now().isoformat()}")
    print(f"  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print("=" * 60)

    return True


if __name__ == '__main__':
    main()