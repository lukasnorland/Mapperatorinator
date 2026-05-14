#!/usr/bin/env bash
# Run 13 (rev2) launcher: Option-1 data-ceiling test.
#
# Resumes Run 12's adapter (logs/2026-04-25/09-51-23/checkpoint-1001/lora, ships as
# `rhythm-skeleton-mt3-v1`) and continues training on the cleaned 2327-sample SnapBeat
# dataset (10 corrupt charts removed 2026-05-12) with Run 12's low-LR cosine tail.
#
# Hypothesis: more data + warm start beats Run 12's 75.53% timing_acc baseline.
#   - Pre-registered gates: step 1000 ≥ 75.5%, final ≥ 76% on the new 234-sample test set,
#     and re-eval against the old 67-sample test split for unambiguous champion status.
#
# The original Run 13 (fresh-from-v31 cold start, 3000 steps) was discarded because cold-start
# wasted ~600 steps relative to a warm resume; see docs/SNAPBEAT_TRAINING_RESULTS.md.
#
# Usage:
#   source /home/norland/envs/mapperatorinator/bin/activate
#   bash scripts/launch_run13.sh
#
# Output is tee'd to logs/run13.log (mirrors Run 9–12 convention).

set -euo pipefail

cd "$(dirname "$0")/.."

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: torch.cuda.is_available() is False."
    echo "       Fix the NVIDIA driver (kernel module not loaded) before launching Run 13."
    echo "       Suggested: sudo apt install --reinstall nvidia-dkms-580-open && sudo modprobe nvidia"
    exit 1
fi

mkdir -p logs
echo "=== Run 13 (rev2) launch: $(date -Iseconds) ==="
echo "Config: configs/train/snapbeat_lora.yaml"
echo "  train_dataset_end=2093, test_dataset_end=2327 (cleaned 2327-sample set)"
echo "  base_lr=5e-5, total_steps=2000, lora_resume_path=Run 12 adapter (rhythm-skeleton-mt3-v1)"
echo

python osuT5/train.py --config-name snapbeat_lora 2>&1 | tee logs/run13.log
