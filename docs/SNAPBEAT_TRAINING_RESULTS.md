# SnapBeat LoRA Training Results

All runs use `train=snapbeat_lora` config with `OliBomby/Mapperatorinator-v31` as the pretrained base.

---

## Run 1: 2026-03-20 14:36 (Steps 0–500)

- **Log**: `logs/2026-03-20/14-36-24/`
- **Dataset**: `datasets/amaremix`
- **Config overrides**: `logging.log_with=tensorboard`, custom dataset paths
- **Notable settings**: `batch_size=16`, `grad_acc=8`, `total_steps=2000` (stopped early at 500), `timing_random_offset=1`, `add_timing=false`, `add_timing_points=false`, `dt_augment_prob=0.3`, LoRA targets: `k_proj, v_proj, q_proj, out_proj`

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 0.9232 | 36.9%     | 76.4%        | 88.2%     | 61.6%      |
| 400  | 0.8348 | 41.7%     | 79.4%        | 89.8%     | 65.1%      |

> Stopped at step 500 (checkpoint saved, no eval at 500).

---

## Run 2: 2026-03-23 09:34 (Steps 0–2001)

- **Log**: `logs/2026-03-23/09-34-43/`
- **Dataset**: `datasets/dataset`
- **Config overrides**: none (default `snapbeat_lora.yaml`)
- **Notable settings**: Same as Run 1 but with `batch_size=16`, `grad_acc=8`, `total_steps=2000`, `timing_random_offset=1`, `add_timing=false`, `add_timing_points=false`, `dt_augment_prob=0.3`, LoRA targets: `k_proj, v_proj, q_proj, out_proj`

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 0.9224 | 36.7%     | 76.5%        | 88.3%     | 61.6%      |
| 400  | 0.8329 | 42.1%     | 79.4%        | 89.7%     | 65.0%      |
| 600  | 0.7988 | 45.0%     | 80.8%        | 90.4%     | 65.8%      |
| 800  | 0.7793 | 46.4%     | 81.4%        | 90.6%     | 66.6%      |
| 1000 | 0.7717 | 46.8%     | 81.9%        | 90.6%     | 66.8%      |
| 1200 | 0.7598 | 47.9%     | 82.4%        | 90.9%     | 67.1%      |
| 1400 | 0.7559 | 48.2%     | 82.7%        | 90.9%     | 67.2%      |
| 1600 | 0.7525 | 48.2%     | 82.7%        | 91.0%     | 67.3%      |
| 1800 | 0.7511 | 48.5%     | 82.8%        | 91.0%     | 67.3%      |
| 2000 | 0.7515 | 48.7%     | 82.7%        | 91.0%     | 67.3%      |
| 2001 | 0.7513 | 48.5%     | 82.6%        | 90.9%     | 67.3%      |

> Plateaued around step 1600–2000. Timing accuracy capped at ~48.7%.

---

## Run 3: 2026-03-23 17:09 (Steps 0–500)

- **Log**: `logs/2026-03-23/17-09-07/`
- **Dataset**: `datasets/dataset`
- **Config overrides**: none
- **Notable config changes vs Run 2**:
  - `batch_size=128`, `grad_acc=64` (same effective batch, larger micro-batch)
  - `total_steps=1000`
  - `add_timing=true`, `add_timing_points=true` (newly enabled)
  - `timing_random_offset=0` (was 1)
  - `dt_augment_prob=0.15` (was 0.3)
  - LoRA targets expanded: added `fc1, fc2`

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 0.7490 | 54.8%     | 81.3%        | 89.9%     | 65.1%      |
| 400  | 0.6799 | 60.5%     | 83.7%        | 90.9%     | 67.0%      |
| 500  | 0.6572 | 62.3%     | 84.3%        | 91.3%     | 67.3%      |

> Big jump in timing accuracy (+14pp vs Run 2 at comparable steps). Key factors: enabling `add_timing`/`add_timing_points`, expanding LoRA targets to `fc1`/`fc2`.

---

## Run 4: 2026-03-24 11:48 (Steps 500–1000, resumed from Run 3)

- **Log**: `logs/2026-03-24/11-48-22/`
- **Dataset**: `datasets/dataset`
- **Resumed from**: `logs/2026-03-23/17-09-07/checkpoint-500`
- **Config**: Same as Run 3, with `compile=false` (to avoid OOM on 12GB GPU)

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 600  | 0.7100 | 61.9%     | 84.2%        | 91.0%     | 67.4%      |
| 800  | 0.6998 | 62.5%     | 84.4%        | 91.1%     | 67.5%      |
| 1000 | 0.6978 | 62.7%     | 84.5%        | 91.2%     | 67.5%      |
| 1001 | 0.6978 | 62.7%     | 84.5%        | 91.2%     | 67.5%      |

> Marginal gains from 500→1000 as cosine LR schedule wound down to 0. Training completed.

---

## Run 5: 2026-03-24 21:31 (Steps 0–2001, fresh training)

- **Log**: `logs/2026-03-24/21-31-48/`
- **Dataset**: `datasets/dataset`
- **Config overrides**: none
- **Notable config changes vs Run 3/4**:
  - `rhythm_weight=5.0` (was 3.0) — upweight TIME_SHIFT tokens in loss
  - `label_smoothing=0.05` (was 0.0) — slight smoothing for timing generalization
  - `dt_augment_prob=0.0` (was 0.15) — disabled, SnapBeat has fixed BPM
  - `total_steps=2000` (fresh run, not resumed)

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 2.763  | 51.6%     | 80.0%        | 89.4%     | 63.6%      |
| 400  | 2.479  | 57.8%     | 82.7%        | 90.5%     | 66.5%      |
| 600  | 2.339  | 60.3%     | 83.8%        | 91.0%     | 67.2%      |
| 800  | 2.266  | 62.0%     | 84.3%        | 91.1%     | 67.3%      |
| 1000 | 2.210  | 63.1%     | 84.7%        | 91.3%     | 67.6%      |
| 1200 | 2.196  | 63.8%     | 84.9%        | 91.2%     | 67.7%      |
| 1400 | 2.184  | 64.5%     | 84.9%        | 91.2%     | 67.9%      |
| 1600 | 2.180  | 64.4%     | 85.0%        | 91.3%     | 67.9%      |
| 1800 | 2.180  | 64.6%     | 85.0%        | 91.3%     | 67.9%      |
| 2000 | 2.178  | 64.6%     | 85.0%        | 91.3%     | 67.9%      |
| 2001 | 2.178  | 64.6%     | 85.0%        | 91.3%     | 67.9%      |

> Best timing accuracy: 64.6% at step 1800–2001. Plateaued after step ~1400. Test loss not directly comparable to earlier runs due to `rhythm_weight=5` and `label_smoothing=0.05`.

---

## Summary Comparison (Best Eval per Run)

| Run | Steps | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|-----|-------|--------|-----------|--------------|-----------|------------|
| 1   | 400   | 0.8348 | 41.7%     | 79.4%        | 89.8%     | 65.1%      |
| 2   | 2001  | 0.7513 | 48.5%     | 82.6%        | 90.9%     | 67.3%      |
| 3   | 500   | 0.6572 | 62.3%     | 84.3%        | 91.3%     | 67.3%      |
| 4   | 1000  | 0.6978 | 62.7%     | 84.5%        | 91.2%     | 67.5%      |
| 5   | 2001  | 2.178* | 64.6%     | 85.0%        | 91.3%     | 67.9%      |

\* Run 5 loss not directly comparable due to `rhythm_weight=5` and `label_smoothing=0.05`.

### Key Takeaways

1. **Enabling `add_timing` and `add_timing_points`** was the single biggest improvement, boosting timing accuracy from ~48% to ~62%.
2. **Expanding LoRA targets** to include `fc1` and `fc2` (feed-forward layers) gave the model more capacity to learn.
3. **Reducing `dt_augment_prob`** from 0.3 to 0.15 and setting `timing_random_offset=0` likely reduced noise in timing data.
4. Run 3+4 achieved in 1000 steps what Run 2 couldn't in 2000 steps, showing config matters more than training length.
5. Most gains happened in the first 500 steps of Run 3; the 500→1000 continuation (Run 4) added only ~0.4pp timing accuracy.
6. **Run 5** pushed timing accuracy to **64.6%** (+1.9pp over Run 4) with `rhythm_weight=5`, `label_smoothing=0.05`, and `dt_augment_prob=0.0`. Gains were diminishing — most improvement came by step 1000, with only +1.5pp from 1000→2000.

---

## Next Steps: Improvement Suggestions

The gap between timing_acc (64.6%) and fuzzy_timing_acc (85.0%) shows ~20% of predictions are off by just 1-2 steps (0.1-0.2ms). The model understands rhythm well but lacks precision — pointing toward regularization and richer context as the most promising directions.

### High Impact

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 1 | **More training data** | 410 samples is the biggest bottleneck. Every config trick has diminishing returns on a tiny dataset. Even 2-3x more charts would likely beat any hyperparameter change. | Add more charts to `datasets/dataset/json/` and `audio/`, update `train_dataset_end`/`test_dataset_end` |
| 2 | **Context type: `timing` -> `map`** | Currently the model gets no input context (audio only). Feeding BPM/timing points as decoder input gives explicit rhythmic structure to anchor predictions on — free information currently withheld. | `context_types: [{in: [timing], out: [map]}]` |
| 3 | **Timing offset augmentation** | Creates cheap data variations by jittering note times by +/-1 step (+/-0.1ms). Helps the model generalize instead of memorizing exact offsets from 410 samples. | `timing_random_offset: 1` |

### Medium Impact

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 4 | **Increase label smoothing** | More smoothing prevents overfitting to exact timing values on a small dataset. | `label_smoothing: 0.1` (from 0.05) |
| 5 | **Increase LoRA dropout** | More regularization for a small dataset — the model may be memorizing rather than generalizing. | `lora_dropout: 0.1-0.15` (from 0.05) |
| 6 | **Reduce effective batch size** | With 410 samples, each epoch is tiny relative to batch. Halving grad_acc doubles gradient updates per epoch for more diverse parameter updates. | `grad_acc: 32` (from 64) |
| 7 | **Lower LR + more steps** | Slower, steadier convergence on small data. Run 5 plateaued at step ~1400; a lower LR might find a better minimum. | `base_lr: 0.0001, base_lr_2: 0.00005, total_steps: 4000` |

### Lower Impact / Experimental

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 8 | **Reduce LoRA rank** | Less capacity = less overfitting risk. Rank 64 may be overkill for 410 samples. | `lora.r: 32` (from 64) |
| 9 | **Snapping augmentation** | Randomly perturb snapping values during training for robustness. | `snapping_random_prob: 0.1` |
| 10 | **Multi-context training** | Train with both `[timing] -> [map]` and `[none] -> [map]` so the model learns both modes. | Add second entry to `context_types` and `context_weights` |
