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

## Run 6: 2026-04-07 15:31 (Steps 0–2001, fresh training)

- **Log**: `logs/2026-04-07/15-31-50/`
- **Dataset**: `datasets/dataset` (expanded: 670 total files, 603 train / 67 test)
- **Config overrides**: `compile=false`, `optim.batch_size=64`, `optim.grad_acc=64`
- **Notable config changes vs Run 5**:
  - **47% more training data**: 603 train / 67 test (was 410 / 46)
  - `batch_size=64`, `grad_acc=64` (was 128/64) — effective batch halved to 64 due to 12GB VRAM constraint
  - `compile=false` — Triton/inductor backend unavailable on this machine
  - 1 corrupt audio file (`7f65a210-...wav`) skipped each epoch (HTML instead of WAV, download error)

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 2.818  | 50.3%     | 79.1%        | 90.5%     | 64.5%      |
| 400  | 2.489  | 57.1%     | 82.1%        | 91.7%     | 67.6%      |
| 600  | 2.342  | 60.1%     | 83.1%        | 92.3%     | 68.4%      |
| 800  | 2.249  | 62.4%     | 84.0%        | 92.4%     | 68.8%      |
| 1000 | 2.213  | 63.0%     | 84.2%        | 92.4%     | 68.9%      |
| 1200 | 2.179  | 63.4%     | 84.4%        | 92.5%     | 69.0%      |
| 1400 | 2.152  | 64.0%     | 84.5%        | 92.6%     | 69.1%      |
| 1600 | 2.152  | 64.2%     | 84.6%        | 92.6%     | 69.2%      |
| 1800 | 2.143  | 64.5%     | 84.6%        | 92.6%     | 69.1%      |
| 2000 | 2.144  | 64.5%     | 84.6%        | 92.5%     | 69.1%      |
| 2001 | 2.144  | 64.5%     | 84.6%        | 92.5%     | 69.1%      |

> Best timing accuracy: 64.5% at step 1800–2001. Plateaued after step ~1400. More data significantly improved other_acc (+1.3pp) and column_acc (+1.3pp) vs Run 5, but timing accuracy was flat (-0.1pp). The extra data helped note placement but not timing precision.

---

## Run 7: 2026-04-09 14:48 (Steps 0–4001, fresh training)

- **Log**: `logs/2026-04-09/14-48-24/`
- **Dataset**: `datasets/dataset` (669 total files, 602 train / 67 test — 1 corrupt sample removed)
- **Config overrides**: `compile=false`
- **Notable config changes vs Run 6**:
  - `timing_random_offset=1` (was 0) — jitter note times ±1 step during training
  - `context_types: timing→map` (was `none→map`) — **intended** to feed BPM/timing points as decoder input, but SnapBeatDataset didn't support it and silently fell back to `none` context
  - `base_lr=0.0001` (was 0.0002), `base_lr_2=0.00005` (was 0.0001) — halved
  - `total_steps=4000` (was 2000), `warmup_steps=400` (was 200) — doubled

### Eval Results

Eval metrics were not logged during training (context type filter mismatch — eval expected `timing` context but dataset produced `none`). Post-training evaluation via `test.py`:

| Metric | Value |
|--------|-------|
| Loss | 0.717 |
| Timing Acc | 55.4% |
| Fuzzy Timing | 84.5% |
| Other Acc | 92.5% |
| Column Acc | 69.2% |

> **`timing_random_offset=1` destroyed exact timing accuracy** (-9.1pp vs Run 6) while fuzzy timing was unchanged. The jitter trained the model to be imprecise. The `context_types: timing` config had no effect because SnapBeatDataset hardcoded `ContextType.NONE` — fixed post-Run 7 by implementing `SnapBeatParser.parse_timing()` and updating `SnapBeatDataset._process_chart()`. Lower LR + 4000 steps provided no benefit over Run 6's 2000 steps.

---

## Run 8: 2026-04-13 11:33 → 2026-04-15 19:56 (Steps 0–2001, fresh training; paused/resumed at step 1500)

- **Log**: `logs/2026-04-13/11-33-50/` (steps 0–1500), resumed at `logs/2026-04-15/09-49-00/` (steps 1500–2001)
- **Dataset**: `datasets/dataset` (669 total, 602 train / 67 test)
- **Config overrides**: `compile=false`
- **Notable config changes vs Run 6**:
  - **Timing context to decoder is now active**: `context_types: [{in: [timing], out: [map]}]`. `SnapBeatParser.parse_timing()` emits TIMING_POINT/MEASURE/BEAT events from BPM and `SnapBeatDataset._process_chart()` feeds them as decoder input context (the change that was intended for Run 7 but silently no-opped).
  - Same Run 6 training hyperparameters otherwise: `rhythm_weight=5.0`, `label_smoothing=0.05`, `timing_random_offset=0`, `base_lr=0.0002`, `total_steps=2000`, `batch_size=64`, `grad_acc=64`.

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 2.088  | 67.3%     | 82.4%        | 90.7%     | 65.1%      |
| 400  | 1.803  | 72.2%     | 84.3%        | 92.0%     | 67.9%      |
| 600  | 1.742  | 73.6%     | 84.9%        | 92.4%     | 68.5%      |
| 800  | 1.707  | 74.3%     | 85.2%        | 92.6%     | 69.1%      |
| 1000 | 1.679  | 74.8%     | 85.4%        | 92.7%     | 69.2%      |
| 1200 | 1.670  | 75.0%     | 85.6%        | 92.7%     | 69.4%      |
| 1400 | 1.669  | 75.1%     | 85.5%        | 92.8%     | 69.3%      |
| 1600 | 1.658  | 75.2%     | 85.7%        | 92.8%     | 69.5%      |
| 1800 | 1.657  | 75.2%     | 85.6%        | 92.8%     | 69.6%      |
| 2000 | 1.658  | **75.3%** | 85.6%        | **92.8%** | **69.6%**  |
| 2001 | 1.658  | **75.3%** | 85.6%        | **92.8%** | **69.6%**  |

> **Breakthrough run.** Feeding the BPM beat grid to the decoder broke the 64.6% timing plateau, jumping timing accuracy by **+10.7pp** over Run 5 and **+10.8pp** over Run 6 — all other metrics also reached new bests. Timing accuracy crossed 67% by step 200 (where Run 5 was at 51.6%) and kept climbing. The 10pp gap to fuzzy_timing shrank to ~10.3pp (from ~20pp), indicating the model can now pick the exact step much more often.

---

## Run 9: 2026-04-20 10:27 → 2026-04-21 15:37 (machine-interrupted at step 1470/2000)

- **Log**: `logs/2026-04-20/10-27-16/` (checkpoints saved: 500, 1000); stdout mirror at `logs/run9.log`
- **Dataset**: `datasets/dataset` (669 total, 602 train / 67 test)
- **Config overrides**: `compile=false`
- **Notable config changes vs Run 8**:
  - `rhythm_weight=8.0` (was 5.0) — push TIME_SHIFT loss weight higher to try to break past Run 8's 75.3% plateau.
  - Everything else identical to Run 8.

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 3.145  | 67.3%     | 82.3%        | 90.6%     | 65.1%      |
| 400  | 2.720  | 72.2%     | 84.3%        | 92.0%     | 67.8%      |
| 600  | 2.624  | 73.5%     | 84.9%        | 92.4%     | 68.7%      |
| 800  | 2.571  | 74.3%     | 85.2%        | 92.7%     | 69.2%      |
| 1000 | 2.532  | 74.6%     | 85.3%        | 92.7%     | 69.3%      |
| 1200 | 2.510  | 74.9%     | 85.4%        | 92.8%     | 69.5%      |
| 1400 | 2.501  | 74.95%    | 85.5%        | 92.8%     | 69.6%      |

> **`rhythm_weight=8.0` did not help.** Run 9 tracked just under Run 8's curve the entire way; last evaluated timing_acc (74.95% at step 1400) was 0.35pp below Run 8's best (75.3% at step 2001), with eval gains slowing to +0.04pp per 200 steps. Train loss kept falling (~2.0 → 0.47) while eval plateaued — classic generalization gap on a 602-sample dataset. Training was interrupted at step 1470 by a machine restart before the final ~600 steps of the cosine schedule; given the flat trajectory and the 0.35pp gap, completion was unlikely to beat Run 8. Takeaway: the plateau is a regularization / capacity problem, not a loss-weighting problem. Higher `rhythm_weight` makes the model fit train harder, not generalize better.

---

## Run 10: 2026-04-21 17:03 → 2026-04-23 08:56 (Steps 0–2001, fresh training; completed)

- **Log**: `logs/2026-04-21/17-03-51/` (checkpoints saved: 500, 1000, 1500, 2000, 2001)
- **Dataset**: `datasets/dataset` (669 total, 602 train / 67 test)
- **Config overrides**: `compile=false`
- **Notable config changes vs Run 8**:
  - `rhythm_weight: 5.0` (reverted from Run 9's 8.0 — Run 9 proved 8.0 is strictly worse than 5.0)
  - `label_smoothing: 0.1` (was 0.05)
  - `lora_dropout: 0.1` (was 0.05)
- **Pre-registered hypothesis**: regularization bundle would close Run 9's train/test gap and lift the plateau above Run 8's 75.3%.

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 2.196  | 65.71%    | 82.0%        | 90.2%     | 64.3%      |
| 400  | 1.881  | 71.21%    | 83.9%        | 91.7%     | 67.1%      |
| 600  | 1.807  | 72.76%    | 84.7%        | 92.3%     | 68.0%      |
| 800  | 1.772  | 73.40%    | 84.9%        | 92.5%     | 68.4%      |
| 1000 | 1.737  | 74.03%    | 85.1%        | 92.6%     | 68.9%      |
| 1200 | 1.720  | 74.29%    | 85.2%        | 92.7%     | 69.1%      |
| 1400 | 1.708  | 74.51%    | 85.4%        | 92.7%     | 69.1%      |
| 1600 | 1.697  | **74.55%** (peak) | 85.4% | 92.7% | 69.1%      |
| 1800 | 1.698  | 74.50%    | 85.4%        | 92.8%     | 69.3%      |
| 2000 | 1.697  | 74.53%    | 85.42%       | 92.76%    | 69.18%     |
| 2001 | 1.697  | 74.53%    | 85.42%       | 92.76%    | 69.18%     |

> **Regularization bundle under-performed.** Final timing_acc **74.53%** landed 0.77pp below Run 8's 75.3%. Peak (74.55%) occurred at step 1600, then drifted; last 600 steps added noise, not signal. Train/test gap *did* tighten — train loss ~0.55 vs test 1.70 (≈3.1×) compared to Run 9's ~0.47/2.50 (≈5.3×) — so the regularization worked mechanically, just net-negative on peak accuracy. All auxiliary metrics also dipped slightly vs Run 8 (fuzzy −0.3pp, other −0.04pp, column −0.4pp). Conclusion: Run 8's `label_smoothing=0.05` / `lora_dropout=0.05` already sat at a favorable trade-off; doubling both over-regularized. Further regularization is a dead end on this dataset.

---

## Run 11: 2026-04-23 09:13 → 2026-04-25 01:49 (Steps 0–2001, fresh training; completed)

- **Log**: `logs/2026-04-23/09-12-54/` (checkpoints saved: 500, 1000, 1500, 2000, 2001)
- **Dataset**: `datasets/dataset` (669 total, 602 train / 67 test)
- **Config overrides**: `compile=false`
- **Notable config changes vs Run 8**:
  - `lora.r: 64 → 128`
  - `lora.lora_alpha: 128 → 256` (keep α/r = 2 so effective scale unchanged)
  - Reverted `label_smoothing: 0.1 → 0.05` and `lora_dropout: 0.1 → 0.05` (Run 8 values, undoing Run 10's regularization)
  - Kept `rhythm_weight: 5.0`
- **Trainable params**: **51.9M** (2.00× Run 10's 25.95M); total 272.1M (frozen base unchanged at 220.2M).
- **VRAM**: 7.8 GB / 12 GB (+256 MiB vs Run 10).
- **Pre-registered hypothesis**: Run 8 may be LoRA-rank-bottlenecked; doubling rank gives the adapter more capacity to absorb the timing-conditional mapping. Final ≥76% = new baseline; 75.0–75.8% = marginal tie; <75% = capacity is not the bottleneck.

### Eval Results

| Step | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 2.094  | 67.08%    | 82.55%       | 90.70%    | 65.11%     |
| 400  | 1.832  | 72.07%    | 84.47%       | 92.21%    | 67.82%     |
| 600  | 1.761  | 73.19%    | 85.03%       | 92.59%    | 68.54%     |
| 800  | 1.723  | 74.23%    | 85.29%       | 92.59%    | 69.20%     |
| 1000 | 1.707  | 74.64%    | 85.62%       | 92.81%    | 69.41%     |
| 1200 | 1.695  | 74.72%    | 85.65%       | 92.82%    | 69.44%     |
| 1400 | 1.688  | 74.90%    | 85.77%       | 92.76%    | 69.56%     |
| 1600 | 1.675  | 75.09%    | 85.79%       | 92.87%    | 69.63%     |
| 1800 | 1.673  | **75.17%** (peak) | **85.83%** | 92.83% | **69.64%** |
| 2000 | 1.673  | 75.12%    | 85.84%       | 92.81%    | 69.63%     |
| 2001 | 1.673  | 75.12%    | 85.84%       | 92.81%    | 69.63%     |

> **Capacity hypothesis essentially rejected — tied Run 8 within noise.** Final 75.12% landed **0.18pp below Run 8's 75.3%**; peak (75.17% at step 1800) was 0.13pp short. Train/test gap stayed clean (~0.45 / 1.67 ≈ 3.7×) — no overfitting, just a hard accuracy ceiling. The step 1000 pre-reg gate (≥75%) was missed by 0.36pp but the run was let to finish for a clean comparison number. Run 11 did beat Run 10 by **+0.59pp** (74.53% → 75.12%), confirming r=128 is *slightly* better than r=64 + over-regularization, but doubling trainable params (26M → 52M) for a tied result with Run 8 isn't a net win. **Run 8 remains champion.** All three single-axis sweeps after Run 8 (loss-weighting in Run 9, regularization in Run 10, capacity in Run 11) failed to beat it — the plateau is now very likely a **data ceiling** at 602 train samples with this hyperparameter family.

---

## Run 12: Run 8 resume with low-LR cosine tail

- **Resume from**: `logs/2026-04-15/09-49-00/checkpoint-2000/lora` (Run 8 adapter weights only — added `lora_resume_path` field + `set_peft_model_state_dict` load path in `osuT5/train.py`, since `accelerator.load_state` would have restored Run 8's spent cosine schedule and immediately exited).
- **Config changes vs Run 8**:
  - `optim.base_lr: 0.0002 → 0.00005` (1/4 of Run 8's LR)
  - `optim.base_lr_2: 0.0001 → 0.000025`
  - `optim.warmup_steps: 200 → 50` (short warmup since model already converged)
  - `optim.total_steps: 2000 → 1000`
  - Everything else identical to Run 8 (`r=64`, `α=128`, `rhythm_weight=5`, `label_smoothing=0.05`, `lora_dropout=0.05`, timing context active).
- **Trainable params**: 25,952,256 (matches Run 8 exactly — confirms PiSSA round-trip preserved adapter shape).

### Eval at intermediate checkpoints

| Step | Loss   | timing_acc | fuzzy_timing | other_acc | column_acc |
|------|--------|-----------|--------------|-----------|------------|
| 200  | 1.6563 | 75.31%    | 85.67%       | 92.81%    | **69.67%** |
| 400  | 1.6555 | 75.40%    | 85.71%       | 92.88%    | 69.72%     |
| 600  | 1.6586 | 75.44%    | 85.71%       | 92.90%    | 69.55%     |
| 800  | 1.6588 | 75.47%    | 85.73%       | 92.89%    | 69.60%     |
| 1000 | 1.6588 | **75.53%** | **85.77%** | **92.90%** | 69.56%     |

> **New best (marginal): +0.23pp over Run 8.** Final 75.53% beats Run 8's 75.30% — first sweep since Run 8 to produce a real (above-noise, monotonic) gain, but small. Pre-registered gates: step 200 ≥ 75.3% **met** (75.31% — confirms PiSSA round-trip worked); step 500 ≥ 75.5% **missed** (~75.42% interpolated); final ≥ 76% for new-champion declaration **missed**. Train/test gap stayed clean at ~3.7× (~0.45 / 1.66) — no overfitting, just a slow climb. **Schedule axis confirmed as the only post-Run-8 lever that moves the needle, but the magnitude (+0.23pp) is too small to break the apparent ceiling.** All four cheap single-axis sweeps from Run 8 are now exhausted: loss-weighting (Run 9), regularization (Run 10), capacity (Run 11), and schedule (Run 12). The plateau at ~75.5% is now strongly attributed to the **data ceiling** at 602 train samples. Run 12 ships as the new baseline (`lukasnorland/rhythm-skeleton-mt3-v1`) — the +0.23pp is small but free given the artifact already exists.

---

## Run 13 (rev2): Run 12 warm-resume on the cleaned + expanded dataset

> **The data-ceiling test the project had been pointing at since Run 6.** Resumes Run 12's adapter on a much larger and cleaner SnapBeat dataset, continuing the same low-LR cosine tail. Ships as `lukasnorland/rhythm-skeleton-mt3-v2` — the new production baseline.

- **Log**: `logs/2026-05-12/15-34-47/` (final checkpoint: `checkpoint-2001/lora`)
- **Resume from**: `logs/2026-04-25/09-51-23/checkpoint-1001/lora` (Run 12 adapter — i.e., `lukasnorland/rhythm-skeleton-mt3-v1`) via `lora_resume_path` (PiSSA shape from base, weights from Run 12).
- **Dataset**: full SnapBeat corpus on `./datasets/dataset` after 2026-05-12 cleanup that **removed 10 corrupt audio/JSON pairs** (9 stems; 1 stem mapped to 2 UUID duplicates). Net: 2337 → 2327 paired samples. Discovered after Run 13's original cold-start version was showing degenerate behavior on a few hot-zone samples; root cause was audio/chart mismatch in the affected files.
  - `train`: `[0, 2093)` → **2093 samples** (~3.5× Run 12's 602).
  - `test`:  `[2093, 2327)` → **234 samples** (~3.5× Run 12's 67).
- **Original (discarded) Run 13**: a *fresh* cold-start at `base_lr=2e-4` on the same data. Pre-cleanup, ran for ~1500 steps, climbed visibly but wasted ~600 steps re-discovering what the Run 12 adapter already knew. **Logs and checkpoints purged after switching to the warm-resume rev2** (user request — clean history). All references in the table below are to rev2.
- **Config deltas vs Run 12** (everything else identical — same `r=64`, `α=128`, `rhythm_weight=5`, `label_smoothing=0.05`, `lora_dropout=0.05`, `timing` context active, Muon optimizer):
  - `data.train_dataset_path` / `test_dataset_path`: `piano7` subset → full `dataset` (mixed origin).
  - `data.train_dataset_end`: 602 → **2093**; `test_dataset_start/end`: 602/669 → **2093/2327**.
  - `optim.total_steps`: 1000 → **2000** (≈ 4.5 epochs at 2093 train samples, vs Run 12's 1.66 epochs at 602).
  - `optim.warmup_steps`: 50 (unchanged; resumes converged weights).
  - `optim.base_lr` / `base_lr_2`: 5e-5 / 2.5e-5 (unchanged; same low-LR cosine tail family as Run 12).

### Eval at intermediate checkpoints (new 234-sample test set)

| Step  | Loss   | timing_acc | fuzzy_timing | other_acc | column_acc |
|-------|--------|-----------|--------------|-----------|------------|
| 400   | 1.9844 | 69.17%    | 81.95%       | 91.69%    | 65.85%     |
| 800   | 1.9815 | 69.20%    | 81.98%       | 91.72%    | 65.96%     |
| 1200  | 1.9807 | 69.19%    | 81.98%       | 91.70%    | 65.92%     |
| 2000  | **1.9806** | **69.21%** | **82.00%** | **91.71%** | **65.99%** |

> Eval at step 2000 = epoch boundary (3199 iterations × 1254 s ≈ 21 min on this 12GB GPU). Curve is essentially flat between step 1200 and 2000 — the LR cosine has annealed to ~0 by step 1900 and the model has converged on the new data distribution.

### Apples-to-apples comparison: Run 12 adapter re-evaluated on the same 234-sample test set

The new test set is a different (harder, larger) split than the piano7-only one Run 12 was originally evaluated on. To isolate the *model* improvement from the *test-set* shift, the Run 12 adapter (`logs/2026-04-25/09-51-23/checkpoint-1001/lora`) was loaded fresh against the Run 13 rev2 config and the same test split. (`osuT5/test.py` was patched to tolerate batches missing the osu!-only `sample_weights` field — see `KeyError: 'sample_weights'` traceback; minimal `.get(...)` + `None`-aware branch into `calc_loss`.)

Eval results on the **same** new 234-sample test set:

| Model                        | Loss    | timing_acc | fuzzy_timing | other_acc | column_acc |
|------------------------------|---------|-----------|--------------|-----------|------------|
| Run 12 adapter (v1, no retraining) | 2.5061  | 64.29%    | 78.68%       | 90.77%    | 61.14%     |
| **Run 13 rev2 (v2, final)**        | **1.9806** | **69.21%** | **82.00%** | **91.71%** | **65.99%** |
| **Δ (v2 − v1)**              | **−0.5255** | **+4.92pp** | **+3.32pp** | **+0.94pp** | **+4.85pp** |

> **Data-ceiling hypothesis: REJECTED.** Run 13 rev2 beats Run 12 by **+4.92pp exact timing**, **+3.32pp fuzzy timing**, **+0.94pp other**, and **+4.85pp column** on the same evaluation set — the largest absolute timing gain since Run 8 broke the 65% plateau. Loss also drops by 0.53, confirming the improvement is not just argmax shuffling. The Run 9-12 conclusion that "we're at the data ceiling at 602 train samples" was **correct as an explanation of the plateau, and Run 13 rev2 is the test that confirms it**: 3.5× more (and cleaner) data + Run 12's converged warm-start + a fresh low-LR cosine tail moved every metric meaningfully.

> **Why Run 13's `0.6921` does NOT look better than Run 12's logged `0.7553`:** the two numbers were measured on *different* test sets. Run 12's logged `0.7553` was on the easier 67-sample piano7 split that included the 9–10 corrupt charts and was statistically noisy. On the *same* cleaned 234-sample split, Run 12 itself only scores `0.6429`. Cross-test-set absolute comparisons across the 2026-05-12 dataset boundary are not meaningful — use the apples-to-apples table above when comparing v1 and v2.

### Ships as `lukasnorland/rhythm-skeleton-mt3-v2`

Run 13 rev2 supersedes v1 as the production baseline. Per the versioned naming scheme established with v1, the v1 artifact is preserved unchanged for A/B comparisons. Update `_GAME_CODE_REGISTRY["MT3"]` in `snapbeat_inference.py` to resolve `MT3 → lukasnorland/rhythm-skeleton-mt3-v2`; see `docs/SNAPBEAT_FINETUNING.md`.

---

## Summary Comparison (Best Eval per Run)

| Run | Steps | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc | Notes |
|-----|-------|--------|-----------|--------------|-----------|------------|-------|
| 1   | 400   | 0.8348 | 41.7%     | 79.4%        | 89.8%     | 65.1%      | |
| 2   | 2001  | 0.7513 | 48.5%     | 82.6%        | 90.9%     | 67.3%      | |
| 3   | 500   | 0.6572 | 62.3%     | 84.3%        | 91.3%     | 67.3%      | +timing, +fc1/fc2 |
| 4   | 1000  | 0.6978 | 62.7%     | 84.5%        | 91.2%     | 67.5%      | resumed from Run 3 |
| 5   | 2001  | 2.178* | 64.6%     | 85.0%        | 91.3%     | 67.9%      | prior best timing |
| 6   | 2001  | 2.144* | 64.5%     | 84.6%        | 92.6%     | 69.2%      | +47% data |
| 7   | 4001  | 0.717  | 55.4%     | 84.5%        | 92.5%     | 69.2%      | timing_offset=1 hurt, timing context was no-op |
| 8   | 2001  | 1.658* | **75.3%** | **85.7%**    | **92.8%** | **69.6%**  | **best all metrics** — timing context in decoder |
| 9   | 1400† | 2.501* | 74.95%    | 85.5%        | 92.8%     | 69.6%      | rhythm_weight=8 underperformed Run 8; interrupted at step 1470 |
| 10  | 2001  | 1.697* | 74.53%    | 85.4%        | 92.8%     | 69.2%      | dropout+smoothing both → 0.1; over-regularized, −0.77pp vs Run 8 |
| 11  | 2001  | 1.673* | 75.12% (peak 75.17% @ 1800) | 85.84% | 92.81% | 69.63% | r=128, α=256; tied Run 8 within noise (−0.18pp), capacity not the bottleneck |
| 12  | 1000  | 1.659* | **75.53%**‡ | **85.77%**‡ | **92.90%**‡ | 69.56%‡ | Run 8 resume, base_lr=5e-5; +0.23pp over Run 8; shipped as `rhythm-skeleton-mt3-v1`. Numbers on old 67-sample test set. |
| 12  | (re-eval) | 2.506* | 64.29%§ | 78.68%§ | 90.77%§ | 61.14%§ | Same v1 adapter, **new 234-sample cleaned test set** — apples-to-apples vs Run 13 rev2 |
| 13 rev2 | 2000 | 1.981* | **69.21%**§ | **82.00%**§ | **91.71%**§ | **65.99%**§ | **new baseline** — Run 12 resume on cleaned 2327-sample dataset (2093 train / 234 test); +4.92pp timing vs Run 12 on same test set; ships as `rhythm-skeleton-mt3-v2` |

\* Runs 5–6, Run 9, Run 10, Run 11, Run 12, and Run 13 rev2 loss not directly comparable to earlier runs due to `rhythm_weight` / `label_smoothing` differences.
† Run 9 last eval step before machine interrupt — not a final-step result.
‡ Run 12's original eval: 67-sample piano7 test split (smaller, easier, included 9 corrupt charts removed 2026-05-12). **Not comparable** to Run 13 rev2 numbers.
§ Run 12 re-eval and Run 13 rev2: 234-sample cleaned mixed-origin test split. Comparable to each other.

### Key Takeaways

1. **Enabling `add_timing` and `add_timing_points`** was the single biggest improvement, boosting timing accuracy from ~48% to ~62% (Run 2→3).
2. **Expanding LoRA targets** to include `fc1` and `fc2` (feed-forward layers) gave the model more capacity to learn.
3. **Reducing `dt_augment_prob`** from 0.3 to 0.15 and setting `timing_random_offset=0` likely reduced noise in timing data.
4. Run 3+4 achieved in 1000 steps what Run 2 couldn't in 2000 steps, showing config matters more than training length.
5. **Run 5** pushed timing accuracy to **64.6%** (+1.9pp over Run 4) with `rhythm_weight=5`, `label_smoothing=0.05`, and `dt_augment_prob=0.0`.
6. **Run 6** added 47% more training data (603 vs 410 samples). This improved **column_acc by +1.3pp** and **other_acc by +1.3pp**, but timing accuracy was essentially unchanged (-0.1pp). This confirms that **timing precision is not data-limited**.
7. **Run 7** tested three changes simultaneously — all were ineffective or harmful:
   - `timing_random_offset=1` **destroyed exact timing** (-9.1pp) while fuzzy timing was unchanged. The jitter makes the model imprecise.
   - `context_types: timing→map` had **no effect** because SnapBeatDataset hardcoded `ContextType.NONE` — the timing context was never actually fed to the decoder. Fixed post-Run 7.
   - Lower LR (0.0001) + 4000 steps provided **no benefit** over Run 6's settings (2000 steps, 0.0002 LR).
8. **Run 8 broke the timing plateau.** With the decoder now receiving the BPM beat grid as input context (via the post-Run 7 fix), exact timing accuracy jumped from 64.6% → **75.3%** (+10.7pp) without any other config changes vs Run 6. Fuzzy timing, other_acc, and column_acc also reached new bests. This confirms the plateau was caused by the decoder lacking explicit beat-grid information, not by data or capacity limits.
9. **Run 9 ruled out higher `rhythm_weight`.** Bumping `rhythm_weight` from 5.0 → 8.0 on top of Run 8's config produced 74.95% timing_acc at step 1400 — 0.35pp *below* Run 8, with eval saturating while train loss kept falling. The Run 8 → Run 9 delta isolates the effect: loss re-weighting pushes the model to fit train harder, not to generalize.
10. **Run 10 ruled out more regularization.** Doubling both `label_smoothing` (0.05 → 0.1) and `lora_dropout` (0.05 → 0.1) on top of Run 8's config produced 74.53% final — 0.77pp *below* Run 8. The train/test gap *did* tighten (3.1× vs Run 9's 5.3×), so the regularization was mechanically effective but traded away peak accuracy. Combined with Run 9, this brackets the answer: Run 8's `rhythm_weight=5`, `label_smoothing=0.05`, `lora_dropout=0.05` already sit at a near-optimal trade-off for this dataset. Further loss-weighting / regularization tuning is exhausted — the remaining axes are **capacity** (LoRA rank) and **schedule** (resume with lower LR).
11. **Run 11 ruled out more capacity.** Doubling LoRA rank (64 → 128) with α scaled in step (128 → 256, keeping α/r=2) on top of Run 8's regularization produced 75.12% final / 75.17% peak — **0.18pp below Run 8** (within noise; effectively a tie). Train/test gap stayed clean at 3.7× — no overfitting from the extra params, just no extra signal from them either. Doubling trainable params from 26M → 52M for a tied result is a net loss. This now closes the third single-axis sweep against Run 8: **loss-weighting (Run 9), regularization (Run 10), and capacity (Run 11) all failed to beat 75.3%.** The plateau is most likely a **data ceiling** at 602 train samples, not a hyperparameter problem. Remaining cheap test: **schedule** — resume Run 8 with low LR (Run 12). After that, the realistic levers are data collection or accepting 75.3% and shipping.
12. **Run 12 confirmed schedule axis is marginal — and exhausts the cheap-sweep budget.** Resuming Run 8's adapter weights with `base_lr=5e-5` over 1000 steps produced a clean monotonic climb (75.31% → 75.40% → 75.44% → 75.47% → 75.53%) — the first sweep since Run 8 to produce an above-noise gain, but the magnitude was only **+0.23pp**. Pre-registered new-champion gate (≥76%) was missed by 0.47pp. Implementation note: `accelerator.load_state` would have restored Run 8's spent cosine schedule and exited at step 0 — added a `lora_resume_path` config field that loads adapter weights only via `set_peft_model_state_dict`, leaving the freshly-built optimizer + scheduler intact. **All four cheap single-axis sweeps from Run 8 are now exhausted (loss / regularization / capacity / schedule)**, each landing within ±0.8pp of 75.3%. The bracketing argument is tight: it isn't the loss weighting, the regularization, the rank, or the schedule. The conclusion most consistent with the data is a **dataset-size ceiling** at 602 samples for this architecture and tokenizer. Run 12 ships as the new baseline (`lukasnorland/rhythm-skeleton-mt3-v1`) since the +0.23pp is essentially free given the artifact already exists; further compute should pivot to data collection.
13. **Run 13 rev2 confirmed the data-ceiling diagnosis was correct — by breaking it.** Resuming v1's adapter on a 3.5× larger and cleaned dataset (2093 train / 234 test, after removing 10 corrupt audio/JSON pairs) and continuing the Run 12 low-LR cosine tail for 2000 steps moved every metric on the same evaluation set: **+4.92pp timing**, +3.32pp fuzzy, +0.94pp other, **+4.85pp column**, −0.53 loss. The diagnosis from takeaways 9–12 was right (it really was data, not hyperparameters), so the prescription worked the first time it was tried. Two methodological lessons from the process: **(a) cross-test-set comparisons are silent traps** — Run 12's logged 75.53% on the old 67-sample piano7 split looked ~6pp better than Run 13's 69.21% on the new 234-sample split, but on the *same* test set Run 12 only scores 64.29% and the model actually improved by ~5pp. Apples-to-apples re-eval of any reference model is mandatory whenever the test set changes. **(b) Data-quality audits are high-leverage** — the original Run 13 cold-start had hot-zone failures that traced to 9 audio/chart mismatches in the dataset (1 stem had 2 UUID duplicates → 10 paired files); removing those before re-launch as rev2 was a tiny edit that probably saved a lot of compute and let the warm-start strategy land cleanly. Ships as `lukasnorland/rhythm-skeleton-mt3-v2`.

---

## Next Steps: Improvement Suggestions

Run 13 rev2 is the new baseline `lukasnorland/rhythm-skeleton-mt3-v2`. Three changes have now meaningfully moved timing accuracy: **feeding the beat grid to the decoder** (Run 8, +10.7pp), **low-LR cosine tail** (Run 12, +0.23pp), and **3.5× more cleaned training data** (Run 13 rev2, +4.92pp on the same test set). The data-ceiling diagnosis from Runs 9–12 was correct; the prescription worked. Next priorities: **(a)** preserve the current eval split going forward so successor runs are directly comparable to v2; **(b)** investigate column accuracy (the weakest metric at 65.99%) as the next high-leverage axis now that timing has moved; **(c)** consider another data expansion + re-train cycle since the curve at step 2000 is flat but the model still has loss headroom.

### Tested and Confirmed

| # | Change | Result | Run |
|---|--------|--------|-----|
| 1 | **Feed BPM beat grid to decoder** (`context_types: timing→map` + `SnapBeatParser.parse_timing()`) | **+10.7pp exact timing** (64.6% → 75.3%), new bests on all metrics | Run 8 |
| 13 | **Low-LR cosine tail** on Run 8 weights (`base_lr=5e-5`, 1000 steps, fresh optimizer/scheduler via new `lora_resume_path`) | **+0.23pp** over Run 8 (75.30% → 75.53%), monotonic climb across 5 evals; missed ≥76% gate but free gain on top of existing artifact | Run 12 |
| 14 | **3.5× more cleaned training data** (602 → 2093 train samples after removing 10 corrupt audio/JSON pairs); warm-resume Run 12 adapter; continue low-LR cosine tail for 2000 steps | **+4.92pp timing**, +3.32pp fuzzy, +0.94pp other, **+4.85pp column**, −0.53 loss on the same 234-sample test set vs Run 12. Breaks the data ceiling — ships as `rhythm-skeleton-mt3-v2`. | Run 13 rev2 |

### Tested and Rejected

| # | Change | Result | Run |
|---|--------|--------|-----|
| ~~2~~ | `timing_random_offset=1` | -9.1pp exact timing, 0pp fuzzy — harmful | Run 7 |
| ~~3~~ | Lower LR (0.0001) + 4000 steps | No improvement over 0.0002 / 2000 steps | Run 7 |
| ~~4+5~~ | `label_smoothing: 0.1` + `lora_dropout: 0.1` | -0.77pp final vs Run 8; tightened gap but traded peak accuracy | Run 10 |
| ~~7~~ | `rhythm_weight: 5 → 8` | -0.35pp timing @ step 1400 vs Run 8; widens train/test gap without eval gain | Run 9 |
| ~~11~~ | More training data (+47%, 410 → 603 samples) | +1.3pp column/other, 0pp timing — *pre-Run-8 timing context*; revisit at scale | Run 6 |
| ~~12~~ | `lora.r: 64 → 128`, `lora_alpha: 128 → 256` (keep α/r=2) | -0.18pp final / -0.13pp peak vs Run 8 — tied within noise; doubled trainable params (26M → 52M) for no gain | Run 11 |

### Highest Priority Next

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 19 | **Another data expansion + warm-resume cycle on v2** | Run 13 rev2 confirmed the data axis is live: +4.92pp timing from 3.5× data. Collect more SnapBeat charts (especially varied mappers and BPM ranges), audit for audio/chart mismatches (the 10-file cleanup in this round was high-leverage), and warm-resume v2 with the same low-LR cosine tail. First experiment to repeat the recipe. | Add more JSON+audio pairs to `datasets/dataset`, `lora_resume_path: <v2 path>`, keep `total_steps: 2000` |
| 20 | **Column-accuracy investigation** | At 65.99%, column placement is now the lowest non-timing metric and has been the slowest-moving across all runs (varied by only ±1pp across Runs 6–13 vs ±15pp on timing). Possibly an inherent ceiling on mania lane assignment without per-mapper conditioning, or a tokenization-level limit. Worth a focused audit before more architectural changes. | Audit failure modes on `column_acc` (which UUIDs / which lane configurations); consider per-mapper conditioning |

### Medium Impact (compute fallbacks if data work is blocked)

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 6 | **Reduce effective batch size** | Run 6 used effective batch 64 (vs 128 in Run 5) and achieved comparable timing with better column/other acc. Halving again to 32 would double gradient updates per epoch. | `optim.batch_size=32, optim.grad_acc=32` |
| 15 | **Middle-ground regularization at r=128** | Combines Run 11's capacity with measured regularization (between Run 8's 0.05/0.05 and Run 10's 0.1/0.1). Only "compound" config not yet tested. Lower priority since Run 11 didn't show overfitting that needed correcting. | `lora.r: 128`, `lora_alpha: 256`, `label_smoothing: 0.075`, `lora_dropout: 0.075` |
| 18 | **Longer Run-12-style tail** (e.g. 2000–3000 steps at `base_lr=5e-5`) | Run 12's curve was still slowly climbing at step 1000 (+0.06pp from step 800 → 1000). Doubling step budget might extract another 0.1–0.2pp. Same family of experiment as Run 12; cheap. | `optim.total_steps: 2000–3000` |

### Lower Impact / Experimental

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 8 | **Reduce LoRA rank** | Less capacity = less overfitting risk. Rank 64 with 602 samples may still be overkill. | `lora.r: 32` (from 64) |
| 9 | **Snapping augmentation** | Randomly perturb snapping values during training for robustness. | `snapping_random_prob: 0.1` |
| 10 | **Multi-context training** | Train with both `[timing] → [map]` and `[none] → [map]` so the model learns both modes. | Add second entry to `context_types` and `context_weights` |
| 16 | **Base LR sweep at Run 8 config** | We never varied `base_lr` at the Run 8 setup (always 2e-4). Try 1e-4 and 5e-5 fresh runs (not just resume). Lower priority than Run 12 which tests the same axis more cheaply. | `optim.base_lr: 0.0001` or `0.00005` |
| 17 | **Target module sweep** | Currently `[q, k, v, out_proj, fc1, fc2]`. Try attention-only (drop fc1/fc2) or add `embed_tokens`. Cheap to test if Run 12 is uninformative. | `lora.target_modules: [...]` |
