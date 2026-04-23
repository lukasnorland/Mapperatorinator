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

## Run 11: planned — test LoRA capacity

- **Config changes vs Run 8** (applied in `configs/train/snapbeat_lora.yaml` for the run):
  - `lora.r: 64 → 128`
  - `lora.lora_alpha: 128 → 256` (keep α/r = 2)
  - Revert `label_smoothing: 0.1 → 0.05` (Run 8 value)
  - Revert `lora_dropout: 0.1 → 0.05` (Run 8 value)
  - Keep `rhythm_weight: 5.0`
- **Rationale**: Run 9 tested "fit harder" (higher `rhythm_weight`) — worse. Run 10 tested "regularize more" — also worse. The remaining untested axis is **capacity**: Run 8 may be LoRA-rank-bottlenecked rather than data- or loss-bottlenecked. Doubling rank gives the adapter more dimensions to absorb the timing-conditional mapping without touching base weights. α doubles in step to keep effective scale (`α/r`) constant, so LR dynamics don't silently shift.
- **Risk**: more parameters on 602 samples could overfit harder. Mitigation: keep Run 8's regularization (don't stack capacity + weaker reg), and watch train/test loss gap at step 1000–1400 — if it blows past Run 9's 5.3× before eval improves over Run 8, abort.
- **Pre-registered expectation**:
  - Step 200 timing_acc ≥ 67% (match or beat Run 8's 67.3%) — early read on whether capacity is helping.
  - Step 1000 timing_acc ≥ 75% (Run 8 was at 74.8% at step 1000) — would justify completing the run.
  - Final timing_acc ≥ 76% would make Run 11 the new baseline; 75.0–75.8% = marginal; <75% = capacity is not the bottleneck and the 602-sample dataset is the ceiling.
- **VRAM check needed before launch**: r=128 doubles LoRA params (~52M trainable vs ~26M). Current run uses 7.5 GB / 12 GB; likely fits, but first step should be confirmed before leaving unattended.
- **Fallback if Run 11 fails**: resume from Run 8's `checkpoint-2000` with `base_lr=5e-5`, ~500 additional steps — refines the existing best instead of re-searching.

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

\* Runs 5–6, Run 9, and Run 10 loss not directly comparable to earlier runs due to `rhythm_weight` / `label_smoothing` differences.
† Run 9 last eval step before machine interrupt — not a final-step result.

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

---

## Next Steps: Improvement Suggestions

Run 8 broke the timing plateau: exact timing accuracy reached **75.3%** (up from 64.6%), and the gap to fuzzy_timing_acc (85.7%) narrowed to ~10pp (from 20pp). Runs 9 and 10 both tried to exceed this from different angles (more loss-weighting, more regularization) and both underperformed. **Run 8 remains the best model.** The remaining untested axis is LoRA capacity — Run 11 tests that hypothesis before falling back to resume-and-polish from Run 8's checkpoint.

### Tested and Confirmed

| # | Change | Result | Run |
|---|--------|--------|-----|
| 1 | **Feed BPM beat grid to decoder** (`context_types: timing→map` + `SnapBeatParser.parse_timing()`) | **+10.7pp exact timing** (64.6% → 75.3%), new bests on all metrics | Run 8 |

### In Progress

| # | Change | Rationale | Run |
|---|--------|-----------|-----|
| 12 | `lora.r: 64 → 128`, `lora.lora_alpha: 128 → 256`; revert smoothing/dropout to Run 8 | After Run 9 (rhythm_weight) and Run 10 (regularization) both failed, capacity is the remaining untested axis. | Run 11 |

### Tested and Rejected

| # | Change | Result | Run |
|---|--------|--------|-----|
| ~~2~~ | `timing_random_offset=1` | -9.1pp exact timing, 0pp fuzzy — harmful | Run 7 |
| ~~3~~ | Lower LR (0.0001) + 4000 steps | No improvement over 0.0002 / 2000 steps | Run 7 |
| ~~4+5~~ | `label_smoothing: 0.1` + `lora_dropout: 0.1` | -0.77pp final vs Run 8; tightened gap but traded peak accuracy | Run 10 |
| ~~7~~ | `rhythm_weight: 5 → 8` | -0.35pp timing @ step 1400 vs Run 8; widens train/test gap without eval gain | Run 9 |
| ~~11~~ | More training data (+47%) | +1.3pp column/other, 0pp timing — not data-limited | Run 6 |

### Medium Impact (next run candidates after Run 11)

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 6 | **Reduce effective batch size** | Run 6 used effective batch 64 (vs 128 in Run 5) and achieved comparable timing with better column/other acc. Halving again to 32 would double gradient updates per epoch. | `optim.batch_size=32, optim.grad_acc=32` |
| 13 | **Resume Run 8 with lower LR for the tail** | If Run 11 also underperforms, resume from Run 8's `checkpoint-2000` for ~500 steps at `base_lr=5e-5` to refine the existing best instead of re-searching. | Resume from Run 8 `checkpoint-2000`, override `base_lr=5e-5` |
| 14 | **Middle-ground regularization** | Run 10 proved 0.1/0.1 over-regularizes but tightened the gap. If Run 11's r=128 overfits, try `label_smoothing=0.075` + `lora_dropout=0.075` on top of r=128 — halfway between Run 8 and Run 10. | `label_smoothing: 0.075`, `lora_dropout: 0.075` |

### Lower Impact / Experimental

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 8 | **Reduce LoRA rank** | Less capacity = less overfitting risk. Rank 64 with 602 samples may still be overkill. | `lora.r: 32` (from 64) |
| 9 | **Snapping augmentation** | Randomly perturb snapping values during training for robustness. | `snapping_random_prob: 0.1` |
| 10 | **Multi-context training** | Train with both `[timing] → [map]` and `[none] → [map]` so the model learns both modes. | Add second entry to `context_types` and `context_weights` |
