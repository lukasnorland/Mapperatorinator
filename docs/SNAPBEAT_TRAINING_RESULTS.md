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

## Summary Comparison (Best Eval per Run)

| Run | Steps | Loss   | Timing Acc | Fuzzy Timing | Other Acc | Column Acc | Notes |
|-----|-------|--------|-----------|--------------|-----------|------------|-------|
| 1   | 400   | 0.8348 | 41.7%     | 79.4%        | 89.8%     | 65.1%      | |
| 2   | 2001  | 0.7513 | 48.5%     | 82.6%        | 90.9%     | 67.3%      | |
| 3   | 500   | 0.6572 | 62.3%     | 84.3%        | 91.3%     | 67.3%      | +timing, +fc1/fc2 |
| 4   | 1000  | 0.6978 | 62.7%     | 84.5%        | 91.2%     | 67.5%      | resumed from Run 3 |
| 5   | 2001  | 2.178* | **64.6%** | **85.0%**    | 91.3%     | 67.9%      | **best timing** |
| 6   | 2001  | 2.144* | 64.5%     | 84.6%        | **92.6%** | **69.2%**  | **best column/other**, +47% data |
| 7   | 4001  | 0.717  | 55.4%     | 84.5%        | 92.5%     | 69.2%      | timing_offset=1 hurt, timing context was no-op |

\* Runs 5–6 loss not directly comparable to earlier runs due to `rhythm_weight=5` and `label_smoothing=0.05`.

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

---

## Next Steps: Improvement Suggestions

Timing accuracy has plateaued at ~64.5% across Runs 5–6. The 20pp gap to fuzzy_timing_acc (84.6%) shows the model gets timing approximately right but can't pick the exact step. Run 7 ruled out jitter augmentation and lower LR as solutions. The remaining high-value change is feeding timing context to the decoder — now implemented correctly.

### High Impact (Run 8)

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 1 | **Feed timing context to decoder** | `SnapBeatParser.parse_timing()` now generates TIMING_POINT/MEASURE/BEAT events from BPM. `SnapBeatDataset` passes them as decoder input when `context_types: timing→map`. This gives the model an explicit beat grid to anchor timing predictions on. Was intended for Run 7 but the code path didn't exist — now implemented. | `context_types: [{in: [timing], out: [map]}]` (already set) |

### Tested and Rejected

| # | Change | Result | Run |
|---|--------|--------|-----|
| ~~2~~ | `timing_random_offset=1` | -9.1pp exact timing, 0pp fuzzy — harmful | Run 7 |
| ~~3~~ | Lower LR (0.0001) + 4000 steps | No improvement over 0.0002 / 2000 steps | Run 7 |
| ~~11~~ | More training data (+47%) | +1.3pp column/other, 0pp timing — not data-limited | Run 6 |

### Medium Impact (after Run 8)

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 4 | **Increase label smoothing** | More smoothing prevents overfitting to exact timing values. Current 0.05 may be too conservative. | `label_smoothing: 0.1` (from 0.05) |
| 5 | **Increase LoRA dropout** | More regularization — the model may be memorizing timing patterns rather than generalizing. | `lora_dropout: 0.1–0.15` (from 0.05) |
| 6 | **Reduce effective batch size** | Run 6 used effective batch 64 (vs 128 in Run 5) and achieved comparable timing with better column/other acc. Halving again to 32 would double gradient updates per epoch. | `optim.batch_size=32, optim.grad_acc=32` |
| 7 | **Higher rhythm_weight** | Pushing timing token weight even higher (e.g. 8–10) may force the model to allocate more capacity to timing at the expense of other tokens (which are already at 92.6%). | `rhythm_weight: 8.0` (from 5.0) |

### Lower Impact / Experimental

| # | Suggestion | Rationale | Config change |
|---|-----------|-----------|---------------|
| 8 | **Reduce LoRA rank** | Less capacity = less overfitting risk. Rank 64 with 602 samples may still be overkill. | `lora.r: 32` (from 64) |
| 9 | **Snapping augmentation** | Randomly perturb snapping values during training for robustness. | `snapping_random_prob: 0.1` |
| 10 | **Multi-context training** | Train with both `[timing] → [map]` and `[none] → [map]` so the model learns both modes. | Add second entry to `context_types` and `context_weights` |
