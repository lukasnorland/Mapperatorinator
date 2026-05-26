# SnapBeat Fine-Tuning Guide

This guide explains how to fine-tune Mapperatorinator on a SnapBeat dataset using the implemented pipeline.

---

## 1. Prerequisites

- **Conda environment**: `mapperatorinator` with all dependencies installed.
- **ffmpeg**: Required for audio loading (`conda install -n mapperatorinator ffmpeg`).
- **torchaudio**: Required for spectrogram generation (`pip install torchaudio`).
- **W&B** (optional): The default config uses tensorboard. To use W&B instead, pass `logging.log_with=wandb` on the CLI.

### Platform notes

- **Linux** is the primary training platform. `torch.compile`, Triton, and multi-worker dataloaders all work as expected.
- **Windows + CUDA**: `torch.compile` is skipped automatically (Triton is unavailable). Multi-worker `IterableDataset` loaders can deadlock at epoch boundaries; use `dataloader.num_workers=0` if training hangs between epochs.

---

## 2. Dataset layout

The dataset follows a UUID-matched directory structure:

```text
{dataset_root}/
  json/    *.json    (SnapBeat v1.0 / MT3 charts)
  audio/   *.mp3|*.wav|*.flac|*.ogg  (matched by UUID stem)
```

Each JSON file is matched to its audio file by UUID filename stem. The dataset class (`SnapBeatDataset`) auto-discovers these pairs.

**Neither JSON nor audio is committed to git** (too large / licensing). You must populate `datasets/dataset/json/` with **456** SnapBeat chart JSON files and `datasets/dataset/audio/` with matching audio files (same UUID stems). The default `snapbeat_lora` config expects 456 pairs split as 410 train / 46 test.

If your audio lives in a separate directory, set `data.snapbeat_audio_path` to that folder (see [datasets/dataset/audio/README.md](../datasets/dataset/audio/README.md)).

---

## 3. Pipeline architecture

### Components

| Component | File | Role |
|-----------|------|------|
| Parser | `osuT5/osuT5/dataset/snapbeat_parser.py` | SnapBeat JSON → `(events, event_times)` using mania EventTypes; `parse_timing()` generates beat grid from BPM |
| Dataset | `osuT5/osuT5/dataset/snapbeat_dataset.py` | Loads audio + chart pairs, builds training samples |
| Model wiring | `osuT5/osuT5/utils/model_utils.py` | Routes `dataset_type=snapbeat` to SnapBeatDataset/Parser |
| Config | `configs/train/snapbeat_lora.yaml` | LoRA fine-tuning config (456-chart `datasets/dataset` split) |
| Inference | `snapbeat_inference.py` | Generate charts from audio using fine-tuned model |
| Converter | `snapbeat_converter.py` | Convert .osu mania output to SnapBeat JSON |

### Token mapping

- **short** → `CIRCLE` + `TIME_SHIFT` + `SNAPPING` + `MANIA_COLUMN`
- **long/zigzag** → `HOLD_NOTE` ... `HOLD_NOTE_END` (same as mania hold)
- **Lane** → `MANIA_COLUMN` with value 0..(N-1)
- **Time** → `TIME_SHIFT` in milliseconds (normalized to relative steps by dataset)

Only mania-compatible tokens are used. No tokenizer changes needed.

### Training flow

1. Hydra loads config → pretrained V31 model + LoRA adapters
2. `get_dataloaders()` detects `dataset_type=snapbeat` → creates `SnapBeatParser` + `SnapBeatDataset`
3. Dataset discovers (json, audio) pairs, slices by train/test indices
4. For each pair: load audio → parse JSON → build frames + token sequences → yield batches
5. Accelerator trains with LoRA, checkpoints periodically

---

## 4. Running training

From the repo root (relative paths are resolved against the original working directory automatically):

```bash
python osuT5/train.py --config-name snapbeat_lora
```

To use absolute paths (e.g. if your dataset is elsewhere):

```bash
python osuT5/train.py --config-name snapbeat_lora \
  data.train_dataset_path=/path/to/your/dataset_root \
  data.test_dataset_path=/path/to/your/dataset_root
```

On **PowerShell**, use a single line or backtick (`` ` ``) continuation — **not** `^`:

```powershell
python osuT5/train.py --config-name snapbeat_lora `
  data.train_dataset_path=D:/path/to/your/dataset_root `
  data.test_dataset_path=D:/path/to/your/dataset_root
```

Override any hyperparameter via CLI:

```bash
python osuT5/train.py --config-name snapbeat_lora \
  optim.batch_size=8 optim.total_steps=1000
```

### Config highlights (`snapbeat_lora.yaml`)

- Pretrained: `OliBomby/Mapperatorinator-v31` with LoRA (r=64, alpha=128, PiSSA init, targets: q/k/v/out_proj + fc1/fc2)
- 669 total samples: 602 train / 67 test
- Optimizer: Muon, lr=2e-4/1e-4, batch=64, grad_acc=64
- 2000 total steps, 200 warmup, eval every 200, checkpoint every 500
- `rhythm_weight: 5.0` (upweights TIME_SHIFT tokens in loss for timing accuracy — Run 9 proved 8.0 underperforms 5.0)
- `label_smoothing: 0.1` / `lora_dropout: 0.1` (current file state, Run 10). Run 10 showed 0.1/0.1 over-regularizes (-0.77pp vs Run 8's 0.05/0.05). Run 11 plan reverts both to **0.05** and tests `lora.r: 128` / `lora_alpha: 256` instead.
- DT augmentation: disabled (`dt_augment_prob: 0.0`) — SnapBeat charts have fixed BPM
- Timing context: `context_types: timing→map` — decoder receives BPM/beat grid as input via `SnapBeatParser.parse_timing()`
- Timing features enabled: `add_timing`, `add_timing_points`, `add_snapping` (provides BPM and beat structure in encoder)
- `timing_random_offset: 0` (jitter hurts exact timing — Run 7 confirmed -9pp)
- LoRA targets: attention projections + feedforward layers (`fc1`, `fc2`)
- Mania-only tokens: gamemode, keycount, hold_note_ratio (other osu! tokens disabled)

### Resuming from a checkpoint

```bash
python osuT5/train.py --config-name snapbeat_lora \
  checkpoint_path=/path/to/logs/YYYY-MM-DD/HH-MM-SS/checkpoint-500
```

Checkpoints are saved under the Hydra output directory (`logs/`). To reduce risk of lost progress, lower `checkpoint.every_steps` (e.g. `checkpoint.every_steps=100`).

---

## 5. Inference

Generate a SnapBeat chart from audio:

```bash
python snapbeat_inference.py audio_path="song.mp3" gamemode=3 keycount=4
```

This generates an .osu file via the standard pipeline, then converts to SnapBeat JSON using `snapbeat_converter.py`.

To use the fine-tuned LoRA checkpoint from HuggingFace (works on any machine):

```bash
python snapbeat_inference.py \
  audio_path="song.mp3" \
  lora_path="lukasnorland/rhythm-skeleton-mt3-v2" \
  gamemode=3 keycount=4 difficulty=5.0
```

Or with a local checkpoint path (if training logs are available):

```bash
python snapbeat_inference.py \
  audio_path="song.mp3" \
  lora_path="logs/2026-05-12/15-34-47/checkpoint-2001/lora" \
  gamemode=3 keycount=4 difficulty=5.0
```

The base model (`OliBomby/Mapperatorinator-v31`) is loaded from the default `configs/inference/v31.yaml` config. The LoRA weights are merged at load time.

You can also use the general inference script directly:

```bash
python inference.py \
  audio_path="song.mp3" \
  output_path="./output/" \
  model_path="OliBomby/Mapperatorinator-v31" \
  lora_path="lukasnorland/rhythm-skeleton-mt3-v2" \
  gamemode=3 difficulty=5.0
```

**Available checkpoints** (see [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md) for full eval metrics):

> **Test-set note:** v1 (Run 12) and the runs before it were evaluated on a smaller 67-sample piano7 test split that included 10 corrupt audio/JSON pairs (removed 2026-05-12). v2 (Run 13 rev2) is evaluated on the cleaned 234-sample split. The two splits are **not directly comparable**; the apples-to-apples re-eval of v1 on the v2 split (row 2 below) is the correct baseline for comparing v1 ↔ v2.

| Checkpoint | Timing Acc | Fuzzy Timing | Other Acc | Column Acc | Eval set | Notes |
|------------|-----------|--------------|-----------|------------|---------|-------|
| `lukasnorland/rhythm-skeleton-mt3-v2` | **69.2%** | **82.0%** | **91.7%** | **66.0%** | 234-sample cleaned | **Current baseline** — Run 13 rev2 (Run 12 resume on 2093-sample cleaned dataset) |
| `lukasnorland/rhythm-skeleton-mt3-v1` (apples-to-apples) | 64.3% | 78.7% | 90.8% | 61.1% | 234-sample cleaned | v1 re-evaluated on v2's test set; **+4.92pp timing gap vs v2** |
| `logs/2026-05-12/15-34-47/checkpoint-2001/lora` | 69.2% | 82.0% | 91.7% | 66.0% | 234-sample cleaned | Run 13 rev2 local path (same weights as v2) |
| `lukasnorland/rhythm-skeleton-mt3-v1` (original) | 75.5% | 85.8% | 92.9% | 69.6% | 67-sample piano7 | Run 12 — historical number on easier pre-cleanup split; **not comparable to v2** |
| `logs/2026-04-25/09-51-23/checkpoint-1001/lora` | 75.5% | 85.8% | 92.9% | 69.6% | 67-sample piano7 | Run 12 local path (same weights as v1) |
| `lukasnorland/rhythm-skeleton-mt3` | 75.3% | 85.7% | 92.8% | 69.6% | 67-sample piano7 | Run 8 (superseded by v1, then v2) |
| `logs/2026-04-15/09-49-00/checkpoint-2001/lora` | 75.3% | 85.7% | 92.8% | 69.6% | 67-sample piano7 | Run 8 local path (same weights) |
| `logs/2026-04-07/15-31-50/checkpoint-2001/lora` | 64.5% | 84.6% | 92.6% | 69.2% | 67-sample piano7 | Run 6, 603 train samples |
| `logs/2026-03-24/21-31-48/checkpoint-2001/lora` | 64.6% | 85.0% | 91.3% | 67.9% | 67-sample piano7 | Run 5 local path (prior best timing) |
| `logs/2026-04-09/14-48-24/checkpoint-4001/lora` | 55.4% | 84.5% | 92.5% | 69.2% | 67-sample piano7 | Run 7 — timing_offset=1 hurt timing |
| `logs/2026-03-24/11-48-22/checkpoint-1001/lora` | 62.7% | 84.5% | 91.2% | 67.5% | 67-sample piano7 | Run 4 (resumed from Run 3) |
| `logs/2026-03-23/17-09-07/checkpoint-500/lora` | 62.3% | 84.3% | 91.3% | 67.3% | 67-sample piano7 | Run 3 (first add_timing run) |

---

## 6. Known issues and fixes

### Special tokens in decoder labels (fixed)

Input-only special tokens (GAMEMODE, MANIA_KEYCOUNT, HOLD_NOTE_RATIO) have token IDs beyond the model's output vocabulary size. These were incorrectly included in decoder labels, causing CUDA device-side asserts during training. Fixed by computing `start_label_index` after adding special tokens so labels only contain event tokens within the output vocab range.

### W&B 401 when not logged in (fixed)

The config default `logging.mode: online` was passed into `wandb.init()` even when the user selected offline mode interactively, causing a 401. Fixed: `mode` is only forwarded when explicitly set to something other than `online`. The `snapbeat_lora` config defaults to tensorboard, avoiding W&B entirely.

### Hydra cwd breaks relative dataset paths (fixed)

Hydra changes the working directory to `logs/...` at startup, so relative paths like `./datasets/dataset` resolved incorrectly. Fixed in `setup_args()` which now resolves relative data paths against `hydra.utils.get_original_cwd()`.

### Windows: torch.compile fails (Triton unavailable, auto-skipped)

`torch.compile` requires Triton, which is not available on Windows CUDA. `train.py` detects `sys.platform == "win32"` and skips compilation automatically.

### Windows: training hangs at epoch boundaries

Multi-worker `IterableDataset` with `persistent_workers=True` can deadlock when a new epoch restarts the iterator. SnapBeat dataloaders use `persistent_workers=False` automatically. If training still hangs, use `dataloader.num_workers=0`.


---

## Changelog

### 2026-05-14 — Run 13 rev2 published as `rhythm-skeleton-mt3-v2` (data-ceiling broken; new baseline)

Run 13 rev2 (Run 12 warm-resume on the cleaned 2327-sample SnapBeat dataset, 2000 steps at `base_lr=5e-5`) reached **69.21% timing accuracy** on the new 234-sample test set — vs **64.29%** for the v1 adapter on the same test set, i.e. **+4.92pp timing**, **+3.32pp fuzzy timing**, **+0.94pp other**, **+4.85pp column**, **−0.53 loss**. The data-ceiling hypothesis from Runs 9–12 (timing plateau at ~75.3% on the old split was caused by 602-sample dataset size, not hyperparameters) is now confirmed and broken. Shipped as the new production baseline.

| | Before (v1) | After (v2) |
|---|---|---|
| **Production LoRA** | `lukasnorland/rhythm-skeleton-mt3-v1` (Run 12) | `lukasnorland/rhythm-skeleton-mt3-v2` (Run 13 rev2) |
| **Train samples** | 602 (piano7-only subset) | 2093 (full cleaned dataset; 10 corrupt pairs removed) |
| **Test samples** | 67 (piano7-only) | 234 (cleaned mixed origin) |
| **Steps** | 1000 | 2000 |
| **Source of truth** | `logs/2026-04-25/09-51-23/checkpoint-1001/lora` | `logs/2026-05-12/15-34-47/checkpoint-2001/lora` |
| **Visibility** | private | private |

What changed:

- **Dataset cleanup (2026-05-12)**: removed 9 audio/JSON-mismatched stems (10 paired files — one stem had 2 UUID duplicates) from `datasets/dataset`. These were discovered during the original Run 13 cold-start, which had hot-zone failures on the affected samples. Cleanup recipe: map piano7-style stems to dataset UUIDs by `songMeta.songName`, delete matching `json/<uuid>.json` + `audio/<uuid>.*`.
- **Dataset expansion**: trained on the full SnapBeat corpus (2327 samples post-cleanup, vs the 669-sample piano7-only subset used through Run 12). 90/10 split: 2093 train / 234 test.
- **Warm-resume strategy preserved**: same `lora_resume_path` mechanism as Run 12 (loads adapter weights only, leaves optimizer/scheduler fresh). The first attempted Run 13 was a *cold* start at `base_lr=2e-4`; it was discarded after ~1500 steps because warm-resume from Run 12 was strictly faster.
- **Patched `osuT5/test.py` for SnapBeat**: the stock `test.py` runs a `test_noise` pass that hard-requires the osu!-only `rhythm_complexities.csv` and reads `sample_weights` from each batch. SnapBeat batches don't carry that field. Patched with a `try/except` around the noise pass and a `.get(...)`-with-`None`-branch into `calc_loss` (which already handles `None`). Both fixes are minimal and backward-compatible with osu! configs.

Follow-up tasks (status as of 2026-05-14):

- [x] **Pushed v2 to HuggingFace** as `lukasnorland/rhythm-skeleton-mt3-v2` (private). Source of truth: `logs/2026-05-12/15-34-47/checkpoint-2001/lora`.
- [x] **Flipped `_GAME_CODE_REGISTRY["MT3"]`** in [`snapbeat_inference.py`](../snapbeat_inference.py) → `rhythm-skeleton-mt3-v2`. Inference defaults now resolve to v2; the v1 repo is preserved unchanged for A/B comparisons.
- [x] **Updated `Dockerfile.deploy`** — `LORA_REPO` build-arg default is `rhythm-skeleton-mt3-v2` and the canonical build tag is `snapbeat-lora:mt3-v2`.
- [x] **Updated `docs/SNAPBEAT_DEPLOY_OPTION_A.md`** — all forward-looking refs point at v2; the v1↔v2 naming-scheme example in §4 still shows v1 → `v1` mapping intentionally as historical illustration.
- [x] **Rebuilt the Docker image** as `snapbeat-lora:mt3-v2` (the prior `snapbeat-lora:dev` image stays around as a v1 A/B reference until you choose to retire it).
- [ ] **Redeploy** — any running v1 deployments need a `docker pull` + restart against the new `snapbeat-lora:mt3-v2` image. (Out of scope for the model team; left to the deployment owner.)

Methodological note for future runs:

- **Eval-set boundary**: any future regression check must re-evaluate the reference model on the v2 234-sample test split. Cross-test-set absolute comparisons across the 2026-05-12 dataset boundary will silently mislead — Run 12 looked like it dropped 6pp under v2 conditions purely because the new split is harder; the actual on-the-same-set improvement is +4.92pp.

### 2026-04-26 — Run 12 published as `rhythm-skeleton-mt3-v1` (new baseline + naming-scheme shift)

Run 12 (Run 8 resume with `base_lr=5e-5` low-LR cosine tail, 1000 steps) reached **75.53%** timing accuracy — **+0.23pp over Run 8**, the only post-Run-8 sweep to produce an above-noise gain. Shipped as the new production baseline.

| | Before | After |
|---|---|---|
| **Production LoRA** | `lukasnorland/rhythm-skeleton-mt3` (Run 8, 75.30%) | `lukasnorland/rhythm-skeleton-mt3-v1` (Run 12, 75.53%) |
| **Naming scheme** | `rhythm-skeleton-<game>` (replaced in place per run) | `rhythm-skeleton-<game>-v<N>` (versioned; future = `-v2`, `-v3`...) |
| **Visibility** | private | private |

Why the naming-scheme shift:

- Versioned suffix preserves prior baselines as inspectable artifacts (the old `rhythm-skeleton-mt3` repo is intentionally kept for A/B comparisons).
- Pinning Docker images to `:v1` / `:v2` becomes unambiguous; reverting a regression no longer needs a HF revision SHA lookup.
- Cost: future BH/DR variants now live at `rhythm-skeleton-bh-v1`, `rhythm-skeleton-dr-v1` (registry stub in `snapbeat_inference.py` updated).

Downstream impact:

- `_GAME_CODE_REGISTRY["MT3"]` in [`snapbeat_inference.py`](../snapbeat_inference.py) now resolves to `lukasnorland/rhythm-skeleton-mt3-v1`. Anyone using `lora_path=...` overrides in scripts must update them.
- Existing Docker images built against `rhythm-skeleton-mt3` still work (the old repo is unchanged); only newly built images pick up v1. Rebuild + redeploy to roll forward.
- `HF_TOKEN` permissions: same fine-grained read access requirement, just point it at the new repo path.

Code change shipped alongside this run: `osuT5/train.py` gained a `lora_resume_path` field that loads adapter weights only via `set_peft_model_state_dict`, leaving the freshly-built optimizer + scheduler intact. `accelerator.load_state` would have restored Run 8's spent cosine schedule and exited at step 0 — the new path is required for any future low-LR resume experiment.

Source of truth: `logs/2026-04-25/09-51-23/checkpoint-1000/lora`.

### 2026-04-23 — HuggingFace rename + privacy + account migration

Three changes landed together on HuggingFace; all published references in this repo now point to the new canonical path. No model weights changed.

| Before | After |
|---|---|
| **Account** | `luannnguyen` | `lukasnorland` |
| **Best LoRA repo** | `luannnguyen/snapbeat-lora-v8` | `lukasnorland/rhythm-skeleton-mt3` |
| **Prior LoRA (v5)** | `luannnguyen/snapbeat-lora-v5` | *deleted* |
| **Visibility** | public | **private** |

Why:

- **Rename to `rhythm-skeleton-mt3`** — establishes a game-code-anchored family. Future variants slot in as `rhythm-skeleton-bh` and `rhythm-skeleton-dr` without re-versioning.
- **Private** — the LoRA is proprietary and distributed via baked Docker images, not direct HF pulls.
- **Account rename** — personal preference; HF auto-redirects at the API level while the old handle still exists.
- **v5 deleted** — superseded by Run 8 (`rhythm-skeleton-mt3`); v5 was kept previously only for historical A/B comparisons in the table above, which have been removed.

Downstream impact:

- `_GAME_CODE_REGISTRY["MT3"]` in [`snapbeat_inference.py`](../snapbeat_inference.py) now resolves to `lukasnorland/rhythm-skeleton-mt3`.
- Anyone calling `snapbeat_inference.py` (or `inference.py`) with `lora_path=...` must:
  1. Update the path to `lukasnorland/rhythm-skeleton-mt3`.
  2. Have a HuggingFace token with **Read** access to that repo, exported as `HF_TOKEN`.
- Old local HF caches under `~/.cache/huggingface/hub/models--luannnguyen--snapbeat-lora-v8` are now dead weight; safe to delete.
- Dockerized deploy (see Option A plan under `.cursor/plans/`) now uses a BuildKit secret to pass `HF_TOKEN` at image-build time; runtime needs no token because weights are baked in.

Checkpoint hosting of the current best (Run 8): `lukasnorland/rhythm-skeleton-mt3` ← `luannnguyen/snapbeat-lora-v8` (deleted) ← `logs/2026-04-15/09-49-00/checkpoint-2001/lora` (local source of truth).

### 2026-04-15 — Run 8 training complete (timing context breakthrough)

Run 8 finished 2000 steps (paused/resumed once at step 1500 due to a machine restart). The post-Run 7 timing-context fix — decoder now receives BPM/beat-grid events generated by `SnapBeatParser.parse_timing()` — broke the 64.6% timing plateau:

| Metric | Run 5 (prior best timing) | Run 6 (prior best column/other) | Run 8 | Delta vs best prior |
|--------|---------------------------|----------------------------------|-------|---------------------|
| Timing Acc | 64.6% | 64.5% | **75.3%** | **+10.7pp** |
| Fuzzy Timing | 85.0% | 84.6% | **85.7%** | +0.7pp |
| Other Acc | 91.3% | 92.6% | **92.8%** | +0.2pp |
| Column Acc | 67.9% | 69.2% | **69.6%** | +0.4pp |

Run 8 beats every prior run on every metric. The gap from exact→fuzzy timing narrowed from ~20pp to ~10pp, indicating the model can now pick the exact step far more often. Config was identical to Run 6 except `context_types: [{in: [timing], out: [map]}]` is now actually honored by the dataset.

Best checkpoint: `logs/2026-04-15/09-49-00/checkpoint-2001/lora` (hosted as `lukasnorland/rhythm-skeleton-mt3`).

### 2026-04-13 — Run 7 results + timing context implementation

Run 7 finished 4000 steps (25 epochs). Three changes were tested simultaneously:

| Change | Result | Verdict |
|--------|--------|---------|
| `timing_random_offset=1` | -9.1pp exact timing, 0pp fuzzy | **Harmful** — reverted to 0 |
| `context_types: timing→map` | No effect (code path didn't exist) | **Fixed** — see below |
| Lower LR (0.0001) + 4000 steps | No improvement over 0.0002/2000 | **Unnecessary** — reverted |

Post-Run 7 implementation: `SnapBeatParser.parse_timing()` now generates TIMING_POINT/MEASURE/BEAT events from chart BPM metadata. `SnapBeatDataset._process_chart()` detects `ContextType.TIMING` in the config and feeds the beat grid to the decoder as input context. This was the intended architectural change for Run 7 but the dataset code didn't support it — now it does.

Also fixed: `test.py` now supports `lora_path` config for evaluating LoRA checkpoints against the base model, and handles tensorboard-only runs (no wandb crash).

Config reverted to Run 6 settings + timing context for Run 8: `timing_random_offset=0`, `base_lr=0.0002`, `total_steps=2000`, `context_types: timing→map`.

### 2026-04-09 — Run 6 training complete (expanded dataset)

Run 6 finished 2000 steps (13 epochs) with 47% more training data (603 train / 67 test, up from 410/46). Final eval results:

| Metric | Run 5 (best prior) | Run 6 | Delta |
|--------|-------------------|-------|-------|
| Timing Acc | 64.6% | 64.5% | -0.1pp |
| Fuzzy Timing | 85.0% | 84.6% | -0.4pp |
| Other Acc | 91.3% | 92.6% | **+1.3pp** |
| Column Acc | 67.9% | 69.2% | **+1.3pp** |

Key finding: more data significantly improved note placement (column +1.3pp, other +1.3pp) but timing accuracy is unchanged. This confirms timing precision is not data-limited — it requires architectural changes (feeding BPM/timing context to the decoder, timing augmentation) rather than more samples. See [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md) for updated improvement suggestions.

Config changes: `compile=false` (Triton unavailable), `batch_size=64, grad_acc=64` (12GB VRAM constraint), dataset split updated to 603/67. One corrupt audio file (`7f65a210-...wav`) was skipped during training.

Best checkpoint: `logs/2026-04-07/15-31-50/checkpoint-2001/lora`.

### 2026-03-26 — Run 5 training complete

Run 5 finished 2000 steps (19 epochs). Final eval results:

| Metric | Run 4 (best prior) | Run 5 | Delta |
|--------|-------------------|-------|-------|
| Timing Acc | 62.7% | 64.6% | +1.9pp |
| Fuzzy Timing | 84.5% | 85.0% | +0.5pp |
| Other Acc | 91.2% | 91.3% | +0.1pp |
| Column Acc | 67.5% | 67.9% | +0.4pp |

Best checkpoint: `logs/2026-03-24/21-31-48/checkpoint-2001/lora`. Gains were diminishing after step ~1400. See [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md) for full step-by-step eval.

### 2026-03-24 — Timing accuracy improvements (Run 5 config)

Changes to `configs/train/snapbeat_lora.yaml` targeting timing accuracy (plateaued at ~62.7% in Run 4):

| Setting | Before | After | Rationale |
|---------|--------|-------|-----------|
| `data.rhythm_weight` | 3.0 (inherited default) | 5.0 | Upweights TIME_SHIFT tokens in cross-entropy loss so the model prioritizes timing predictions |
| `data.label_smoothing` | 0.0 (inherited default) | 0.05 | Helps the model generalize timing predictions instead of overfitting exact offsets |
| `data.dt_augment_prob` | 0.15 | 0.0 | SnapBeat charts have fixed BPM; DT augmentation adds noise to timing signal |
| `optim.total_steps` | 1000 | 2000 | Run 3 was still improving at step 500 but Run 4 stalled because cosine LR had decayed to 0. Fresh 2000-step schedule gives more room to converge |
| `optim.warmup_steps` | 100 | 200 | Scaled proportionally with total_steps |
