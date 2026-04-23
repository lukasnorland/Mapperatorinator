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
  lora_path="lukasnorland/rhythm-skeleton-mt3" \
  gamemode=3 keycount=4 difficulty=5.0
```

Or with a local checkpoint path (if training logs are available):

```bash
python snapbeat_inference.py \
  audio_path="song.mp3" \
  lora_path="logs/2026-04-15/09-49-00/checkpoint-2001/lora" \
  gamemode=3 keycount=4 difficulty=5.0
```

The base model (`OliBomby/Mapperatorinator-v31`) is loaded from the default `configs/inference/v31.yaml` config. The LoRA weights are merged at load time.

You can also use the general inference script directly:

```bash
python inference.py \
  audio_path="song.mp3" \
  output_path="./output/" \
  model_path="OliBomby/Mapperatorinator-v31" \
  lora_path="lukasnorland/rhythm-skeleton-mt3" \
  gamemode=3 difficulty=5.0
```

**Available checkpoints** (see [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md) for full eval metrics):

| Checkpoint | Timing Acc | Fuzzy Timing | Other Acc | Column Acc | Notes |
|------------|-----------|--------------|-----------|------------|-------|
| `lukasnorland/rhythm-skeleton-mt3` | **75.3%** | **85.7%** | **92.8%** | **69.6%** | **Best all metrics** — Run 8 (timing context in decoder) |
| `logs/2026-04-15/09-49-00/checkpoint-2001/lora` | 75.3% | 85.7% | 92.8% | 69.6% | Run 8 local path (same weights) |
| `logs/2026-04-07/15-31-50/checkpoint-2001/lora` | 64.5% | 84.6% | 92.6% | 69.2% | Run 6, 603 train samples |
| `logs/2026-03-24/21-31-48/checkpoint-2001/lora` | 64.6% | 85.0% | 91.3% | 67.9% | Run 5 local path (prior best timing) |
| `logs/2026-04-09/14-48-24/checkpoint-4001/lora` | 55.4% | 84.5% | 92.5% | 69.2% | Run 7 — timing_offset=1 hurt timing |
| `logs/2026-03-24/11-48-22/checkpoint-1001/lora` | 62.7% | 84.5% | 91.2% | 67.5% | Run 4 (resumed from Run 3) |
| `logs/2026-03-23/17-09-07/checkpoint-500/lora` | 62.3% | 84.3% | 91.3% | 67.3% | Run 3 (first add_timing run) |

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
