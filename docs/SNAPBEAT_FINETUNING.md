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
| Parser | `osuT5/osuT5/dataset/snapbeat_parser.py` | SnapBeat JSON → `(events, event_times)` using mania EventTypes |
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
- 456 total samples: 410 train / 46 test
- Optimizer: Muon, lr=2e-4/1e-4, batch=128, grad_acc=64
- 2000 total steps, 200 warmup, eval every 200, checkpoint every 500
- `rhythm_weight: 5.0` (upweights TIME_SHIFT tokens in loss for timing accuracy)
- `label_smoothing: 0.05` (helps timing generalization)
- DT augmentation: disabled (`dt_augment_prob: 0.0`) — SnapBeat charts have fixed BPM
- Timing context enabled: `add_timing`, `add_timing_points`, `add_snapping` (provides BPM and beat structure)
- `timing_random_offset: 0` (no jitter on timing labels for exact-match accuracy)
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
  lora_path="luannnguyen/snapbeat-lora-v5" \
  gamemode=3 keycount=4 difficulty=5.0
```

Or with a local checkpoint path (if training logs are available):

```bash
python snapbeat_inference.py \
  audio_path="song.mp3" \
  lora_path="logs/2026-03-24/21-31-48/checkpoint-2001/lora" \
  gamemode=3 keycount=4 difficulty=5.0
```

The base model (`OliBomby/Mapperatorinator-v31`) is loaded from the default `configs/inference/v31.yaml` config. The LoRA weights are merged at load time.

You can also use the general inference script directly:

```bash
python inference.py \
  audio_path="song.mp3" \
  output_path="./output/" \
  model_path="OliBomby/Mapperatorinator-v31" \
  lora_path="luannnguyen/snapbeat-lora-v5" \
  gamemode=3 difficulty=5.0
```

**Available checkpoints** (see [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md) for full eval metrics):

| Checkpoint | Timing Acc | Fuzzy Timing | Notes |
|------------|-----------|--------------|-------|
| `luannnguyen/snapbeat-lora-v5` | 64.6% | 85.0% | **Best** — Run 5, hosted on HuggingFace |
| `logs/2026-03-24/21-31-48/checkpoint-2001/lora` | 64.6% | 85.0% | Same as above (local path) |
| `logs/2026-03-24/11-48-22/checkpoint-1001/lora` | 62.7% | 84.5% | Run 4 (resumed from Run 3) |
| `logs/2026-03-23/17-09-07/checkpoint-500/lora` | 62.3% | 84.3% | Run 3 (first add_timing run) |

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
