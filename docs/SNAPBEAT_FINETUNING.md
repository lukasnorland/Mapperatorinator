# SnapBeat Fine-Tuning Guide

This guide explains how to fine-tune Mapperatorinator on a SnapBeat dataset using the implemented pipeline.

---

## 1. Prerequisites

- **Conda environment**: `mapperatorinator` with all dependencies installed.
- **ffmpeg**: Required for audio loading (`conda install -n mapperatorinator ffmpeg`).
- **torchaudio**: Required for spectrogram generation (`pip install torchaudio`).
- **W&B** (optional): If not using W&B, pass `logging.log_with=tensorboard` on the CLI.

---

## 2. Dataset layout

The dataset follows a UUID-matched directory structure:

```text
datasets/amaremix/
  json/    *.json    (SnapBeat v1.0 charts)
  audio/   *.mp3|*.wav|*.flac  (matched by UUID stem)
```

Each JSON file is matched to its audio file by UUID filename stem. The dataset class (`SnapBeatDataset`) auto-discovers these pairs.

---

## 3. Pipeline architecture

### Components

| Component | File | Role |
|-----------|------|------|
| Parser | `osuT5/osuT5/dataset/snapbeat_parser.py` | SnapBeat JSON → `(events, event_times)` using mania EventTypes |
| Dataset | `osuT5/osuT5/dataset/snapbeat_dataset.py` | Loads audio + chart pairs, builds training samples |
| Model wiring | `osuT5/osuT5/utils/model_utils.py` | Routes `dataset_type=snapbeat` to SnapBeatDataset/Parser |
| Config | `configs/train/snapbeat_lora.yaml` | LoRA fine-tuning config for amaremix dataset |
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

From the repo root:

```bash
python osuT5/train.py --config-name snapbeat_lora
```

If dataset paths don't resolve (Hydra changes cwd), use absolute paths:

```bash
python osuT5/train.py --config-name snapbeat_lora \
  data.train_dataset_path=/absolute/path/to/datasets/amaremix \
  data.test_dataset_path=/absolute/path/to/datasets/amaremix
```

To skip W&B and use tensorboard:

```bash
python osuT5/train.py --config-name snapbeat_lora \
  logging.log_with=tensorboard \
  data.train_dataset_path=/absolute/path/to/datasets/amaremix \
  data.test_dataset_path=/absolute/path/to/datasets/amaremix
```

Override any hyperparameter via CLI:

```bash
python osuT5/train.py --config-name snapbeat_lora \
  optim.batch_size=8 optim.total_steps=1000
```

### Config highlights (`snapbeat_lora.yaml`)

- Pretrained: `OliBomby/Mapperatorinator-v31` with LoRA (r=64, alpha=128, PiSSA init)
- 456 total samples: 410 train / 46 test
- Optimizer: Muon, lr=2e-4/1e-4, batch=16, grad_acc=8
- 2000 total steps, eval every 200, checkpoint every 500
- DT augmentation: prob=0.3, speed range [1.1, 1.4]
- Mania-only tokens: gamemode, keycount, hold_note_ratio (other osu! tokens disabled)

---

## 5. Inference

Generate a SnapBeat chart from audio:

```bash
python snapbeat_inference.py audio_path="song.mp3" gamemode=3 keycount=4
```

This generates an .osu file via the standard pipeline, then converts to SnapBeat JSON using `snapbeat_converter.py`.

To use a fine-tuned LoRA checkpoint, point to the adapter directory.

---

## 6. Known issues and fixes

### Special tokens in decoder labels (fixed)

Input-only special tokens (GAMEMODE, MANIA_KEYCOUNT, HOLD_NOTE_RATIO) have token IDs beyond the model's output vocabulary size. These were incorrectly included in decoder labels, causing CUDA device-side asserts during training. Fixed by computing `start_label_index` after adding special tokens so labels only contain event tokens within the output vocab range.

### W&B tracker crash when using tensorboard (fixed)

`accelerator.get_tracker("wandb")` throws `ValueError` instead of returning `None` when W&B is not the active tracker. Fixed with try/except in `train_utils.py`.
