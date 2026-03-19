# SnapBeat Fine-Tuning Guide

This guide explains **how fine-tuning works** when you already have a SnapBeat dataset, and what you need to implement to run it.

---

## 1. How fine-tuning is run (high level)

1. **Entry point**: Training is started from the project root with:
   ```bash
   python osuT5/train.py
   ```
   Hydra loads a config from `configs/train/` (e.g. `v29`, `v30`, `lora.yaml`).

2. **Flow**:
   - `get_dataloaders(tokenizer, args, shared)` builds train/test dataloaders.
   - It creates an **parser** (`OsuParser` by default) and calls `get_dataset(args, test=..., parser=parser, tokenizer=..., shared=...)`.
   - The dataset yields **batches** with: `frames` (audio spectrogram), `decoder_input_ids`, `decoder_attention_mask`, `labels`, and optional conditioning.
   - If `enable_lora: true`, the base model is frozen and only LoRA adapters are trained.

3. **Config overrides** (CLI):
   ```bash
   python osuT5/train.py \
     data.dataset_type=snapbeat \
     data.train_dataset_path=/path/to/your/snapbeat_dataset \
     data.test_dataset_path=/path/to/your/snapbeat_test \
     pretrained_path=OliBomby/Mapperatorinator-v31 \
     enable_lora=true
   ```

So: **fine-tuning = same `train.py`, different `dataset_type` and data paths**, plus (optionally) LoRA and pretrained path. The missing piece is the **SnapBeat dataset type** and a **parser** that turns SnapBeat JSON into the same event sequences the model expects.

---

## 2. What the training pipeline expects from a dataset

Each **training sample** must be a dict with at least:

| Key | Shape/type | Meaning |
|-----|------------|--------|
| `frames` | `(src_seq_len - 1) * hop_length` float32 | Audio frame samples for one window (will be turned into mel spectrogram by the model). |
| `decoder_input_ids` | `(tgt_seq_len,)` int64 | Decoder input: special tokens + context + [SOS] + event tokens (padded with `pad_id`). |
| `decoder_attention_mask` | `(tgt_seq_len,)` bool | `True` where not padding. |
| `labels` | `(tgt_seq_len,)` int64 | Same length; use `-100` for positions to ignore (e.g. padding and some context). |

Optional: `beatmap_idx`, `difficulty`, `mapper_idx`, `song_position`, `sample_weights`, etc., for conditioning or loss weighting.

The **event sequence** for the decoder is produced by a **parser**: it turns one chart (e.g. one `.osu` or one SnapBeat JSON) into `(events, event_times)` where:

- `events`: list of `Event(EventType.XXX, value)`.
- `event_times`: list of int (milliseconds), same length as `events`.

The tokenizer then encodes each `Event` to token ids. So for SnapBeat you need a **SnapBeat → (events, event_times)** step that reuses the same `Event`/`EventType` vocabulary (e.g. mania-like: `TIME_SHIFT`, `MANIA_COLUMN`, `HOLD_NOTE`, `HOLD_NOTE_END`, etc.).

---

## 3. Dataset layout your SnapBeat dataset should follow

The existing MMRS dataset uses:

- A **metadata table** (e.g. Parquet) with one row per chart, and columns such as:
  - Path to the chart file (e.g. `.osu` or `.json`).
  - Path to the **audio file** for that chart.
  - Optional: difficulty, mapper, year, etc., for conditioning.
- A **directory layout** where each chart and its audio can be resolved from those paths.

For SnapBeat you have two options.

### Option A: Reuse a simple directory + manifest (recommended to start)

Use a **manifest file** (e.g. CSV or JSON) that lists each chart and its audio path, for example:

```text
/path/to/snapbeat_dataset/
  manifest.csv          # chart_path, audio_path [, difficulty, ...]
  charts/
    song1_snapbeat.json
    song2_snapbeat.json
  audio/
    song1.mp3
    song2.mp3
```

`manifest.csv` example:

```csv
chart_path,audio_path
charts/song1_snapbeat.json,audio/song1.mp3
charts/song2_snapbeat.json,audio/song2.mp3
```

Your SnapBeat dataset class will:

1. Read the manifest.
2. For each row, load `audio_path` and `chart_path`.
3. Run the SnapBeat parser on the JSON to get `(events, event_times)`.
4. Use the same sequence-building and tokenization logic as MMRS (frames from audio, tokens from events).

### Option B: Match MMRS layout

Alternatively, match the MMRS layout and metadata so you can reuse more of `MmrsDataset`:

- `metadata.parquet` with columns including: `BeatmapSetFolder`, `BeatmapFile`, `AudioFile`, and optionally `ModeInt=3`, `StarRating`, etc.
- Under `data/{BeatmapSetFolder}/`: the SnapBeat JSON (as “beatmap”) and the audio file.

Then you only need a **SnapBeat parser** and a way to feed its output into the same `(events, event_times)` pipeline; the rest can stay as in MMRS (with a separate “snapbeat” dataset type that uses your parser instead of loading `.osu`).

---

## 4. What you need to implement

### 4.1 SnapBeat parser (same role as `OsuParser`)

- **Location**: e.g. `osuT5/osuT5/dataset/snapbeat_parser.py`.
- **Interface**: One method that takes your chart representation (e.g. a SnapBeat dict or path to JSON) and returns `(events, event_times)` in the same format as `OsuParser.parse(beatmap, speed)`.
- **Token mapping** (recommended for transfer learning):
  - **short** → same as mania tap: `TIME_SHIFT` → `MANIA_COLUMN` → `CIRCLE` (or a single note event the tokenizer already knows).
  - **long** → `TIME_SHIFT` → `MANIA_COLUMN` → `HOLD_NOTE` → (optional sustain events) → `TIME_SHIFT` (end time) → `HOLD_NOTE_END`.
  - **Lane** → `MANIA_COLUMN` with value 0..(N-1) for N lanes (tokenizer already supports 0–17).
  - **Time** → `TIME_SHIFT` in milliseconds (e.g. 10 ms steps, same as osu! pipeline).

Use the existing `Event` and `EventType` from `osuT5/event.py` and `osuT5/tokenizer.py` so no tokenizer changes are needed if you stay within mania-like events.

### 4.2 SnapBeat dataset class

- **Location**: e.g. `osuT5/osuT5/dataset/snapbeat_dataset.py`.
- **Role**: Same as `MmrsDataset`: yield training samples (frames + decoder_input_ids + labels + masks).
- **Steps per chart**:
  1. Load audio (reuse `load_audio_file` from `data_utils.py`).
  2. Build frames and frame times (reuse or mirror `_get_frames` from MMRS).
  3. Run **SnapBeat parser** to get `(events, event_times)` for that chart.
  4. Build context (e.g. “out” context = map context = this event sequence; “in” context can be minimal or none).
  5. Call the same sequence-creation and tokenization helpers used in MMRS: `_create_sequences`, `_normalize_time_shifts`, `_tokenize_sequence`, `_pad_frame_sequence`, `_pad_and_split_token_sequence`, so that each sample has `frames`, `decoder_input_ids`, `decoder_attention_mask`, `labels`.

You can either:

- Subclass or copy `BeatmapDatasetIterable` and swap in SnapBeat loading + SnapBeat parser, or  
- Implement a smaller `SnapBeatDataset(IterableDataset)` that still produces the same batch keys and shapes.

### 4.3 Wire the new dataset type

- In `osuT5/osuT5/utils/model_utils.py`, in `get_dataset()`:
  - Add:
    ```python
    elif args.data.dataset_type == "snapbeat":
        return SnapBeatDataset(args=args.data, test=test, **kwargs)
    ```
  - Pass the same `parser` and `tokenizer` (and `shared`) that you use for other types. For SnapBeat, `parser` can be your SnapBeat parser instance (you may need a small adapter so the same `get_dataloaders` can instantiate either OsuParser or SnapBeatParser based on `dataset_type`).
- In `get_dataloaders`, you currently do:
  ```python
  parser = OsuParser(args, tokenizer)
  ```
  For `dataset_type == "snapbeat"`, use your SnapBeat parser instead of (or in addition to) `OsuParser` so that the dataset receives a parser that returns `(events, event_times)` from a SnapBeat chart.

### 4.4 Config

- **DataConfig** in `osuT5/config.py`: Ensure `train_dataset_path`, `test_dataset_path`, and any new keys (e.g. `snapbeat_manifest`) are defined and used by `SnapBeatDataset`.
- **Training config**: Add or copy a config (e.g. `configs/train/snapbeat_lora.yaml`) that sets:
  - `data.dataset_type: snapbeat`
  - `data.train_dataset_path` and `data.test_dataset_path` to your SnapBeat dataset root (or leave empty and override via CLI).
  - `pretrained_path: OliBomby/Mapperatorinator-v31` (or the checkpoint you want to fine-tune).
  - `enable_lora: true` and your LoRA settings (e.g. from `lora.yaml`).
  - Optionally `pretrained_t5_compat: true` if you ever change tokenizer/vocab.

No tokenizer or `EventType` changes are required if you map SnapBeat notes to existing mania events.

---

## 5. End-to-end fine-tuning steps (once implemented)

1. **Prepare dataset**: SnapBeat JSONs + audio files + manifest (or Parquet) as above.
2. **Run training** (from repo root):
   ```bash
   python osuT5/train.py \
     --config-name snapbeat_lora \
     data.train_dataset_path=/path/to/snapbeat_train \
     data.test_dataset_path=/path/to/snapbeat_test
   ```
   Or with inline overrides:
   ```bash
   python osuT5/train.py \
     data.dataset_type=snapbeat \
     data.train_dataset_path=/path/to/snapbeat_train \
     data.test_dataset_path=/path/to/snapbeat_test \
     pretrained_path=OliBomby/Mapperatorinator-v31 \
     enable_lora=true
   ```
3. **Checkpoints**: Saved under the path configured in your train config (e.g. under `../` or `tensorboard_logs`). Use the best or last checkpoint for inference.
4. **Inference**: Point your inference script (e.g. `snapbeat_inference.py`) to the fine-tuned checkpoint (and LoRA adapter if you used LoRA) so generated charts are in the same token space and style as your SnapBeat dataset.

---

## 6. Summary

| You have | You need to add |
|----------|------------------|
| SnapBeat dataset (JSONs + audio) | Manifest or Parquet that lists chart path + audio path per sample. |
| — | **SnapBeat parser**: SnapBeat JSON → `(events, event_times)` using existing `Event`/`EventType` (mania-like). |
| — | **SnapBeat dataset class**: Load audio + chart, run parser, build frames + token sequences, yield same batch format as MMRS. |
| — | **Wiring**: `get_dataset` + `get_dataloaders` for `dataset_type=snapbeat` and a SnapBeat parser. |
| — | **Config**: `data.dataset_type`, paths, LoRA/pretrained path. |

Fine-tuning itself is “run `train.py` with that config and your data”; the real work is the parser and dataset class that turn your existing SnapBeat dataset into the `(events, event_times)` + audio frames the rest of the pipeline already consumes.
