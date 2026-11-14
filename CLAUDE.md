# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Mapperatorinator generates osu! beatmaps (all gamemodes) from audio. The pipeline is: mel spectrogram → `osuT5` (Whisper-derived transformer, ~219M params per README) produces a token stream of timed events → `Postprocessor` writes the `.osu`. The current model is **v32** (`OliBomby/Mapperatorinator-v32`, backbone `OliBomby/varwhisper-small`), which emits 2 px-precise positions itself via `POS` + `POS_REFINE` tokens. `osu_diffusion` position refinement is an optional legacy stage (`generate_positions`, std/ctb only), **off** in the v31/v32 inference configs. `classifier/` and `rcomplexion/` are separate auxiliary models.

A fork of this repo adds **SnapBeat** support (non-osu rhythm game charts routed through the mania tokenizer). The `amaremix` branch is actively working on SnapBeat LoRA fine-tuning — see `docs/SNAPBEAT_FINETUNING.md` and `docs/SNAPBEAT_TRAINING_RESULTS.md`.

## Environment

- **Python 3.10** is required. Newer Python versions break dependencies.
- Install: `pip install -r requirements.txt` (plus a CUDA/ROCm `torch` build matching your GPU, and `ffmpeg` on the system).
- Training is expected to run in Docker (`docker compose up -d --force-recreate && docker attach mapperatorinator_space`). The compose file mounts `../datasets` → `/workspace/datasets` and expects `WANDB_API_KEY` / `WANDB_ENTITY` in the environment.

## Common commands

Inference (uses Hydra override syntax `key=value`):
```
python inference.py beatmap_path="..." gamemode=0 difficulty=5.5 year=2023 descriptors="['jump aim','clean']" in_context=[TIMING,KIAI]
```
Defaults come from `configs/inference/v32.yaml` (`inference.py` uses `config_name="v32"`; `v32-mini` is a smaller variant). Other inference entry points: `web-ui.py` (GUI wrapper), `cli_inference.sh` (interactive prompts), `mai_mod.py` / `mai_mod_ui.py` (AI modding), `snapbeat_inference.py` (mania → SnapBeat JSON).

Training:
```
python osuT5/train.py -cn v32 train_dataset_path=/workspace/datasets/... test_dataset_path=/workspace/datasets/... train_dataset_end=90 test_dataset_start=90 test_dataset_end=100
# Multi-GPU
torchrun --nproc_per_node=NUM_GPUS osuT5/train.py -cn v32 ...
# Fine-tune / LoRA on top of v32
python osuT5/train.py -cn v32_ft ...
python osuT5/train.py -cn lora_v32 ...
# SnapBeat LoRA (base model: OliBomby/Mapperatorinator-v31)
python osuT5/train.py --config-name snapbeat_lora
```
`-cn` names a file in `configs/train/`. Always pass it: `osuT5/train.py` still defaults to `config_name="v29"`.

Diffusion training: `python osu_diffusion/train.py` (configs under `configs/diffusion/`).

There is no unit test suite and no lint config. `osuT5/test.py` is an evaluation/sampling script, not a pytest suite. `calc_fid.py` computes FID-style metrics for a trained model against a dataset.

## Config system

Hydra + dataclasses. The dataclasses in `config.py` (`InferenceConfig`, `FidConfig`, `MaiModConfig`) and `osuT5/osuT5/config.py` (`TrainConfig`) are the schema of truth; YAMLs override them. Important conventions:

- `configs/inference/`, `configs/train/`, `configs/diffusion/`, `configs/model/` are separate config groups — don't mix.
- Versioned configs (`v28`–`v32`, `v32-mini`, etc.) correspond to model iterations; v32 is current. `configs/inference/legacy/` and `configs/train/legacy/` are archived.
- **Each version is a different token "dialect"**: tokenizer flags in `configs/train/vNN.yaml` (`types_first`, `add_distances`, `position_precision`, `position_split_axes`, `position_refinement`, …) change the vocabulary, so a checkpoint only works with its own train config. v32: type token last, no distance token, `POS` = 32 px cell (x+y combined) + `POS_REFINE` (2 px). v31: type token first, `DISTANCE`, separate `POS_X`/`POS_Y` at 4 px.
- Paths should be relative or placeholder; `setup_args()` resolves relative data paths against `hydra.utils.get_original_cwd()` (Hydra changes cwd to `logs/...` at startup, which would otherwise break `./datasets/...`).
- CLI overrides: `key=value`, `key=[a,b]`, `null` for optional params. `in_context` and `output_type` use `ContextType` enum values: `NONE, TIMING, KIAI, MAP, GD, NO_HS`.
- `beatmap_path` in inference configs auto-fills `audio_path` and `output_path` from the `.osu` file.

## Code structure

- `osuT5/osuT5/event.py` / `tokenizer.py` — event types and the event ⇄ token-ID vocabulary. Time is relative to the window start, quantized to 10 ms. Position precision depends on the config version (see above). Events: hit objects, timing points, kiai, SV, new combos, hitsounds, mania columns. Gamemodes: `0=std, 1=taiko, 2=ctb, 3=mania`.
- `osuT5/osuT5/model/` — `Mapperatorinator` (`modeling_mapperatorinator.py`) wraps a HF seq2seq backbone chosen by `get_backbone_model` (Whisper variants; T5 for legacy) with custom input embeddings + loss. Custom transformer components in `custom_transformers/`.
- `osuT5/osuT5/inference/` — `Preprocessor`, `Processor`, `Postprocessor` (pipeline stages); `server.py` (multiprocess inference); `super_timing_generator.py` (averages 20 timing inferences); `logit_processors.py` (window-boundary masking for seamless long generation).
- `osuT5/osuT5/dataset/` — `ors_dataset`, `mmrs_dataset`, `web_dataset` (v32 default, streams `project-riz/osu-beatmaps` from HF), `snapbeat_dataset`. `dataset_type` (`ors`/`mmrs`/`web`/`snapbeat`) in train config routes to the right dataset and parser (`osu_parser.py` vs `snapbeat_parser.py`); `osuT5/osuT5/utils/model_utils.py` does the wiring.
- `osu_diffusion/` — optional DiT-based position refinement over the last 10% of the noise schedule. Recalculates slider end positions after each step to satisfy required slider lengths. Not used by default since v31/v32 (`generate_positions: false`).
- `diffusion_pipeline.py` — glue between osuT5 output and osu_diffusion refinement (`inference.py`, gated on `generate_positions` and gamemode 0/2).
- `classifier/`, `rcomplexion/` — independent auxiliary models with their own READMEs.

### Seamless long generation

Window length = `(src_seq_len − 1) × hop_length / sample_rate` (see `inference/preprocessor.py`): ≈16.4 s for v32 (`src_seq_len: 2048`, hop 128, 16 kHz), ≈32.8 s for v31 (4096). Long songs use 90% overlap with sequential windows (`lookback: 0.5`, `lookahead: 0.4` in `configs/inference/default.yaml`): each window pre-fills the decoder with the previous tokens from the first 50%; time tokens in the first 50% are masked (can't regenerate), and time tokens in the last 40% are treated as EOS (reserved for next window). See `logit_processors.py` and `processor.py`.

### LoRA

Uses `peft`. Specify `lora_path` (local or HF repo, e.g. `luannnguyen/snapbeat-lora-v8`) at inference. The base model loads from `model_path` / `pretrained_path`, LoRA adapters merge at load time. A LoRA only works on the base version it was trained on: `lora_v32.yaml` → v32, `snapbeat_lora.yaml` → v31, so run SnapBeat LoRAs with the v31 base and config.

## Conventions

- `pathlib.Path` for paths; validate `.osu` extension and existence before loading beatmaps.
- Use `routed_pickle` for safe deserialization (vendored class-allowlist wrapper — `utils/routed_pickle.py`, imported as `from utils import routed_pickle`).
- `get_default_logger()` for user-facing messages; `assert_package_version()` for dependency checks (both defined in `inference.py`).
- `slider.Beatmap` is the osu! file I/O library (custom fork at `github.com/OliBomby/slider.git@gedagedigedagedaoh`).
- `excepthook` (`utils/excepthook.py`) is imported for its side effect (project-wide exception handler) — keep the `# noqa` import near the top of new entry-point scripts.

## Platform notes

- **Windows + CUDA**: `torch.compile` is auto-skipped (Triton unavailable). Multi-worker `IterableDataset` loaders can deadlock at epoch boundaries — set `dataloader.num_workers=0` if training hangs.
- **Linux** is the primary training target; `flash-attn==2.7.4.post1` is installed in the Docker image.
