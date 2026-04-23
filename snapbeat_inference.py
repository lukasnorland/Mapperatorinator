"""SnapBeat chart generation using Mapperatorinator V31.

Runs mania inference and always outputs a SnapBeat JSON file as the primary
result.  The intermediate .osu file is kept alongside for reference.

Usage:
    python snapbeat_inference.py audio_path="path/to/song.mp3" \\
        gamemode=3 keycount=4 difficulty=5.0

All standard Hydra / Mapperatorinator overrides are supported.  SnapBeat-
specific options are passed via the environment or via extra Hydra keys
defined below (prefixed with ``snapbeat_``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig

import excepthook  # noqa – project-wide exception hook

from config import InferenceConfig
from inference import (
    compile_args,
    get_config,
    generate,
    load_model_with_server,
    setup_inference_environment,
)
from snapbeat_converter import osu_mania_to_snapbeat, write_snapbeat_json


# ── Defaults for SnapBeat-specific settings ─────────────────────────────────
# Note: `game_code` is now a real field on InferenceConfig (see config.py),
# so it is read from `args.game_code` rather than popped off the DictConfig.
_SNAPBEAT_DEFAULTS = {
    "snapbeat_song_name": "Generated",
    "snapbeat_visual_speed": 4.5,
    "snapbeat_max_simultaneous": 2,
}

# game_code -> default LoRA to load. Missing entries mean "no model yet"; add
# BH/DR once those LoRAs are trained and pushed to HuggingFace.
_GAME_CODE_REGISTRY = {
    "MT3": "luannnguyen/snapbeat-lora-v8",
    # "BH": "<your-org>/snapbeat-lora-bh-v1",
    # "DR": "<your-org>/snapbeat-lora-dr-v1",
}

_ALLOWED_GAME_CODES = {"MT3", "BH", "DR"}


def _pop_snapbeat_args(cfg: DictConfig) -> dict:
    """Extract and remove SnapBeat-specific keys from the Hydra config."""
    sb = {}
    for key, default in _SNAPBEAT_DEFAULTS.items():
        sb[key] = cfg.pop(key, default) if key in cfg else default
    return sb


@hydra.main(config_path="configs/inference", config_name="v31", version_base="1.1")
def main(cfg: DictConfig) -> None:
    sb_args = _pop_snapbeat_args(cfg)
    args: InferenceConfig = OmegaConf.to_object(cfg)

    # Validate game_code and route lora_path from the registry if needed
    game_code = str(args.game_code).upper()
    if game_code not in _ALLOWED_GAME_CODES:
        print(
            f"ERROR: invalid game_code={args.game_code!r}; "
            f"must be one of {sorted(_ALLOWED_GAME_CODES)}",
            file=sys.stderr,
        )
        sys.exit(2)
    args.game_code = game_code

    if not args.lora_path:
        registered = _GAME_CODE_REGISTRY.get(game_code)
        if registered is None:
            print(
                f"ERROR: no LoRA registered for game_code={game_code!r}. "
                f"Registered codes: {sorted(_GAME_CODE_REGISTRY)}. "
                f"Either train/register a LoRA for {game_code} or pass lora_path=... explicitly.",
                file=sys.stderr,
            )
            sys.exit(3)
        args.lora_path = registered
        print(f"game_code={game_code}: using LoRA {registered!r}")
    else:
        print(f"game_code={game_code}: honoring explicit lora_path={args.lora_path!r}")

    # Force mania mode with sensible defaults for SnapBeat-style charts
    if args.gamemode is None:
        args.gamemode = 3
    if args.keycount is None:
        args.keycount = 4
    if args.hold_note_ratio is None:
        args.hold_note_ratio = 0.3
    if args.scroll_speed_ratio is None:
        args.scroll_speed_ratio = 0.05

    compile_args(args)
    setup_inference_environment(args.seed)

    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=args.use_server,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
    )

    generation_config, beatmap_config = get_config(args)

    _result, result_path, _osz_path = generate(
        args,
        generation_config=generation_config,
        beatmap_config=beatmap_config,
        model=model,
        tokenizer=tokenizer,
    )

    if result_path is None:
        print("ERROR: No .osu file was generated – cannot produce SnapBeat JSON.", file=sys.stderr)
        sys.exit(1)

    # ── Convert to SnapBeat JSON (mandatory) ────────────────────────────
    audio_stem = Path(args.audio_path).stem
    json_path = Path(args.output_path) / f"{audio_stem}_snapbeat.json"

    snapbeat = osu_mania_to_snapbeat(
        osu_path=result_path,
        song_name=sb_args["snapbeat_song_name"],
        audio_path=Path(args.audio_path).name,
        n_lanes=args.keycount,
        visual_speed=sb_args["snapbeat_visual_speed"],
        max_simultaneous=sb_args["snapbeat_max_simultaneous"],
        game_code=args.game_code,
    )

    written = write_snapbeat_json(snapbeat, json_path)
    print(f"SnapBeat JSON saved to {written}")


if __name__ == "__main__":
    main()
