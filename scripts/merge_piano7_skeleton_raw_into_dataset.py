#!/usr/bin/env python3
"""Copy Piano7 ``json_skeleton_raw`` + ``audio`` into unified ``datasets/dataset`` layout.

Writes under ``{dataset_root}/json`` and ``{dataset_root}/audio`` using a new UUID stem
per chart so names never collide with existing SnapBeat UUID charts. Pairing for
training follows ``SnapBeatDataset``: JSON stem must match audio stem.

UUIDs are **deterministic** (UUID v5) from each source JSON stem, so re-running this
script overwrites the same targets and stays idempotent.

Example::

    python scripts/merge_piano7_skeleton_raw_into_dataset.py
    python scripts/merge_piano7_skeleton_raw_into_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

# Namespace chosen for stable v5 IDs (not an RFC reserved namespace).
_PIANO7_DATASET_UUID_NS = uuid.UUID("8c5b8a2e-6f1d-5e4c-9b0a-2c4d6e8f0a1b")


def _new_stem(json_stem: str) -> str:
    return str(uuid.uuid5(_PIANO7_DATASET_UUID_NS, json_stem))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--piano7-root",
        type=Path,
        default=Path("datasets/dataset/piano7"),
        help="Folder containing json_skeleton_raw/ and audio/",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/dataset"),
        help="Folder containing json/ and audio/ (SnapBeat layout)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions only")
    args = ap.parse_args()

    skel = args.piano7_root / "json_skeleton_raw"
    src_audio = args.piano7_root / "audio"
    out_json = args.dataset_root / "json"
    out_audio = args.dataset_root / "audio"

    for d in (skel, src_audio):
        if not d.is_dir():
            print(f"error: missing directory {d.resolve()}", file=sys.stderr)
            return 1
    if not args.dry_run:
        out_json.mkdir(parents=True, exist_ok=True)
        out_audio.mkdir(parents=True, exist_ok=True)

    copied_json = 0
    copied_audio = 0
    skipped = 0

    for jp in sorted(skel.glob("*.json")):
        stem = jp.stem
        mp3 = src_audio / f"{stem}.mp3"
        if not mp3.is_file():
            print(f"skip (no audio): {jp.name} -> expected {mp3.name}", file=sys.stderr)
            skipped += 1
            continue

        new_stem = _new_stem(stem)
        dest_j = out_json / f"{new_stem}.json"
        dest_a = out_audio / f"{new_stem}.mp3"

        if args.dry_run:
            print(f"would copy {jp.name} -> {dest_j.name} + {dest_a.name}")
            copied_json += 1
            copied_audio += 1
            continue

        with open(jp, encoding="utf-8") as f:
            chart = json.load(f)
        sm = chart.get("songMeta")
        if isinstance(sm, dict):
            sm["audioPath"] = f"{new_stem}.mp3"

        with open(dest_j, "w", encoding="utf-8") as f:
            json.dump(chart, f, indent=2, ensure_ascii=False)
            f.write("\n")
        shutil.copy2(mp3, dest_a)
        copied_json += 1
        copied_audio += 1

    print(
        f"done: wrote {copied_json} json, {copied_audio} audio under {args.dataset_root}; "
        f"skipped {skipped}"
    )
    return 0 if skipped == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
