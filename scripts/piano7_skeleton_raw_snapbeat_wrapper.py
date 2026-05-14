#!/usr/bin/env python3
"""Rewrite Piano7 skeleton charts to SnapBeat v1.3 **file** layout (match ``datasets/dataset/json``).

Top-level keys: ``version``, ``format``, ``notes``, ``songMeta``, ``snapBeatMeta``.
(No ``lastModified`` — omitted on purpose.)

- ``version`` → ``1.3``
- ``format`` → ``MT3`` (override with ``--format``)
- ``songMeta.visualSpeed`` is always **``bpm / 30``** (``gameMeta.visualSpeed`` is ignored).
- ``snapBeatMeta`` → ``{}`` unless ``--keep-snap-meta`` is passed (then merge from source)

``notes`` are left unchanged.

Usage::

    python scripts/piano7_skeleton_raw_snapbeat_wrapper.py \\
        --dir datasets/dataset/piano7/json_skeleton_raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _infer_n_lanes(notes: list, default: int = 4) -> int:
    m = 0
    for n in notes:
        if isinstance(n, dict) and "lane" in n:
            try:
                m = max(m, int(n["lane"]))
            except (TypeError, ValueError):
                continue
    return max(default, m) if m else default


def build_snapbeat_wrapper(
    data: dict,
    *,
    format_code: str = "MT3",
    snap_empty: bool = True,
) -> dict:
    notes = data.get("notes")
    if not isinstance(notes, list):
        notes = []

    sm = dict(data.get("songMeta") or {})
    gm = dict(data.get("gameMeta") or {})

    n_lanes = int(gm.get("nLanes", sm.get("nLanes", 0)) or 0)
    if n_lanes <= 0:
        n_lanes = _infer_n_lanes(notes)

    n_lanes_meta = int(sm.get("nLanesMeta", n_lanes) or n_lanes)
    bpm_val = float(sm.get("bpm", 120.0) or 120.0)
    visual_speed = round(bpm_val / 30.0, 10)

    song_meta = {
        "songName": sm.get("songName", ""),
        "audioPath": sm.get("audioPath", ""),
        "audioVolume": float(sm.get("audioVolume", 1.0)),
        "bpm": bpm_val,
        "audioOffset": int(sm.get("audioOffset", 0) or 0),
        "audioDuration": float(sm.get("audioDuration", 0.0)),
        "nLanes": n_lanes,
        "nLanesMeta": n_lanes_meta,
        "visualSpeed": visual_speed,
    }

    sb = {} if snap_empty else dict(data.get("snapBeatMeta") or {})

    return {
        "version": "1.3",
        "format": format_code,
        "notes": notes,
        "songMeta": song_meta,
        "snapBeatMeta": sb,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("datasets/dataset/piano7/json_skeleton_raw"),
    )
    ap.add_argument("--format", default="MT3", dest="format_code")
    ap.add_argument(
        "--keep-snap-meta",
        action="store_true",
        help="Keep / merge snapBeatMeta from source instead of {}",
    )
    args = ap.parse_args()
    root = args.dir.resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(1)

    n = 0
    for fp in sorted(root.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        out = build_snapbeat_wrapper(
            data,
            format_code=args.format_code,
            snap_empty=not args.keep_snap_meta,
        )
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.write("\n")
        n += 1

    print(f"Rewrote {n} chart wrappers under {root}")


if __name__ == "__main__":
    main()
