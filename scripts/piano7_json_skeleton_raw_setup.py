#!/usr/bin/env python3
"""Clone ``json_skeleton`` → ``json_skeleton_raw`` and normalize ``swipe`` → ``short``.

Swipe audit (Piano7 skeleton):

- No note had ``variant`` (including no ``\"strong\"``).
- All had a ``variants`` list (e.g. ``['down', 'flick']``, ``['right']``).

Converted shorts match minimal Piano7 taps: ``type``, ``time``, ``lane`` only.

Usage::

    python scripts/piano7_json_skeleton_raw_setup.py --recreate
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_DEFAULT_SOURCE = Path("datasets/dataset/piano7/json_skeleton")
_DEFAULT_TARGET = Path("datasets/dataset/piano7/json_skeleton_raw")


def swipe_to_minimal_short(note: dict) -> bool:
    if not isinstance(note, dict) or note.get("type") != "swipe":
        return False
    t = note["time"]
    lane = note["lane"]
    note.clear()
    note.update({"type": "short", "time": t, "lane": lane})
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    ap.add_argument("--target", type=Path, default=_DEFAULT_TARGET)
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="Delete target then copy source (required if target exists)",
    )
    ap.add_argument(
        "--clone-only",
        action="store_true",
        help="Only copy source → target; do not convert swipes",
    )
    args = ap.parse_args()

    src = args.source.resolve()
    dst = args.target.resolve()
    if not src.is_dir():
        print("Missing source:", src, file=sys.stderr)
        sys.exit(1)

    if dst.exists():
        if not args.recreate:
            print("Target exists; use --recreate:", dst, file=sys.stderr)
            sys.exit(1)
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    print("Cloned", src, "->", dst)

    if args.clone_only:
        return

    converted = 0
    files_hit = 0
    for fp in sorted(dst.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        n_here = 0
        for note in notes:
            if swipe_to_minimal_short(note):
                n_here += 1
        if n_here:
            files_hit += 1
            converted += n_here
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")

    print(f"Converted {converted} swipe -> short in {files_hit} files")


if __name__ == "__main__":
    main()