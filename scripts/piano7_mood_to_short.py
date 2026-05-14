#!/usr/bin/env python3
"""Turn every ``type: mood`` note into a plain ``short`` tap and drop ``metas``.

Other keys (``time``, ``lane``, ``controls`` if any) are normalized for a minimal
short note: ``metas`` / empty ``controls`` are removed so the object matches
typical Piano7 shorts.

Usage::

    python scripts/piano7_mood_to_short.py --dir datasets/dataset/piano7/json --dry-run
    python scripts/piano7_mood_to_short.py --dir datasets/dataset/piano7/json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert_mood_notes(notes: list) -> int:
    """Mutate notes in place. Returns number of mood notes converted."""
    changed = 0
    for n in notes:
        if not isinstance(n, dict) or n.get("type") != "mood":
            continue
        n["type"] = "short"
        n.pop("metas", None)
        # Mood rows should not be holds; strip if present.
        n.pop("controls", None)
        n.pop("variant", None)
        n.pop("variants", None)
        changed += 1
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("datasets/dataset/piano7/json"),
        help="Directory of Piano7 *.json files",
    )
    ap.add_argument("--dry-run", action="store_true", help="Count only; do not write")
    args = ap.parse_args()
    root = args.dir.resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(1)

    files = sorted(root.glob("*.json"))
    total = 0
    touched = 0

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        n = convert_mood_notes(notes)
        if n == 0:
            continue
        touched += 1
        total += n
        if args.dry_run:
            continue
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    if args.dry_run:
        print(f"Would convert {total} mood notes in {touched} files")
        return
    print(f"Converted {total} mood -> short (metas removed) in {touched} files under {root}")


if __name__ == "__main__":
    main()
