#!/usr/bin/env python3
"""Remove the ``layer`` key from every note in Piano7 JSON charts.

Usage::

    python scripts/piano7_strip_note_layer.py --dir datasets/dataset/piano7/json --dry-run
    python scripts/piano7_strip_note_layer.py --dir datasets/dataset/piano7/json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def strip_layers(notes: list) -> tuple[list, int]:
    """Return (possibly same list, n_stripped). Mutates dict notes in place."""
    stripped = 0
    for n in notes:
        if isinstance(n, dict) and "layer" in n:
            del n["layer"]
            stripped += 1
    return notes, stripped


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
    total_stripped = 0
    touched = 0

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        _, n = strip_layers(notes)
        if n == 0:
            continue
        touched += 1
        total_stripped += n
        if args.dry_run:
            continue
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    if args.dry_run:
        print(f"Would strip {total_stripped} layer fields in {touched} files")
        return
    print(f"Stripped {total_stripped} layer fields in {touched} files under {root}")


if __name__ == "__main__":
    main()
