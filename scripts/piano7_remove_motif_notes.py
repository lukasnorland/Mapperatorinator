#!/usr/bin/env python3
"""Drop Piano7 JSON notes where ``layer == \"motif\"``.

Keeps notes with ``layer`` in ``main`` / ``main2``, notes with no ``layer`` key,
and any other layer value except ``motif``.

Usage::

    python scripts/piano7_remove_motif_notes.py --dir datasets/dataset/piano7/json --dry-run
    python scripts/piano7_remove_motif_notes.py --dir datasets/dataset/piano7/json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def filter_notes(notes: list) -> tuple[list, int, int]:
    """Return (filtered_notes, kept_count, removed_count)."""
    removed = 0
    kept: list = []
    for n in notes:
        if isinstance(n, dict) and n.get("layer") == "motif":
            removed += 1
            continue
        kept.append(n)
    return kept, len(kept), removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("datasets/dataset/piano7/json"),
        help="Directory of Piano7 *.json files",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write; print summary (use --verbose for per-file lines)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="With --dry-run, print each file that would change",
    )
    args = ap.parse_args()
    root = args.dir.resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(1)

    files = sorted(root.glob("*.json"))
    total_removed = 0
    total_before = 0
    changed_files = 0

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        before = len(notes)
        new_notes, _, removed = filter_notes(notes)
        total_before += before
        total_removed += removed
        if removed == 0:
            continue
        changed_files += 1
        if args.dry_run:
            if args.verbose:
                print(f"{fp.name}: {before} -> {len(new_notes)} (-{removed})")
            continue
        data["notes"] = new_notes
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    if args.dry_run:
        print(f"Totals: {total_before} notes, would remove {total_removed} motif rows in {changed_files} files")
        return

    print(
        f"Updated {changed_files} files under {root}; "
        f"removed {total_removed} motif notes (from {total_before} total rows scanned)."
    )


if __name__ == "__main__":
    main()
