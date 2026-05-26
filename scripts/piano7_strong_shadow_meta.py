#!/usr/bin/env python3
"""Ensure every note with ``variant == \"strong\"`` has shadow meta ``value: 2``.

Matches ``datasets/dataset/json`` convention::

    {\"key\": \"shadow\", \"value\": 2}

Existing ``metas`` entries are kept; any prior ``shadow`` entry is replaced.

Usage::

    python scripts/piano7_strong_shadow_meta.py --dir datasets/dataset/piano7/json_skeleton
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHADOW_META = {"key": "shadow", "value": 2}


def ensure_shadow_meta(notes: list) -> int:
    """Mutate *notes* in place. Returns count of notes updated."""
    changed = 0
    for n in notes:
        if not isinstance(n, dict) or n.get("variant") != "strong":
            continue
        metas = [m for m in (n.get("metas") or []) if isinstance(m, dict) and m.get("key") != "shadow"]
        metas.append(dict(SHADOW_META))
        n["metas"] = metas
        changed += 1
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("datasets/dataset/piano7/json_skeleton"),
        help="Directory of chart JSON files to update",
    )
    args = ap.parse_args()
    root = args.dir.resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(1)

    files_changed = 0
    notes_updated = 0
    for fp in sorted(root.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        n = ensure_shadow_meta(notes)
        if n == 0:
            continue
        files_changed += 1
        notes_updated += n
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(f"Updated {notes_updated} strong notes in {files_changed} files under {root}")


if __name__ == "__main__":
    main()
