#!/usr/bin/env python3
"""Convert every ``zigzag`` note in ``datasets/dataset/json`` to ``long``.

Each ``long`` keeps ``variant`` and ``metas``. ``controls`` becomes a single
endpoint: last path time, lane forced to the note's ``lane`` (matches other
``long`` rows in this corpus). Strong holds get ``\"variant\": \"strong\"`` on
the control point, same as existing ``long`` notes.

Usage::

    python scripts/dataset_json_zigzag_to_long.py --dry-run
    python scripts/dataset_json_zigzag_to_long.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert_zigzag_to_long(note: dict) -> bool:
    if not isinstance(note, dict) or note.get("type") != "zigzag":
        return False
    ctr = note.get("controls")
    if not isinstance(ctr, list) or not ctr:
        note["type"] = "long"
        return True
    last = ctr[-1]
    if not isinstance(last, dict) or last.get("time") is None or note.get("lane") is None:
        note["type"] = "long"
        note["controls"] = [{"time": note["time"], "lane": note["lane"]}]
        return True
    t_end = last["time"]
    lane = note["lane"]
    v = note.get("variant", "")
    if v == "strong":
        note["controls"] = [{"time": t_end, "lane": lane, "variant": "strong"}]
    else:
        note["controls"] = [{"time": t_end, "lane": lane}]
    note["type"] = "long"
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("datasets/dataset/json"),
        help="SnapBeat JSON directory",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.dir.resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(1)

    total = 0
    files = 0
    for fp in sorted(root.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        n_here = sum(1 for note in notes if convert_zigzag_to_long(note))
        if n_here == 0:
            continue
        files += 1
        total += n_here
        if args.dry_run:
            continue
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    if args.dry_run:
        print(f"Would convert {total} zigzag -> long in {files} files")
    else:
        print(f"Converted {total} zigzag -> long in {files} files under {root}")


if __name__ == "__main__":
    main()
