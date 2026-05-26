#!/usr/bin/env python3
"""Set every ``long`` note's ``controls[*].lane`` to the note's top-level ``lane``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def align_long_control_lanes(notes: list) -> int:
    """Mutate notes in place. Returns count of control points updated."""
    updated = 0
    for n in notes:
        if not isinstance(n, dict) or n.get("type") != "long":
            continue
        if "lane" not in n:
            continue
        parent_lane = n["lane"]
        for c in n.get("controls") or []:
            if not isinstance(c, dict):
                continue
            if c.get("lane") != parent_lane:
                c["lane"] = parent_lane
                updated += 1
    return updated


def process_dir(root: Path) -> tuple[int, int]:
    """Returns (files_changed, control_points_updated)."""
    changed_files = 0
    total_pts = 0
    for fp in sorted(root.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        n = align_long_control_lanes(notes)
        if n == 0:
            continue
        changed_files += 1
        total_pts += n
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return changed_files, total_pts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dirs",
        nargs="+",
        type=Path,
        default=[
            Path("datasets/dataset/piano7/json"),
            Path("datasets/dataset/piano7/json_skeleton"),
        ],
        help="Directories to process",
    )
    args = ap.parse_args()
    for d in args.dirs:
        root = d.resolve()
        if not root.is_dir():
            print("Skip (not a dir):", root, file=sys.stderr)
            continue
        cf, pts = process_dir(root)
        print(f"{root}: updated {pts} control lane(s) in {cf} files")


if __name__ == "__main__":
    main()
