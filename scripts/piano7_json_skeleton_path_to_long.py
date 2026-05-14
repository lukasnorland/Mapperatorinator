#!/usr/bin/env python3
"""Clone Piano7 ``json/`` to ``json_skeleton/`` and simplify path notes to ``long``.

- Every ``zigzag``, ``wavy``, and ``trace`` becomes ``type: long``.
- ``controls`` is replaced by a single endpoint: the last point of the original path
  (same duration as the full path; end lane = last control's lane).
- ``wavy`` and ``trace`` get ``variant: "strong"`` (on the note). ``zigzag`` has no
  ``variant`` key so it matches native Piano7 ``long`` rows.

Usage::

    python scripts/piano7_json_skeleton_path_to_long.py --recreate
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_PATH_TYPES = frozenset({"zigzag", "wavy", "trace"})


def convert_path_notes_to_long(notes: list) -> tuple[int, int, int]:
    """Mutate *notes* in place. Returns (n_zigzag, n_wavy, n_trace) converted."""
    nz = nw = nt = 0
    for n in notes:
        if not isinstance(n, dict):
            continue
        orig = n.get("type")
        if orig not in _PATH_TYPES:
            continue
        controls = n.get("controls")
        if not isinstance(controls, list) or not controls:
            continue
        last = controls[-1]
        if not isinstance(last, dict):
            continue
        t_end = last.get("time")
        lane_end = last.get("lane")
        if t_end is None or lane_end is None:
            continue

        n["type"] = "long"
        n["controls"] = [{"time": t_end, "lane": lane_end}]

        if orig in ("wavy", "trace"):
            n["variant"] = "strong"
        else:
            n.pop("variant", None)

        n.pop("variants", None)
        # Keep metas if any; Piano7 path notes rarely have them

        if orig == "zigzag":
            nz += 1
        elif orig == "wavy":
            nw += 1
        else:
            nt += 1
    return nz, nw, nt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/dataset/piano7/json"),
        help="Source directory (Piano7 JSON)",
    )
    ap.add_argument(
        "--target",
        type=Path,
        default=Path("datasets/dataset/piano7/json_skeleton"),
        help="Clone + conversion output directory",
    )
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="Remove target directory before copying (recommended)",
    )
    ap.add_argument(
        "--convert-only",
        action="store_true",
        help="Skip copy; only run conversion on existing --target",
    )
    args = ap.parse_args()

    src = args.source.resolve()
    dst = args.target.resolve()
    if not src.is_dir():
        print("Missing source dir:", src, file=sys.stderr)
        sys.exit(1)

    if not args.convert_only:
        if dst.exists():
            if not args.recreate:
                print("Target exists; pass --recreate to replace, or --convert-only", file=sys.stderr)
                sys.exit(1)
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("Cloned", src, "->", dst)

    total_z = total_w = total_t = 0
    n_files = 0
    for fp in sorted(dst.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        z, w, t = convert_path_notes_to_long(notes)
        if z or w or t:
            n_files += 1
            total_z += z
            total_w += w
            total_t += t
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(
        "Converted zigzag -> long:", total_z,
        "| wavy -> long (variant strong):", total_w,
        "| trace -> long (variant strong):", total_t,
    )
    print("Files with at least one conversion:", n_files)


if __name__ == "__main__":
    main()
