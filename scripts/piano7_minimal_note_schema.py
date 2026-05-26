#!/usr/bin/env python3
"""Apply minimal note shape (same rules as ``json_skeleton_raw``).

**Short** — only ``type``, ``time``, ``lane``.  
If ``variant == \"strong\"``, also ``variant`` and ``metas`` (shadow only).

**Long** — ``type``, ``time``, ``lane``, ``controls`` with a single
``{\"time\": z, \"lane\": y}`` where ``y`` is the note's lane.  
If ``variant == \"strong\"``, also ``variant`` and ``metas`` (shadow only).
No ``variant`` on the control object.

Does **not** add empty ``variant`` / ``metas`` / ``controls`` when unused.
Other ``metas`` entries (e.g. ``mini``, ``bg_color``) are **removed** — only
``shadow`` is kept for strong notes.

Default ``--dir``: ``datasets/dataset/json``.

Usage::

    python scripts/piano7_minimal_note_schema.py
    python scripts/piano7_minimal_note_schema.py --dir datasets/dataset/piano7/json_skeleton_raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHADOW = [{"key": "shadow", "value": 2}]


def normalize_note(n: dict) -> None:
    if not isinstance(n, dict):
        return
    t = n.get("type")
    if t == "short":
        out: dict = {"type": "short", "time": n["time"], "lane": n["lane"]}
        if n.get("variant") == "strong":
            out["variant"] = "strong"
            out["metas"] = [dict(SHADOW[0])]
        n.clear()
        n.update(out)
        return
    if t == "long":
        lane = n["lane"]
        ctr = n.get("controls")
        if isinstance(ctr, list) and ctr:
            last = ctr[-1]
            z = last.get("time", n["time"]) if isinstance(last, dict) else n["time"]
        else:
            z = n["time"]
        out = {
            "type": "long",
            "time": n["time"],
            "lane": lane,
            "controls": [{"time": z, "lane": lane}],
        }
        if n.get("variant") == "strong":
            out["variant"] = "strong"
            out["metas"] = [dict(SHADOW[0])]
        n.clear()
        n.update(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("datasets/dataset/json"),
        help="Chart JSON directory (e.g. datasets/dataset/json or json_skeleton_raw)",
    )
    args = ap.parse_args()
    root = args.dir.resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(1)

    changed_files = 0
    for fp in sorted(root.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        for note in notes:
            normalize_note(note)
        changed_files += 1
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(f"Normalized notes in {changed_files} files under {root}")


if __name__ == "__main__":
    main()
