#!/usr/bin/env python3
"""Normalize every ``long`` note to SnapBeat v1.3 shape (same as ``datasets/dataset/json``).

For each ``long``:

- ``variant``: string (``\"\"`` or ``\"strong\"``).
- ``metas``: always a list (``[]`` if none). ``strong`` notes keep / add ``shadow`` → ``2``.
- ``controls``: exactly **one** endpoint ``{time, lane}``; if ``variant == \"strong\"`` on the
  note, the control also includes ``\"variant\": \"strong\"`` (matches ``dataset/json``).
- Strips stray keys on control points (e.g. ``type``, ``variants`` from Piano7 glitches).

Default directory: ``datasets/dataset/piano7/json_skeleton``.

Usage::

    python scripts/piano7_normalize_longs_snapbeat.py --dir datasets/dataset/piano7/json_skeleton
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHADOW_META = {"key": "shadow", "value": 2}


def normalize_long(n: dict) -> bool:
    """Mutate *n* in place. Returns True if *n* was a long that was updated."""
    if not isinstance(n, dict) or n.get("type") != "long":
        return False

    ctr = n.get("controls")
    if not isinstance(ctr, list) or not ctr:
        return False

    end = ctr[-1]
    if not isinstance(end, dict):
        return False
    t_end = end.get("time")
    lane_end = end.get("lane")
    if t_end is None or lane_end is None:
        return False

    v = n.get("variant", "")
    if not isinstance(v, str):
        v = str(v) if v is not None else ""

    if v == "strong":
        n["variant"] = "strong"
        n["controls"] = [{"time": t_end, "lane": lane_end, "variant": "strong"}]
        metas = [m for m in (n.get("metas") or []) if not (isinstance(m, dict) and m.get("key") == "shadow")]
        metas.append(dict(SHADOW_META))
        n["metas"] = metas
    else:
        n["variant"] = v if v else ""
        n["controls"] = [{"time": t_end, "lane": lane_end}]
        if not isinstance(n.get("metas"), list):
            n["metas"] = []

    n.pop("variants", None)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("datasets/dataset/piano7/json_skeleton"),
        help="Directory of chart JSON files",
    )
    args = ap.parse_args()
    root = args.dir.resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(1)

    files_touched = 0
    longs_updated = 0
    for fp in sorted(root.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        notes = data.get("notes")
        if not isinstance(notes, list):
            continue
        n_changed = sum(1 for note in notes if normalize_long(note))
        if n_changed == 0:
            continue
        files_touched += 1
        longs_updated += n_changed
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(f"Normalized {longs_updated} long notes in {files_touched} files under {root}")


if __name__ == "__main__":
    main()
