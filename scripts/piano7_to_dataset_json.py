#!/usr/bin/env python3
"""Convert Piano7-style JSON (v2.3, gameMeta + layered notes) to SnapBeat v1.3 JSON.

The layout under ``datasets/dataset/json`` matches ``snapbeat_converter.osu_mania_to_snapbeat``:
top-level ``format`` / ``lastModified``, ``songMeta`` with ``nLanes`` / ``nLanesMeta`` / ``visualSpeed``,
and homogeneous ``notes`` entries (``variant``, ``controls``, ``metas`` on every note).

Examples::

    python scripts/piano7_to_dataset_json.py --report
    python scripts/piano7_to_dataset_json.py \\
        --input datasets/dataset/piano7/json/4_Dynamite.json \\
        --output /tmp/4_Dynamite_v13.json
    python scripts/piano7_to_dataset_json.py \\
        --input-dir datasets/dataset/piano7/json \\
        --output-dir datasets/dataset/piano7/json_v13
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _merge_metas(existing: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(existing or [])
    for e in extra:
        if e not in out:
            out.append(e)
    return out


def piano7_dict_to_snapbeat_v13(
    chart: dict[str, Any],
    *,
    format_code: str = "MT3",
    layer: str | None = "main",
    drop_mood: bool = True,
) -> dict[str, Any]:
    """Return a new dict in SnapBeat JSON v1.3 shape (compatible with ``SnapBeatParser``).

    Args:
        chart: Parsed Piano7 JSON (typically ``version`` 2.3).
        format_code: Top-level ``format`` string.
        layer: If set, keep only notes whose ``layer`` is this value (Piano7 duplicates
            ``main`` / ``motif`` streams). Use ``None`` to keep all notes.
        drop_mood: If True, omit ``mood`` notes (section markers, not hits).
    """
    song = dict(chart.get("songMeta") or {})
    game = dict(chart.get("gameMeta") or {})
    n_lanes = int(game.get("nLanes", song.get("nLanes", 4)))
    visual_speed = float(game.get("visualSpeed", song.get("visualSpeed", 4.0)))

    song["nLanes"] = n_lanes
    song["nLanesMeta"] = int(song.get("nLanesMeta", n_lanes))
    song["visualSpeed"] = visual_speed

    out_notes: list[dict[str, Any]] = []

    for raw in chart.get("notes") or []:
        if layer is not None and "layer" in raw and raw.get("layer") != layer:
            continue
        t = raw.get("type", "short")
        if drop_mood and t == "mood":
            continue

        metas = list(raw.get("metas") or [])
        if raw.get("variants") is not None:
            metas = _merge_metas(
                metas,
                [{"key": "piano7_variants", "value": json.dumps(raw["variants"], ensure_ascii=False)}],
            )
        if raw.get("layer"):
            metas = _merge_metas(metas, [{"key": "piano7_layer", "value": str(raw["layer"])}])

        if t == "short":
            out_notes.append({
                "type": "short",
                "time": float(raw["time"]),
                "variant": "",
                "lane": int(raw["lane"]),
                "controls": [],
                "metas": metas,
            })
        elif t == "long":
            controls = raw.get("controls") or []
            cleaned = [{k: v for k, v in c.items() if k in ("time", "lane", "variant")} for c in controls]
            out_notes.append({
                "type": "long",
                "time": float(raw["time"]),
                "variant": str(raw.get("variant", "")),
                "lane": int(raw["lane"]),
                "controls": cleaned,
                "metas": metas,
            })
        elif t == "zigzag":
            controls = raw.get("controls") or []
            cleaned = [{k: v for k, v in c.items() if k in ("time", "lane", "variant")} for c in controls]
            out_notes.append({
                "type": "zigzag",
                "time": float(raw["time"]),
                "variant": str(raw.get("variant", "")),
                "lane": int(raw["lane"]),
                "controls": cleaned,
                "metas": metas,
            })
        elif t == "swipe":
            out_notes.append({
                "type": "short",
                "time": float(raw["time"]),
                "variant": "",
                "lane": int(raw["lane"]),
                "controls": [],
                "metas": _merge_metas(metas, [{"key": "piano7_type", "value": "swipe"}]),
            })
        elif t in ("trace", "wavy"):
            controls = raw.get("controls") or []
            cleaned = [{k: v for k, v in c.items() if k in ("time", "lane", "variant")} for c in controls]
            out_notes.append({
                "type": "zigzag",
                "time": float(raw["time"]),
                "variant": "",
                "lane": int(raw["lane"]),
                "controls": cleaned,
                "metas": _merge_metas(metas, [{"key": "piano7_type", "value": t}]),
            })
        else:
            out_notes.append({
                "type": "short",
                "time": float(raw["time"]),
                "variant": "",
                "lane": int(raw["lane"]),
                "controls": [],
                "metas": _merge_metas(metas, [{"key": "piano7_type", "value": str(t)}]),
            })

    out_notes.sort(key=lambda n: (n["time"], n["lane"], n["type"]))

    return {
        "version": "1.3",
        "format": format_code,
        "lastModified": _iso_now(),
        "notes": out_notes,
        "songMeta": song,
        "snapBeatMeta": dict(chart.get("snapBeatMeta") or {}),
    }


def print_schema_report() -> None:
    from collections import Counter
    import random

    def sample_stats(root: Path, n: int = 80) -> dict[str, Any]:
        files = list(root.glob("*.json"))
        random.seed(42)
        pick = random.sample(files, min(n, len(files)))
        top = Counter()
        ver = Counter()
        note_keys = Counter()
        types = Counter()
        for fp in pick:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
            for k in d:
                top[k] += 1
            ver[str(d.get("version", "?"))] += 1
            for note in (d.get("notes") or [])[:400]:
                types[note.get("type", "?")] += 1
                for nk in note:
                    note_keys[nk] += 1
        return {
            "files": len(files),
            "sampled": len(pick),
            "top_level_keys": dict(top),
            "versions": dict(ver),
            "note_keys": dict(note_keys.most_common(12)),
            "note_types": dict(types),
        }

    piano = _ROOT / "datasets/dataset/piano7/json"
    cur = _ROOT / "datasets/dataset/json"
    print("Piano7:", sample_stats(piano))
    print("Current dataset/json:", sample_stats(cur))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="Print schema stats and exit")
    ap.add_argument("--input", type=Path, help="Single Piano7 JSON file")
    ap.add_argument("--output", type=Path, help="Output path for single-file conversion")
    ap.add_argument("--input-dir", type=Path, help="Directory of Piano7 JSON files")
    ap.add_argument("--output-dir", type=Path, help="Output directory (mirrors basenames)")
    ap.add_argument("--format", default="MT3", dest="format_code", help="Top-level format string")
    ap.add_argument("--layer", default="main", help="Keep only this layer (use 'all' for no filter)")
    ap.add_argument("--keep-mood", action="store_true", help="Keep mood notes as short taps with metas")
    args = ap.parse_args()

    if args.report:
        print_schema_report()
        return

    layer: str | None
    if args.layer == "all":
        layer = None
    else:
        layer = args.layer

    if args.input and args.output:
        with open(args.input, encoding="utf-8") as f:
            chart = json.load(f)
        out = piano7_dict_to_snapbeat_v13(
            chart,
            format_code=args.format_code,
            layer=layer,
            drop_mood=not args.keep_mood,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Wrote", args.output)
        return

    if args.input_dir and args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(args.input_dir.glob("*.json"))
        for fp in paths:
            with open(fp, encoding="utf-8") as f:
                chart = json.load(f)
            out = piano7_dict_to_snapbeat_v13(
                chart,
                format_code=args.format_code,
                layer=layer,
                drop_mood=not args.keep_mood,
            )
            outp = args.output_dir / fp.name
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        print("Converted", len(paths), "files ->", args.output_dir)
        return

    ap.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
