"""Convert osu! mania .osu beatmaps to SnapBeat JSON format (v1.3)."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from slider import Beatmap, Circle, HoldNote


def osu_mania_to_snapbeat(
    osu_path: str | Path,
    song_name: str = "Generated",
    audio_path: str = "",
    n_lanes: int = 4,
    visual_speed: float = 4.5,
    max_simultaneous: int = 2,
) -> dict:
    """Convert an osu! mania .osu beatmap file to a SnapBeat-compliant dict.

    Args:
        osu_path: Path to the generated .osu file.
        song_name: Value for songMeta.songName.
        audio_path: Value for songMeta.audioPath.
        n_lanes: Number of playable lanes (written to songMeta.nLanes).
        visual_speed: Scroll speed written to songMeta.visualSpeed.
        max_simultaneous: Maximum simultaneous notes allowed per timestamp.

    Returns:
        A dict matching SnapBeat JSON v1.3 schema.
    """
    beatmap = Beatmap.from_path(osu_path)
    key_count = int(beatmap.circle_size)

    bpm = _extract_bpm(beatmap)
    hit_objects = beatmap.hit_objects(stacking=False)

    if not hit_objects:
        return _empty_snapbeat(song_name, audio_path, bpm, n_lanes, visual_speed)

    last_ho = hit_objects[-1]
    end_time = last_ho.end_time if isinstance(last_ho, HoldNote) else last_ho.time
    audio_duration = end_time.total_seconds()

    notes = _build_notes(hit_objects, key_count, n_lanes)
    notes = _enforce_simultaneous_cap(notes, max_simultaneous)
    notes = _enforce_lane_spacing(notes, n_lanes)
    notes.sort(key=lambda n: (n["time"], n["lane"]))

    return {
        "version": "1.3",
        "format": "MT3",
        "lastModified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "songMeta": {
            "songName": song_name,
            "audioPath": audio_path,
            "audioVolume": 1,
            "bpm": bpm,
            "audioOffset": 0,
            "audioDuration": audio_duration,
            "nLanes": n_lanes,
            "nLanesMeta": n_lanes,
            "visualSpeed": visual_speed,
        },
        "snapBeatMeta": {"showInfoPanel": True},
    }


def write_snapbeat_json(snapbeat: dict, output_path: str | Path) -> str:
    """Serialise a SnapBeat dict to a JSON file and return the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapbeat, f, indent=2, ensure_ascii=False)
    return str(output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_bpm(beatmap: Beatmap) -> float:
    for tp in beatmap.timing_points:
        if tp.parent is None and tp.ms_per_beat > 0:
            return 60000.0 / tp.ms_per_beat
    return 120.0


def _ho_to_lane(ho, key_count: int, n_lanes: int) -> int:
    """Map osu! X coordinate to a 1-indexed lane, clamped to [1, n_lanes]."""
    lane = int(ho.position[0] * key_count / 512) + 1
    return max(1, min(lane, n_lanes))


def _build_notes(hit_objects, key_count: int, n_lanes: int) -> list[dict]:
    notes: list[dict] = []
    for ho in hit_objects:
        lane = _ho_to_lane(ho, key_count, n_lanes)
        time_sec = ho.time.total_seconds()

        if isinstance(ho, HoldNote):
            notes.append({
                "type": "long",
                "time": time_sec,
                "variant": "",
                "lane": lane,
                "controls": [{"time": ho.end_time.total_seconds(), "lane": lane}],
                "metas": [],
            })
        elif isinstance(ho, Circle):
            notes.append({
                "type": "short",
                "time": time_sec,
                "variant": "",
                "lane": lane,
                "controls": [],
                "metas": [],
            })
    return notes


def _enforce_simultaneous_cap(notes: list[dict], cap: int) -> list[dict]:
    """Keep at most *cap* notes per unique timestamp (prefer lower lanes)."""
    if cap <= 0:
        return notes

    groups: dict[float, list[dict]] = defaultdict(list)
    for n in notes:
        groups[n["time"]].append(n)

    result: list[dict] = []
    for t in sorted(groups):
        group = sorted(groups[t], key=lambda n: n["lane"])
        result.extend(group[:cap])
    return result


def _enforce_lane_spacing(notes: list[dict], n_lanes: int) -> list[dict]:
    """Ensure simultaneous notes are not on adjacent lanes (gap >= 2).

    When a pair violates the rule the converter tries to reassign one note to
    the nearest non-adjacent, unoccupied lane.  If that is impossible the note
    is dropped so only a single tap remains.
    """
    groups: dict[float, list[dict]] = defaultdict(list)
    for n in notes:
        groups[n["time"]].append(n)

    result: list[dict] = []
    for t in sorted(groups):
        group = sorted(groups[t], key=lambda n: n["lane"])
        if len(group) <= 1:
            result.extend(group)
            continue

        group = _fix_adjacent_pair(group, n_lanes)
        result.extend(group)
    return result


def _fix_adjacent_pair(group: list[dict], n_lanes: int) -> list[dict]:
    """Fix a group of (at most 2) simultaneous notes so no two are adjacent."""
    if len(group) < 2:
        return group

    a, b = group[0], group[1]
    if abs(a["lane"] - b["lane"]) >= 2:
        return group

    occupied = {a["lane"], b["lane"]}
    moved = _try_reassign(b, a["lane"], occupied, n_lanes)
    if moved is not None:
        return [a, moved]

    moved = _try_reassign(a, b["lane"], occupied, n_lanes)
    if moved is not None:
        return [moved, b]

    return [a]


def _try_reassign(
    note: dict,
    anchor_lane: int,
    occupied: set[int],
    n_lanes: int,
) -> Optional[dict]:
    """Try to move *note* to the nearest lane that is >= 2 away from *anchor_lane*."""
    candidates = [
        lane for lane in range(1, n_lanes + 1)
        if lane not in occupied and abs(lane - anchor_lane) >= 2
    ]
    if not candidates:
        return None

    best = min(candidates, key=lambda l: abs(l - note["lane"]))
    reassigned = {**note, "lane": best}
    if reassigned.get("controls"):
        reassigned["controls"] = [
            {**ctrl, "lane": best} for ctrl in reassigned["controls"]
        ]
    return reassigned


def _empty_snapbeat(
    song_name: str, audio_path: str, bpm: float, n_lanes: int, visual_speed: float,
) -> dict:
    return {
        "version": "1.3",
        "format": "MT3",
        "lastModified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [],
        "songMeta": {
            "songName": song_name,
            "audioPath": audio_path,
            "audioVolume": 1,
            "bpm": bpm,
            "audioOffset": 0,
            "audioDuration": 0,
            "nLanes": n_lanes,
            "nLanesMeta": n_lanes,
            "visualSpeed": visual_speed,
        },
        "snapBeatMeta": {"showInfoPanel": True},
    }
