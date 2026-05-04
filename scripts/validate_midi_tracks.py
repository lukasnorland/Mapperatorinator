"""Validate that every MIDI file in a directory contains tracks named "main" and "relation".

Usage:
    python scripts/validate_midi_tracks.py [MIDI_DIR]

Defaults MIDI_DIR to ``datasets/dataset/midi`` (relative to repo root).
Exits with code 0 if every file passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import mido

REQUIRED_TRACKS = {"main", "relation"}


def get_track_names(midi_path: Path) -> list[str]:
    """Return the list of track names (lower-cased, stripped) for a MIDI file."""
    mid = mido.MidiFile(midi_path)
    names: list[str] = []
    for track in mid.tracks:
        track_name = ""
        for msg in track:
            if msg.type == "track_name":
                track_name = msg.name
                break
        names.append(track_name.strip().lower())
    return names


def validate_file(midi_path: Path) -> tuple[bool, str]:
    """Return (is_valid, reason). reason is empty string when valid."""
    try:
        names = get_track_names(midi_path)
    except Exception as exc:  # corrupt / unreadable file
        return False, f"unreadable: {exc!r}"

    present = set(names)
    missing = REQUIRED_TRACKS - present
    if missing:
        return False, f"missing tracks: {sorted(missing)}; tracks present: {names}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "midi_dir",
        nargs="?",
        default="datasets/dataset/midi",
        type=Path,
        help="Directory containing .mid files (default: datasets/dataset/midi)",
    )
    parser.add_argument(
        "--show-extra",
        action="store_true",
        help="Also report files that have the required tracks but include extras.",
    )
    args = parser.parse_args()

    midi_dir: Path = args.midi_dir
    if not midi_dir.is_dir():
        print(f"ERROR: {midi_dir} is not a directory", file=sys.stderr)
        return 2

    midi_files = sorted(p for p in midi_dir.iterdir() if p.suffix.lower() in {".mid", ".midi"})
    if not midi_files:
        print(f"No .mid/.midi files found in {midi_dir}")
        return 0

    total = len(midi_files)
    failures: list[tuple[Path, str]] = []
    extras: list[tuple[Path, list[str]]] = []
    track_count_dist: Counter[int] = Counter()

    for i, path in enumerate(midi_files, 1):
        ok, reason = validate_file(path)
        if not ok:
            failures.append((path, reason))
        else:
            try:
                names = get_track_names(path)
                track_count_dist[len(names)] += 1
                if args.show_extra and (set(names) - REQUIRED_TRACKS - {""}):
                    extras.append((path, names))
            except Exception:
                pass

        if i % 50 == 0 or i == total:
            print(f"  scanned {i}/{total} ...", file=sys.stderr)

    print()
    print(f"Validated {total} MIDI files in {midi_dir}")
    print(f"  passed: {total - len(failures)}")
    print(f"  failed: {len(failures)}")
    if track_count_dist:
        dist = ", ".join(f"{n} tracks: {c}" for n, c in sorted(track_count_dist.items()))
        print(f"  track-count distribution (passing files): {dist}")

    if failures:
        print()
        print("Failures:")
        for path, reason in failures:
            print(f"  - {path.name}: {reason}")

    if args.show_extra and extras:
        print()
        print(f"Files with extra (non-required) named tracks: {len(extras)}")
        for path, names in extras[:20]:
            print(f"  - {path.name}: {names}")
        if len(extras) > 20:
            print(f"  ... and {len(extras) - 20} more")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
