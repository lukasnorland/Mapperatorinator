"""Parse SnapBeat JSON charts into (events, event_times) compatible with the osuT5 tokenizer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..event import Event, EventType


class SnapBeatParser:
    """Convert a SnapBeat JSON chart to (events, event_times) using mania-compatible EventTypes.

    Token mapping:
        short  -> CIRCLE + TIME_SHIFT + SNAPPING + MANIA_COLUMN
        long   -> HOLD_NOTE ... HOLD_NOTE_END  (same as mania hold)
        zigzag -> treated as HOLD_NOTE ... HOLD_NOTE_END  (start to last control)
    """

    def __init__(self, *, types_first: bool = True, add_snapping: bool = True,
                 add_timing_points: bool = True, sustain_interval: Optional[int] = None):
        self.types_first = types_first
        self.add_snapping = add_snapping
        self.add_timing_points = add_timing_points
        self.sustain_interval = sustain_interval

    def parse(self, chart: dict, speed: float = 1.0) -> tuple[list[Event], list[int]]:
        """Parse a SnapBeat chart dict into events.

        Args:
            chart: Parsed SnapBeat JSON dict with ``notes`` and ``songMeta``.
            speed: Speed multiplier (for DT augmentation).

        Returns:
            Tuple of (events, event_times) sorted by time, with times in ms.
        """
        song_meta = chart.get("songMeta", {})
        bpm = song_meta.get("bpm", 120.0)
        n_lanes = song_meta.get("nLanes", 4)
        ms_per_beat = 60_000.0 / bpm if bpm > 0 else 500.0

        events: list[Event] = []
        event_times: list[int] = []

        for note in chart.get("notes", []):
            lane = note.get("lane", 1)
            if lane > n_lanes:
                continue

            note_type = note.get("type", "short")
            time_s = note.get("time", 0.0)
            time_ms = int(time_s * 1000 + 0.5)
            column = min(max(lane - 1, 0), n_lanes - 1)

            if note_type == "short":
                self._emit_tap(events, event_times, time_ms, column, ms_per_beat)
            elif note_type in ("long", "zigzag"):
                controls = note.get("controls", [])
                if controls:
                    end_time_s = controls[-1].get("time", time_s)
                else:
                    end_time_s = time_s
                end_time_ms = int(end_time_s * 1000 + 0.5)
                if end_time_ms <= time_ms:
                    self._emit_tap(events, event_times, time_ms, column, ms_per_beat)
                else:
                    self._emit_hold(events, event_times, time_ms, end_time_ms, column, ms_per_beat)

        if len(events) > 0:
            events, event_times = zip(*sorted(zip(events, event_times), key=lambda x: x[1]))
            events, event_times = list(events), list(event_times)

        if speed != 1.0:
            events, event_times = self._apply_speed(events, event_times, speed)

        return events, event_times

    def parse_timing(self, chart: dict, speed: float = 1.0, song_length_ms: Optional[float] = None) -> tuple[list[Event], list[int]]:
        """Generate TIMING_POINT / MEASURE / BEAT events from the chart's BPM.

        SnapBeat charts have a single BPM and assumed 4/4 meter.  This produces
        the same event format as ``OsuParser.parse_timing`` so the model receives
        an explicit beat grid as decoder input context.

        Args:
            chart: Parsed SnapBeat JSON dict.
            speed: Speed multiplier (for DT augmentation).
            song_length_ms: Song duration in ms (used as upper bound for beats).

        Returns:
            Tuple of (events, event_times) sorted by time.
        """
        song_meta = chart.get("songMeta", {})
        bpm = song_meta.get("bpm", 120.0)
        ms_per_beat = 60_000.0 / bpm if bpm > 0 else 500.0
        meter = 4  # SnapBeat assumes 4/4

        # Determine how far to generate beats
        notes = chart.get("notes", [])
        if notes:
            last_time = max(n.get("time", 0.0) for n in notes) * 1000 + 1
        elif song_length_ms is not None:
            last_time = song_length_ms
        else:
            last_time = 10

        events: list[Event] = []
        event_times: list[int] = []
        measure_counter = 0
        time = 0.0

        while time <= last_time:
            time_ms = int(time + 0.5)

            if self.add_timing_points and measure_counter == 0:
                event_type = EventType.TIMING_POINT
            elif measure_counter % meter == 0:
                event_type = EventType.MEASURE
            else:
                event_type = EventType.BEAT

            if self.types_first:
                events.append(Event(event_type))
                event_times.append(time_ms)

            events.append(Event(EventType.TIME_SHIFT, time_ms))
            event_times.append(time_ms)

            if not self.types_first:
                events.append(Event(event_type))
                event_times.append(time_ms)

            measure_counter += 1
            time = measure_counter * ms_per_beat

        if speed != 1.0:
            events, event_times = self._apply_speed(events, event_times, speed)

        return events, event_times

    # ------------------------------------------------------------------
    # Event emission helpers
    # ------------------------------------------------------------------

    def _emit_tap(self, events: list[Event], event_times: list[int],
                  time_ms: int, column: int, ms_per_beat: float) -> None:
        if self.types_first:
            events.append(Event(EventType.CIRCLE))
            event_times.append(time_ms)

        self._add_time_and_snap(events, event_times, time_ms, ms_per_beat)
        events.append(Event(EventType.MANIA_COLUMN, column))
        event_times.append(time_ms)

        if not self.types_first:
            events.append(Event(EventType.CIRCLE))
            event_times.append(time_ms)

    def _emit_hold(self, events: list[Event], event_times: list[int],
                   start_ms: int, end_ms: int, column: int, ms_per_beat: float) -> None:
        # Hold start
        if self.types_first:
            events.append(Event(EventType.HOLD_NOTE))
            event_times.append(start_ms)

        self._add_time_and_snap(events, event_times, start_ms, ms_per_beat)
        events.append(Event(EventType.MANIA_COLUMN, column))
        event_times.append(start_ms)

        if not self.types_first:
            events.append(Event(EventType.HOLD_NOTE))
            event_times.append(start_ms)

        # Sustain events
        if self.sustain_interval:
            t = start_ms + self.sustain_interval
            while t < end_ms - 10:
                if self.types_first:
                    events.append(Event(EventType.HOLD_NOTE_SUSTAIN))
                    event_times.append(t)
                self._add_time_and_snap(events, event_times, t, ms_per_beat, add_snap=False)
                if not self.types_first:
                    events.append(Event(EventType.HOLD_NOTE_SUSTAIN))
                    event_times.append(t)
                t += self.sustain_interval

        # Hold end
        if self.types_first:
            events.append(Event(EventType.HOLD_NOTE_END))
            event_times.append(end_ms)

        self._add_time_and_snap(events, event_times, end_ms, ms_per_beat)

        if not self.types_first:
            events.append(Event(EventType.HOLD_NOTE_END))
            event_times.append(end_ms)

    def _add_time_and_snap(self, events: list[Event], event_times: list[int],
                           time_ms: int, ms_per_beat: float, add_snap: bool = True) -> None:
        events.append(Event(EventType.TIME_SHIFT, time_ms))
        event_times.append(time_ms)

        if not add_snap or not self.add_snapping:
            return

        beats = time_ms / ms_per_beat
        snapping = 0
        for i in range(1, 17):
            if abs(beats - round(beats * i) / i) * ms_per_beat < 2:
                snapping = i
                break
        events.append(Event(EventType.SNAPPING, snapping))
        event_times.append(time_ms)

    @staticmethod
    def _apply_speed(events: list[Event], event_times: list[int], speed: float) -> tuple[list[Event], list[int]]:
        sped = []
        for event in events:
            if event.type == EventType.TIME_SHIFT:
                sped.append(Event(EventType.TIME_SHIFT, int(event.value / speed)))
            else:
                sped.append(event)
        sped_times = [int(t / speed) for t in event_times]
        return sped, sped_times

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_chart(path: str | Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
