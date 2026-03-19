"""IterableDataset for SnapBeat JSON charts paired with audio files."""
from __future__ import annotations

import random
from multiprocessing.managers import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import IterableDataset

from .data_utils import load_audio_file, get_song_length
from .snapbeat_parser import SnapBeatParser
from ..tokenizer import Tokenizer, ContextType, Event, EventType
from ..config import DataConfig

MILISECONDS_PER_SECOND = 1000
LABEL_IGNORE_ID = -100


class SnapBeatDataset(IterableDataset):
    """Dataset that loads SnapBeat JSON + audio pairs from an amaremix-style folder.

    Expected layout::

        {path}/
            json/   *.json   (SnapBeat v1.0 charts)
            audio/  *.mp3|*.wav|*.flac  (matched by UUID stem)
    """

    def __init__(
            self,
            args: DataConfig,
            parser: SnapBeatParser,
            tokenizer: Tokenizer,
            test: bool = False,
            shared: Namespace = None,
            **_kwargs,
    ):
        super().__init__()
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared

        self.path = Path(args.test_dataset_path if test else args.train_dataset_path)
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end

        self.pairs = self._discover_pairs()

        self.frame_seq_len = args.src_seq_len - 1
        self.min_pre_token_len = 4
        self.pre_token_len = args.tgt_seq_len // 2
        self.add_pre_tokens = args.add_pre_tokens
        self.add_empty_sequences = args.add_empty_sequences

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        """Find (json_path, audio_path) pairs in the dataset folder."""
        json_dir = self.path / "json"
        audio_dir = self.path / "audio"

        if not json_dir.is_dir() or not audio_dir.is_dir():
            raise FileNotFoundError(f"Expected json/ and audio/ subdirectories in {self.path}")

        audio_by_stem: dict[str, Path] = {}
        for f in audio_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg"):
                audio_by_stem[f.stem] = f

        pairs: list[tuple[Path, Path]] = []
        for f in sorted(json_dir.iterdir()):
            if f.suffix.lower() != ".json":
                continue
            audio = audio_by_stem.get(f.stem)
            if audio is not None:
                pairs.append((f, audio))

        pairs = pairs[self.start:self.end]
        return pairs

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        pairs = list(self.pairs)
        if not self.test:
            random.shuffle(pairs)

        for json_path, audio_path in pairs:
            try:
                yield from self._process_chart(json_path, audio_path)
            except Exception as e:
                print(f"SnapBeatDataset: skipping {json_path.stem}: {e}")
                continue

    def _get_speed_augment(self) -> float:
        if self.test or random.random() >= self.args.dt_augment_prob:
            return 1.0
        mi, ma = self.args.dt_augment_range
        base = random.random()
        if self.args.dt_augment_sqrt:
            base = np.power(base, 0.5)
        return mi + (ma - mi) * base

    def _process_chart(self, json_path: Path, audio_path: Path):
        speed = self._get_speed_augment()

        audio_samples = load_audio_file(str(audio_path), self.args.sample_rate, speed, self.args.normalize_audio)
        chart = SnapBeatParser.load_chart(json_path)
        song_meta = chart.get("songMeta", {})
        n_lanes = song_meta.get("nLanes", 4)

        events, event_times = self.parser.parse(chart, speed)
        frames, frame_times = self._get_frames(audio_samples)

        hold_count = sum(1 for e in events if e.type == EventType.HOLD_NOTE)
        total_notes = sum(1 for e in events if e.type in (EventType.CIRCLE, EventType.HOLD_NOTE))
        hold_note_ratio = hold_count / total_notes if total_notes > 0 else 0.0
        song_length = get_song_length(audio_samples, self.args.sample_rate)

        out_context = [{
            "events": events,
            "event_times": event_times,
            "extra": {
                "context_type": ContextType.MAP,
                "add_type": self.args.add_out_context_types,
                "id": "out_map",
                "gamemode": 3,
                "beatmap_id": 0,
                "beatmap_idx": 0,
                "difficulty": 5.0,
                "year": 2025,
                "hitsounded": False,
                "song_length": song_length,
                "keycount": n_lanes,
                "hold_note_ratio": hold_note_ratio,
                "scroll_speed_ratio": 0.0,
            },
        }]
        in_context = [{
            "events": [],
            "event_times": [],
            "extra": {
                "context_type": ContextType.NONE,
                "add_type": True,
                "id": "in_none",
            },
        }]

        extra_data = {
            "beatmap_idx": torch.tensor(self.tokenizer.num_classes, dtype=torch.long),
            "mapper_idx": torch.tensor(self.tokenizer.num_mapper_classes, dtype=torch.long),
            "difficulty": torch.tensor(5.0, dtype=torch.float32),
            "special": {
                "gamemode": 3,
                "beatmap_id": 0,
                "beatmap_idx": 0,
                "difficulty": 5.0,
                "year": 2025,
                "hitsounded": False,
                "song_length": song_length,
                "keycount": n_lanes,
                "hold_note_ratio": hold_note_ratio,
                "scroll_speed_ratio": 0.0,
            },
        }

        sequences = self._create_sequences(frames, frame_times, out_context, in_context, extra_data)

        for sequence in sequences:
            self._maybe_change_dataset()
            sequence = self._normalize_time_shifts(sequence, str(json_path))
            sequence = self._tokenize_sequence(sequence)
            sequence = self._pad_frame_sequence(sequence)
            sequence = self._pad_and_split_token_sequence(sequence)
            if not self.add_empty_sequences and ((sequence["labels"] == self.tokenizer.eos_id) | (
                    sequence["labels"] == LABEL_IGNORE_ID)).all():
                continue
            yield sequence

    # ------------------------------------------------------------------
    # Helpers (mirrored from BeatmapDatasetIterable)
    # ------------------------------------------------------------------

    def _get_frames(self, samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        samples = np.pad(samples, [0, self.args.hop_length - len(samples) % self.args.hop_length])
        frames = np.reshape(samples, (-1, self.args.hop_length))
        frames_per_ms = self.args.sample_rate / self.args.hop_length / MILISECONDS_PER_SECOND
        frame_times = np.arange(len(frames)) / frames_per_ms
        return frames, frame_times

    def _create_sequences(
            self,
            frames: npt.NDArray,
            frame_times: npt.NDArray,
            out_context: list[dict],
            in_context: list[dict],
            extra_data: Optional[dict] = None,
    ) -> list[dict]:
        def get_event_indices(events2, event_times2):
            start_indices = []
            event_index = 0
            for current_time in frame_times:
                while event_index < len(events2) and event_times2[event_index] < current_time:
                    event_index += 1
                start_indices.append(event_index)
            end_indices = start_indices[1:] + [start_indices[-1]]
            return start_indices, end_indices

        start_indices, end_indices = {}, {}
        for context in in_context + out_context:
            sid, eid = get_event_indices(context["events"], context["event_times"])
            start_indices[context["extra"]["id"]] = sid
            end_indices[context["extra"]["id"]] = eid

        sequences = []
        n_frames = len(frames)
        offset = random.randint(0, min(self.frame_seq_len, 2000)) if not self.test and random.random() < self.args.frame_offset_augment_prob else 0
        gen_start_frame_x = int(round(self.args.lookback * self.frame_seq_len)) if not self.test and random.random() < self.args.lookback_prob else 0
        gen_end_frame_x = int(round((1 - self.args.lookahead) * self.frame_seq_len))

        for frame_start_idx in range(offset, n_frames - gen_start_frame_x, self.frame_seq_len):
            frame_end_idx = min(frame_start_idx + self.frame_seq_len, n_frames)
            gen_start_frame = min(frame_start_idx + gen_start_frame_x, n_frames - 1)
            gen_end_frame = min(frame_start_idx + gen_end_frame_x, n_frames)
            frame_pre_idx = max(frame_start_idx - self.frame_seq_len, 0)

            def slice_events(context, fsi, fei):
                if len(context["events"]) == 0:
                    return []
                ident = context["extra"]["id"]
                return context["events"][start_indices[ident][fsi]:end_indices[ident][fei - 1]]

            def slice_context(context, fsi, fei):
                result = {"events": slice_events(context, fsi, fei)} | context["extra"]
                result["time"] = frame_times[fsi]
                ident = context["extra"]["id"]
                result["labels_offset"] = start_indices[ident][gen_start_frame] - start_indices[ident][fsi]
                return result

            sequence = {
                "frames": frames[frame_start_idx:frame_end_idx],
                "out_context": [slice_context(c, frame_start_idx, gen_end_frame) for c in out_context],
                "in_context": [slice_context(c, frame_start_idx, frame_end_idx) for c in in_context],
                "song_position": torch.tensor([frame_start_idx / n_frames, frame_end_idx / n_frames], dtype=torch.float32),
            } | extra_data

            sequence["special"] = sequence["special"].copy()
            sequence["special"]["time"] = frame_times[frame_start_idx]

            if self.args.add_pre_tokens or self.args.add_pre_tokens_at_step >= 0:
                sequence["pre_events"] = slice_events(out_context[0], frame_pre_idx, frame_start_idx)

            sequences.append(sequence)

        return sequences

    def _normalize_time_shifts(self, sequence: dict, chart_path: str) -> dict:
        min_t = self.tokenizer.event_range[EventType.TIME_SHIFT].min_value
        max_t = self.tokenizer.event_range[EventType.TIME_SHIFT].max_value
        STEPS_PER_MS = 0.1

        def process(events, start_time):
            for i, event in enumerate(events):
                if event.type == EventType.TIME_SHIFT:
                    t = int((event.value - start_time) * STEPS_PER_MS)
                    if t < min_t or t > max_t:
                        t = np.clip(t, min_t, max_t)
                    events[i] = Event(EventType.TIME_SHIFT, t)
            return events

        if "pre_events" in sequence:
            sequence["pre_events"] = process(sequence["pre_events"], sequence["out_context"][0]["time"])

        for context in sequence["in_context"] + sequence["out_context"]:
            context["events"] = process(context["events"], context["time"])

        return sequence

    def _get_special_tokens(self, context: dict) -> list:
        special_tokens = []
        if "beatmap_id" not in context:
            return special_tokens

        if self.args.add_gamemode_token:
            special_tokens.append(self.tokenizer.encode_gamemode(context["gamemode"]))
        if self.args.add_cs_token and "keycount" in context:
            special_tokens.append(self.tokenizer.cs_unk)
        if self.args.add_keycount_token and "keycount" in context:
            special_tokens.append(self.tokenizer.encode(Event(EventType.MANIA_KEYCOUNT, context["keycount"])))
        if self.args.add_hold_note_ratio_token and "hold_note_ratio" in context:
            special_tokens.append(self.tokenizer.encode_hold_note_ratio(context["hold_note_ratio"]))
        if self.args.add_hitsounded_token:
            special_tokens.append(self.tokenizer.encode(Event(EventType.HITSOUNDED, int(context.get("hitsounded", False)))))

        return special_tokens

    def _tokenize_sequence(self, sequence: dict) -> dict:
        sequence["special_tokens"] = self._get_special_tokens(sequence["special"])

        for context in sequence["in_context"] + sequence["out_context"]:
            tokens = torch.empty(len(context["events"]), dtype=torch.long)
            for i, event in enumerate(context["events"]):
                tokens[i] = self.tokenizer.encode(event)
            context["tokens"] = tokens
            context["special_tokens"] = self._get_special_tokens(context)

        if "pre_events" in sequence:
            pre_tokens = torch.empty(len(sequence["pre_events"]), dtype=torch.long)
            for i, event in enumerate(sequence["pre_events"]):
                pre_tokens[i] = self.tokenizer.encode(event)
            sequence["pre_tokens"] = pre_tokens
            del sequence["pre_events"]

        return sequence

    def _pad_frame_sequence(self, sequence: dict) -> dict:
        frames = torch.from_numpy(sequence["frames"]).to(torch.float32)
        if frames.shape[0] != self.frame_seq_len:
            n = min(self.frame_seq_len, len(frames))
            padded = torch.zeros(self.frame_seq_len, frames.shape[-1], dtype=frames.dtype, device=frames.device)
            padded[:n] = frames[:n]
            sequence["frames"] = torch.flatten(padded)
        else:
            sequence["frames"] = torch.flatten(frames)
        return sequence

    def _pad_and_split_token_sequence(self, sequence: dict) -> dict:
        stl = 1  # SOS
        stl += len(sequence["special_tokens"])
        for context in sequence["in_context"] + sequence["out_context"]:
            if context["add_type"]:
                stl += 2
            stl += len(context["special_tokens"])

        num_tokens = sum(len(c["tokens"]) for c in sequence["out_context"])
        num_pre_tokens = len(sequence.get("pre_tokens", []))
        if self.args.max_pre_token_len > 0:
            num_pre_tokens = min(num_pre_tokens, self.args.max_pre_token_len)
        num_other_tokens = sum(len(c["tokens"]) for c in sequence["in_context"])

        if self.args.center_pad_decoder:
            n = min(self.args.tgt_seq_len - self.pre_token_len - 1, num_tokens)
            m = min(self.pre_token_len - stl + 1, num_pre_tokens)
            o = min(self.pre_token_len - m - stl + 1, num_other_tokens)
            si = self.pre_token_len - m - stl + 1 - o
        else:
            n = min(self.args.tgt_seq_len - stl - min(self.min_pre_token_len, num_pre_tokens), num_tokens)
            m = min(self.args.tgt_seq_len - stl - n, num_pre_tokens)
            o = min(self.args.tgt_seq_len - stl - n - m, num_other_tokens)
            si = 0

        input_tokens = torch.full((self.args.tgt_seq_len,), self.tokenizer.pad_id, dtype=torch.long)
        label_tokens = torch.full((self.args.tgt_seq_len,), LABEL_IGNORE_ID, dtype=torch.long)

        def add_special_tokens(special_tokens, si):
            for token in special_tokens:
                input_tokens[si] = token
                si += 1
            return si

        def add_context(context, si, max_tokens, add_labels=False):
            if context["add_type"]:
                input_tokens[si] = self.tokenizer.context_sos[context["context_type"]]
                if add_labels:
                    label_tokens[si - 1] = self.tokenizer.context_sos[context["context_type"]]
                si += 1

            start_label_index = si + context["labels_offset"]
            si = add_special_tokens(context["special_tokens"], si)

            num_to_add = min(len(context["tokens"]), max_tokens)
            input_tokens[si:si + num_to_add] = context["tokens"][:num_to_add]
            si += num_to_add
            max_tokens -= num_to_add

            if context["add_type"]:
                input_tokens[si] = self.tokenizer.context_eos[context["context_type"]]
                si += 1

            if add_labels:
                label_tokens[start_label_index - 1:si - 1] = input_tokens[start_label_index:si]

            return si, max_tokens

        for context in sequence["in_context"]:
            si, o = add_context(context, si, o)

        si = add_special_tokens(sequence["special_tokens"], si)
        start_random_index = si

        if m > 0:
            input_tokens[si:si + m] = sequence["pre_tokens"][-m:]
            si += m

        input_tokens[si] = self.tokenizer.sos_id
        si += 1
        for context in sequence["out_context"]:
            si, n = add_context(context, si, n, True)
        end_index = si

        label_tokens[end_index - 1] = self.tokenizer.eos_id

        # Timing augmentation
        if self.args.timing_random_offset > 0 or self.args.timing_random_offset_2 > 0:
            def randomize_tokens(tokens):
                offset_tokens = tokens.clone()
                if random.random() < self.args.timing_random_offset_prob:
                    offset_tokens += torch.randint(
                        low=-self.args.timing_random_offset, high=self.args.timing_random_offset + 1, size=tokens.shape)
                if random.random() < self.args.timing_random_offset_prob:
                    offset_tokens += torch.randint(
                        low=-self.args.timing_random_offset_2, high=self.args.timing_random_offset_2 + 1, size=(1,))
                return torch.where(
                    (self.tokenizer.event_start[EventType.TIME_SHIFT] <= tokens) &
                    (tokens < self.tokenizer.event_end[EventType.TIME_SHIFT]),
                    torch.clamp(offset_tokens,
                                self.tokenizer.event_start[EventType.TIME_SHIFT],
                                self.tokenizer.event_end[EventType.TIME_SHIFT] - 1),
                    tokens)

            input_tokens[start_random_index:end_index] = randomize_tokens(input_tokens[start_random_index:end_index])

        if self.args.snapping_random_prob > 0:
            random_snappings = torch.randint_like(
                input_tokens, low=self.tokenizer.event_start[EventType.SNAPPING],
                high=self.tokenizer.event_end[EventType.SNAPPING])
            mask = ((self.tokenizer.event_start[EventType.SNAPPING] <= input_tokens) &
                    (input_tokens < self.tokenizer.event_end[EventType.SNAPPING]))
            mask &= torch.rand_like(input_tokens, dtype=torch.float32) < self.args.snapping_random_prob
            input_tokens = torch.where(mask, random_snappings, input_tokens)

        sequence["decoder_input_ids"] = input_tokens
        sequence["decoder_attention_mask"] = input_tokens != self.tokenizer.pad_id
        sequence["labels"] = label_tokens

        del sequence["out_context"]
        del sequence["in_context"]
        del sequence["special_tokens"]
        del sequence["special"]
        if "pre_tokens" in sequence:
            del sequence["pre_tokens"]

        return sequence

    def _maybe_change_dataset(self):
        if self.shared is None:
            return
        step = self.shared.current_train_step
        if 0 <= self.args.add_empty_sequences_at_step <= step and not self.add_empty_sequences:
            self.add_empty_sequences = True
        if 0 <= self.args.add_pre_tokens_at_step <= step and not self.add_pre_tokens:
            self.add_pre_tokens = True
