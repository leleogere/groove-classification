import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Self

import torch

from utils.utils import load_and_validate_midi_file


@dataclass(slots=True)
class Drum:
    pitch: int
    id: int
    name: str


class DrumList:
    def __init__(self, drums: list[Drum]):
        self.drums = drums
        self._pitch_to_percussion = {perc.pitch: perc for perc in self.drums}
        self._id_to_percussion = {perc.id: perc for perc in self.drums}
        self.pitch_to_id = {perc.pitch: perc.id for perc in self.drums}
        self.id_to_pitch = {perc.id: perc.pitch for perc in self.drums}

    def drum_from_pitch(self, pitch: int) -> Drum:
        return self._pitch_to_percussion[pitch]

    def drum_from_id(self, drum_id: int) -> Drum:
        return self._id_to_percussion[drum_id]

    def __len__(self):
        return len(self.drums)


class AsDictMixin:
    __slots__: list[str]

    def __init__(self, _):
        pass

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


@dataclass(slots=True)
class Note(AsDictMixin):
    start: float
    end: float
    pitch: int
    velocity: int

    @property
    def duration(self):
        return self.end - self.start


@dataclass(slots=True)
class RelativeNote(AsDictMixin):
    pitch: int
    velocity: float
    duration: float
    timeshift: float

    @staticmethod
    def from_note(note: Note, timeshift: float = 0.0):
        return RelativeNote(
            pitch=note.pitch,
            velocity=note.velocity,
            duration=note.duration,
            timeshift=timeshift,
        )


class Track:
    def __init__(
        self,
        midi_path: Path,
        label: str = None,
        bpm: float = None,
        split: str = None,
        label_idx: int = None,
    ):
        self.midi_path = midi_path
        self.label = label
        self.bpm = bpm
        self.split = split
        self.label_idx = label_idx

    def notes(self) -> list[Note]:
        midi = load_and_validate_midi_file(
            self.midi_path,
            only_drums=True,
            unique_track=True,
            unique_tempo=True,
            unique_time_signature=True,
        )
        notes = [
            Note(n.start, n.end, n.pitch, n.velocity)
            for n in sorted(midi.instruments[0].notes, key=lambda x: x.start)
        ]
        return notes

    def relative_notes(self) -> list[RelativeNote]:
        relative_notes = [
            RelativeNote.from_note(n, timeshift=n_next.start - n.start)
            for n, n_next in zip(self.notes(), self.notes()[1:] + [self.notes()[-1]])
        ]
        return relative_notes

    def to_json(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["midi_path"] = str(d["midi_path"])
        return d

    def to_json_file(self, path: Path, create_parents: bool = False, **kwargs) -> None:
        if not path.name.endswith(".json"):
            raise ValueError(f"Path {path} must end with .json")
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, **kwargs)

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        json_dict["midi_path"] = Path(json_dict["midi_path"])
        return cls(**json_dict)

    def copy(self) -> Self:
        return Track.from_json(self.to_json())

    @classmethod
    def from_json_file(cls, path: Path) -> Self:
        with open(path) as f:
            return cls.from_json(json.load(f))

    def __repr__(self) -> str:
        elements = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Track({', '.join(elements)})"

    @classmethod
    def from_partial(cls, track: Self, **kwargs) -> Self:
        d = track.__dict__.copy()
        d.update(kwargs)
        return cls(**d)


class ProcessedTrack(Track):
    def __init__(
        self,
        notes_tensor: torch.Tensor,
        tempo_tensor: torch.Tensor,
        label_idx_tensor: torch.Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.notes_tensor = notes_tensor
        self.tempo_tensor = tempo_tensor
        self.label_idx_tensor = label_idx_tensor
