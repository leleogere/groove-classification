import copy
import json
import os.path
import re
from dataclasses import dataclass
from dataclasses import dataclass
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path
from pathlib import Path
from pathlib import Path
from pathlib import Path
from typing import Any
from typing import Any
from typing import Any
from typing import Self
from typing import Self
from typing import Self
from typing import Self
from typing import Self
from typing import Union

import mido
import numpy as np
import pretty_midi as pm
import symusic
import torch

np.int = int


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


def compute_relative_notes(notes: list[Note]) -> list[RelativeNote]:
    relative_notes = [
        RelativeNote.from_note(n, timeshift=n.start - n_prev.start)
        for n_prev, n in zip([notes[0]] + notes, notes)
    ]
    return relative_notes


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

    def min_pitch(self):
        return min([d.pitch for d in self.drums])

    def max_pitch(self):
        return max([d.pitch for d in self.drums])


class Track:
    def __init__(
        self,
        midi_path: Path,
        label: str = None,
        bpm: float = None,
        time_signature: tuple[int, int] = None,
        split: str = None,
        label_idx: int = None,
        other_data: dict[str, Any] = None,
    ):
        self.midi_path = midi_path
        self.label = label
        self.bpm = int(bpm)  # Allow float BPM?
        self.time_signature = time_signature
        self.split = split
        self.label_idx = label_idx
        self.other_data = other_data if other_data is not None else {}

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
        return compute_relative_notes(self.notes())

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
        d = copy.deepcopy(track.__dict__)
        d.update(kwargs)
        return cls(**d)


class ProcessedTrack(Track):
    def __init__(
        self,
        processed_data: torch.Tensor | Any,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.processed_data = processed_data


def process_label(raw_label: str, keep_sublabel_for: list[str] | None = None) -> str:
    """
    Process a raw label read from a metadata file into a valid label. Labels are supposed to be of the form
    <mainlabel>/<sublabel> where <mainlabel> is the main label and <sublabel> is a sub-label. By default, only the
    main label is kept. To keep sub-labels for some classes, pass a list of classes to keep sub-labels for.
    :param raw_label: Raw label.
    :param keep_sublabel_for: List of classes to keep sub-labels for.
    :return: Processed label.
    """
    if keep_sublabel_for is None:
        keep_sublabel_for = []
    main_label = raw_label.split("/")[0]
    if main_label not in keep_sublabel_for:
        return main_label
    else:
        return raw_label


def parse_bpm_from_file_name(file_name: str) -> float:
    """
    Parse the BPM from a file name like that: 1_funk-groove1_138_beat_4-4.mid
    :param file_name: File name.
    :return: BPM.
    """
    return float(re.search(r"\d+_[a-z0-9\-]+_(\d+|\d+\.\d+)_.*\.mid", file_name).group(1))


def parse_time_signature_from_file_name(file_name: str) -> tuple[int, int]:
    """
    Parse the time signature from a file name like that: 1_funk-groove1_138_beat_4-4.mid
    :param file_name: File name.
    :return: Time signature.
    """
    ts = re.search(r"\d+_[a-z0-9\-]+_\d+_[a-z]+_(\d+)-(\d+).*\.mid", file_name)
    return int(ts.group(1)), int(ts.group(2))


def parse_genre_from_file_name(file_name: str, only_main: bool = True) -> str:
    """
    Parse the genre from a file name like that: 1_funk-groove1_138_beat_4-4.mid
    :param file_name: File name.
    :param only_main: If True, only the main genre is returned.
    :return: Genre.
    """
    genre = re.search(r"\d+_([a-z0-9\-]+)_.+\.mid", file_name).group(1)
    if only_main:
        genre = genre.split("-")[0]
    return genre


def load_and_validate_midi_file(
    midi_path: Path,
    only_drums: bool = False,
    unique_track: bool = False,
    unique_time_signature: bool = False,
    unique_tempo: bool = False,
    backend: str = "pretty_midi",
) -> Union[pm.PrettyMIDI, symusic.Score]:
    if backend == "pretty_midi":
        return load_and_validate_midi_file_prettymidi(
            midi_path,
            only_drums=only_drums,
            unique_track=unique_track,
            unique_time_signature=unique_time_signature,
            unique_tempo=unique_tempo,
        )
    elif backend == "symusic":
        return load_and_validate_midi_file_symusic(
            midi_path,
            only_drums=only_drums,
            unique_track=unique_track,
            unique_time_signature=unique_time_signature,
            unique_tempo=unique_tempo,
        )
    elif backend == "mido":
        return load_and_validate_midi_file_mido(
            midi_path,
            only_drums=only_drums,
            unique_track=unique_track,
            unique_time_signature=unique_time_signature,
            unique_tempo=unique_tempo,
        )
    else:
        raise ValueError(f"Unknown backend {backend}")


def load_and_validate_midi_file_prettymidi(
    midi_path: Path,
    only_drums: bool = False,
    unique_track: bool = False,
    unique_time_signature: bool = False,
    unique_tempo: bool = False,
) -> pm.PrettyMIDI:
    if not midi_path.exists():
        raise FileNotFoundError(f"File {midi_path} does not exist.")
    midi = pm.PrettyMIDI(str(midi_path))
    # Check that there is only one unique drum track
    if unique_track and len(midi.instruments) != 1:
        raise ValueError(f"Expected a unique track, got {len(midi.instruments)} in {midi_path}.")
    if only_drums and not midi.instruments[0].is_drum:
        raise ValueError(f"Expected a drum track, got {midi.instruments} in {midi_path}.")
    # Read tempo (only allow one unique tempo in the whole track)
    tempo_timestamps, bpms = midi.get_tempo_changes()
    if unique_tempo and len(tempo_timestamps) != 1:
        raise ValueError(f"Expected a unique tempo, got {len(tempo_timestamps)} in {midi_path}.")
    # Read time signature (only allow one unique time signature in the whole track)
    time_signatures = midi.time_signature_changes
    if (
        unique_time_signature
        and len(time_signatures) != 1
        and any(
            time_signatures[0].__dict__ != ts.__dict__ for ts in time_signatures
        )  # in case of duplicates
    ):
        raise ValueError(
            f"Expected a unique time signature, got {len(time_signatures)} in {midi_path}."
        )
    return midi


def load_and_validate_midi_file_symusic(
    midi_path: Path,
    only_drums: bool = False,
    unique_track: bool = False,
    unique_time_signature: bool = False,
    unique_tempo: bool = False,
) -> symusic.Score:
    if not midi_path.exists():
        raise FileNotFoundError(f"File {midi_path} does not exist.")
    midi = symusic.Score(midi_path)
    track: symusic.Track = midi.tracks[0]
    # Check that there is only one unique drum track
    if unique_track and len(midi.tracks) != 1:
        raise ValueError(f"Expected a unique track, got {len(midi.tracks)} in {midi_path}.")
    if only_drums and not track.is_drum:
        raise ValueError(f"Expected a drum track, got {midi.instruments} in {midi_path}.")
    # Read tempo (only allow one unique tempo in the whole track)
    tempos = midi.tempos
    if unique_tempo and len(tempos) != 1:
        raise ValueError(f"Expected a unique tempo, got {len(tempos)} in {midi_path}.")
    # Read time signature (only allow one unique time signature in the whole track)
    time_signatures = midi.time_signatures
    if (
        unique_time_signature
        and len(time_signatures) != 1
        and any(time_signatures[0] != ts for ts in time_signatures)  # in case of duplicates
    ):
        raise ValueError(
            f"Expected a unique time signature, got {len(time_signatures)} in {midi_path}."
        )
    return midi.to(symusic.TimeUnit.second)


def load_and_validate_midi_file_mido(
    midi_path: Path,
    only_drums: bool = False,
    unique_track: bool = False,
    unique_time_signature: bool = False,
    unique_tempo: bool = False,
) -> mido.MidiFile:
    if not midi_path.exists():
        raise FileNotFoundError(f"File {midi_path} does not exist.")
    midi = mido.MidiFile(str(midi_path))
    track: mido.MidiTrack = midi.tracks[0]
    # Check that there is only one unique drum track
    if unique_track and len(midi.tracks) != 1:
        raise ValueError(f"Expected a unique track, got {len(midi.tracks)} in {midi_path}.")
    if only_drums:
        raise NotImplementedError("Only drums not implemented for mido.")
    # Read tempo (only allow one unique tempo in the whole track)
    tempos = [msg.tempo for msg in midi.tracks[0] if msg.type == "set_tempo"]
    if unique_tempo and any(tempos[0] != tempo for tempo in tempos):
        raise ValueError(f"Expected a unique tempo, got {len(tempos)} in {midi_path}.")
    # Read time signature (only allow one unique time signature in the whole track)
    time_signatures = [msg for msg in midi.tracks[0] if msg.type == "time_signature"]
    if unique_time_signature and any(time_signatures[0] != ts for ts in time_signatures):
        raise ValueError(
            f"Expected a unique time signature, got {len(time_signatures)} in {midi_path}."
        )
    return midi


def get_midi_paths(path: Path, return_sizes: bool = False):
    midi_exts = [".mid", ".midi"]
    midi_exts.extend([ext.upper() for ext in midi_exts])
    paths = sum([list(path.rglob(f"*{ext}")) for ext in midi_exts], [])
    if return_sizes:
        sizes = [p.stat().st_size for p in paths]
        return paths, sizes
    else:
        return paths


def merge_tracks(midi_path: Path, out_path: Path = None) -> None:
    """
    Merge all tracks of a MIDI file into a single track.

    :param midi_path: Path of the MIDI file to merge.
    :param out_path: Path of the new MIDI file. If None, the original file is overwritten.
    """
    if out_path is None:
        out_path = midi_path
    midi = load_and_validate_midi_file(midi_path, backend="mido")
    midi.tracks = [mido.merge_tracks(midi.tracks)]

    midi.save(out_path)


def dump_json(
    d: dict | list,
    path: Path,
    create_parents: bool = False,
    overwrite: bool = True,
    **kwargs,
):
    if not path.name.endswith(".json"):
        raise ValueError(f"Path {path} must end with .json")
    path.parent.mkdir(parents=create_parents, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"File {path} already exists.")
    with open(path, "w") as f:
        json.dump(d, f, **kwargs)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def find_common_root(paths: list[Path]) -> Path:
    return Path(os.path.commonpath(paths))
