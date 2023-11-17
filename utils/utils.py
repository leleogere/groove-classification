import copy
import csv
import json
import dataclasses
import os.path
from pathlib import Path
from typing import Iterable, Self

import pretty_midi as pm
import tabulate
import torch
from miditoolkit import MidiFile

import numpy as np

np.int = int


@dataclasses.dataclass(slots=True)
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

    def drum_from_id(self, id: int) -> Drum:
        return self._id_to_percussion[id]

    def __len__(self):
        return len(self.drums)


@dataclasses.dataclass(slots=True)
class Note:
    start: float
    end: float
    pitch: int
    velocity: int

    @property
    def duration(self):
        return self.end - self.start

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @staticmethod
    def from_dict(d: dict):
        return Note(**d)


@dataclasses.dataclass(slots=True)
class RelativeNote:
    pitch: int
    velocity: float
    duration: float
    timeshift: float

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @staticmethod
    def from_dict(d: dict):
        return RelativeNote(**d)

    @staticmethod
    def from_note(note: Note, timeshift: float = 0.):
        return RelativeNote(
            pitch=note.pitch,
            velocity=note.velocity,
            duration=note.duration,
            timeshift=timeshift
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

    def notes(self):
        midi = load_and_check_midi_file(self.midi_path, only_drums=True, unique_track=True, unique_tempo=True, unique_time_signature=True)
        notes = [Note(n.start, n.end, n.pitch, n.velocity) for n in sorted(midi.instruments[0].notes, key=lambda x: x.start)]
        return notes

    def relative_notes(self):
        relative_notes = [
            RelativeNote.from_note(n, timeshift=n_next.start - n.start)
            for n, n_next in zip(self.notes(), self.notes()[1:] + [self.notes()[-1]])
        ]
        return relative_notes

    def to_json(self):
        d = self.__dict__.copy()
        d['midi_path'] = str(d['midi_path'])
        return d

    def to_json_file(self, path: Path, create_parents: bool = False, **kwargs):
        if not path.name.endswith('.json'):
            raise ValueError(f"Path {path} must end with .json")
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, **kwargs)

    @classmethod
    def from_json(cls, json_dict: dict):
        json_dict['midi_path'] = Path(json_dict['midi_path'])
        return cls(**json_dict)

    def copy(self):
        return Track.from_json(self.to_json())

    @classmethod
    def from_json_file(cls, path: Path):
        with open(path) as f:
            return cls.from_json(json.load(f))

    def __repr__(self):
        elements = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Track({', '.join(elements)})"

    @classmethod
    def from_partial(cls, track: Self, **kwargs):
        d = track.__dict__.copy()
        d.update(kwargs)
        return cls(**d)


class ProcessedTrack(Track):
    def __init__(
            self,
            data: torch.Tensor,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data


def read_notes(midi_path: Path):
    # raise NotImplementedError("Test.")
    midi = MidiFile(midi_path)
    drum_tracks = [track for track in midi.instruments if track.is_drum]
    if len(drum_tracks) != 1:
        raise ValueError(f"Expected 1 drum track, got {len(drum_tracks)} in {midi_path}.")
    notes = []
    last_start = 0
    for note in sorted(drum_tracks[0].relative_notes, key=lambda x: x.start):
        if note.start < last_start:
            raise ValueError(f"Notes are not sorted by start time in {midi_path}.")
        notes.append(
            RelativeNote(
                pitch=note.pitch,
                velocity=note.velocity,
                duration=note.end - note.start,
                timeshift=note.start - last_start
            )
        )
        last_start = note.start
    return notes


def read_drums_list(file_path: Path):
    with open(file_path) as f:
        reader = csv.DictReader(f)
        drums = DrumList([
            Drum(pitch=int(row["Pitch"]), id=idx, name=row["Roland Mapping"])
            for idx, row in enumerate(reader)
        ])
        return drums


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
    main_label = raw_label.split('/')[0]
    if main_label not in keep_sublabel_for:
        return main_label
    else:
        return raw_label


def load_and_check_midi_file(
        midi_path: Path,
        only_drums: bool = False,
        unique_track: bool = False,
        unique_time_signature: bool = False,
        unique_tempo: bool = False,
) -> pm.PrettyMIDI:
    if not midi_path.exists():
        raise FileNotFoundError(f'File {midi_path} does not exist.')
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
            and any(time_signatures[0].__dict__ != ts.__dict__ for ts in time_signatures)  # in case of duplicates
    ):
        raise ValueError(f"Expected a unique time signature, got {len(time_signatures)} in {midi_path}.")
    return midi


def get_midi_paths(path: Path, return_sizes: bool = False):
    midi_exts = ['.mid', '.midi']
    midi_exts.extend([ext.upper() for ext in midi_exts])
    paths = sum([list(path.rglob(f'*{ext}')) for ext in midi_exts], [])
    if return_sizes:
        sizes = [p.stat().st_size for p in paths]
        return paths, sizes
    else:
        return paths


def dump_json(d: dict, path: Path, create_parents: bool = False, **kwargs):
    if not path.name.endswith('.json'):
        raise ValueError(f"Path {path} must end with .json")
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(d, f, **kwargs)


def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def find_common_root(paths: list[Path]) -> Path:
    return Path(os.path.commonpath(paths))


def _write_labels_info(infos: dict[str, dict[str, dict[str, object]]], path: Path, tablefmt: str = 'rounded_outline'):
    def unique_list(iter: Iterable):
        return list(dict.fromkeys(iter))
    infos = copy.deepcopy(infos)
    labels = unique_list(label for label in infos)
    splits = unique_list(split for label in infos for split in infos[label])
    fields = unique_list(field for label in infos for split in infos[label] for field in infos[label][split])
    # add split "Total" to dict infos
    for label in infos:
        for field in infos[label][splits[0]]:
            infos[label]['total'][field] = sum(infos[label][split][field] for split in splits)
    splits.insert(0, 'total')
    with open(path, 'w') as f:
        for split in splits:
            f.write(f'{split}\n')
            table = []
            for label in labels:
                table.append([label] + [infos[label][split][field] for field in fields])
            f.write(tabulate.tabulate(table, headers=['label'] + list(fields), tablefmt=tablefmt))
            f.write('\n\n')
