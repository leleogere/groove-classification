import json
import os.path
import re
from pathlib import Path
from typing import Union

import mido
import numpy as np
import pretty_midi as pm
import symusic

np.int = int


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
