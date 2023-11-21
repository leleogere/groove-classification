import copy
import json
import os.path
from pathlib import Path
from typing import Iterable

import pretty_midi as pm
import tabulate

import numpy as np

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
    main_label = raw_label.split('/')[0]
    if main_label not in keep_sublabel_for:
        return main_label
    else:
        return raw_label


def load_and_validate_midi_file(
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


def dump_json(d: dict, path: Path, create_parents: bool = False, overwrite: bool = True, **kwargs):
    if not path.name.endswith('.json'):
        raise ValueError(f"Path {path} must end with .json")
    path.parent.mkdir(parents=create_parents, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"File {path} already exists.")
    with open(path, 'w') as f:
        json.dump(d, f, **kwargs)


def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def find_common_root(paths: list[Path]) -> Path:
    return Path(os.path.commonpath(paths))


def write_labels_info(infos: dict[str, dict[str, dict[str, object]]], path: Path, tablefmt: str = 'rounded_outline'):
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
