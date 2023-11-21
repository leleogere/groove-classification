import copy
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import pretty_midi as pm
from tqdm import tqdm

from utils.classes import Track
from utils.utils import load_and_validate_midi_file, get_midi_paths, find_common_root


def sliding_window(seq: Sequence, window_size: int = 2) -> Sequence[Sequence]:
    """
    Returns a sequence of overlapping chunks of the input sequence.
    :param seq: The sequence to iterate over.
    :param window_size: The size of the window.
    :return: A sequence of chunks of the input sequence.
    """
    # Verify the inputs
    if not isinstance(seq, Sequence):
        raise ValueError(f'sequence ({seq}) must be implement __len__')
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError(f'window_size ({window_size}) must be a positive integer.')
    if len(seq) == 0:
        return []
    # Return the windows
    windows = []
    range_end = max(1, len(seq) - window_size + 1)  # ensure that we deal with window_size > len(sequence)
    for i in range(0, range_end):
        windows.append(seq[i:i + window_size])
    return windows


def non_sliding_window(seq: Sequence, window_size: int = 2) -> Sequence[Sequence]:
    """
    Returns a sequence of non-overlapping chunks of the input sequence.
    the defined chunks of input sequence.
    :param seq: The sequence to iterate over.
    :param window_size: The size of the window.
    :return: A sequence of chunks of the input sequence.
    """
    # Verify the inputs
    if not isinstance(seq, Sequence):
        raise ValueError(f'sequence ({seq}) must be implement __len__')
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError(f'window_size ({window_size}) must be a positive integer.')
    if len(seq) == 0:
        return []
    # Return the windows
    windows = [seq[i:i + window_size] for i in range(0, len(seq), window_size)]
    return windows


def _chunk_filename(original_midi_path: Path, start_bar: int, num_bars: int, formatting_number_width: int):
    filename_end = f'startbar={start_bar:0{formatting_number_width}}_numbars={num_bars}'
    return f'{original_midi_path.stem}_{filename_end}.mid'


class Splitter:
    def __init__(self, num_measures_per_chunk: int, chunk_overlap: bool = True):
        self.num_measures_per_chunk = num_measures_per_chunk
        self.chunk_overlap = chunk_overlap

    def _get_chunk_measures(self, num_measures: int) -> Sequence[Sequence[int]]:
        if self.chunk_overlap:
            return sliding_window(list(range(0, num_measures)), window_size=self.num_measures_per_chunk)
        else:
            return non_sliding_window(list(range(0, num_measures)), window_size=self.num_measures_per_chunk)

    def split_midi_file(self, midi_path: Path, out_path: Path, pbar: bool = False) -> list[Path]:
        """
        Split a MIDI file into bars.
        :param midi_path: Path to the MIDI file.
        :param out_path: Output directory. All chunks will be written in that directory.
        :param pbar: Whether to display the note progression.
        :return: List of paths to the chunks.
        :todo: What if a note starts before the chunk but ends after the chunk?
        """
        # Read midi and associated information
        midi = load_and_validate_midi_file(midi_path, unique_track=True, unique_tempo=True, unique_time_signature=True)
        track: pm.Instrument = midi.instruments[0]
        bpm = midi.get_tempo_changes()[1][0]
        time_signature = midi.time_signature_changes[0]

        # Get measure timings
        bar_locations = midi.get_downbeats().tolist()
        measure_duration = bar_locations[1] if len(bar_locations) > 1 else midi.get_end_time()
        chunk_duration = measure_duration * self.num_measures_per_chunk
        if abs(bar_locations[-1] - midi.get_end_time()) > measure_duration / 2:
            bar_locations.append(midi.get_end_time())
        num_measures = len(bar_locations) - 1

        # Prepare output
        formatting_number_width = len(str(num_measures))
        out_path.mkdir(parents=True, exist_ok=True)
        chunk_file_paths = []

        # Prepare loop
        chunk_measures = self._get_chunk_measures(num_measures)
        chunk_measures_starts_and_ends = [(chunk[0], chunk[-1] + 1) for chunk in chunk_measures]
        chunk_time_starts_and_ends = [
            (bar_locations[start], bar_locations[end]) for start, end in chunk_measures_starts_and_ends
        ]
        num_chunks = len(chunk_time_starts_and_ends)
        if num_chunks == 0:
            print(f'No chunks in {midi_path}.')
            return []
        measure_idx_to_chunk_idxs = {
            i: [j for j, measures in enumerate(chunk_measures) if i in measures]
            for i in range(num_measures)
        }

        # Prepare MIDI template
        def create_template() -> pm.PrettyMIDI:
            chunk_template = pm.PrettyMIDI(initial_tempo=bpm, resolution=midi.resolution)
            chunk_template.time_signature_changes.append(time_signature)
            chunk_template.instruments.append(pm.Instrument(track.program, is_drum=track.is_drum, name=track.name))
            return chunk_template
        chunks = [create_template() for _ in range(num_chunks)]

        # Loop over notes
        current_measure = 0
        current_measure_end = bar_locations[1]
        for note in tqdm(sorted(track.notes, key=lambda n: n.start), total=len(track.notes), unit="note", disable=not pbar, leave=False):
            if note.start >= current_measure_end:
                current_measure += 1
                if current_measure >= num_measures:
                    break
                current_measure_end = bar_locations[current_measure + 1]
            chunks_to_add_note_to = measure_idx_to_chunk_idxs[current_measure]
            if note.end > current_measure_end and current_measure + 1 < num_measures:
                chunks_to_add_note_to.extend(measure_idx_to_chunk_idxs[current_measure + 1])
            for chunk_idx in chunks_to_add_note_to:
                chunk_start = chunk_time_starts_and_ends[chunk_idx][0]
                note_start = max(0., note.start - chunk_start)
                note_end = min(chunk_duration, note.end - chunk_start)
                if abs(note_start - note_end) > 1e-8:
                    chunks[chunk_idx].instruments[0].notes.append(pm.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note_start,
                        end=note_end,
                    ))
        # Loop over chunks, fix note timings and save them
        for i, chunk in enumerate(chunks):
            # Don't save empty chunks
            chunk.remove_invalid_notes()
            if not chunk.instruments[0].notes:
                continue
            chunk_file_path = out_path / _chunk_filename(
                original_midi_path=midi_path,
                start_bar=chunk_measures_starts_and_ends[i][0],
                num_bars=len(chunk_measures[i]),
                formatting_number_width=formatting_number_width
            )
            chunk.write(str(chunk_file_path))
            chunk_file_paths.append(chunk_file_path)
        return chunk_file_paths

    def split_midi_dataset(
            self,
            dataset: Path | list[Track],
            out_path: Path,
            only_if_doesnt_exist: bool = False,
            subpbar: bool = False,
    ) -> dict[Path, list[Path]]:
        """
        Split a MIDI dataset into bars.
        :param dataset: Path to the MIDI dataset or list of tracks.
        :param out_path: Output directory. The structure of the dataset will be preserved.
        :param only_if_doesnt_exist: Whether to skip files that already have an associated output directory (it doesn't
            mean that the chunks have already been computed, an error could have occurred during the previous process).
        :param subpbar: Whether to display the note progression for each MIDI file.
        :return: Dictionary mapping each MIDI file to a list of paths to the chunks.
        """
        # Prepare output
        out_path.mkdir(parents=True, exist_ok=True)
        chunk_file_paths = defaultdict(list)

        # Loop over MIDI files
        if isinstance(dataset, list):
            midi_paths = [track.midi_path for track in dataset]
            midi_sizes = [path.stat().st_size for path in midi_paths]
            root_path = find_common_root(midi_paths)
        elif isinstance(dataset, Path):
            midi_paths, midi_sizes = get_midi_paths(dataset, return_sizes=True)
            root_path = dataset
        else:
            raise TypeError(f"dataset must be a Path or a list of Tracks, not {type(dataset)}")
        pbar = tqdm(
            midi_paths,
            desc=f"Splitting MIDI files in chunks of {self.num_measures_per_chunk} measures",
            total=sum(midi_sizes),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        for midi_path, midi_size in zip(midi_paths, midi_sizes):
            current_out_path = out_path / midi_path.relative_to(root_path).with_suffix('')
            if only_if_doesnt_exist and current_out_path.exists():
                chunk_file_paths[midi_path].extend(list(current_out_path.rglob('*.mid')))
                pbar.update(midi_size)
                continue
            current_out_path.parent.mkdir(parents=True, exist_ok=True)
            current_chunk_paths = self.split_midi_file(midi_path, current_out_path, pbar=subpbar)
            chunk_file_paths[midi_path].extend(current_chunk_paths)
            pbar.update(midi_size)
        pbar.close()
        return chunk_file_paths

    def __call__(self, path: Path, out_path: Path) -> list[Path] | dict[Path, list[Path]]:
        """
        Split a MIDI file or dataset into bars.
        :param path: Path to the MIDI file or dataset.
        :param out_path: Output directory. The structure of the dataset will be preserved.
        :return: List of paths to the chunks if path is a MIDI file, dictionary mapping each MIDI file to a list of
            paths to the chunks if path is a folder.
        """
        if path.is_dir():
            return self.split_midi_dataset(path, out_path)
        else:
            return self.split_midi_file(path, out_path)


if __name__ == '__main__':
    # midi_path = Path('data/groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid')
    # midi_path = Path('data/groove_augmented/drummer5/session1/8_latin-venezuelan-merengue_162_beat_5-8/8_latin-venezuelan-merengue_162_beat_5-8_variation=7.mid')
    midi_path = Path('data/groove/drummer10/session1/1_jazz-swing_124_beat_4-4.mid')
    # midi_path = Path('groove_light/drummer1/eval_session/4_soul-groove4_80_beat_4-4.mid')
    # midi_path = Path('groove/drummer7/session2/22_rock_118_beat_4-4.mid')
    # midi_path = Path('groove/drummer2/session3/1_rock-shuffle_85_beat_4-4.mid')
    # midi_path = Path('groove/drummer1/session2/111_rock-prog_110_beat_5-4.mid')
    splitter = Splitter(num_measures_per_chunk=1, chunk_overlap=False)
    # splitter(midi_path, out_path=Path('/tmp/tests'))
    splitter(midi_path, out_path=Path('/tmp/tests'))
