from collections import defaultdict
from pathlib import Path
from typing import Sequence

import pretty_midi as pm
from tqdm import tqdm

from utils.classes import Track
from utils.utils import find_common_root
from utils.utils import get_midi_paths
from utils.utils import load_and_validate_midi_file


def sliding_window(seq: Sequence, window_size: int = 2) -> Sequence[Sequence]:
    """
    Returns a sequence of overlapping chunks of the input sequence.
    :param seq: The sequence to iterate over.
    :param window_size: The size of the window.
    :return: A sequence of chunks of the input sequence.
    """
    # Verify the inputs
    if not isinstance(seq, Sequence):
        raise ValueError(f"sequence ({seq}) must be implement __len__")
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError(f"window_size ({window_size}) must be a positive integer.")
    if len(seq) == 0:
        return []
    # Return the windows
    windows = []
    num_windows = max(
        1, len(seq) - window_size + 1
    )  # max(1, n) to ensure that we deal with window_size > len(sequence)
    for i in range(0, num_windows):
        windows.append(seq[i : i + window_size])
    return windows


def non_sliding_window(seq: Sequence, window_size: int = 2) -> Sequence[Sequence]:
    """
    Returns a sequence of non-overlapping chunks of the input sequence.
    :param seq: The sequence to iterate over.
    :param window_size: The size of the window.
    :return: A sequence of chunks of the input sequence.
    """
    # Verify the inputs
    if not isinstance(seq, Sequence):
        raise ValueError(f"sequence ({seq}) must be implement __len__")
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError(f"window_size ({window_size}) must be a positive integer.")
    if len(seq) == 0:
        return []
    # Return the windows
    windows = [seq[i : i + window_size] for i in range(0, len(seq), window_size)]
    return windows


def _chunk_filename(
    original_midi_path: Path,
    start_bar: int,
    num_bars: int,
    formatting_number_width: int,
):
    filename_end = f"startbar={start_bar:0{formatting_number_width}}_numbars={num_bars}"
    return f"{original_midi_path.stem}_{filename_end}.mid"


class Splitter:
    def __init__(self, num_measures_per_chunk: int, chunk_overlap: bool = True):
        self.num_measures_per_chunk = num_measures_per_chunk
        self.chunk_overlap = chunk_overlap

    def _get_measures_contained_by_each_chunk(self, total_measures: int) -> Sequence[Sequence[int]]:
        """
        Get the measures contained by each chunk.
        :param total_measures:
        :return: A sequence of chunks of the input sequence.
        :example:
            >>> splitter = Splitter(num_measures_per_chunk=3, chunk_overlap=True)
            >>> splitter._get_measures_contained_by_each_chunk(total_measures=8)
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
            >>> splitter = Splitter(num_measures_per_chunk=3, chunk_overlap=False)
            >>> splitter._get_measures_contained_by_each_chunk(total_measures=8)
            [[0, 1, 2], [3, 4, 5], [6, 7]]
        """
        if self.chunk_overlap:
            return sliding_window(
                list(range(0, total_measures)),
                window_size=self.num_measures_per_chunk,
            )
        else:
            return non_sliding_window(
                list(range(0, total_measures)),
                window_size=self.num_measures_per_chunk,
            )

    def split_midi_file(self, path: Path, out_path: Path, pbar: bool = False) -> list[Path]:
        """
        Split a MIDI file into bars.
        :param path: Path to the MIDI file.
        :param out_path: Output directory. All chunks will be written in that directory.
        :param pbar: Whether to display the note progression.
        :return: List of paths to the chunks.
        :todo: What if a note starts before the chunk but ends after the chunk?
        """
        # Read midi and associated information
        midi = load_and_validate_midi_file(
            path,
            unique_track=True,
            unique_tempo=True,
            unique_time_signature=True,
        )
        track: pm.Instrument = midi.instruments[0]
        bpm = midi.get_tempo_changes()[1][0]
        time_signature = midi.time_signature_changes[0]

        # Get the starting time of each measure
        measure_locations = midi.get_downbeats().tolist()
        measure_duration = (
            measure_locations[1] if len(measure_locations) > 1 else midi.get_end_time()
        )
        chunk_duration = measure_duration * self.num_measures_per_chunk
        num_measures = len(measure_locations) - 1
        # The remaining part is dropped (the one between the last measure and the end of the MIDI
        # file) as either not long enough, or at least very likely not representative of the genre

        # Prepare output
        formatting_number_width = len(str(num_measures))
        out_path.mkdir(parents=True, exist_ok=True)
        chunk_file_paths = []

        # Prepare loop
        measures_in_each_chunk = self._get_measures_contained_by_each_chunk(num_measures)
        chunk_measures_start_and_end_measures = [
            (current_chunk_measures[0], current_chunk_measures[-1] + 1)
            for current_chunk_measures in measures_in_each_chunk
        ]
        chunk_measures_start_and_end_times = [
            (measure_locations[start], measure_locations[end])
            for start, end in chunk_measures_start_and_end_measures
        ]
        num_chunks = len(chunk_measures_start_and_end_times)
        if num_chunks == 0:
            # print(f"No chunks in {path}.")
            return []
        measure_idx_to_chunk_idxs = {
            i: {j for j, measures in enumerate(measures_in_each_chunk) if i in measures}
            for i in range(num_measures)
        }
        num_notes_per_measure = [0] * num_measures

        # Prepare MIDI template
        def create_chunk_template() -> pm.PrettyMIDI:
            chunk_template = pm.PrettyMIDI(initial_tempo=bpm, resolution=midi.resolution)
            chunk_template.time_signature_changes.append(time_signature)
            chunk_template.instruments.append(
                pm.Instrument(track.program, is_drum=track.is_drum, name=track.name)
            )
            return chunk_template

        chunks = [create_chunk_template() for _ in range(num_chunks)]

        # Loop over notes
        current_measure = 0
        current_measure_end = measure_locations[1]
        for note in tqdm(
            sorted(track.notes, key=lambda n: n.start),  # make sure notes are sorted by start time
            total=len(track.notes),
            unit="note",
            disable=not pbar,
            leave=False,
        ):
            # If current note starts after the end of the current measure, move to the next measure
            if note.start >= current_measure_end:
                current_measure += 1
                # Break if last measure (remaining part is dropped)
                if current_measure >= num_measures:
                    break
                current_measure_end = measure_locations[current_measure + 1]
            num_notes_per_measure[current_measure] += 1
            # Get all chunks in which the current note should be added
            chunks_to_add_note_to = measure_idx_to_chunk_idxs[current_measure].copy()
            # If the note ends after the end of the current measure, add we need to add it to
            if note.end > current_measure_end and current_measure + 1 < num_measures:
                chunks_to_add_note_to.update(measure_idx_to_chunk_idxs[current_measure + 1])
            # Add the note to each chunk it should be added to
            for chunk_idx in chunks_to_add_note_to:
                # Convert note start and end to chunk-relative time
                chunk_start = chunk_measures_start_and_end_times[chunk_idx][0]
                note_start = max(0.0, note.start - chunk_start)
                note_end = min(chunk_duration, note.end - chunk_start)
                # Filter out notes that are too short and add the note to the chunk
                if abs(note_start - note_end) > 1e-8:
                    chunks[chunk_idx].instruments[0].notes.append(
                        pm.Note(
                            velocity=note.velocity,
                            pitch=note.pitch,
                            start=note_start,
                            end=note_end,
                        )
                    )
        # Loop over all chunks and save them (or drop them if they do not meet the requirements)
        dropped_chunks_file_path = out_path / "dropped_chunks.csv"
        with open(dropped_chunks_file_path, "w") as f:
            f.write("chunk_start,chunk_end,reason\n")
            num_dropped_chunks = 0
            for i, chunk in enumerate(chunks):
                # Fix eventual invalid notes (there should not be any but just in case)
                chunk.remove_invalid_notes()

                # Run some test in order to test if a chunk is worth saving
                num_empty_measures = sum([
                    1
                    for measure_idx in measures_in_each_chunk[i]
                    if num_notes_per_measure[measure_idx] == 0
                ])
                num_notes = len(chunk.instruments[0].notes)
                drop_message = None  # message to log if the chunk is dropped
                if num_notes == 0:
                    # if no notes in the chunk
                    drop_message = "no_notes"
                elif num_notes < 7 * self.num_measures_per_chunk:
                    # if too few notes in the chunk
                    drop_message = "not_enough_notes"
                elif num_empty_measures > self.num_measures_per_chunk // 2:
                    # if more than half of the measures are empty
                    drop_message = "too_many_empty_measures"
                elif i == 0 and chunk.instruments[0].notes[0].start > 0.100:
                    # if no note in the first 100ms if the track (only for first chunk)
                    drop_message = "no_note_on_first_track_beat"

                # Drop the chunk if needed
                if drop_message is not None:
                    # print(f"Chunk {i} of {path} dropped ({drop_message}).")
                    f.write(
                        f"{chunk_measures_start_and_end_times[i][0]},"
                        f"{chunk_measures_start_and_end_times[i][1]},"
                        f"{drop_message}\n"
                    )
                    num_dropped_chunks += 1
                else:  # Save the chunk
                    chunk_file_path = out_path / _chunk_filename(
                        original_midi_path=path,
                        start_bar=chunk_measures_start_and_end_measures[i][0],
                        num_bars=len(measures_in_each_chunk[i]),
                        formatting_number_width=formatting_number_width,
                    )
                    chunk.write(str(chunk_file_path))
                    chunk_file_paths.append(chunk_file_path)

        # Remove the log file if no chunk was dropped
        if num_dropped_chunks == 0:
            dropped_chunks_file_path.unlink()

        return chunk_file_paths

    def split_midi_dataset(
        self,
        dataset: Path | list[Track],
        out_path: Path,
        overwrite_existing: bool = True,
        subpbar: bool = False,
    ) -> dict[Path, list[Path]]:
        """
        Split a MIDI dataset into bars.
        :param dataset: Path to the MIDI dataset or list of tracks.
        :param out_path: Output directory. The structure of the dataset will be preserved.
        :param overwrite_existing: Whether to recompute chunks of files that already have chunks, i.e. files that have
            at least ONE chunk existing, it will not check if all chunks have been computed. It means that interrupting
            the splitting process and resuming it later may result in incomplete splitting for the file being processed
            at the moment of the interruption if overwrite_existing is False.
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
            current_out_path = out_path / midi_path.relative_to(root_path).with_suffix("")
            if current_out_path.exists() and not overwrite_existing:
                chunk_file_paths[midi_path].extend(sorted(list(current_out_path.rglob("*.mid"))))
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


if __name__ == "__main__":
    # test_midi_path = Path('data/groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid')
    # test_midi_path = Path('data/groove_augmented/drummer5/session1/8_latin-venezuelan-merengue_162_beat_5-8/8_latin-venezuelan-merengue_162_beat_5-8_variation=7.mid')
    test_midi_path = Path("data/groove/drummer10/session1/1_jazz-swing_124_beat_4-4.mid")
    # test_midi_path = Path('groove_light/drummer1/eval_session/4_soul-groove4_80_beat_4-4.mid')
    # test_midi_path = Path('groove/drummer7/session2/22_rock_118_beat_4-4.mid')
    # test_midi_path = Path('groove/drummer2/session3/1_rock-shuffle_85_beat_4-4.mid')
    # test_midi_path = Path('groove/drummer1/session2/111_rock-prog_110_beat_5-4.mid')
    splitter = Splitter(num_measures_per_chunk=1, chunk_overlap=False)
    # splitter(test_midi_path, out_path=Path('/tmp/tests'))
    splitter(test_midi_path, out_path=Path("/tmp/split"))
