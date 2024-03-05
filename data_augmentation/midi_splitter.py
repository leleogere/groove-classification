from collections import defaultdict
from pathlib import Path
from typing import Sequence
from typing import TypeVar

import joblib
import numpy as np
import pretty_midi as pm
from tqdm import tqdm

from data_processing.track_processor_grammar import GrammarTrackProcessor
from utils.utils import ProcessedTrack
from utils.utils import Track
from utils.utils import find_common_root
from utils.utils import get_midi_paths
from utils.utils import load_and_validate_midi_file
from utils.utils import merge_tracks
from utils.utils import parse_bpm_from_file_name
from utils.utils import parse_time_signature_from_file_name


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


class SplitterAbstract:
    def __init__(
        self,
        num_measures_per_chunk: int,
        chunk_overlap: bool = True,
    ):
        self.num_measures_per_chunk = num_measures_per_chunk
        self.chunk_overlap = chunk_overlap

    def split_track(
        self,
        track: Track | ProcessedTrack,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> list[Track]:
        raise NotImplementedError

    def split_list_of_tracks(
        self,
        tracks: list[Track | ProcessedTrack],
        output_dir: Path,
        root_path: Path | None = None,
        skip_existing: bool = True,
        n_jobs: int = 1,
        pbar: bool = True,
    ) -> dict[Path, list[Path]]:
        """
        Split a MIDI dataset into bars.
        :param tracks: List of tracks to split.
        :param output_dir: Output directory. The structure of the dataset will be preserved.
        :param root_path: If None, the output structure will be based on the nearest common parent
            directory of the MIDI files in the dataset and the given path.
        :param skip_existing: Whether skip the splitting of a MIDI file if the output directory
            already exists.
        :param n_jobs: Number of parallel tracks to split in parallel.
        :param pbar: Whether to display the note progression for each MIDI file.
        :return: Dictionary mapping each MIDI file to a list of paths to the chunks.
        """
        # Prepare output
        output_dir.mkdir(parents=True, exist_ok=True)

        # Loop over files
        midi_paths = [track.midi_path for track in tracks]
        midi_sizes = [path.stat().st_size for path in midi_paths]
        if root_path is None:
            root_path = find_common_root(midi_paths)

        def _split_track(track: Track | ProcessedTrack) -> list[Track]:
            """Wrapper to split a track to be used in parallel."""
            base_path = output_dir / track.midi_path.relative_to(root_path).with_suffix("")
            return self.split_track(
                track,
                output_dir=base_path,
                skip_existing=skip_existing,
            )

        # Split data in parallel
        # TODO: use "unordered_generator" when available in joblib?
        split_tracks_generator = joblib.Parallel(n_jobs=n_jobs, return_as="generator")(
            joblib.delayed(_split_track)(tr) for tr in tracks
        )

        # Loop instead of list comprehension to track progression
        chunked_tracks = {}
        pbar = tqdm(
            midi_paths,
            desc=f"Splitting MIDI files in chunks of {self.num_measures_per_chunk} measures",
            total=sum(midi_sizes),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not pbar,
        )
        for track, chunks, size in zip(tracks, split_tracks_generator, midi_sizes):
            chunked_tracks[track] = chunks
            pbar.update(size)
        pbar.close()
        return chunked_tracks

    def _get_measures_contained_by_each_chunk(self, total_measures: int) -> Sequence[Sequence[int]]:
        """
        Get the measures contained by each chunk.
        :param total_measures:
        :return: A sequence of chunks of the input sequence.
        :example:
            >>> splitter = MidiSplitter(num_measures_per_chunk=3, chunk_overlap=True)
            >>> splitter._get_measures_contained_by_each_chunk(total_measures=8)
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
            >>> splitter = MidiSplitter(num_measures_per_chunk=3, chunk_overlap=False)
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


Splitter = TypeVar("Splitter", bound=SplitterAbstract)


class MidiSplitter(SplitterAbstract):
    def __init__(
        self,
        num_measures_per_chunk: int,
        chunk_overlap: bool = True,
    ):
        super().__init__(
            num_measures_per_chunk=num_measures_per_chunk,
            chunk_overlap=chunk_overlap,
        )

    def split_track(
        self,
        track: Track,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> list[Track]:
        """
        Split a MIDI file into bars.
        :param track: Track to split.
        :param output_dir: Output directory. All chunks will be written in that directory.
        :param skip_existing: Whether to skip the splitting of the MIDI file if the output
        :return: List of paths to the chunks.
        :todo: What if a note starts before the chunk but ends after the chunk?
        """
        # Skip if the output directory already exists
        if output_dir.exists() and skip_existing:
            paths = get_midi_paths(output_dir)
            tracks = [Track.from_partial(track, midi_path=path) for path in paths]
            return tracks

        # Read midi and associated information
        midi = load_and_validate_midi_file(
            track.midi_path,
            unique_track=True,
            unique_tempo=True,
            unique_time_signature=True,
        )
        pm_track: pm.Instrument = midi.instruments[0]
        bpm = track.bpm
        time_signature = track.time_signature

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
        output_dir.mkdir(parents=True, exist_ok=True)
        chunked_tracks = []

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
            chunk_template.time_signature_changes.append(
                pm.TimeSignature(time_signature[0], time_signature[1], 0.0)
            )
            chunk_template.instruments.append(
                pm.Instrument(pm_track.program, is_drum=pm_track.is_drum, name=pm_track.name)
            )
            return chunk_template

        chunks = [create_chunk_template() for _ in range(num_chunks)]

        # Loop over notes
        current_measure = 0
        current_measure_end = measure_locations[1]
        for note in sorted(pm_track.notes, key=lambda n: n.start):
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
        dropped_chunks_file_path = output_dir / "dropped_chunks.csv"
        with open(dropped_chunks_file_path, "w") as f:
            f.write("chunk_start,chunk_end,reason\n")
            num_dropped_chunks = 0
            for i, chunk in enumerate(chunks):
                # Fix eventual invalid notes (there should not be any but just in case)
                chunk.remove_invalid_notes()

                # Run some test in order to test if a chunk is worth saving
                num_empty_measures = sum(
                    [
                        1
                        for measure_idx in measures_in_each_chunk[i]
                        if num_notes_per_measure[measure_idx] == 0
                    ]
                )
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
                    chunk_file_path = output_dir / _chunk_filename(
                        original_midi_path=track.midi_path,
                        start_bar=chunk_measures_start_and_end_measures[i][0],
                        num_bars=len(measures_in_each_chunk[i]),
                        formatting_number_width=formatting_number_width,
                    )
                    chunk.write(str(chunk_file_path))
                    merge_tracks(chunk_file_path)
                    chunked_tracks.append(Track.from_partial(track, midi_path=chunk_file_path))

        # Remove the log file if no chunk was dropped
        if num_dropped_chunks == 0:
            dropped_chunks_file_path.unlink()

        return chunked_tracks


class RuleSequenceSplitter(SplitterAbstract):
    def __init__(
        self,
        grammar_processor: GrammarTrackProcessor,
        num_measures_per_chunk: int,
        chunk_overlap: bool = True,
        split_on: int = -1,
    ):
        super().__init__(
            num_measures_per_chunk=num_measures_per_chunk,
            chunk_overlap=chunk_overlap,
        )
        self.grammar_processor = grammar_processor
        self.split_on = split_on

    def split_track(
        self,
        track: Track,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> list[Track]:
        measure_split_id = self.grammar_processor.grammar.SIMPLE_MEASURE_SPLIT_RULE.simple_id
        sequence = self.grammar_processor.track_to_sequence(track, none_if_parsing_failed=True)
        if sequence is None:
            return []
        rules, midi_indices = zip(*sequence)
        measure_indices = np.argwhere(np.array(rules) == measure_split_id).flatten().tolist() + [
            len(rules)
        ]
        num_measures = len(measure_indices) - 1

        # Create chunks
        formatting_number_width = len(str(num_measures))
        chunks = []
        if self.chunk_overlap:
            range_ = range(0, num_measures - self.num_measures_per_chunk + 1)
        else:
            range_ = range(0, num_measures, self.num_measures_per_chunk)
        if range_.stop > num_measures:
            breakpoint()
        for chunk_idx, current_index in enumerate(range_):
            start_measure_idx = current_index
            end_measure_idx = min(current_index + self.num_measures_per_chunk, num_measures)
            start, end = measure_indices[start_measure_idx], measure_indices[end_measure_idx]
            chunk_rules = rules[start:end]
            chunk_indices = midi_indices[start:end]
            chunk_name = _chunk_filename(
                track.midi_path,
                measure_indices[current_index],
                self.num_measures_per_chunk,
                formatting_number_width,
            )
            chunk = Track.from_partial(
                track,
                other_data={
                    "chunk_name": chunk_name,
                    "sequence": list(zip(chunk_rules, chunk_indices)),
                },
            )
            chunks.append(chunk)
        # for chunk_idx, measures in enumerate(measures_per_chunk):
        #     start, end = measures[0], measures[1]
        #     idx_start, idx_end = measure_indices[start], measure_indices[end]
        #     chunk_rules = rules[idx_start:idx_end]
        #     chunk_indices = midi_indices[idx_start:idx_end]
        #     chunk_name = _chunk_filename(track.midi_path, idx_start, idx_end, formatting_number_width)
        #     chunk = ProcessedTrack.from_partial(
        #         track,
        #         midi_path=track.midi_path.with_suffix("") / chunk_name,
        #         notes_tensor=(chunk_rules, chunk_indices),
        #     )
        #     chunks.append(chunk)

        return chunks


if __name__ == "__main__":
    from pprint import pprint

    TEST = "track"
    # splitter = MidiSplitter(
    #     num_measures_per_chunk=4,
    #     chunk_overlap=True,
    # )
    splitter = RuleSequenceSplitter(
        grammar_processor=GrammarTrackProcessor(
            Path("utils/grammar.wta"),
            sequences_dir=Path("/tmp/sequences_tests"),
        ),
        num_measures_per_chunk=4,
        chunk_overlap=True,
    )
    if TEST == "track":
        MIDI_PATH = Path("../groove_data/groove/drummer1/session1/1_funk_80_beat_4-4.mid")
        # MIDI_PATH = Path('data/groove_augmented/drummer5/session1/8_latin-venezuelan-merengue_162_beat_5-8/8_latin-venezuelan-merengue_162_beat_5-8_variation=7.mid')
        # MIDI_PATH = Path('data/groove/drummer10/session1/1_jazz-swing_124_beat_4-4.mid')
        # MIDI_PATH = Path('groove_light/drummer1/eval_session/4_soul-groove4_80_beat_4-4.mid')
        # MIDI_PATH = Path('groove/drummer7/session2/22_rock_118_beat_4-4.mid')
        # MIDI_PATH = Path('groove/drummer2/session3/1_rock-shuffle_85_beat_4-4.mid')
        # MIDI_PATH = Path('groove/drummer1/session2/111_rock-prog_110_beat_5-4.mid')
        OUT_PATH = Path("/tmp/split_tracks")
        track = Track(
            midi_path=MIDI_PATH,
            bpm=parse_bpm_from_file_name(MIDI_PATH.name),
            time_signature=parse_time_signature_from_file_name(MIDI_PATH.name),
        )
        chunks = splitter.split_track(
            track,
            output_dir=OUT_PATH,
            skip_existing=False,
        )
        pprint(chunks)
        for chunk in chunks:
            print(chunk.midi_path)
            print(chunk.processed_data)
            print()
    elif TEST == "dataset":
        DATASET_PATH = Path("../groove_data/groove/drummer1/session1")
        OUT_PATH = Path("/tmp/dataset_split")
        midi_paths = get_midi_paths(DATASET_PATH)
        midi_paths = [path for path in midi_paths if "_beat_" in path.name]
        tracks = [
            Track(
                midi_path=path,
                bpm=parse_bpm_from_file_name(path.name),
                time_signature=parse_time_signature_from_file_name(path.name),
            )
            for path in midi_paths
        ]
        augmented_tracks = splitter.split_list_of_tracks(
            tracks,
            output_dir=OUT_PATH,
            skip_existing=False,
            n_jobs=1,
        )
    else:
        raise ValueError(f"ERROR: {TEST=}")
