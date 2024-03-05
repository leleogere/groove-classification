import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import joblib
import torch

import pyqparse

from data_processing.track_processors_base import TrackProcessor
from utils.utils import ProcessedTrack
from utils.utils import Track
from utils.constants import DRUMS
from utils.grammar import Grammar
from utils.grammar import Node
from utils.utils import load_and_validate_midi_file


pyqparse.set_verbosity_level(1)

RuleWithIndices = tuple[int, tuple[int, int] | None]


class GrammarTrackProcessor(TrackProcessor):
    short_name = "grammar"

    def __init__(self, weights_file: Path, use_tempo: bool = True, sequences_dir: Path = None):
        super().__init__(use_tempo=use_tempo)
        self.grammar_file = weights_file
        self.grammar = Grammar.from_file(self.grammar_file)
        # self.scoring_env.set_order(False)  # To get the best trees (equivalent of option -max)
        # self.scoring_env.set_trs(pyqparse.Mode.Drum)
        # self.scoring_env.import_WTA(str(self.grammar_file), pyqparse.WeightCode.Tropical)
        # self.scoring_env.set_debug(False)
        if sequences_dir is None:
            sequences_dir = Path(tempfile.mkdtemp(prefix="production_rules_sequences_"))
        self.sequences_dir = sequences_dir
        self.memory = joblib.Memory(self.sequences_dir, verbose=0)
        self.track_to_sequence = self.memory.cache(self.track_to_sequence, ignore=["self"])
        self.sequence_to_tensor = self.memory.cache(self.sequence_to_tensor, ignore=["self"])

    def track_to_sequence(
        self,
        track: Track,
        none_if_parsing_failed: bool = False,
    ) -> list[RuleWithIndices] | None:
        """
        Convert a track to a sequence of rules and indices.

        :param track: Track to convert.
        :param none_if_parsing_failed: If True, return None if parsing failed. If False, propagate
            the exception.
        :return: Sequence of rules and indices.
        """
        # try:
        #     tree = self._build_tree(track)
        # except ValueError as e:
        #     if none_if_parsing_failed:
        #         print(f"Error in tree built from {track.midi_path}: {e}")
        #         return None
        #     else:
        #         raise e
        # sequence = tree.get_production_rule_ids_and_midi_indices(simplified_ids=True)
        # return sequence

        # Run the parsing as a subprocess bc pyqparse is sometimes result in a segfault when
        # dealing with complex rythms. This way, the segfault is contained in the subprocess, and
        # does not crash the main process.
        environ = os.environ.copy()
        try:
            json_tree = subprocess.run(
                [
                    "python",
                    "utils/_parse_midi_subprocess.py",
                    str(track.midi_path),
                    "--tempo",
                    str(track.bpm),
                    "--time-signature",
                    str(track.time_signature[0]),
                    str(track.time_signature[1]),
                    "--grammar-file",
                    str(self.grammar_file),
                ],
                check=True,
                capture_output=True,
                env=environ,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error in tree built from {track.midi_path}: {e.stderr.decode()}")
            if none_if_parsing_failed:
                return None
            else:
                raise e
        sequence = json.loads(json_tree.stdout)
        return sequence

    def sequence_to_tensor(self, sequence: list[RuleWithIndices], track: Track) -> torch.Tensor:
        """
        Convert a list of rules and with MIDI event indices to a tensor. The first n rows contain
        the OHE rules, and the next m rows contain the HE drums pitches (only for terminal rules).

        :param sequence: Sequence of rules and indices.
        :param track: Track to get the notes from.
        :param rules: List of production rules.
        :param indices: List of indices in the format [start, end).
        :return: Tensor of shape (len(rules), num_simplified_rules + num_drum_pitches).
        """
        rules, indices = zip(*sequence)
        if len(indices) != len(rules):
            raise ValueError("Length of rules and indices must be the same.")
        # Convert [start, end) to range(start, end)
        indices = map(
            lambda start_end: range(start_end[0], start_end[1]) if start_end is not None else [],
            indices,
        )
        notes_pitches = self._get_notes_from_indices(track, indices)
        rules_tensor = torch.zeros(len(rules), self.grammar.num_simplified_rules())
        rules_tensor[range(len(rules)), rules] = 1
        drums_tensor = torch.ones(len(notes_pitches), len(DRUMS))  # * 0.1  # For debugging in plots
        for i, pitches in enumerate(notes_pitches):
            ids = [DRUMS.pitch_to_id[p] for p in pitches]
            drums_tensor[i, ids] = 1
        return torch.cat([rules_tensor, drums_tensor], dim=1)

    def track_to_tensor_with_positions(
        self,
        track: Track,
        none_if_parsing_failed: bool = False,
    ) -> torch.Tensor | None:
        if "sequence" in track.other_data:
            sequence = track.other_data["sequence"]
        else:
            sequence = self.track_to_sequence(track, none_if_parsing_failed=none_if_parsing_failed)
        positions = list(range(len(sequence)))  # TODO: tree positional encoding?
        return self.sequence_to_tensor(sequence, track), positions

    # def export_last_parsed_file_to_svg(self, path: Path, delete_mei: bool = True):
    #     # Export MEI file to /tmp
    #     mei_path = Path(tempfile.mktemp(suffix=".mei"))
    #     self.scoring_env.create_voicing_drum("funk")
    #     self.scoring_env.export_MEI(str(mei_path))
    #     # Convert MEI to SVG
    #     tk = verovio.toolkit()
    #     tk.loadFile(str(mei_path))
    #     tk.renderToSVGFile(str(path))
    #     # Delete MEI file
    #     if delete_mei:
    #         mei_path.unlink()

    def _initialize_a_scoring_env(self):
        scoring_env = pyqparse.ScoringEnv()
        scoring_env.set_order(False)  # To get the best trees (equivalent of option -max)
        scoring_env.set_trs(pyqparse.Mode.Drum)
        scoring_env.import_WTA(str(self.grammar_file), pyqparse.WeightCode.Tropical)
        scoring_env.set_debug(False)
        return scoring_env

    def _build_tree(self, track: Track):
        scoring_env = self._initialize_a_scoring_env()
        # scoring_env = self.scoring_env
        scoring_env.import_file(
            filename=str(track.midi_path),
            format=pyqparse.Format.MIDI,
            mode=pyqparse.Mode.Drum,
            tracknb=0,
        )
        scoring_env.open_score(str(track.midi_path))
        assert track.time_signature == (4, 4)  # Only 4/4 supported for now
        scoring_env.update_time_sig(track.time_signature[0], track.time_signature[1])
        num_beat_in_measure = scoring_env.get_bar_beats()
        measure_duration = (60 / track.bpm) * num_beat_in_measure
        scoring_env.set_tempo(mod=pyqparse.Tempo.Constant, bardur=measure_duration)

        raw_tree = scoring_env.best_tree(pyqparse.Model.DrumAgostini, 1)
        scoring_env.close_score()
        tree = Node(raw_tree, self.grammar)
        # print(tree)
        return tree

    def _get_notes_from_indices(
        self,
        track: Track,
        indices: Iterable[int] | Iterable[Iterable | int],
    ) -> list[int | list]:
        """
        Get the notes from a list of indices. The list of indices can be an arbitrary nested list
        of iterables containing integers, and the shape of the structure will be preserved in the
        output.

        :param track: Track to get the notes from.
        :param indices: List (or nested lists) of indices.
        :return: List (or nested lists) of note pitches.
        """
        midi = load_and_validate_midi_file(
            track.midi_path,
            unique_track=True,
            unique_tempo=True,
            unique_time_signature=True,
            backend="mido",
        )

        midi_note_events = [event for event in midi.tracks[0] if event.type.startswith("note_")]
        import numpy as np

        indices = list(indices)

        tmp = np.concatenate(list(map(list, indices)))
        if tmp.max() >= len(midi_note_events):
            breakpoint()  # TODO: BLOCKED HERE
        # indices = list(indices)
        # print(indices)

        def _recursive_get_notes_from_indices(int_or_iter):
            # print(int_or_list)
            if isinstance(int_or_iter, int):
                if int_or_iter >= len(midi_note_events):
                    print(
                        f"Warning: index {int_or_iter} is out of range {len(midi_note_events)} for"
                        f" track {track.midi_path}"
                    )
                event = midi_note_events[int_or_iter]
                return event.note if event.type == "note_on" else None
            elif isinstance(int_or_iter, Iterable):
                return list(
                    filter(  # Filter for None (note_off events)
                        lambda x: x is not None,
                        map(_recursive_get_notes_from_indices, int_or_iter),
                    )
                )
            else:
                raise ValueError(f"Expected int or iterable, got {type(int_or_iter)}")

        note_pitches = _recursive_get_notes_from_indices(indices)
        # print(x)
        return note_pitches

    @property
    def output_dim(self) -> int:
        # As many features as there are drums ("height" of the piano roll)
        return self.grammar.num_simplified_rules() + len(DRUMS)

    def collate_fn(self, batch: list[ProcessedTrack]) -> dict[str, torch.Tensor]:
        """
        Filter out tracks that errored during processing and collate the rest.
        """
        filtered_batch = [pt for pt in batch if pt.processed_data is not None]
        return super().collate_fn(filtered_batch)


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     MIDI_DIR = Path("/tmp/augmented_tracks")
#     GRAMMAR_FILE = Path("utils/grammar.wta")
#     SVG_PATHS = Path("/tmp/svg")
#
#     Path("/tmp/trees").mkdir(exist_ok=True)
#     bpm_file = MIDI_DIR / "bpm.json"
#
#     import json
#
#     with open(bpm_file, "r") as f:
#         bpm_dict = json.load(f)
#
#     from tqdm import tqdm
#
#     processor = GrammarTrackProcessor(GRAMMAR_FILE, sequences_dir=Path("/tmp/joblib_cache"))
#
#     for midi_file in tqdm(sorted(MIDI_DIR.glob("**/*.mid"))):
#         print("MIDI file:", midi_file)
#         track = Track(
#             midi_path=midi_file,
#             bpm=bpm_dict[midi_file.name],
#             time_signature=(4, 4),
#         )
#         # print(processor.get_rules_and_indices(track))
#         # processor.export_last_parsed_file_to_svg(SVG_PATHS / f"{midi_file.stem}.svg")
#         tensor = processor.notes_to_tensor(track)
#         plt.imshow(tensor.T, aspect="auto", interpolation="nearest")
#         plt.grid()
#         plt.show()


if __name__ == "__main__":
    from utils.utils import parse_bpm_from_file_name, parse_time_signature_from_file_name

    # MIDI_PATH = Path("../groove_data/groove/drummer1/session1")
    # midi_paths = list(MIDI_PATH.rglob("*_funk_*.mid"))
    MIDI_PATH = Path("/tmp/augmented_tracks")
    midi_paths = list(MIDI_PATH.rglob("*.mid"))
    print(midi_paths)
    print(f"{len(midi_paths) = }")
    tracks = [
        Track(
            midi_path=midi_path,
            bpm=parse_bpm_from_file_name(midi_path.name),
            time_signature=parse_time_signature_from_file_name(midi_path.name),
        )
        for midi_path in midi_paths
    ]
    GRAMMAR_FILE = Path("utils/grammar.wta")

    processor = GrammarTrackProcessor(GRAMMAR_FILE, sequences_dir=None)

    for track in tracks:
        sequence = processor.track_to_sequence(track, none_if_parsing_failed=True)
        print(sequence)
        if sequence is not None:
            tensor = processor.sequence_to_tensor(sequence, track)
            print(tensor)
