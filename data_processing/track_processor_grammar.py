import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

import joblib
import pyqparse
import torch

from data_processing.track_processors_base import TrackProcessor
from utils.constants import DRUMS
from utils.grammar import Grammar
from utils.grammar import Node
from utils.utils import Note
from utils.utils import ProcessedTrack
from utils.utils import Track
from utils.utils import load_and_validate_midi_file

pyqparse.set_verbosity_level(1)

RuleWithIndices = tuple[int, tuple[int, int] | None]


class GrammarTrackProcessor(TrackProcessor):
    short_name = "grammar"

    def __init__(
        self,
        weights_file: Path,
        use_tempo: bool = True,
        sequences_dir: Path = None,
        tree_positional_encoding: bool = False,
    ):
        super().__init__(use_tempo=use_tempo)
        self.grammar_file = weights_file
        self.grammar = Grammar.from_file(self.grammar_file)
        # self.scoring_env.set_order(False)  # To get the best trees (equivalent of option -max)
        # self.scoring_env.set_trs(pyqparse.Mode.Drum)
        # self.scoring_env.import_WTA(str(self.grammar_file), pyqparse.WeightCode.Tropical)
        # self.scoring_env.set_debug(False)
        self.tree_positional_encoding = tree_positional_encoding
        if sequences_dir is not None:
            self.sequences_dir = sequences_dir / self.str_state()
            self.memory = joblib.Memory(self.sequences_dir, verbose=0)
            self.track_to_tree = self.memory.cache(self.track_to_tree, ignore=["self"])
            # self.tree_to_tensor = self.memory.cache(self.tree_to_tensor, ignore=["self"])

    def str_state(self) -> str:
        """String representation of the state of the track processor.

        :return: String representation of the state of the track processor.
        """
        return "_-_".join(
            [
                self.name,
                f"use_tempo={self.use_tempo}",
                f"grammar_file={self.grammar_file.name}",
                f"tree_positional_encoding={self.tree_positional_encoding}",
            ]
        )

    def track_to_tree(
        self,
        track: Track,
        none_if_parsing_failed: bool = False,
    ) -> Node | None:
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
        # dealing with complex rhythms. This way, the segfault is contained in the subprocess, and
        # does not crash the main process.
        # Not very clean but no other choice for now.
        script_location = Path(__file__).parent.parent / "utils" / "_parse_midi_subprocess.py"
        environ = os.environ.copy()
        try:
            json_tree = subprocess.run(
                [
                    "python",
                    script_location,
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
            if none_if_parsing_failed:
                print(f"Error in tree built from {track.midi_path}: {e.stderr.decode()}")
                return None
            else:
                raise RuntimeError(f"Error in parsing {track.midi_path}: {e.stderr.decode()}")
        tree = Node.from_dict(json.loads(json_tree.stdout), self.grammar)
        return tree

    def tree_to_tensor(self, tree: Node, track: Track) -> torch.Tensor:
        """
        Convert a list of rules and with MIDI event indices to a tensor. The first n rows contain
        the OHE rules, and the next m rows contain the HE drums pitches (only for terminal rules).

        :param tree: Tree of rules.
        :param track: Track to get the notes from.
        :param rules: List of production rules.
        :param indices: List of indices in the format [start, end).
        :return: Tensor of shape (len(rules), num_simplified_rules + num_drum_pitches).
        """
        rules_and_indices = tree.traverse(
            lambda node: (
                node.id,
                (node.midi_start_index, node.midi_last_index) if node.has_midi_events else None,
            )
        )
        rules, indices = zip(*rules_and_indices)
        assert len(indices) == len(rules), "Length of rules and indices must be the same."
        # Convert [start, end) to range(start, end)
        indices = map(
            lambda start_end: range(start_end[0], start_end[1]) if start_end is not None else [],
            indices,
        )
        notes_per_rule: list[list[Note]] = self._get_notes_from_indices(track, indices)
        rules_tensor = torch.zeros(len(rules), self.grammar.num_simplified_rules())
        rules_tensor[range(len(rules)), rules] = 1
        drums_tensor = torch.zeros(len(rules), len(DRUMS))  # * 0.1  # For debugging in plots
        for i, notes in enumerate(notes_per_rule):
            ids = [DRUMS.pitch_to_id[n.pitch] for n in notes]
            drums_tensor[i, ids] = torch.FloatTensor(
                [n.velocity / self.velocity_norm for n in notes]
            )
        return torch.cat([rules_tensor, drums_tensor], dim=1)

    def track_to_tensor_with_positions(
        self,
        track: Track,
    ) -> torch.Tensor | None:
        if "tree" in track.other_data:
            tree = Node.from_dict(track.other_data["tree"], self.grammar)
        else:
            tree = self.track_to_tree(track)
        if self.tree_positional_encoding:
            positions = tree.get_positions()
        else:
            positions = tree.traverse(lambda node: node.id)
        return self.tree_to_tensor(tree, track), positions

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

    def _get_notes_from_indices(
        self,
        track: Track,
        indices: Iterable[int] | Iterable[Iterable | int],
    ) -> list[Note | list]:
        """
        Get the notes from a list of indices. The list of indices can be an arbitrary nested list
        of iterables containing integers, and the shape of the structure will be preserved in the
        output.

        :param track: Track to get the notes from.
        :param indices: List (or nested lists) of indices.
        :return: List (or nested lists) of notes.
        """
        midi = load_and_validate_midi_file(
            track.midi_path,
            unique_track=True,
            unique_tempo=True,
            unique_time_signature=True,
            backend="mido",
        )

        midi_note_events = [event for event in midi.tracks[0] if event.type.startswith("note_")]

        def _recursive_get_notes_from_indices(int_or_iter):
            # print(int_or_list)
            if isinstance(int_or_iter, int):
                if int_or_iter >= len(midi_note_events):
                    print(
                        f"Warning: index {int_or_iter} is out of range {len(midi_note_events)} for"
                        f" track {track.midi_path}"
                    )
                event = midi_note_events[int_or_iter]
                if event.type == "note_on" and event.velocity > 0:
                    return Note(start=None, end=None, pitch=event.note, velocity=event.velocity)
                else:
                    return None
            elif isinstance(int_or_iter, Iterable):
                return list(
                    filter(  # Filter for None (note_off events)
                        lambda x: x is not None,
                        map(_recursive_get_notes_from_indices, int_or_iter),
                    )
                )
            else:
                raise ValueError(f"Expected int or iterable, got {type(int_or_iter)}")

        notes = _recursive_get_notes_from_indices(indices)
        # print(x)
        return notes

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
    import matplotlib.pyplot as plt

    MIDI_PATH = Path("../groove_data/groove/drummer1/session1")
    # midi_paths = list(MIDI_PATH.rglob("*_funk_80_*.mid"))[:1]
    midi_paths = [
        Path(
            # "../groove_data/split_measures=4/groove/drummer1/session1/1_funk_80_beat_4-4/1_funk_80_beat_4-4_variation=original/1_funk_80_beat_4-4_variation=original_startbar=04_numbars=4.mid",
            "../groove_data/split_measures=4/groove/drummer1/session1/1_funk_80_beat_4-4/1_funk_80_beat_4-4_variation=original/1_funk_80_beat_4-4_variation=original_startbar=08_numbars=4.mid",
        )
    ]
    # MIDI_PATH = Path("/tmp/augmented_tracks")
    # midi_paths = list(MIDI_PATH.rglob("*.mid"))
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

    processor = GrammarTrackProcessor(
        GRAMMAR_FILE,
        sequences_dir=None,
        tree_positional_encoding=True,
    )
    from pprint import pprint

    pprint(processor.grammar.basic_rules)

    for track in tracks:
        tree = processor.track_to_tree(track, none_if_parsing_failed=True)
        print(tree.to_string_top_down())
        print(tree.to_string_left_right())
        if tree is not None:
            tensor, positions = processor.track_to_tensor_with_positions(track)
            print(torch.tensor(positions).shape)

            # subplots with colorbars
            fig, axs = plt.subplots(2)
            rules = axs[0].imshow(tensor.T, aspect="auto", interpolation="nearest")
            plt.colorbar(rules)
            pe = axs[1].imshow(torch.tensor(positions).T, aspect="auto", interpolation="nearest")
            plt.colorbar(pe, ticks=range(processor.grammar.max_depth))
            plt.show()
