import tempfile
from pathlib import Path

import joblib
import torch

import pyqparse
import verovio

from data_processing.track_processors_base import TrackProcessor
from utils.classes import Track
from utils.grammar import Node


class GrammarTrackProcessor(TrackProcessor):
    short_name = "pianoroll"

    def __init__(self, grammar_file: Path, use_tempo: bool = True, sequences_dir: Path = None):
        super().__init__(use_tempo=use_tempo)
        self.grammar_file = grammar_file
        self.scoring_env = self._initialize_a_scoring_env()
        # self.scoring_env.set_order(False)  # To get the best trees (equivalent of option -max)
        # self.scoring_env.set_trs(pyqparse.Mode.Drum)
        # self.scoring_env.import_WTA(str(self.grammar_file), pyqparse.WeightCode.Tropical)
        # self.scoring_env.set_debug(False)
        self.cache_dir = sequences_dir
        self.memory = joblib.Memory(self.cache_dir, verbose=1)
        # self.get_rules_and_indices = self.memory.cache(self.get_rules_and_indices)

    def _initialize_a_scoring_env(self):
        scoring_env = pyqparse.ScoringEnv()
        scoring_env.set_order(False)  # To get the best trees (equivalent of option -max)
        scoring_env.set_trs(pyqparse.Mode.Drum)
        scoring_env.import_WTA(str(self.grammar_file), pyqparse.WeightCode.Tropical)
        scoring_env.set_debug(False)
        return scoring_env

    def notes_to_tensor(self, track: Track):
        raise NotImplementedError

    def build_tree(self, track: Track):
        scoring_env = self.scoring_env
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
        tree = Node(raw_tree)
        print(tree)
        with open(f"/tmp/trees/{track.midi_path.name}.txt", "w") as f:
            # s = tree_to_string_with_offset(tree)
            # print(s)
            f.write(tree.to_expanded_tree_string(display_midi_indices=True))
        return tree

    def export_last_parsed_file_to_svg(self, path: Path, delete_mei: bool = True):
        # Export MEI file to /tmp
        mei_path = Path(tempfile.mktemp(suffix=".mei"))
        self.scoring_env.create_voicing_drum("funk")
        self.scoring_env.export_MEI(str(mei_path))
        # Convert MEI to SVG
        tk = verovio.toolkit()
        tk.loadFile(str(mei_path))
        tk.renderToSVGFile(str(path))
        # Delete MEI file
        if delete_mei:
            mei_path.unlink()

    def get_rules_and_indices(self, track: Track):
        tree = self.build_tree(track)
        return tree.get_production_rule_ids_and_midi_indices()

    @property
    def output_dim(self) -> int:
        # As many features as there are drums ("height" of the piano roll)
        raise NotImplementedError("n_rules + n_drums+ flam?")


# if __name__ == "__main__":
#     MIDI_FILE = Path("../groove_data/groove/drummer1/session1/1_funk_80_beat_4-4.mid")
#     GRAMMAR_FILE = Path("utils/grammar.wta")
#
#     track = Track(
#         midi_path=MIDI_FILE,
#         bpm=80,
#         time_signature=(4, 4),
#     )
#
#     processor = PianoRollTrackProcessor(GRAMMAR_FILE)
#     processor.create_tree(track)


if __name__ == "__main__":
    MIDI_DIR = Path("/tmp/augmented_tracks")
    GRAMMAR_FILE = Path("utils/grammar.wta")
    SVG_PATHS = Path("/tmp/svg")

    Path("/tmp/trees").mkdir(exist_ok=True)
    bpm_file = MIDI_DIR / "bpm.json"

    import json

    with open(bpm_file, "r") as f:
        bpm_dict = json.load(f)

    from tqdm import tqdm

    processor = GrammarTrackProcessor(GRAMMAR_FILE, sequences_dir=Path("/tmp/joblib_cache"))

    for midi_file in tqdm(sorted(MIDI_DIR.glob("**/*.mid"))):
        print("MIDI file:", midi_file)
        track = Track(
            midi_path=midi_file,
            bpm=bpm_dict[midi_file.name],
            time_signature=(4, 4),
        )
        print(processor.get_rules_and_indices(track))
        processor.export_last_parsed_file_to_svg(SVG_PATHS / f"{midi_file.stem}.svg")
