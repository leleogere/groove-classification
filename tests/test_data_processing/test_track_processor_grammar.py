from importlib.resources import files
from pathlib import Path
import unittest

import numpy as np

from data_processing.track_processor_grammar import GrammarTrackProcessor
from utils.utils import Track


class TestTrackProcessorGrammar(unittest.TestCase):
    def setUp(self):
        self.processor_tree_encoding = GrammarTrackProcessor(
            weights_file=Path(files("tests.resources").joinpath("grammar.wta")),
            tree_positional_encoding=True,
        )
        self.processor_seq_encoding = GrammarTrackProcessor(
            weights_file=Path(files("tests.resources").joinpath("grammar.wta")),
            tree_positional_encoding=False,
        )
        self.track = Track(
            midi_path=Path(files("tests.resources").joinpath("midi.mid")),
            bpm=80,
            time_signature=(4, 4),
        )

    def test_str_state(self):
        self.assertEqual(
            self.processor_tree_encoding.str_state(),
            "GrammarTrackProcessor_-_use_tempo=True_-_grammar_file=grammar.wta_-_tree_positional_encoding=True",
        )

    def test_output_dim(self):
        self.assertEqual(self.processor_tree_encoding.output_dim, 37)

    def test_track_to_tensor_with_positions_with_tree_encoding(self):
        tensor, positions = self.processor_tree_encoding.track_to_tensor_with_positions(self.track)
        self.assertEqual(tensor.shape, (48, 37))
        self.assertEqual(len(positions), len(tensor))
        self.assertTrue(
            np.allclose(tensor, np.load(files("tests.resources").joinpath("midi_grammar.npy")))
        )
        self.assertTrue(
            np.allclose(
                positions, np.load(files("tests.resources").joinpath("midi_tree_positions.npy"))
            )
        )

    def test_track_to_tensor_with_positions_with_seq_encoding(self):
        tensor, positions = self.processor_seq_encoding.track_to_tensor_with_positions(self.track)
        self.assertEqual(tensor.shape, (48, 37))
        self.assertEqual(len(positions), len(tensor))
        self.assertTrue(
            np.allclose(tensor, np.load(files("tests.resources").joinpath("midi_grammar.npy")))
        )
        self.assertListEqual(positions, list(range(48)))
