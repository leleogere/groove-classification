from importlib.resources import files
from pathlib import Path
import unittest

import numpy as np

from data_processing.track_processor_tokenizer import TokenizerTrackProcessor
from utils.utils import Track


class TestTrackProcessorTokenizer(unittest.TestCase):
    def setUp(self):
        self.processor = TokenizerTrackProcessor(type="midilike")
        self.track = Track(midi_path=Path(files("tests.resources").joinpath("midi.mid")), bpm=80)

    def test_str_state(self):
        self.assertEqual(
            self.processor.str_state(), "TokenizerTrackProcessor_-_type=midilike_-_use_tempo=True"
        )

    def test_output_dim(self):
        self.assertEqual(self.processor.output_dim, 1)

    def test_track_to_tensor_with_positions(self):
        tensor, positions = self.processor.track_to_tensor_with_positions(self.track)
        self.assertEqual(tensor.shape, (181, 1))
        self.assertEqual(len(positions), len(tensor))
        self.assertTrue(
            np.allclose(tensor, np.load(files("tests.resources").joinpath("midi_tokenizer.npy")))
        )
        self.assertTrue(np.allclose(positions, range(181)))
