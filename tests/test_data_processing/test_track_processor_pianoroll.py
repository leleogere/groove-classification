from importlib.resources import files
from pathlib import Path
import unittest

import numpy as np

from data_processing.track_processor_pianoroll import PianoRollTrackProcessor
from utils.utils import Track


class TestTrackProcessorPianoroll(unittest.TestCase):
    def setUp(self):
        self.processor = PianoRollTrackProcessor(resolution=20)
        self.track = Track(midi_path=Path(files("tests.resources").joinpath("midi.mid")), bpm=80)

    def test_str_state(self):
        self.assertEqual(
            self.processor.str_state(), "PianoRollTrackProcessor_-_use_tempo=True_-_resolution=20"
        )

    def test_output_dim(self):
        self.assertEqual(self.processor.output_dim, 22)

    def test_track_to_tensor_with_positions(self):
        tensor, positions = self.processor.track_to_tensor_with_positions(self.track)
        self.assertEqual(tensor.shape, (118, 22))
        self.assertEqual(len(positions), len(tensor))
        self.assertTrue(
            np.allclose(tensor, np.load(files("tests.resources").joinpath("midi_pianoroll.npy")))
        )
        self.assertTrue(np.allclose(positions, [i / self.processor.resolution for i in range(118)]))
