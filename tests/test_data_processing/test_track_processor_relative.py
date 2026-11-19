from importlib.resources import files
from pathlib import Path
import unittest

import numpy as np

from data_processing.track_processor_relative import RelativeMultivariateTrackProcessor
from utils.utils import Track


class TestTrackProcessorRelative(unittest.TestCase):
    def setUp(self):
        self.processor = RelativeMultivariateTrackProcessor()
        self.track = Track(midi_path=Path(files("tests.resources").joinpath("midi.mid")), bpm=80)

    def test_str_state(self):
        self.assertEqual(
            self.processor.str_state(), "RelativeMultivariateTrackProcessor_-_use_tempo=True"
        )

    def test_output_dim(self):
        self.assertEqual(self.processor.output_dim, 25)

    def test_track_to_tensor_with_positions(self):
        tensor, positions = self.processor.track_to_tensor_with_positions(self.track)
        self.assertEqual(tensor.shape, (44, 25))
        self.assertEqual(len(positions), len(tensor))
        self.assertTrue(
            np.allclose(tensor, np.load(files("tests.resources").joinpath("midi_relative.npy")))
        )
        self.assertTrue(
            np.allclose(
                positions, np.load(files("tests.resources").joinpath("midi_relative_positions.npy"))
            )
        )
