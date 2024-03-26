import numpy as np
import torch

from data_processing.track_processors_base import TrackProcessor
from utils.utils import Track


class PianoRollTrackProcessor(TrackProcessor):
    short_name = "pianoroll"

    def __init__(self, resolution: int = 20, use_tempo: bool = True):
        super().__init__(use_tempo=use_tempo)
        self.resolution = resolution

    def str_state(self) -> str:
        """String representation of the state of the track processor.

        :return: String representation of the state of the track processor.
        """
        return "_-_".join(
            [
                self.name,
                f"use_tempo={self.use_tempo}",
                f"resolution={self.resolution}",
            ]
        )

    def track_to_tensor_with_positions(self, track: Track):
        notes = track.notes()
        start = notes[0].start
        end = max(note.end for note in notes)  # last starting note might not be the last ending one
        length = end - start
        num_steps = int(length * self.resolution)
        piano_roll = torch.zeros(len(self.drums), num_steps)
        for note in notes:
            i_start = int((note.start - start) * self.resolution)
            i_end = int((note.end - start) * self.resolution)
            piano_roll[self.drums.pitch_to_id[note.pitch], i_start:i_end] = (
                note.velocity / self.velocity_norm
            )
        positions = np.linspace(0, length, num_steps).tolist()
        return piano_roll.transpose(0, 1), positions

    @property
    def output_dim(self) -> int:
        # As many features as there are drums ("height" of the piano roll)
        return len(self.drums)
