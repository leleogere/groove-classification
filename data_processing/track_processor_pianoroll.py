from pathlib import Path

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
        positions = (np.arange(0, num_steps) / self.resolution).tolist()
        return piano_roll.transpose(0, 1), positions

    @property
    def output_dim(self) -> int:
        # As many features as there are drums ("height" of the piano roll)
        return len(self.drums)


class PianoRollTrackProcessorNoVelocity(PianoRollTrackProcessor):
    short_name = "pianoroll_no_velocity"

    def track_to_tensor_with_positions(self, track: Track):
        tensor, positions = super().track_to_tensor_with_positions(track)
        # binarize the tensor
        tensor = (tensor > 0).float()
        return tensor, positions


class PianoRollTrackProcessorNoPitches(PianoRollTrackProcessor):
    short_name = "pianoroll_no_pitches"

    def track_to_tensor_with_positions(self, track: Track):
        tensor, positions = super().track_to_tensor_with_positions(track)
        # remove pitch information by flattening the tensor (take mean when multiple notes at same time?)
        num_notes = (tensor > 0).sum(dim=1, keepdim=True).float()
        num_notes[num_notes == 0] = 1  # to avoid division by zero
        tensor = tensor.sum(dim=1, keepdim=True) / num_notes
        return tensor, positions

    @property
    def output_dim(self) -> int:
        # Only one as pitches flattened
        return 1


class PianoRollTrackProcessorNoPitchesNoVelocity(PianoRollTrackProcessor):
    short_name = "pianoroll_no_pitches_no_velocity"

    def track_to_tensor_with_positions(self, track: Track):
        tensor, positions = super().track_to_tensor_with_positions(track)
        # remove pitch and velocity information by flattening the tensor (keep number of notes)
        tensor = (tensor > 0).sum(dim=1, keepdim=True).float()
        return tensor, positions

    @property
    def output_dim(self) -> int:
        # Only one as pitches flattened
        return 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    midi = Path(
        "../../papiers/groove_classification/example/example2_notefixed_durationfixed_randomized.mid"
    )
    track = Track(midi, bpm=80)

    for processor in [
        PianoRollTrackProcessor,
        PianoRollTrackProcessorNoVelocity,
        PianoRollTrackProcessorNoPitches,
        PianoRollTrackProcessorNoPitchesNoVelocity,
    ]:
        processor = processor()
        tensor, positions = processor.track_to_tensor_with_positions(track)
        print(tensor.shape)

        plt.figure()
        plt.imshow(tensor.transpose(0, 1), aspect="auto")
        plt.yticks(range(len(processor.drums)), range(len(processor.drums)))
        plt.xlabel("Time steps")
        plt.ylabel("Drum ID")
        plt.title(processor.short_name)
        plt.grid(alpha=0.3)
        plt.show(block=False)

    plt.show()


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     midi = Path("../../papiers/groove_classification/example/example2_fixed_durations_fixed.mid")
#     track = Track(midi, bpm=80)
#
#     processor = PianoRollTrackProcessor()
#     tensor, positions = processor.track_to_tensor_with_positions(track)
#     print(tensor.shape)
#
#     plt.imshow(tensor.transpose(0, 1), aspect="auto")
#     # range on yaxis
#     # plt.yticks(range(len(processor.drums)), [drum.name for drum in processor.drums])
#     plt.yticks(range(len(processor.drums)), range(len(processor.drums)))
#     # Titles
#     plt.xlabel("Time steps")
#     plt.ylabel("Drum ID")
#     plt.title("Piano Roll")
#     # Grid
#     plt.grid(alpha=0.3)
#     plt.show()
