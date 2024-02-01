from collections import defaultdict

import torch

from data_processing.track_processors_base import TrackProcessor
from utils.classes import RelativeNote
from utils.classes import Track


class RelativeMultivariateTrackProcessor(TrackProcessor):
    short_name = "multivariate"
    time_norm = 100  # so duration and timeshift are closer to 0-5

    def __init__(self, use_tempo: bool = True):
        super().__init__(use_tempo=use_tempo)

    def notes_to_tensor(self, track: Track):
        # Turn a list of dict (RelativeNote.__slots__) into a dict of lists.
        note_attributes = defaultdict(list)
        for note in track.relative_notes():
            for key in RelativeNote.__slots__:
                note_attributes[key].append(getattr(note, key))
            # Append drum id for one-hot encoding
            note_attributes["id"].append(self.drums.pitch_to_id[note.pitch])

        vector_components = [
            # the 22 first components are the one-hot encoding of the drum id
            torch.nn.functional.one_hot(
                torch.tensor(note_attributes["id"]),
                num_classes=len(self.drums),
            ).float(),
            # the 3 others are the normalized velocity, duration and timeshift
            *[
                torch.tensor(note_attributes[key], dtype=torch.float).unsqueeze(-1) / norm
                for key, norm in zip(
                    ["velocity", "duration", "timeshift"],
                    [self.velocity_norm, self.time_norm, self.time_norm],
                )
            ],
        ]
        vector = torch.cat(vector_components, dim=-1)
        return vector

    @property
    def output_dim(self) -> int:
        # As many features as there are drums + 3 (velocity, duration, timeshift)
        return len(self.drums) + 3
