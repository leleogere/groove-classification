from collections import defaultdict
from enum import Enum

import torch
from torch import LongTensor
from torch import nn

from utils.classes import ProcessedTrack
from utils.classes import RelativeNote
from utils.classes import Track
from utils.constants import DRUMS


class TrackProcessor:
    short_name: str = None
    padding_value = 0
    velocity_norm = 127  # max velocity
    drums = DRUMS

    def __init__(self, use_tempo: bool = True):
        """Operator that defines how to transform a Track into a tensor.

        :param use_tempo: Whether to use the tempo as an input feature.
        """
        self.use_tempo = use_tempo

    @property
    def name(self) -> str:
        """Name of the track processor.

        :return: Name of the track processor.
        """
        return self.__class__.__name__

    def notes_to_tensor(self, track: Track) -> torch.Tensor:
        """Operation that turn the *note* events of a track into a tensor. The shape of the
        resulting tensor should be (num_notes, num_features).

        :param track: Track to process.
        :return: Tensor representation of the track's notes.
        """
        raise NotImplementedError

    def process_track(self, track: Track) -> ProcessedTrack:
        """Process a track.

        :param track: Track to process.
        :return: Processed track, containing the tensor representation of the track's notes, tempo
            and label.
        """
        return ProcessedTrack(
            notes_tensor=self.notes_to_tensor(track),
            **track.__dict__,
        )

    def collate_samples_fn(self, batch_of_notes_tensors: list[torch.Tensor]) -> torch.Tensor:
        """Collate function that stacks the note attributes of a batch of tracks.

        :param batch_of_notes_tensors: Batch of notes tensors of shape (num_notes, num_features)
            (computed by self.notes_to_tensor).
        :return: Stacked tensor of shape (batch_size, num_notes, num_features).
        """
        if any(sample.ndim != 2 for sample in batch_of_notes_tensors):
            raise ValueError("All tensors must have 2 dimensions (num_notes, num_features).")
        # Pad all sequence to the same by adding vectors of self.padding_value (of size
        # num_features) at the *beginning* of each sequence (see https://arxiv.org/abs/1903.07288).
        max_sequence_length = max(notes_tensor.shape[0] for notes_tensor in batch_of_notes_tensors)
        padded = [
            nn.functional.pad(
                input=notes_tensor,
                pad=(0, 0, max_sequence_length - notes_tensor.shape[0], 0),
                value=self.padding_value,
            )
            for notes_tensor in batch_of_notes_tensors
        ]
        stacked = torch.stack(padded)
        return stacked

    def collate_tempos_fn(self, batch_of_bpm: list[float]) -> torch.Tensor:
        """Collate function that stacks a list of bpm.

        :param batch_of_bpm: Batch of bpm.
        :return: Stacked tensor of shape (batch_size, 1).
        """
        return torch.FloatTensor(batch_of_bpm).unsqueeze(-1)

    def collate_labels_fn(self, batch_of_label_idxs: list[int]) -> torch.Tensor:
        """Collate function that stacks a list of label_idx. Note that the label_idx must be
        encoded as integers beforehand.

        :param batch_of_label_idxs: Batch of label indices.
        :return: Stacked tensor of shape (batch_size,).
        """
        return torch.LongTensor(batch_of_label_idxs)

    def collate_fn(self, batch: list[ProcessedTrack]) -> dict[str, torch.Tensor]:
        """Collate function that stacks the attributes of a batch of tracks.

        :param batch: Batch of processed tracks (computed by self.process_track).
        :return: Stacked tensors of shape (batch_size, ...).
        """
        results = {
            "samples": self.collate_samples_fn([pt.notes_tensor for pt in batch]),
            "labels": self.collate_labels_fn([pt.label_idx for pt in batch]),
        }
        if self.use_tempo:
            results["tempos"] = self.collate_tempos_fn([pt.bpm for pt in batch])
        return results

    @property
    def output_dim(self) -> int:
        """Output dimension of the track processor.

        :return: Output dimension of the track processor."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.process_track(*args, **kwargs)


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
            # 22 first components are the one-hot encoding of the drum id
            torch.nn.functional.one_hot(
                torch.tensor(note_attributes["id"]), num_classes=len(self.drums)
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


class PianoRollTrackProcessor(TrackProcessor):
    short_name = "pianoroll"

    def __init__(self, resolution: int = 20, use_tempo: bool = True):
        super().__init__(use_tempo=use_tempo)
        self.resolution = resolution

    def notes_to_tensor(self, track: Track):
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
        return piano_roll.transpose(0, 1)  # change to (timestep, features)

    @property
    def output_dim(self) -> int:
        # As many features as there are drums ("height" of the piano roll)
        return len(self.drums)


class TrackProcessors(Enum):
    multivariate = RelativeMultivariateTrackProcessor
    pianoroll = PianoRollTrackProcessor

    def __str__(self):
        return self.name

    @staticmethod
    def from_str(s: str):
        try:
            return TrackProcessors[s]
        except KeyError:
            raise ValueError(
                f"Cannot convert {s} to TrackProcessors. Valid values are {list(TrackProcessors)}."
            )
