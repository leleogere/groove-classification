import copy

import torch
from torch import nn

from utils.constants import DRUMS
from utils.utils import ProcessedTrack
from utils.utils import Track


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

    def str_state(self) -> str:
        """String representation of the state of the track processor.

        :return: String representation of the state of the track processor.
        """
        return "_-_".join([self.name, f"use_tempo={self.use_tempo}"])

    def track_to_tensor_with_positions(
        self,
        track: Track,
    ) -> tuple[torch.Tensor, list[float | int]]:
        """
        Operation that turns the *note* events of a track into a tensor. The shape of the
        resulting tensor should be (num_notes, num_features). This function also return the
        position (whatever that means for this processor) of each element in the tensor.

        :param track: Track to process.
        :return: A tuple containing the tensor representation of a track and a list of the position
            of each of its elements.
        """
        raise NotImplementedError

    def track_to_tensor(self, track: Track) -> torch.Tensor:
        """Operation that turn the *note* events of a track into a tensor. The shape of the
        resulting tensor should be (num_notes, num_features).

        :param track: Track to process.
        :return: Tensor representation of the track's notes.
        """
        return self.track_to_tensor_with_positions(track)[0]

    def process_track(self, track: Track) -> ProcessedTrack:
        """Process a track.

        :param track: Track to process.
        :return: Processed track, containing the tensor representation of the track's notes, tempo
            and label.
        """
        tensor, positions = self.track_to_tensor_with_positions(track)
        track_attributes_copy = copy.deepcopy(track.__dict__)
        track_attributes_copy["other_data"]["positions"] = positions
        return ProcessedTrack(processed_data=self.track_to_tensor(track), **track_attributes_copy)

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

    def collate_positions_fn(
        self,
        batch_of_positions: list[list[float | int]] | list[list[list[float | int]]],
        max_len: int = None,
    ) -> torch.Tensor:
        """Collate function that stacks a list of positions.

        :param batch_of_positions: Batch of positions.
        :param max_len: Maximum length of the sequences if already known. If None, the maximum
            length of the sequences in the batch is used.
        :return: Stacked tensor of shape (batch_size, num_notes).
        """
        if all(pos is None for pos in batch_of_positions):
            return None
        if max_len is None:
            max_len = max(len(positions) for positions in batch_of_positions)
        pad_maker = (
            (lambda pos: (max_len - len(pos), 0))
            if isinstance(batch_of_positions[0][0], int | float)
            else (lambda pos: (0, 0, max_len - len(pos), 0))  # for vector pos like tree encoding
        )
        padded = [
            nn.functional.pad(
                input=torch.FloatTensor(positions),
                pad=pad_maker(positions),
                value=self.padding_value,
            )
            for positions in batch_of_positions
        ]
        stacked = torch.stack(padded)
        return stacked

    def collate_fn(self, batch: list[ProcessedTrack]) -> dict[str, torch.Tensor]:
        """Collate function that stacks the attributes of a batch of tracks.

        :param batch: Batch of processed tracks (computed by self.process_track).
        :return: Stacked tensors of shape (batch_size, ...).
        """
        collated_samples = self.collate_samples_fn([pt.processed_data for pt in batch])
        results = {
            "midi_paths": [pt.midi_path for pt in batch],
            "samples": collated_samples,
            "labels": self.collate_labels_fn([pt.label_idx for pt in batch]),
            "_positions": self.collate_positions_fn(
                [pt.other_data["positions"] for pt in batch], max_len=collated_samples.shape[1]
            ),
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
