from collections import defaultdict
from enum import Enum

import torch
from torch import nn, LongTensor

from utils.classes import Track, RelativeNote, ProcessedTrack
from utils.constants import DRUMS


class TrackProcessor:
    short_name = 'abstract_track_processor'

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def notes_to_tensor(self, track: Track) -> torch.Tensor:
        raise NotImplementedError

    def process_track(self, track: Track) -> ProcessedTrack:
        return ProcessedTrack(
            data=self.notes_to_tensor(track),
            **track.__dict__,
        )

    def collate_samples_fn(self, batch: list[ProcessedTrack]) -> torch.Tensor:
        return NotImplementedError

    def collate_labels_fn(self, batch: list[ProcessedTrack | Track]) -> torch.Tensor:
        return LongTensor([sample.label_idx for sample in batch])

    def collate_fn(self, batch: list[ProcessedTrack]) -> dict[str, torch.Tensor]:
        return {
            'samples': self.collate_samples_fn(batch),
            'labels': self.collate_labels_fn(batch),
        }

    def output_dim(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.process_track(*args, **kwargs)


class TrackProcessorWithTempo(TrackProcessor):
    def __init__(self, use_tempo: bool = True):
        super().__init__()
        self.use_tempo = use_tempo

    def tempo_to_tensor(self, track: Track) -> torch.Tensor:
        return torch.tensor(track.bpm, dtype=torch.float32)

    def collate_tempos_fn(self, batch: list[ProcessedTrack | Track]) -> torch.Tensor:
        return torch.stack([self.tempo_to_tensor(sample) for sample in batch])

    def collate_fn(self, batch: list[ProcessedTrack]) -> dict[str, torch.Tensor]:
        results = super().collate_fn(batch)
        if self.use_tempo:
            results['tempos'] = self.collate_tempos_fn(batch)
        return results


class RelativeMultivariateTrackProcessor(TrackProcessorWithTempo):
    short_name = 'multivariate'
    velocity_norm = 127  # max velocity
    time_norm = 100  # so duration and timeshift are closer to 0-5
    drums = DRUMS

    def __init__(self, use_tempo: bool = True):
        super().__init__(use_tempo=use_tempo)

    def notes_to_tensor(self, track: Track):
        note_attributes = defaultdict(list)
        for note in track.relative_notes():
            for key in RelativeNote.__slots__:
                note_attributes[key].append(getattr(note, key))
            note_attributes['id'].append(self.drums.pitch_to_id[note.pitch])

        vector = torch.cat(
            [
                torch.nn.functional.one_hot(torch.tensor(note_attributes['id']), num_classes=len(self.drums)).float(),
                torch.tensor(note_attributes["velocity"], dtype=torch.float).unsqueeze(-1) / self.velocity_norm,
                torch.tensor(note_attributes["duration"], dtype=torch.float).unsqueeze(-1) / self.time_norm,
                torch.tensor(note_attributes["timeshift"], dtype=torch.float).unsqueeze(-1) / self.time_norm,
            ],
            dim=-1
        )
        return vector

    def collate_samples_fn(self, batch: list[ProcessedTrack]) -> torch.Tensor:
        x = [sample.data for sample in batch]
        padding_value = 0
        max_len = max(sample.shape[-2] for sample in x)
        padded = [nn.functional.pad(seq, (0, 0, max_len - seq.shape[-2], 0), value=padding_value) for seq in x]
        stacked = torch.stack(padded)
        return stacked

    def output_dim(self):
        return (len(self.drums) + 3,)


class PianoRollTrackProcessor(RelativeMultivariateTrackProcessor):
    short_name = 'pianoroll'
    drums = DRUMS

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
            piano_roll[self.drums.pitch_to_id[note.pitch], i_start:i_end] = note.velocity / self.velocity_norm
        return piano_roll.transpose(0, 1)  # change to (timestep, features)

    def output_dim(self):
        return (len(self.drums),)


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
            raise ValueError()
