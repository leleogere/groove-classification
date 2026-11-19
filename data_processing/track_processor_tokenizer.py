from pathlib import Path

import miditok
import symusic
import torch

from data_processing.track_processors_base import TrackProcessor
from utils.utils import Track


class TokenizerTrackProcessor(TrackProcessor):
    short_name = "tokenizer"
    allowed_types = ["midilike", "remi", "tsd", "structured"]

    def __init__(self, type: str, use_tempo: bool = True):
        super().__init__(use_tempo=use_tempo)
        self.type = type
        self.tokenizer = self._instantiate_tokenizer()

    def _instantiate_tokenizer(self) -> miditok.MIDITokenizer:
        config = miditok.TokenizerConfig(
            pitch_range=(self.drums.min_pitch(), self.drums.max_pitch()),
        )
        match self.type.lower():
            case "midilike":
                return miditok.MIDILike(config)
            case "remi":
                return miditok.REMI(config)
            case "tsd":
                return miditok.TSD(config)
            case "structured":
                return miditok.Structured(config)
            case _:
                raise ValueError(f"Unknown tokenizer: {self.type} (allowed: {self.allowed_types})")

    def str_state(self, separator: str = "_-_") -> str:
        """String representation of the state of the track processor.

        :return: String representation of the state of the track processor.
        """
        return separator.join(
            [
                self.name,
                f"type={self.type}",
                f"use_tempo={self.use_tempo}",
            ]
        )

    def track_to_tensor_with_positions(self, track: Track):
        midi = symusic.Score(track.midi_path)
        tokens = self.tokenizer.midi_to_tokens(midi)[0]
        return torch.tensor(tokens).unsqueeze(-1), list(range(len(tokens)))

    @property
    def output_dim(self) -> int:
        return 1


if __name__ == "__main__":
    from utils.utils import parse_bpm_from_file_name, parse_time_signature_from_file_name

    # MIDI_PATH = Path("../groove_data/groove/drummer1/session1")
    # midi_paths = list(MIDI_PATH.rglob("*_funk_80_*.mid"))[:1]
    midi_paths = [
        Path(
            # "../groove_data/split_measures=4/groove/drummer1/session1/1_funk_80_beat_4-4/1_funk_80_beat_4-4_variation=original/1_funk_80_beat_4-4_variation=original_startbar=04_numbars=4.mid",
            "../groove_data/split_measures=4/groove/drummer1/session1/1_funk_80_beat_4-4/1_funk_80_beat_4-4_variation=original/1_funk_80_beat_4-4_variation=original_startbar=08_numbars=4.mid",
            # "../groove_data/groove/drummer1/session1/1_funk_80_beat_4-4.mid",
        )
    ]
    # MIDI_PATH = Path("/tmp/augmented_tracks")
    # midi_paths = list(MIDI_PATH.rglob("*.mid"))
    print(midi_paths)
    print(f"{len(midi_paths) = }")
    tracks = [
        Track(
            midi_path=midi_path,
            bpm=parse_bpm_from_file_name(midi_path.name),
            time_signature=parse_time_signature_from_file_name(midi_path.name),
        )
        for midi_path in midi_paths
    ]

    processor = TokenizerTrackProcessor("midilike")

    for track in tracks:
        tensor = processor.track_to_tensor(track)
        print(tensor)
