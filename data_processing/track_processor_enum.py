from enum import Enum

from data_processing.track_processor_grammar import GrammarTrackProcessor
from data_processing.track_processor_pianoroll import PianoRollTrackProcessor
from data_processing.track_processor_relative import RelativeMultivariateTrackProcessor


class TrackProcessors(Enum):
    relative = RelativeMultivariateTrackProcessor
    pianoroll = PianoRollTrackProcessor
    grammar = GrammarTrackProcessor

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
