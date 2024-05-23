from enum import Enum

from data_processing.track_processor_grammar import GrammarTrackProcessor
from data_processing.track_processor_grammar import GrammarTrackProcessorNoPitches
from data_processing.track_processor_grammar import GrammarTrackProcessorNoPitchesNoVelocity
from data_processing.track_processor_grammar import GrammarTrackProcessorNoVelocity
from data_processing.track_processor_pianoroll import PianoRollTrackProcessor
from data_processing.track_processor_pianoroll import PianoRollTrackProcessorNoPitches
from data_processing.track_processor_pianoroll import PianoRollTrackProcessorNoPitchesNoVelocity
from data_processing.track_processor_pianoroll import PianoRollTrackProcessorNoVelocity
from data_processing.track_processor_relative import RelativeMultivariateTrackProcessor
from data_processing.track_processor_relative import RelativeMultivariateTrackProcessorNoPitches
from data_processing.track_processor_relative import (
    RelativeMultivariateTrackProcessorNoPitchesNoVelocity,
)
from data_processing.track_processor_relative import RelativeMultivariateTrackProcessorNoVelocity
from data_processing.track_processor_tokenizer import TokenizerTrackProcessor
from data_processing.track_processor_tokenizer import TokenizerTrackProcessorNoPitches
from data_processing.track_processor_tokenizer import TokenizerTrackProcessorNoPitchesNoVelocity
from data_processing.track_processor_tokenizer import TokenizerTrackProcessorNoVelocity


class TrackProcessors(Enum):
    relative = RelativeMultivariateTrackProcessor
    pianoroll = PianoRollTrackProcessor
    grammar = GrammarTrackProcessor
    tokenizer = TokenizerTrackProcessor

    relative_no_velocity = RelativeMultivariateTrackProcessorNoVelocity
    relative_no_pitches = RelativeMultivariateTrackProcessorNoPitches
    relative_no_pitches_no_velocity = RelativeMultivariateTrackProcessorNoPitchesNoVelocity

    pianoroll_no_velocity = PianoRollTrackProcessorNoVelocity
    pianoroll_no_pitches = PianoRollTrackProcessorNoPitches
    pianoroll_no_pitches_no_velocity = PianoRollTrackProcessorNoPitchesNoVelocity

    tokenizer_no_velocity = TokenizerTrackProcessorNoVelocity
    tokenizer_no_pitches = TokenizerTrackProcessorNoPitches
    tokenizer_no_pitches_no_velocity = TokenizerTrackProcessorNoPitchesNoVelocity

    grammar_no_velocity = GrammarTrackProcessorNoVelocity
    grammar_no_pitches = GrammarTrackProcessorNoPitches
    grammar_no_pitches_no_velocity = GrammarTrackProcessorNoPitchesNoVelocity

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
