from dataclasses import dataclass
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

from data_processing.track_processors import TrackProcessors


@dataclass(frozen=True)
class ScriptParameters:
    # Data parameters
    data_path: Path
    processed_data_path: Path
    results_path: Path
    # Track processor parameters
    track_processor_type: TrackProcessors
    track_processor_pianoroll_resolution: int
    # Training parameters
    training_batch_size: int
    training_num_epochs: int
    training_patience: int
    # Hyperparameters
    hp_lstm_hidden_size: int
    hp_lstm_num_layers: int
    hp_dropout_rate: float
    hp_lr: float
    hp_reg: float
    hp_num_measures: int
    hp_measures_overlap: bool
    hp_imfw: bool
    hp_use_tempo: bool
    hp_use_augmented_dataset: bool
    hp_num_augmentations: int
    # Others
    min_tracks_per_label: int

    def __str__(self):
        return pformat(self.__dict__)

    def to_serializable_dict(self) -> dict:
        return {
            k: v if isinstance(v, (int, float, str, bool)) else str(v)
            for k, v in self.__dict__.items()
        }


def bool_from_str(s: str) -> bool:
    if s.lower() in ['true', '1']:
        return True
    elif s.lower() in ['false', '0']:
        return False
    else:
        raise ValueError(f'Cannot convert {s} to bool.')


def parse_args() -> ScriptParameters:
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, allow_abbrev=False)

    # Data group
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data-path', type=Path, required=True, help='Path to the dataset.')
    data_group.add_argument('--processed-data-path', type=Path, required=True, help='Path to stored the processed data.')
    data_group.add_argument('--results-path', type=Path, required=True, help='Path to the results.')

    # Track processor group
    track_processor_group = parser.add_argument_group('Track processor')
    track_processor_group.add_argument('--track-processor', type=TrackProcessors.from_str, required=True, choices=list(TrackProcessors), help='Track processor type.')
    track_processor_group.add_argument('--track-processor-pianoroll-resolution', type=int, default=30, help=f'Pianoroll resolution in steps per second (only for {TrackProcessors.pianoroll}).')

    # Training group
    training_group = parser.add_argument_group('Training')
    training_group.add_argument('--training-batch-size', type=int, default=128, help='Batch size for training.')
    training_group.add_argument('--training-num-epochs', type=int, default=100, help='Number of epochs for training.')
    training_group.add_argument('--training-patience', type=int, default=40, help='Number of epochs for training.')

    # Hyperparameters group
    hp_group = parser.add_argument_group('Hyperparameters')
    hp_group.add_argument('--hp-lstm-hidden-size', type=int, default=32, help='Number of hidden units in the LSTM layer.')
    hp_group.add_argument('--hp-lstm-num-layers', type=int, default=2, help='Number of layers in the LSTM layer.')
    hp_group.add_argument('--hp-dropout-rate', type=float, default=0.3, help='Dropout rate.')
    hp_group.add_argument('--hp-lr', type=float, default=1e-3, help='Learning rate.')
    hp_group.add_argument('--hp-reg', type=float, default=1e-4, help='Regularization parameter.')
    hp_group.add_argument('--hp-num-measures', type=int, default=1, help='Number of measures to use.')
    hp_group.add_argument('--hp-measures-overlap', type=bool_from_str, help='Whether to use overlapping measures.', default=False)
    hp_group.add_argument('--hp-imfw', type=bool_from_str, help='Whether to use the IMFW dataset.', default=False)
    hp_group.add_argument('--hp-use-tempo', type=bool_from_str, help='Whether to use the tempo.', default=True)
    hp_group.add_argument('--hp-use-augmented-dataset', type=bool_from_str, help='Whether to use the augmented dataset.', default=True)
    hp_group.add_argument('--hp-num-augmentations', type=int, default=2, help='Number of augmentations to use.')

    # Others group
    others_group = parser.add_argument_group('Others')
    others_group.add_argument('--min-tracks-per-label', type=int, default=30, help='Minimum number of tracks per label.')

    args = parser.parse_args()

    script_params = ScriptParameters(
        data_path=args.data_path,
        processed_data_path=args.processed_data_path,
        results_path=args.results_path,
        track_processor_type=args.track_processor,
        track_processor_pianoroll_resolution=args.track_processor_pianoroll_resolution,
        training_batch_size=args.training_batch_size,
        training_num_epochs=args.training_num_epochs,
        training_patience=args.training_patience,
        hp_lstm_hidden_size=args.hp_lstm_hidden_size,
        hp_lstm_num_layers=args.hp_lstm_num_layers,
        hp_dropout_rate=args.hp_dropout_rate,
        hp_lr=args.hp_lr,
        hp_reg=args.hp_reg,
        hp_num_measures=args.hp_num_measures,
        hp_measures_overlap=args.hp_measures_overlap,
        hp_imfw=args.hp_imfw,
        hp_use_tempo=args.hp_use_tempo,
        hp_use_augmented_dataset=args.hp_use_augmented_dataset,
        hp_num_augmentations=args.hp_num_augmentations,
        min_tracks_per_label=args.min_tracks_per_label,
    )

    return script_params




if __name__ == '__main__':
    print(parse_args())
