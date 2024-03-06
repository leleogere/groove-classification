from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

from data_processing.track_processor_enum import TrackProcessors
from models.genre_classifier import GenreClassifierType


@dataclass(frozen=True)
class ScriptParameters:
    # Data parameters
    data_path: Path
    processed_data_path: Path
    results_path: Path
    # Training parameters
    training_batch_size: int
    training_num_epochs: int
    training_patience: int
    training_learning_rate: float
    training_l2_regularization: float
    # Track processor parameters
    track_processor_type: TrackProcessors
    track_processor_pianoroll_resolution: int
    track_processor_grammar_weights_file: Path
    # Models parameters
    model_type: GenreClassifierType
    model_lstm_hidden_size: int
    model_lstm_num_layers: int
    model_transformer_num_heads: int
    model_transformer_num_features_per_head: int
    model_transformer_feedforward_dim: int
    model_transformer_num_layers: int
    model_dropout_rate: float
    # Data augmentation parameters
    num_measures_per_chunk: int
    chunk_overlap: bool
    num_augmentations: int
    # Others parameters
    min_tracks_per_label: int
    use_imfw: bool
    use_tempo: bool

    def __str__(self):
        return pformat(self.__dict__)

    def to_serializable_dict(self) -> dict:
        return {
            k: v if isinstance(v, (int, float, str, bool)) else str(v)
            for k, v in self.__dict__.items()
        }


def bool_from_str(s: str) -> bool:
    if s.lower() in ["true", "1"]:
        return True
    elif s.lower() in ["false", "0"]:
        return False
    else:
        raise ValueError(f"Cannot convert {s} to bool.")


def parse_args() -> ScriptParameters:
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, allow_abbrev=False)

    # Data group
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the dataset.",
    )
    data_group.add_argument(
        "--processed-data-path",
        type=Path,
        required=True,
        help="Path to stored the processed data.",
    )
    data_group.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to the results.",
    )

    # Training group
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--training-batch-size",
        type=int,
        default=128,
        help="Batch size for training.",
    )
    training_group.add_argument(
        "--training-num-epochs",
        type=int,
        default=100,
        help="Number of epochs for training.",
    )
    training_group.add_argument(
        "--training-patience",
        type=int,
        default=40,
        help="Number of epochs for training.",
    )
    training_group.add_argument(
        "--training-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    training_group.add_argument(
        "--training-l2-regularization",
        type=float,
        default=1e-4,
        help="Regularization parameter.",
    )

    # Track processor group
    track_processor_group = parser.add_argument_group("Track processor")
    track_processor_group.add_argument(
        "--track-processor",
        type=TrackProcessors.from_str,
        required=True,
        choices=list(TrackProcessors),
        help="Track processor type.",
    )
    track_processor_group.add_argument(
        "--track-processor-pianoroll-resolution",
        type=int,
        default=30,
        help=f"Pianoroll resolution in steps per second (only for {TrackProcessors.pianoroll}).",
    )
    track_processor_group.add_argument(
        "--track-processor-grammar-weights-file",
        type=Path,
        help=(
            "File containing the grammar and the weights associated to each rule (only for"
            f" {TrackProcessors.grammar})."
        ),
    )

    # Models group
    models_group = parser.add_argument_group("Models")
    models_group.add_argument(
        "--model",
        type=GenreClassifierType.from_str,
        required=True,
        choices=list(GenreClassifierType),
        help="Genre classifier type.",
    )
    models_group.add_argument(
        "--model-dropout-rate",
        type=float,
        default=0.3,
        help="Dropout rate.",
    )
    # LSTM
    models_group.add_argument(
        "--model-lstm-hidden-size",
        type=int,
        default=32,
        help="Number of hidden units in the LSTM layer.",
    )
    models_group.add_argument(
        "--model-lstm-num-layers",
        type=int,
        default=2,
        help="Number of layers in the LSTM layer.",
    )
    # Transformer
    models_group.add_argument(
        "--model-transformer-num-heads",
        type=int,
        default=4,
        help="Number of heads in the transformer.",
    )
    models_group.add_argument(
        "--model-transformer-num-features-per-head",
        type=int,
        default=4,
        help="Number of features per head in the transformer.",
    )
    models_group.add_argument(
        "--model-transformer-feedforward-dim",
        type=int,
        default=128,
        help="Feedforward dimension in the transformer.",
    )
    models_group.add_argument(
        "--model-transformer-num-layers",
        type=int,
        default=2,
        help="Number of layers in the transformer.",
    )

    # Data augmentation group
    augmentation_group = parser.add_argument_group("Hyperparameters")
    augmentation_group.add_argument(
        "--num-measures-per-chunk",
        type=int,
        default=1,
        help="Number of measures to use in each chunk.",
    )
    augmentation_group.add_argument(
        "--chunk-overlap",
        type=bool_from_str,
        help="Whether to use overlapping chunks.",
        default=False,
    )
    augmentation_group.add_argument(
        "--num-augmentations",
        type=int,
        default=0,
        help="Number of augmentations to use (0 for none).",
    )

    # Others group
    others_group = parser.add_argument_group("Others")
    others_group.add_argument(
        "--min-tracks-per-label",
        type=int,
        default=30,
        help="Minimum number of tracks per label.",
    )
    augmentation_group.add_argument(
        "--use-imfw",
        type=bool_from_str,
        help="Whether to weight the loss with the inverse median frequency of classes.",
        default=False,
    )
    augmentation_group.add_argument(
        "--use-tempo",
        type=bool_from_str,
        help="Whether let the model access the tempo.",
        default=True,
    )

    args = parser.parse_args()

    script_params = ScriptParameters(
        data_path=args.data_path,
        processed_data_path=args.processed_data_path,
        results_path=args.results_path,
        training_batch_size=args.training_batch_size,
        training_num_epochs=args.training_num_epochs,
        training_patience=args.training_patience,
        training_learning_rate=args.training_learning_rate,
        training_l2_regularization=args.training_l2_regularization,
        track_processor_type=args.track_processor,
        track_processor_pianoroll_resolution=args.track_processor_pianoroll_resolution,
        track_processor_grammar_weights_file=args.track_processor_grammar_weights_file,
        model_type=args.model,
        model_lstm_hidden_size=args.model_lstm_hidden_size,
        model_lstm_num_layers=args.model_lstm_num_layers,
        model_transformer_num_heads=args.model_transformer_num_heads,
        model_transformer_num_features_per_head=args.model_transformer_num_features_per_head,
        model_transformer_feedforward_dim=args.model_transformer_feedforward_dim,
        model_transformer_num_layers=args.model_transformer_num_layers,
        model_dropout_rate=args.model_dropout_rate,
        num_measures_per_chunk=args.num_measures_per_chunk,
        chunk_overlap=args.chunk_overlap,
        num_augmentations=args.num_augmentations,
        min_tracks_per_label=args.min_tracks_per_label,
        use_imfw=args.use_imfw,
        use_tempo=args.use_tempo,
    )

    return script_params


if __name__ == "__main__":
    print(parse_args())
