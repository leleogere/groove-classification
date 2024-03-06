import argparse
import copy
import gc
import itertools
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tabulate
import torch
import yaml

from cli.parser import ScriptParameters
from data_processing.track_processor_enum import TrackProcessors
from models.genre_classifier import GenreClassifierType
from training.training_script import main


@dataclass(frozen=True)
class HyperParameter:
    name: str
    values: list

    def __post_init__(self):
        if not all(isinstance(v, self.type) or v is None for v in self.values):
            raise ValueError(f"All values must be of the same type: {type(self.values[0])}")

    @property
    def type(self):
        return type(self.values[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the dataset.")
    parser.add_argument(
        "--processed-data-path", type=Path, required=True, help="Path to store the processed data."
    )
    parser.add_argument("--results-path", type=Path, required=True, help="Path to the results.")
    parser.add_argument(
        "--grammar-path",
        type=Path,
        required=True,
        help="Path to the weights file for the grammar processor.",
    )
    parser.add_argument(
        "--start-at", type=int, default=0, help="Index of the first hyperparameter set to run."
    )
    parser.add_argument(
        "--skip-already-tested-hps",
        action="store_true",
        help="Skip already tested hyperparameters.",
    )
    args = parser.parse_args()

    HYPERPARAMETERS = [
        HyperParameter("TrackProcessorType", ["grammar", "relative", "pianoroll"]),
        HyperParameter("TrackProcessorPianorollResolution", [30]),
        HyperParameter("Model", ["lstm", "transformer"]),
        HyperParameter("LstmHiddenSize", [8, 16, 32]),  # , 64, 96, 128]),
        HyperParameter("LstmNumLayers", [1, 2, 3]),
        HyperParameter("TransformerNumHeads", [2, 4, 8]),
        HyperParameter("TransformerNumFeaturesPerHead", [2, 4, 8, 16, 32]),
        HyperParameter("TransformerFFDim", [8, 16, 32]),
        HyperParameter("TransformerNumLayers", [1, 2, 3]),
        HyperParameter("DropoutRate", [0.3, 0.5]),
        # HyperParameter('DropoutRate', [0.5]),
        # HyperParameter('lr', np.logspace(-4, -2, 3).tolist()),
        HyperParameter("LearningRate", [1e-3]),
        # HyperParameter('reg', np.logspace(-4, -2, 3).tolist()),
        HyperParameter("L2Regularization", [1e-4, 5e-4]),
        HyperParameter("NumMeasures", [1, 2, 4, 8]),
        HyperParameter("MeasuresOverlap", [True, False]),
        HyperParameter("IMFW", [True, False]),
        HyperParameter("UseTempo", [True, False]),
        HyperParameter("NumAugmentations", [0, 2, 5, 10]),
    ]

    domain = list(
        dict(zip(map(lambda hp: hp.name, HYPERPARAMETERS), x))
        for x in itertools.product(*map(lambda hp: hp.values, HYPERPARAMETERS))
    )  # See https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists/40623158
    print(f"Number of hyperparameter sets before filtering: {len(domain)}")

    # def is_valid(hps: dict) -> bool:
    #     if (
    #         not hps["TrackProcessorType"] == "pianoroll"
    #         and hps["TrackProcessorPianorollResolution"] != 30
    #     ):
    #         return False
    #     if hps["Model"] == "lstm":
    #         if hps["TransformerNumHeads"] != 4:
    #             return False
    #         if hps["TransformerNumFeaturesPerHead"] != 4:
    #             return False
    #         if hps["TransformerFFDim"] != 32:
    #             return False
    #         if hps["TransformerNumLayers"] != 2:
    #             return False
    #     elif hps["Model"] == "transformer":
    #         if hps["LstmHiddenSize"] != 8:
    #             return False
    #         if hps["LstmNumLayers"] != 2:
    #             return False
    #     else:
    #         raise ValueError(f"Unknown model: {hps['Model']}")
    #     return True

    def conditional_set_none(
        hps: dict,
        invalid_rules: list[tuple[tuple[str, str], str]] = None,
        inplace: bool = True,
    ) -> dict:
        if invalid_rules is None:
            invalid_rules = [
                # Pop transformer parameters for LSTM model
                (("Model", "lstm"), r"Transformer"),
                # Pop LSTM parameters for transformer model
                (("Model", "transformer"), "Lstm"),
                # Pop pianoroll parameters for relative processor
                (("TrackProcessorType", "relative"), r"TrackProcessorPianoroll"),
                # Pop pianoroll parameters for grammar processor
                (("TrackProcessorType", "grammar"), r"TrackProcessorPianoroll"),
            ]
        if not inplace:
            hps = copy.deepcopy(hps)
        for (key_to_filter_on, value_to_filter_on), pattern_to_pop in invalid_rules:
            if hps[key_to_filter_on] == value_to_filter_on:
                keys_to_pop = filter(lambda key: key.startswith(pattern_to_pop), hps.keys())
                for key_to_pop in keys_to_pop:
                    hps[key_to_pop] = None
        return hps

    # Set to None useless hyperparameters
    domain = [conditional_set_none(hps, inplace=True) for hps in domain]
    # Remove duplicates (using fact that keys are unique in dict)
    hashes = {hash(frozenset(trial.items())): trial for trial in domain}
    domain = list(hashes.values())
    print(f"Number of hyperparameter sets after filtering: {len(domain)}")

    random.seed(22)
    random.shuffle(domain)
    START_AT = args.start_at

    for i, hparams in enumerate(domain):
        if i < START_AT:
            continue
        params = ScriptParameters(
            data_path=args.data_path,
            processed_data_path=args.processed_data_path,
            results_path=args.results_path,
            training_batch_size=64,
            training_num_epochs=2,
            training_patience=40,
            training_learning_rate=hparams["LearningRate"],
            training_l2_regularization=hparams["L2Regularization"],
            track_processor_type=TrackProcessors.from_str(hparams["TrackProcessorType"]),
            track_processor_pianoroll_resolution=hparams["TrackProcessorPianorollResolution"],
            track_processor_grammar_weights_file=args.grammar_path,
            model_type=GenreClassifierType.from_str(hparams["Model"]),
            model_lstm_hidden_size=hparams["LstmHiddenSize"],
            model_lstm_num_layers=hparams["LstmNumLayers"],
            model_transformer_num_heads=hparams["TransformerNumHeads"],
            model_transformer_num_features_per_head=hparams["TransformerNumFeaturesPerHead"],
            model_transformer_feedforward_dim=hparams["TransformerFFDim"],
            model_transformer_num_layers=hparams["TransformerNumLayers"],
            model_dropout_rate=hparams["DropoutRate"],
            num_measures_per_chunk=hparams["NumMeasures"],
            chunk_overlap=hparams["MeasuresOverlap"],
            num_augmentations=hparams["NumAugmentations"],
            min_tracks_per_label=10,
            use_imfw=hparams["IMFW"],
            use_tempo=hparams["UseTempo"],
        )
        print("=" * 80)
        print(f"Run {i + 1}/{len(domain)}")
        print(
            tabulate.tabulate(
                list(hparams.items()),
                headers=["Hyperparameter", "Value"],
                tablefmt="rounded_outline",
            )
        )

        # Checking if same hyperparameters already tested
        if args.skip_already_tested_hps:
            existing = False
            ignored_params = ["training_batch_size", "training_num_epochs", "training_patience"]
            if params.results_path.exists():
                new_hparams = copy.deepcopy(params.to_serializable_dict())
                for ignored_param in ignored_params:
                    new_hparams.pop(ignored_param)
                hparam_files = list(params.results_path.rglob("**/hparams.yaml"))
                for hparam_file in hparam_files:
                    if hparam_file.exists():
                        with open(hparam_file, "r") as f:
                            try:
                                hparams = yaml.safe_load(f)
                            except yaml.YAMLError as exc:
                                print(f"Error in loading {hparam_file}: exc")
                                continue
                        for ignored_param in ignored_params:
                            hparams.pop(ignored_param)
                        if hparams == new_hparams:
                            existing = True
                            break
            if existing:
                print(
                    "Skipping run as same hyperparameters already tested in"
                    f" '{hparam_file.parent}'."
                )
                continue

        main(params)
        # Try to release GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        # Check if we should stop the search
        if Path("/tmp/groove-stop").exists():
            print("Stopping early")
            Path("/tmp/groove-stop").unlink()
            (params.results_path / f"end_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt").touch()
            break
