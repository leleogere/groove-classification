import argparse
import copy
import gc
import inspect
import itertools
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import click
import tabulate
import torch

from cli.parser import ScriptParameters
from cli.parser import bool_from_str
from data_processing.track_processor_enum import TrackProcessors
from models.genre_classifier import GenreClassifierType
from training.training_script import main as train_one_model


@dataclass(frozen=True)
class HyperParameter:
    name: str
    values: list

    def __post_init__(self):
        print(self.name, self.values)
        import time

        time.sleep(0.01)
        if not all(isinstance(v, self.type) or v is None for v in self.values):
            raise ValueError(f"All values must be of the same type: {type(self.values[0])}")

    @property
    def type(self):
        return type(self.values[0])


def split_parameter_callback(type_: Callable = str, sep: str = ",", strip: bool = True):
    def callback(ctx, param, s):
        caster = type_
        if s is None or s == "" or s.lower() == "none":
            return [None]
        if type_ == bool:
            caster = bool_from_str
        split = s.split(sep)
        if strip:
            split = list(map(str.strip, split))
        for i, x in enumerate(split):
            try:
                split[i] = caster(x)
            except ValueError:
                raise click.BadParameter(f"The element '{x}' cannot be casted into {type_}.")
        return split

    return callback


# fmt: off
@click.command(help="Hyperparameter tuning script. Except --start-at, all options allow to specify multiple values separated by a comma (no spaces).")
@click.option("--data-path", callback=split_parameter_callback(click.Path(dir_okay=True, file_okay=False, readable=True, writable=True, executable=True, path_type=Path)), required=True, help="Path to the dataset.", metavar='PATH')
@click.option("--processed-data-path", callback=split_parameter_callback(click.Path(dir_okay=True, file_okay=False, readable=True, writable=True, executable=True, path_type=Path)), required=True, help="Path to store the processed data.", metavar='PATH')
@click.option("--results-path", callback=split_parameter_callback(click.Path(dir_okay=True, file_okay=False, readable=True, writable=True, executable=True, path_type=Path)), required=True, help="Path to the results.", metavar='PATH')
@click.option("--start-at", type=int, default=0, help="Index of the first hyperparameter set to run.", metavar='INT')
@click.option("--track-processor-type", callback=split_parameter_callback(TrackProcessors.from_str), default=",".join([p.name for p in TrackProcessors]), help=f"Type of track processor ({', '.join([p.name for p in TrackProcessors])}).", metavar='STR')
@click.option("--track-processor-pianoroll-resolution", callback=split_parameter_callback(int), default="30,50", help="Pianoroll resolution.", metavar='INT')
@click.option("--track-processor-grammar-weights-file", callback=split_parameter_callback(click.Path(exists=True, dir_okay=False, readable=True, path_type=Path)), default=None, help="Path to the weights file for the grammar processor.", metavar='PATH')
@click.option("--model-type", callback=split_parameter_callback(GenreClassifierType.from_str), default=",".join([p.name for p in GenreClassifierType]), help=f"Type of model ({', '.join([p.name for p in GenreClassifierType])}).", metavar='STR')
@click.option("--model-lstm-hidden-size", callback=split_parameter_callback(int), default="8,16,32,64", help="Hidden size of the LSTM model.", metavar='INT')
@click.option("--model-lstm-num-layers", callback=split_parameter_callback(int), default="1,2,3", help="Number of layers in the LSTM model.", metavar='INT')
@click.option("--model-transformer-num-heads", callback=split_parameter_callback(int), default="2,4,8", help="Number of heads in the transformer model.", metavar='INT')
@click.option("--model-transformer-num-features-per-head", callback=split_parameter_callback(int), default="2,4,8", help="Number of features per head in the transformer model.", metavar='INT')
@click.option("--model-transformer-feedforward-dim", callback=split_parameter_callback(int), default="8,16,32", help="Feedforward dimension in the transformer model.", metavar='INT')
@click.option("--model-transformer-num-layers", callback=split_parameter_callback(int), default="1,2,3", help="Number of layers in the transformer model.", metavar='INT')
@click.option("--model-dropout-rate", callback=split_parameter_callback(float), default="0.1,0.3,0.5", help="Dropout rate.", metavar='FLOAT')
@click.option("--training-learning-rate", callback=split_parameter_callback(float), default="1e-3", help="Learning rate.", metavar='FLOAT')
@click.option("--training-l2-regularization", callback=split_parameter_callback(float), default="0.0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3", help="L2 regularization.", metavar='FLOAT')
@click.option("--training-batch-size", callback=split_parameter_callback(int), default="64", help="Batch size.", metavar='INT')
@click.option("--training-num-epochs", callback=split_parameter_callback(int), default="1000", help="Number of epochs.", metavar='INT')
@click.option("--training-patience", callback=split_parameter_callback(int), default="300", help="Patience.", metavar='INT')
@click.option("--num-measures-per-chunk", callback=split_parameter_callback(int), default="1,2,4,8", help="Number of measures in each chunk.", metavar='INT')
@click.option("--chunk-overlap", callback=split_parameter_callback(bool), default="True", help="Whether chunks overlap.", metavar='BOOL')
@click.option("--use-imfw", callback=split_parameter_callback(bool), default="True,False", help="Whether to use IMFW.", metavar='BOOL')
@click.option("--use-tempo", callback=split_parameter_callback(bool), default="True,False", help="Whether to use tempo.", metavar='BOOL')
@click.option("--num-augmentations", callback=split_parameter_callback(int), default="0,10", help="Number of augmentations.", metavar='INT')
# fmt: on
def main(
    data_path: Path,
    processed_data_path: Path,
    results_path: Path,
    track_processor_type: TrackProcessors,
    track_processor_pianoroll_resolution: int,
    track_processor_grammar_weights_file: Path,
    model_type: GenreClassifierType,
    model_lstm_hidden_size: int,
    model_lstm_num_layers: int,
    model_transformer_num_heads: int,
    model_transformer_num_features_per_head: int,
    model_transformer_feedforward_dim: int,
    model_transformer_num_layers: int,
    model_dropout_rate: float,
    training_learning_rate: float,
    training_l2_regularization: float,
    training_batch_size: int,
    training_num_epochs: int,
    training_patience: int,
    num_measures_per_chunk: int,
    chunk_overlap: bool,
    use_imfw: bool,
    use_tempo: bool,
    num_augmentations: int,
    start_at: int,
):
    _, _, _, params = inspect.getargvalues(inspect.currentframe())
    print(params)
    hparams = [HyperParameter(k, v) for k, v in params.items() if k not in ["start_at"]]

    domain = list(
        dict(zip(map(lambda hp: hp.name, hparams), x))
        for x in itertools.product(*map(lambda hp: hp.values, hparams))
    )  # See https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists/40623158
    print(f"Number of hyperparameter sets before filtering: {len(domain)}")

    def conditional_set_none(
        hps: dict,
        invalid_rules: list[tuple[tuple[str, str], str]] = None,
        inplace: bool = True,
    ) -> dict:
        """
        Set to None hyperparameters that are not relevant for the current model (like LSTM
        parameters for a transformer model for example).

        :param hps: Dictionary of hyperparameters
        :param invalid_rules: List of rules to set to None some hyperparameters (in the form of
            ((key_to_filter_on, value_to_filter_on), pattern_to_pop))
        :param inplace: Whether to modify the input dictionary or return a new one
        :return: New dictionary of hyperparameters with some values set to None
        """
        if invalid_rules is None:
            invalid_rules = [
                # Pop transformer parameters for LSTM model
                (("model_type", GenreClassifierType.lstm), r"model_transformer_"),
                # Pop LSTM parameters for transformer model
                (("model_type", GenreClassifierType.transformer), "model_lstm_"),
                # Pop pianoroll parameters for relative processor
                (("track_processor_type", TrackProcessors.relative), r"track_processor_pianoroll_"),
                # Pop pianoroll parameters for grammar processor
                (("track_processor_type", TrackProcessors.grammar), r"track_processor_pianoroll_"),
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

    # Use a file to stop training after the current run
    pid_file = Path(f"/tmp/groove-stop-{os.getpid()}")
    if pid_file.exists():  # Remove an eventual file from a previous run
        pid_file.unlink()

    # Loop over hyperparameters
    for i, cur_hparams in enumerate(domain):
        if i < start_at:
            continue
        training_params = ScriptParameters(
            min_tracks_per_label=30,
            **cur_hparams,
        )
        print("=" * 80)
        print(f"Run {i + 1}/{len(domain)}")
        print(
            tabulate.tabulate(
                list(cur_hparams.items()),
                headers=["Hyperparameter", "Value"],
                tablefmt="rounded_outline",
            )
        )
        #
        #     # # Checking if same hyperparameters already tested
        #     # if args.skip_already_tested_hps:
        #     #     existing = False
        #     #     ignored_params = ["training_batch_size", "training_num_epochs", "training_patience"]
        #     #     if params.results_path.exists():
        #     #         new_hparams = copy.deepcopy(params.to_serializable_dict())
        #     #         for ignored_param in ignored_params:
        #     #             new_hparams.pop(ignored_param)
        #     #         hparam_files = list(params.results_path.rglob("**/hparams.yaml"))
        #     #         for hparam_file in hparam_files:
        #     #             if hparam_file.exists():
        #     #                 with open(hparam_file, "r") as f:
        #     #                     try:
        #     #                         hparams = yaml.safe_load(f)
        #     #                     except yaml.YAMLError as exc:
        #     #                         print(f"Error in loading {hparam_file}: exc")
        #     #                         continue
        #     #                 for ignored_param in ignored_params:
        #     #                     hparams.pop(ignored_param)
        #     #                 if hparams == new_hparams:
        #     #                     existing = True
        #     #                     break
        #     #     if existing:
        #     #         print(
        #     #             "Skipping run as same hyperparameters already tested in"
        #     #             f" '{hparam_file.parent}'."
        #     #         )
        #     #         continue

        train_one_model(training_params)
        # Try to release GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        # Check if we should stop the search
        if pid_file.exists():
            print("Stopping early")
            pid_file.unlink()
            (
                training_params.results_path / f"end_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            ).touch()
            break


if __name__ == "__main__":
    # Get terminal width
    max_content_width = shutil.get_terminal_size().columns
    main(max_content_width=max_content_width)
