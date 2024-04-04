import copy
import gc
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import partial
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
        if not all(isinstance(v, self.type) or v is None for v in self.values):
            raise ValueError(f"All values must be of the same type: {type(self.values[0])}")

    @property
    def type(self):
        return type(self.values[0])


class RunGenerator:
    """
    Generator that yields a dictionary with the hyperparameters to test. It iterates over all the
    possible combinations of the hyperparameters in a random order. It makes sure that the same
    combination is not tested twice.
    """

    def __init__(
        self,
        hparams: list[HyperParameter],
        fix_run: Callable[[dict], dict] = lambda x: x,
        seed: int = None,
    ):
        self.hparams = hparams
        self.fix_run = fix_run
        self.already_tested = set()
        self.max_num_elements = math.prod(len(hp.values) for hp in hparams)
        gamma = 0.5772156649
        hn = (
            gamma
            + math.log(self.max_num_elements)
            + 1 / (2 * self.max_num_elements)
            - 1 / (12 * self.max_num_elements**2)
        )  # approx sum(1 / n for n in range(1, n + 1))
        avg_num_tries = self.max_num_elements * hn
        self.max_attempts_per_trial = math.sqrt(
            avg_num_tries
        )  # how much before considering no more possible combinations
        self.random = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        run = {}
        hashed_run = hash(frozenset(run.items()))

        tries = 0
        while hashed_run in self.already_tested or tries == 0:
            for hp in self.hparams:
                run[hp.name] = self.random.choice(hp.values)
            run = self.fix_run(run)
            hashed_run = hash(frozenset(run.items()))
            tries += 1
            if tries >= self.max_attempts_per_trial:
                raise StopIteration

        self.already_tested.add(hashed_run)
        if len(self.already_tested) == self.max_attempts_per_trial:
            raise StopIteration
        return run


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
@click.command(help="Hyperparameter tuning script. Except --start-at and --override-metadata, all options allow to specify multiple values separated by a comma (no spaces).")
@click.option("--data-path", callback=split_parameter_callback(click.Path(dir_okay=True, file_okay=False, readable=True, writable=True, executable=True, path_type=Path)), required=True, help="Path to the dataset.", metavar='PATH')
@click.option("--processed-data-path", callback=split_parameter_callback(click.Path(dir_okay=True, file_okay=False, readable=True, writable=True, executable=True, path_type=Path)), required=True, help="Path to store the processed data.", metavar='PATH')
@click.option("--results-path", callback=split_parameter_callback(click.Path(dir_okay=True, file_okay=False, readable=True, writable=True, executable=True, path_type=Path)), required=True, help="Path to the results.", metavar='PATH')
@click.option("--override-metadata", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path), help="Path to the metadata file to override the default one.", metavar='PATH')
@click.option("--start-at", type=int, default=0, help="Index of the first hyperparameter set to run.", metavar='INT')
@click.option("--track-processor-type", callback=split_parameter_callback(TrackProcessors.from_str), default=",".join([p.name for p in TrackProcessors]), help=f"Type of track processor ({', '.join([p.name for p in TrackProcessors])}).", metavar='STR')
@click.option("--track-processor-pianoroll-resolution", callback=split_parameter_callback(int), default="30,50", help="Pianoroll resolution.", metavar='INT')
@click.option("--track-processor-grammar-weights-file", callback=split_parameter_callback(click.Path(exists=True, dir_okay=False, readable=True, path_type=Path)), default=None, help="Path to the weights file for the grammar processor.", metavar='PATH')
@click.option("--track-processor-grammar-tree-positional-encoding", callback=split_parameter_callback(bool), default="True,False", help="Whether to use tree positional encoding for the grammar processor, or standard sequence positional encoding.", metavar='BOOL')
@click.option("--track-processor-tokenizer-type", callback=split_parameter_callback(str), default="midilike,remi,tsd,structured", help=f"Type of tokenizer for the tokenizer track processor (midilike, remi, tsd, structured).", metavar='STR')
@click.option("--track-processor-tokenizer-embedding-dim", callback=split_parameter_callback(int), default="32,64", help="Embedding dimension for the tokenizer track processor.", metavar='INT')
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
@click.option("--num-measures-per-chunk", callback=split_parameter_callback(int), default="2,4,8", help="Number of measures in each chunk.", metavar='INT')
@click.option("--chunk-overlap", callback=split_parameter_callback(bool), default="True", help="Whether chunks overlap.", metavar='BOOL')
@click.option("--use-imfw", callback=split_parameter_callback(bool), default="True,False", help="Whether to use IMFW.", metavar='BOOL')
@click.option("--use-tempo", callback=split_parameter_callback(bool), default="True,False", help="Whether to use tempo.", metavar='BOOL')
@click.option("--num-augmentations", callback=split_parameter_callback(int), default="0,10", help="Number of augmentations.", metavar='INT')
# fmt: on
def main(
    start_at: int,
    override_metadata: Path,
    **params,
):
    for pname, pvalue in params.items():
        print(f"{pname}: {pvalue}")
    hparams = [HyperParameter(k, v) for k, v in params.items()]

    def conditional_set_none(
        hps: dict,
        invalid_rules: list[tuple[tuple[str, str], str]],
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
        if not inplace:
            hps = copy.deepcopy(hps)
        for (key_to_filter_on, value_to_filter_on), pattern_to_pop in invalid_rules:
            if hps[key_to_filter_on] == value_to_filter_on:
                keys_to_pop = filter(lambda key: key.startswith(pattern_to_pop), hps.keys())
                for key_to_pop in keys_to_pop:
                    hps[key_to_pop] = None
        return hps

    invalid_rules = [
        # Pop transformer parameters for LSTM model
        (("model_type", GenreClassifierType.lstm), r"model_transformer_"),
        # Pop LSTM parameters for transformer model
        (("model_type", GenreClassifierType.transformer), "model_lstm_"),
        # Pop parameters non-relevant to the track processor
        # For each processor p1:
        #   For each processor p2 != p1:
        #     Create a rule that set to None the parameters of p2 for p1
        #     (e.g. for p1=relative, set to None the pianoroll and grammar parameters)
        #     ((track_processor_type, p1), r"track_processor_{p2}_")
        *sum(
            [
                [
                    (
                        ("track_processor_type", selected_tp),
                        f"track_processor_{other_tp.name}_",
                    )
                    for other_tp in TrackProcessors
                    if other_tp != selected_tp
                ]
                for selected_tp in TrackProcessors
            ],
            [],
        ),
        # Pop tree positional encoding for LSTM models
        (
            ("model_type", GenreClassifierType.lstm),
            "track_processor_grammar_tree_positional_encoding",
        ),
    ]

    run_generator = RunGenerator(
        hparams,
        fix_run=partial(conditional_set_none, invalid_rules=invalid_rules),
        seed=22,
    )

    # Use a file to stop training after the current run
    pid_file = Path(f"/tmp/groove-stop-{os.getpid()}")
    if pid_file.exists():  # Remove an eventual file from a previous run
        pid_file.unlink()

    # Loop over hyperparameters
    for i, cur_hparams in enumerate(run_generator):
        if i < start_at:
            continue
        training_params = ScriptParameters(override_metadata=override_metadata, **cur_hparams)
        print("=" * 80)
        print(f"Run {i + 1}")
        print(
            tabulate.tabulate(
                list(cur_hparams.items()),
                headers=["Hyperparameter", "Value"],
                tablefmt="rounded_outline",
            )
        )

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
