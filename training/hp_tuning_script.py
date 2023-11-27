import argparse
import gc
import itertools
from dataclasses import dataclass
import random
from datetime import datetime
from pathlib import Path
from pprint import pformat

import torch
import yaml

from cli.parser import ScriptParameters
from data_processing.track_processors import TrackProcessors
from training.training_script import main


@dataclass(frozen=True)
class HyperParameter:
    name: str
    values: list

    def __post_init__(self):
        if not all(isinstance(v, self.type) or v is None for v in self.values):
            raise ValueError(f'All values must be of the same type: {type(self.values[0])}')

    @property
    def type(self):
        return type(self.values[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument('--data-path', type=Path, required=True, help='Path to the dataset.')
    parser.add_argument('--processed-data-path', type=Path, required=True, help='Path to store the processed data.')
    parser.add_argument('--results-path', type=Path, required=True, help='Path to the results.')
    parser.add_argument('--start-at', type=int, default=0, help='Index of the first hyperparameter set to run.')
    args = parser.parse_args()

    HYPERPARAMETERS = [
        HyperParameter('TrackProcessorType', ['multivariate', 'pianoroll']),
        HyperParameter('TrackProcessorPianorollResolution', [30]),
        HyperParameter('LstmHiddenSize', [8, 16, 32, 64, 96, 128]),
        HyperParameter('LstmNumLayers', [1, 2, 3]),
        HyperParameter('DropoutRate', [0.1, 0.3, 0.5]),
        # HyperParameter('DropoutRate', [0.5]),
        # HyperParameter('lr', np.logspace(-4, -2, 3).tolist()),
        HyperParameter('lr', [1e-3]),
        # HyperParameter('reg', np.logspace(-4, -2, 3).tolist()),
        HyperParameter('reg', [1e-4, 5e-4]),
        HyperParameter('NumMeasures', [1, 2, 4, 8]),
        HyperParameter('MeasuresOverlap', [True, False]),
        HyperParameter('IMFW', [True, False]),
        HyperParameter('UseTempo', [True, False]),
        HyperParameter('UseAugmentedDataset', [True, False]),
        HyperParameter('NumAugmentations', [0, 2, 5, 10]),
    ]

    domain = list(
        dict(zip(map(lambda hp: hp.name, HYPERPARAMETERS), x))
        for x in itertools.product(*map(lambda hp: hp.values, HYPERPARAMETERS))
    )  # See https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists/40623158

    def is_valid(hps: dict) -> bool:
        if hps['UseAugmentedDataset'] and hps['NumAugmentations'] == 0:
            return False
        if not hps['UseAugmentedDataset'] and hps['NumAugmentations'] != 0:
            return False
        if not hps['TrackProcessorType'] == 'pianoroll' and hps['TrackProcessorPianorollResolution'] != 30:
            return False
        return True

    domain = list(filter(is_valid, domain))

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
            track_processor_type=TrackProcessors.from_str(hparams['TrackProcessorType']),
            track_processor_pianoroll_resolution=hparams['TrackProcessorPianorollResolution'],
            training_batch_size=64,
            training_num_epochs=2,
            training_patience=40,
            hp_use_tempo=hparams['UseTempo'],
            hp_num_measures=hparams['NumMeasures'],
            hp_measures_overlap=hparams['MeasuresOverlap'],
            hp_imfw=hparams['IMFW'],
            hp_use_augmented_dataset=hparams['UseAugmentedDataset'],
            hp_num_augmentations=hparams['NumAugmentations'],
            hp_lstm_hidden_size=hparams['LstmHiddenSize'],
            hp_lstm_num_layers=hparams['LstmNumLayers'],
            hp_dropout_rate=hparams['DropoutRate'],
            hp_lr=hparams['lr'],
            hp_reg=hparams['reg'],
            min_tracks_per_label=20,
        )
        print("=" * 80)
        print(f'Running {i + 1}/{len(domain)}:\n{pformat(params.__dict__)}')

        # Checking if same hyperparameters already tested
        existing = False
        if params.results_path.exists():
            hparam_files = list(params.results_path.rglob('**/hparams.yaml'))
            for hparam_file in hparam_files:
                if hparam_file.exists():
                    with open(hparam_file, 'r') as f:
                        try:
                            hparams = yaml.safe_load(f)
                        except yaml.YAMLError as exc:
                            print(f"Error in loading {hparam_file}: exc")
                            continue
                    if hparams == params.to_serializable_dict():
                        existing = True
                        break
        if existing:
            print(f"Skipping run as same hyperparameters already tested in '{hparam_file.parent}'.")
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
