from datetime import datetime
from pathlib import Path

import lightning as L
import lightning.pytorch as pl
import torch

from cli.parser import ScriptParameters
from cli.parser import parse_args
from data_processing.track_processors import TrackProcessor
from dataset.groove_dataset import GrooveMidiModule
from models.multivariate_classifier import GenreClassifier


def main(params: ScriptParameters):
    # Setup track processor
    track_processor_param_prefix = f"track_processor_{params.track_processor_type.name}"
    track_processor_params = {
        parameter[len(track_processor_param_prefix) + 1 :]: getattr(params, parameter)
        for parameter in params.__dict__.keys()
        if parameter.startswith(track_processor_param_prefix)
    }
    track_processor_params["use_tempo"] = params.hp_use_tempo
    track_processor: TrackProcessor = params.track_processor_type.value(**track_processor_params)
    groove_module = GrooveMidiModule(
        path=params.data_path,
        batch_size=params.training_batch_size,
        track_processor=track_processor,
        min_tracks_per_label=params.min_tracks_per_label,
        cache_location=params.processed_data_path / "cache",
        split_dataset_into_measures=True,
        split_dataset_into_measures_kwargs=dict(
            split_path=params.processed_data_path / "split",
            num_measures_per_chunk=params.hp_num_measures,
            chunk_overlap=params.hp_measures_overlap,
        ),
        augment_dataset=params.hp_use_augmented_dataset,
        augment_dataset_kwargs=dict(
            augmented_path=params.processed_data_path / "augmented",
            num_augmentations=params.hp_num_augmentations,
            seed=22,
        ),
    )

    # Setup data
    groove_module.prepare_data()
    groove_module.setup(stage="")

    # Compute inverse median frequency weights
    if params.hp_imfw:
        class_count = {
            label: sum(
                groove_module.label_infos[label][split]["chunks"]
                for split in ["train", "validation"]
            )
            for label in groove_module.unique_labels
        }
        total = sum(class_count.values())
        class_frequency = {label: count / total for label, count in class_count.items()}
        sorted_class_frequency = sorted(class_frequency.values())
        median_idx = len(sorted_class_frequency) // 2
        median_freq = (
            sorted_class_frequency[median_idx]
            if len(sorted_class_frequency) % 2 == 1
            else (sorted_class_frequency[median_idx - 1] + sorted_class_frequency[median_idx]) / 2
        )
        weights_c = {label: median_freq / freq for label, freq in class_frequency.items()}
        weights_c = torch.tensor(list(weights_c.values()), dtype=torch.float32)
    else:
        weights_c = None
    loss = torch.nn.CrossEntropyLoss(weight=weights_c)

    # Create model
    model = GenreClassifier(
        num_features=track_processor.output_dim()[0],
        lstm_hidden_size=params.hp_lstm_hidden_size,
        lstm_num_layers=params.hp_lstm_num_layers,
        num_classes=len(groove_module.id_to_label),
        class_names=list(groove_module.label_to_id.keys()),
        dropout_rate=params.hp_dropout_rate,
        use_tempo=params.hp_use_tempo,
        loss_function=loss,
        hparams_to_log=params.to_serializable_dict(),
    )

    # Prepare callbacks
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val/loss",
        min_delta=0.00,
        patience=params.training_patience,
        mode="min",
    )
    params.results_path.mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(
        save_dir=params.results_path,
        name=datetime.now().strftime("%Y-%m-%d"),
        default_hp_metric=False,
    )
    logger_dir = Path(logger.log_dir)
    logger_dir.mkdir(parents=True, exist_ok=False)
    groove_module.export_label_infos(logger_dir / "label_infos.txt")

    # Train model
    print("Training model...")
    trainer = L.Trainer(
        max_epochs=params.training_num_epochs,
        logger=logger,
        callbacks=[early_stopping],
        devices=1,
    )
    trainer.fit(model, datamodule=groove_module)


if __name__ == "__main__":
    parameters = parse_args()
    main(parameters)
