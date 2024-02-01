import os
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch import loggers as loggers, callbacks as callbacks
import torch

from cli.parser import ScriptParameters
from cli.parser import parse_args
from data_processing.track_processors_base import TrackProcessor
from dataset.groove_dataset import GrooveMidiModule


def main(params: ScriptParameters):
    # Setup track processor
    track_processor_param_prefix = f"track_processor_{params.track_processor_type.name}"
    track_processor_params = {
        parameter[len(track_processor_param_prefix) + 1 :]: getattr(params, parameter)
        for parameter in params.__dict__.keys()
        if parameter.startswith(track_processor_param_prefix)
    }
    track_processor_params["use_tempo"] = params.use_tempo
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
            num_measures_per_chunk=params.num_measures_per_chunk,
            chunk_overlap=params.chunk_overlap,
        ),
        augment_dataset=params.num_augmentations > 0,
        augment_dataset_kwargs=dict(
            augmented_path=params.processed_data_path / "augmented",
            num_augmentations=params.num_augmentations,
            seed=22,
        ),
    )

    # Setup data
    groove_module.prepare_data()
    groove_module.setup(stage="")

    # Compute inverse median frequency weights
    if params.use_imfw:
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
    if params.model_type.name == "lstm":
        model_specific_params = {
            "hidden_size": params.model_lstm_hidden_size,
            "num_layers": params.model_lstm_num_layers,
        }
    elif params.model_type.name == "transformer":
        model_specific_params = {
            "num_heads": params.model_transformer_num_heads,
            "num_features_per_head": params.model_transformer_num_features_per_head,
            "feedforward_dim": params.model_transformer_feedforward_dim,
            "num_layers": params.model_transformer_num_layers,
        }
    else:
        raise ValueError(f"Unknown model name: {params.model_type.name}")
    model = params.model_type.value(
        input_dim=track_processor.output_dim,
        **model_specific_params,
        num_classes=len(groove_module.id_to_label),
        dropout_rate=params.model_dropout_rate,
        lr=params.training_learning_rate,
        use_tempo=params.use_tempo,
        class_names=groove_module.unique_labels,
        name=params.model_type.name,
        loss_function=loss,
        hparams_to_log=params.to_serializable_dict(),
    )

    # Prepare callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val/loss",
        min_delta=0.00,
        patience=params.training_patience,
        mode="min",
    )
    params.results_path.mkdir(parents=True, exist_ok=True)
    try:
        project = os.environ["WANDB_PROJECT"]
    except KeyError:
        raise KeyError("Please set the WANDB_PROJECT environment variable to your project name")
    logger = loggers.WandbLogger(
        save_dir=str(params.results_path),
        project=project,
        log_model=True,
    )
    # logger_dir = Path(logger.save_dir)
    # logger_dir.mkdir(parents=True, exist_ok=False)
    # groove_module.export_label_infos(logger_dir / "label_infos.txt")
    print("Label infos:")
    print(groove_module.label_infos_to_string())

    # Train model
    print("Training model...")
    trainer = L.Trainer(
        max_epochs=params.training_num_epochs,
        logger=logger,
        callbacks=[early_stopping],
        devices=1,
    )
    trainer.fit(model, datamodule=groove_module)

    logger.experiment.finish()

    groove_module.memory.reduce_size(bytes_limit="100G")  # TODO: make this configurable


if __name__ == "__main__":
    parameters = parse_args()
    main(parameters)
