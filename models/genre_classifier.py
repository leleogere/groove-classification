import time
from enum import Enum
from typing import Callable

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MulticlassF1Score

from models.utils import TupleExtractor
from utils.confusion_matrix import confusion_matrix_pipeline


class GenreClassifier(L.LightningModule):
    _train_key = "train"
    _val_key = "val"

    def __init__(
        self,
        base_module: nn.Module,
        base_module_output_dim: int,
        num_classes: int,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        use_tempo: bool = False,
        class_names: list[str] = None,
        name: str = None,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
        hparams_to_log: dict = None,
    ):
        super().__init__()
        self.use_tempo = use_tempo
        self.base_module = base_module
        self.base_module_output_dim = base_module_output_dim
        self.dropout = nn.Dropout(dropout_rate)
        if self.use_tempo:
            self.base_module_output_dim += 1
        self.batch_normalization = nn.BatchNorm1d(self.base_module_output_dim)
        self.linear = nn.Linear(self.base_module_output_dim, num_classes)
        self.lr = lr
        self.num_classes = num_classes
        self.name = name
        self.class_names = (
            class_names if class_names is not None else [f"Class_{i}" for i in range(num_classes)]
        )
        self.criterion = loss_function
        self.metrics = nn.ModuleDict({
            "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
            "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
        })
        if hparams_to_log is not None:
            self.save_hyperparameters(hparams_to_log)
        self.accumulated_predictions = {}
        self._reset_accumulated_predictions()

    def forward(self, x, tempo=None):
        # print(f"Before module: {x.shape = }")
        x = self.base_module(x)
        # print(f"After module: {x.shape = }")
        x = x[:, -1, :]  # Keep only last step
        # Add tempo
        if self.use_tempo:
            x = torch.cat([x, tempo], dim=-1)
        # print(f'Before batch norm: {x.shape = }')
        x = self.batch_normalization(x)
        # print(f'After batch norm: {x.shape = }')
        x = self.dropout(x)
        # print(f'After dropout: {x.shape = }')
        x = self.linear(x)
        # print(f'End forward: {x.shape = }')
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def common_step(self, batch, key: str):
        # Get data
        x = batch["samples"]
        y = batch["labels"]
        tempo = batch["tempos"] if self.use_tempo else None
        # Apply model
        y_hat = self.forward(x, tempo)
        # Compute and log loss
        loss = self.criterion(y_hat, y)
        self.log(f"{key}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Accumulate prediction for global metrics
        self.accumulated_predictions[key]["y_true"].append(y.detach())
        self.accumulated_predictions[key]["y_pred"].append(torch.argmax(y_hat, dim=-1).detach())
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, self._train_key)

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, self._val_key)

    def on_common_epoch_end(self, key: str):
        y_true = torch.cat(self.accumulated_predictions[key]["y_true"], dim=0)
        y_pred = torch.cat(self.accumulated_predictions[key]["y_pred"], dim=0)
        # Compute metrics
        metrics_dict = {k: v(y_pred, y_true) for k, v in self.metrics.items()}
        self.log_dict(
            {f"{key}/{k}": v for k, v in metrics_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        image = confusion_matrix_pipeline(
            y_true.cpu(),
            y_pred.cpu(),
            self.class_names,
            normalization="true",
        )  # .transpose((2, 0, 1))
        self.logger.log_image(f"{key}_confusion_matrix", [image], self.current_epoch)
        # Reset accumulated predictions
        self._reset_accumulated_predictions(key)

    def on_train_epoch_end(self):
        self.on_common_epoch_end(self._train_key)

    def on_validation_epoch_end(self):
        self.on_common_epoch_end(self._val_key)

    def on_train_start(self) -> None:
        # Log hyperparameters with first dummy metrics in order for them to appear in TensorBoard
        to_log = {}
        for part in ["train", "val"]:
            to_log.update({f"{part}/{k}": float("NaN") for k in self.metrics.keys()})
            to_log.update({f"{part}/loss": float("NaN")})
        self.logger.log_hyperparams(self.hparams)

    def _reset_accumulated_predictions(self, key: str | None = None):
        def default_dict():
            return {"y_true": [], "y_pred": []}

        if key is None:
            self.accumulated_predictions = {
                k: default_dict() for k in [self._train_key, self._val_key]
            }
        else:
            self.accumulated_predictions[key] = default_dict()


class GenreClassifierLSTM(GenreClassifier):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        use_tempo: bool = False,
        class_names: list[str] = None,
        name: str = None,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
        hparams_to_log: dict = None,
    ):
        lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )
        element_extractor = TupleExtractor(index=0)
        module = nn.Sequential(lstm, element_extractor)
        super().__init__(
            base_module=module,
            base_module_output_dim=2 * hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lr=lr,
            use_tempo=use_tempo,
            class_names=class_names,
            name=name,
            loss_function=loss_function,
            hparams_to_log=hparams_to_log,
        )


class GenreClassifierTransformer(GenreClassifier):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_features_per_head: int,
        feedforward_dim: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        use_tempo: bool = False,
        class_names: list[str] = None,
        name: str = None,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
        hparams_to_log: dict = None,
    ):
        # torch need embed_dim % nheads == 0, so we add a linear layer to adjust the input dimension
        encoder_input_dim = num_heads * num_features_per_head
        linear = nn.Linear(input_dim, encoder_input_dim)
        encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=encoder_input_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout_rate,
            ),
            num_layers=num_layers,
        )
        module = nn.Sequential(linear, encoder)
        super().__init__(
            base_module=module,
            base_module_output_dim=encoder_input_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lr=lr,
            use_tempo=use_tempo,
            class_names=class_names,
            name=name,
            loss_function=loss_function,
            hparams_to_log=hparams_to_log,
        )


class GenreClassifierType(Enum):
    lstm = GenreClassifierLSTM
    transformer = GenreClassifierTransformer

    def __str__(self):
        return self.name

    @staticmethod
    def from_str(s: str):
        try:
            return GenreClassifierType[s]
        except KeyError:
            raise ValueError(
                f"Cannot convert {s} to TrackProcessors. Valid values are"
                f" {list(GenreClassifierType)}."
            )
