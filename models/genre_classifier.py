import time
from enum import Enum
from typing import Callable
from typing import Type

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MulticlassF1Score

from models.positional_encoding import ContinuousPositionalEncoder
from models.positional_encoding import TreePositionalEncoder
from utils.confusion_matrix import confusion_matrix_pipeline


class GenreClassifier(L.LightningModule):
    _train_key = "train"
    _val_key = "val"
    _test_key = "test"

    def __init__(
        self,
        base_module: nn.Module,
        base_module_output_dim: int,
        num_classes: int,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
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
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.name = name
        self.class_names = (
            class_names if class_names is not None else [f"Class_{i}" for i in range(num_classes)]
        )
        self.criterion = loss_function
        self.metrics = nn.ModuleDict(
            {
                "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
                # "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
                # "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
            }
        )
        if hparams_to_log is not None:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            hparams_to_log["num_parameters"] = num_params
            self.save_hyperparameters(hparams_to_log)
        self.accumulated_predictions = {}
        self._reset_accumulated_predictions()

    def forward(self, x, **kwargs):
        # print(f"Before module: {x.shape = }")
        x = self.base_module(x, **kwargs)
        # print(f"After module: {x.shape = }")
        x = x[:, -1, :]  # Keep only last step
        # Add tempo
        if self.use_tempo:
            x = torch.cat([x, kwargs["tempo"]], dim=-1)
        # print(f'Before batch norm: {x.shape = }')
        x = self.batch_normalization(x)
        # print(f'After batch norm: {x.shape = }')
        x = self.dropout(x)
        # print(f'After dropout: {x.shape = }')
        x = self.linear(x)
        # print(f'End forward: {x.shape = }')
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def common_step(self, batch, key: str):
        # Get data
        x = batch["samples"]
        y = batch["labels"]
        pos = batch["_positions"]
        tempo = batch["tempos"] if self.use_tempo else None
        # Apply model
        y_hat = self.forward(x, tempo=tempo, positions=pos)
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

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, self._test_key)

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
        self.logger.log_image(f"{key}_confusion_matrix", images=[image], step=self.global_step)
        # Reset accumulated predictions
        self._reset_accumulated_predictions(key)

    def on_train_epoch_end(self):
        self.on_common_epoch_end(self._train_key)

    def on_validation_epoch_end(self):
        self.on_common_epoch_end(self._val_key)

    def on_test_epoch_end(self):
        self.on_common_epoch_end(self._test_key)

    # def on_train_start(self) -> None:
    #     # Log hyperparameters with first dummy metrics in order for them to appear in TensorBoard
    #     to_log = {}
    #     for part in ["train", "val"]:
    #         to_log.update({f"{part}/{k}": float("NaN") for k in self.metrics.keys()})
    #         to_log.update({f"{part}/loss": float("NaN")})
    #     self.logger.log_hyperparams(self.hparams)

    def _reset_accumulated_predictions(self, key: str | None = None):
        def default_dict():
            return {"y_true": [], "y_pred": []}

        if key is None:
            self.accumulated_predictions = {
                k: default_dict() for k in [self._train_key, self._val_key, self._test_key]
            }
        else:
            self.accumulated_predictions[key] = default_dict()


class LSTMModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )

    def forward(self, x, **kwargs):
        x, _ = self.lstm(x)
        return x


class GenreClassifierLSTM(GenreClassifier):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_tempo: bool = False,
        class_names: list[str] = None,
        name: str = None,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
        hparams_to_log: dict = None,
    ):
        bidirectional = True
        module = LSTMModule(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )

        super().__init__(
            base_module=module,
            base_module_output_dim=2 * hidden_size if bidirectional else hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lr=lr,
            weight_decay=weight_decay,
            use_tempo=use_tempo,
            class_names=class_names,
            name=name,
            loss_function=loss_function,
            hparams_to_log=hparams_to_log,
        )


class TransformerModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_features_per_head: int,
        feedforward_dim: int,
        num_layers: int,
        positional_encoder: Type[nn.Module],
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        # torch need embed_dim % nheads == 0, so we add a linear layer to adjust the input dimension
        encoder_input_dim = num_heads * num_features_per_head
        self.linear = nn.Linear(input_dim, encoder_input_dim)
        self.relu = nn.ReLU()
        self.pe = positional_encoder(d_model=encoder_input_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=encoder_input_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout_rate,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, **kwargs):
        x = self.linear(x)
        x = self.relu(x)
        if isinstance(self.pe, ContinuousPositionalEncoder | TreePositionalEncoder):
            x = self.pe(x, kwargs["positions"])
        else:
            x = self.pe(x)
        x = self.encoder(x)
        return x


class GenreClassifierTransformer(GenreClassifier):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_features_per_head: int,
        feedforward_dim: int,
        num_layers: int,
        num_classes: int,
        positional_encoder: Type[nn.Module],
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_tempo: bool = False,
        class_names: list[str] = None,
        name: str = None,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
        hparams_to_log: dict = None,
    ):
        # torch need embed_dim % nheads == 0, so we add a linear layer to adjust the input dimension
        module = TransformerModule(
            input_dim=input_dim,
            num_heads=num_heads,
            num_features_per_head=num_features_per_head,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            positional_encoder=positional_encoder,
            dropout_rate=dropout_rate,
        )
        super().__init__(
            base_module=module,
            base_module_output_dim=module.linear.out_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lr=lr,
            weight_decay=weight_decay,
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
