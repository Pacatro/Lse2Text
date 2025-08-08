import lightning as L
from torch import nn
import torch
from torchmetrics import (
    MetricCollection,
    Accuracy,
    F1Score,
    Precision,
    Recall,
)
from dataclasses import dataclass, field

from app.core.config import settings


@dataclass
class ModelConfig:
    input_channel: int = settings.img_channels
    hidden_units: list[int] = field(default_factory=lambda: settings.hidden_units)
    adapt_size: tuple[int, int] = settings.adapt_size
    p: float = settings.p
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    lr: float = 1e-3
    weight_decay: float = 1e-5


class Lse2TextModel(L.LightningModule):
    def __init__(
        self,
        config: ModelConfig,
        num_classes: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.example_input_array = torch.randn(
            (1, settings.img_channels, settings.img_width, settings.img_height)
        )
        self.config = config
        self.num_classes: int = num_classes

        layers = []
        input = self.config.input_channel

        for h in self.config.hidden_units:
            layers += [
                nn.Conv2d(
                    in_channels=input,
                    out_channels=h,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(h),
                nn.MaxPool2d(2),
            ]
            input = h

        self.features = nn.Sequential(*layers)

        self.adapt_pool = nn.AdaptiveAvgPool2d(self.config.adapt_size)
        flat_dim = (
            self.config.hidden_units[-1]
            * self.config.adapt_size[0]
            * self.config.adapt_size[1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.p),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.p),
            nn.Linear(in_features=4096, out_features=self.num_classes),
        )

        metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=self.num_classes),
                "precision": Precision(
                    task="multiclass", average="macro", num_classes=self.num_classes
                ),
                "recall": Recall(
                    task="multiclass", average="macro", num_classes=self.num_classes
                ),
                "f1": F1Score(
                    task="multiclass", average="macro", num_classes=self.num_classes
                ),
            }
        )

        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adapt_pool(x)
        return self.classifier(x)

    def _step(
        self,
        batch: tuple[torch.Tensor, int],
        prefix: str,
        metrics: MetricCollection | None = None,
    ) -> float:
        x, y = batch
        logits = self(x)

        loss = self.config.loss_fn(logits, y)
        self.log(f"{prefix}/loss", loss, prog_bar=True)

        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        if metrics:
            metrics.update(preds, y)

        return loss

    def training_step(self, batch: tuple[torch.Tensor, int]) -> float:
        return self._step(batch, prefix="train")

    def validation_step(self, batch: tuple[torch.Tensor, int]) -> float:
        return self._step(batch, prefix="val", metrics=self.val_metrics)

    def test_step(self, batch: tuple[torch.Tensor, int]) -> float:
        return self._step(batch, prefix="test", metrics=self.test_metrics)

    def predict_step(self, batch: tuple[torch.Tensor, int]):
        logits = self(batch[0])
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        return preds

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
