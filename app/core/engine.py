from pathlib import Path
from typing import Type
import lightning as L
from torch import nn
import torch
from torchmetrics import (
    MetricCollection,
    Accuracy,
    F1Score,
    Precision,
    Recall,
    ConfusionMatrix,
    Metric,
)

from app.core.config import settings


class Lse2TextModel(L.LightningModule):
    def __init__(
        self,
        model: Type[nn.Module],
        config: dict,
        loss_fn: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        cm_img_path: Path | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn, cm_img_path"])
        self.example_input_array = torch.randn(
            (1, settings.img_channels, settings.img_width, settings.img_height)
        )
        self.loss_fn = nn.CrossEntropyLoss() if not loss_fn else loss_fn
        self.model = model(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.config = config
        self.num_classes: int = int(self.config["out_channels"])
        self.cm_img_path = cm_img_path

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

        self.test_confmat = (
            ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
            if cm_img_path
            else None
        )

        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(
        self,
        batch: tuple[torch.Tensor, int],
        prefix: str,
        metrics: MetricCollection | None = None,
        confmat: Metric | None = None,
    ) -> float:
        x, y = batch
        logits = self(x)

        loss = self.loss_fn(logits, y)
        self.log(f"{prefix}/loss", loss, prog_bar=True)

        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        if metrics:
            metrics.update(preds, y)

        if confmat:
            confmat.update(preds, y)

        return loss

    def training_step(self, batch: tuple[torch.Tensor, int]) -> float:
        return self._step(batch, prefix="train")

    def validation_step(self, batch: tuple[torch.Tensor, int]) -> float:
        return self._step(batch, prefix="val", metrics=self.val_metrics)

    def test_step(self, batch: tuple[torch.Tensor, int]) -> float:
        return self._step(
            batch, prefix="test", metrics=self.test_metrics, confmat=self.test_confmat
        )

    def predict_step(self, batch: tuple[torch.Tensor, int]):
        logits = self(batch[0])
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        return preds

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def on_test_epoch_start(self):
        self.test_metrics.reset()
        if self.test_confmat:
            self.test_confmat.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def on_test_end(self):
        if self.test_confmat:
            fig, _ = self.test_confmat.plot()
            fig.savefig(self.cm_img_path)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
