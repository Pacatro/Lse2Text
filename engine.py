import lightning as L
from torch import nn
import torch
from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall


class Lse2TextModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss_fn: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.loss_fn = nn.CrossEntropyLoss() if not loss_fn else loss_fn
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        self.metrics = MetricCollection(
            Accuracy(num_classes=self.num_classes, task="multiclass"),
            Precision(num_classes=self.num_classes, task="multiclass"),
            Recall(num_classes=self.num_classes, task="multiclass"),
            F1Score(num_classes=self.num_classes, task="multiclass"),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        preds = torch.softmax(input=logits, dim=1)
        return logits, preds

    def _step(self, batch: tuple[torch.Tensor, int], prefix: str) -> float:
        x, y = batch
        logits, preds = self(x)
        loss = self.loss_fn(logits, y)
        self.log(f"{prefix}/loss", loss, prog_bar=True, on_epoch=True)
        self.metrics.update(preds, y)
        return loss

    def training_step(self, batch: torch.Tensor) -> float:
        return self._step(batch, prefix="train")

    def validation_step(self, batch: torch.Tensor) -> float:
        return self._step(batch, prefix="val")

    def test_step(self, batch: torch.Tensor) -> float:
        return self._step(batch, prefix="test")

    def on_validation_epoch_start(self):
        self.metrics.reset()

    def on_test_epoch_start(self):
        self.metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        self.log_dict(metrics)

    def on_test_epoch_end(self):
        metrics = self.metrics.compute()
        self.log_dict(metrics)

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
