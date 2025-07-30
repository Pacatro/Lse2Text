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

        metrics = MetricCollection(
            {
                "accuracy": Accuracy(num_classes=self.num_classes, task="multiclass"),
                "precision": Precision(num_classes=self.num_classes, task="multiclass"),
                "recall": Recall(num_classes=self.num_classes, task="multiclass"),
                "f1": F1Score(num_classes=self.num_classes, task="multiclass"),
            }
        )

        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    def _step(
        self,
        batch: tuple[torch.Tensor, int],
        prefix: str,
        metrics: MetricCollection | None = None,
    ) -> float:
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, y)
        self.log(f"{prefix}/loss", loss, prog_bar=True, on_epoch=True)

        if metrics:
            metrics.update(preds, y)

        return loss

    def training_step(self, batch: torch.Tensor) -> float:
        return self._step(batch, prefix="train")

    def validation_step(self, batch: torch.Tensor) -> float:
        return self._step(batch, prefix="val", metrics=self.val_metrics)

    def test_step(self, batch: torch.Tensor) -> float:
        return self._step(batch, prefix="test", metrics=self.test_metrics)

    def predict_step(self, batch: torch.Tensor):
        x, _ = batch
        return self(x)

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

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
