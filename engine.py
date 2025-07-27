import lightning as L
from torch import nn


class Lse2TextModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.loss_fn = loss_fn if not None else nn.CrossEntropyLoss()

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat
