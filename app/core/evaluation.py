from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import lightning as L
from typing import Type
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

from app.core.engine import Lse2TextModel
from app.core.config import settings
from app.core.lse_dm import LseDataModule


def cross_validation(
    model_cls: Type[nn.Module],
    dm: LseDataModule,
    k: int = settings.k,
    batch_size: int = settings.batch_size,
    epochs: int = settings.epochs,
) -> list[dict[str, float]]:
    results = []

    kf = KFold(n_splits=settings.k, shuffle=True, random_state=42)

    dm.setup("fit")

    ds = dm.dataset
    idxs = np.arange(len(ds))

    model_config = {
        "input_channel": settings.img_channels,
        "out_channels": settings.classes,
        "hidden_units": [128, 64, 32],
        "adapt_size": (4, 4),
        "p": 0.5,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(idxs)):
        if settings.verbose:
            print(f"Fold {fold + 1}/{k}")

        ds_train = Subset(ds, train_idx.tolist())
        ds_val = Subset(ds, val_idx.tolist())

        train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            ds_val, batch_size=batch_size, shuffle=False, num_workers=4
        )

        model = Lse2TextModel(model=model_cls, config=model_config)

        early_stopping = EarlyStopping(
            monitor=settings.monitoring_metric,
            patience=3,
            mode="min",
            verbose=settings.verbose,
        )

        checkpoint = ModelCheckpoint(
            monitor=settings.monitoring_metric,
            mode="min",
            filename="best-model",
            save_last=True,
        )

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            callbacks=[early_stopping, checkpoint],
            enable_model_summary=settings.verbose,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        metrics = trainer.validate(model=model, dataloaders=val_loader)
        results.append(metrics[0])

    if settings.verbose:
        print(f"Results:\n{results}")

    return results
