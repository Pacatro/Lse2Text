from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import lightning as L
from typing import Type
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from engine import Lse2TextModel
import config
from train_dm import TrainLseDataModule


def cross_validation(
    model_cls: Type[nn.Module],
    dm: TrainLseDataModule,
    k: int = config.K,
    batch_size: int = config.BATCH_SIZE,
    epochs: int = config.EPOCHS,
) -> list[dict[str, float]]:
    results = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    dm.setup("fit")

    ds = dm.dataset

    idxs = list(range(len(ds)))

    for fold, (train_idx, val_idx) in enumerate(kf.split(idxs)):
        if config.state["verbose"]:
            print(f"Fold {fold + 1}/{k}")

        ds_train = Subset(ds, train_idx)
        ds_val = Subset(ds, val_idx)

        train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            ds_val, batch_size=batch_size, shuffle=False, num_workers=4
        )

        model = Lse2TextModel(
            model=model_cls(
                input_channel=config.IMG_CHANNELS, out_channels=config.CLASSES
            ),
            num_classes=config.CLASSES,
        )

        early_stopping = EarlyStopping(
            monitor=config.MONITORING_METRIC,
            patience=3,
            mode="min",
            verbose=config.state["verbose"],
        )

        checkpoint = ModelCheckpoint(
            monitor=config.MONITORING_METRIC,
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
            enable_model_summary=config.state["verbose"],
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        metrics = trainer.validate(model=model, dataloaders=val_loader)
        results.append(metrics[0])

    if config.state["verbose"]:
        print(f"Results:\n{results}")

    return results
