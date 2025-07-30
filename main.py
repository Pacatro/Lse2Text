from datamodule import LSEDataModule
import config
from model import VGG
from engine import Lse2TextModel
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import matplotlib.pyplot as plt


def save_image(img: torch.Tensor, label: str, path: str):
    plt.imshow(img.squeeze())
    plt.title(label)
    plt.savefig(path)


def main():
    dm = LSEDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
    )

    model = Lse2TextModel(
        model=VGG(
            input_channel=config.IMG_CHANNELS,
            out_shape=config.CLASSES,
            hidden_units=[10],
        ),
        num_classes=config.CLASSES,
    )
    print(model)

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=3,
        mode="min",
        verbose=True,
    )

    checkpoint = ModelCheckpoint(monitor="val/loss", mode="min", filename="best-model")

    trainer = L.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint],
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.validate(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
