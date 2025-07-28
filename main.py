from datamodule import LSEDataModule
import config
from model import VGG
from engine import Lse2TextModel
import lightning as L
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
        out_channels=config.IMG_CHANNELS,
    )

    dm.setup("fit")
    img, label = dm.train_dataset[0]
    save_image(img, label, f"train_{label}.png")

    model = Lse2TextModel(
        model=VGG(
            input_channel=config.IMG_CHANNELS,
            out_shape=config.CLASSES,
            hidden_units=[10],
        ),
        num_classes=config.CLASSES,
    )
    print(model)

    trainer = L.Trainer(max_epochs=config.EPOCHS)
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
