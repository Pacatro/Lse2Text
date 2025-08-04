from train_dm import TrainLseDataModule
import config
from models import CNN_01
from engine import Lse2TextModel
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def save_image(img: torch.Tensor, label: str, path: str):
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(label)
    plt.savefig(path)


def train_model(out_model: str = "model.pt"):
    dm = TrainLseDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
    )

    model = Lse2TextModel(
        model=CNN_01(input_channel=config.IMG_CHANNELS, out_channels=config.CLASSES),
        num_classes=config.CLASSES,
    )

    early_stopping = EarlyStopping(
        monitor=config.MONITORING_METRIC,
        patience=3,
        mode="min",
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        monitor=config.MONITORING_METRIC,
        mode="min",
        filename="best-model",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint],
        fast_dev_run=config.FAST_DEV_RUN,
    )

    trainer.fit(model=model, datamodule=dm)
    metrics = trainer.test(model=model, datamodule=dm)

    df = pd.DataFrame(metrics, index=["value"]).T
    df.to_csv(f"{config.METRICS_FOLDER}/test_metrics.csv")

    Path(checkpoint.best_model_path).rename(out_model)


def predict_model(model_path: str = "model.pt", max: int = 20):
    cnn = CNN_01(input_channel=config.IMG_CHANNELS, out_channels=config.CLASSES)

    dm = TrainLseDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
    )

    dm.setup("predict")
    classes = dm.dataset.classes

    model = Lse2TextModel.load_from_checkpoint(model_path, model=cnn)
    trainer = L.Trainer()
    preds = trainer.predict(model=model, datamodule=dm)

    assert preds, "There are no predictions"
    assert len(preds) == len(dm.predict_dataset), (
        "Predictions and dataset sizes are different"
    )

    for i in range(max):
        pred = preds[i].item()
        _, label = dm.predict_dataset[i]
        print(f"Prediction: {classes[pred]} | Label: {classes[label]}")


def main():
    # train_model()
    predict_model()


if __name__ == "__main__":
    main()
