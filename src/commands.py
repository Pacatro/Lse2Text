import typer
from typing_extensions import Annotated
from pathlib import Path
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import json

import config
from train_dm import TrainLseDataModule
from models import CnnV1
from engine import Lse2TextModel
from evaluation import cross_validation

app = typer.Typer(no_args_is_help=True)


@app.command(
    help="Train a model with the given parameters and save it to the given path."
)
def train(
    out_model: Annotated[
        str,
        typer.Option("--out-model", "-o", help="Model path"),
    ] = "model.pt",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of train epochs"),
    ] = 10,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size"),
    ] = config.BATCH_SIZE,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Run in debug mode"),
    ] = config.FAST_DEV_RUN,
    save_config: Annotated[
        bool,
        typer.Option("--save-config", "-s", help="Save config"),
    ] = False,
):
    metrics_folder = f"{config.METRICS_FOLDER}/test_metrics.csv"

    if config.state["verbose"] and not debug:
        print(f"IMG SIZE: ({config.IMG_WIDTH}, {config.IMG_HEIGHT})")
        print(f"IMG CHANNELS: {config.IMG_CHANNELS}")
        print(f"CLASSES: {config.CLASSES}")
        print(f"EPOCHS: {epochs}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"MONITORING METRIC: {config.MONITORING_METRIC}")
        print(f"METRICS FOLDER: {metrics_folder}")
        print(f"OUT MODEL: {out_model}")

    dm = TrainLseDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
        batch_size=batch_size,
    )

    model = Lse2TextModel(
        model=CnnV1(input_channel=config.IMG_CHANNELS, out_channels=config.CLASSES),
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
        fast_dev_run=debug,
        enable_model_summary=config.state["verbose"],
    )

    trainer.fit(model=model, datamodule=dm)
    metrics = trainer.test(model=model, datamodule=dm)

    if not debug:
        df = pd.DataFrame(metrics, index=["value"]).T
        df.to_csv(metrics_folder)
        Path(checkpoint.best_model_path).rename(out_model)

    if save_config:
        Path(config.CONFIG_FOLDER).mkdir(parents=True, exist_ok=True)
        with open(
            f"{config.CONFIG_FOLDER}/{model.model.__class__.__name__}.json", "w"
        ) as f:
            json.dump(model.config, f, indent=4)


@app.command(help="Runs inference with the given model.")
def predict(
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Model path"),
    ] = "model.pt",
    max: Annotated[
        int,
        typer.Option("--max-predictions", "-p", help="Max predictions"),
    ] = 20,
):
    cnn = CnnV1(input_channel=config.IMG_CHANNELS, out_channels=config.CLASSES)

    dm = TrainLseDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
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


@app.command(help="Runs a K-Fold Cross Validation evaluation.")
def eval(
    k: Annotated[
        int, typer.Option("--folds", "-k", help="The number of folds for CV")
    ] = 5,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size")
    ] = config.BATCH_SIZE,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of train epochs")
    ] = config.EPOCHS,
):
    dm = TrainLseDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
    )

    results = cross_validation(
        model_cls=CnnV1,
        dm=dm,
        k=k,
        batch_size=batch_size,
        epochs=epochs,
    )

    pd.DataFrame(results).to_csv(f"{config.METRICS_FOLDER}/cv_metrics.csv")
