import typer
import onnx
import onnxruntime as ort
import numpy as np
from typing_extensions import Annotated
from pathlib import Path
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
# import json

import config
from lse_dm import LseDataModule
from models import CnnV1
from engine import Lse2TextModel
from evaluation import cross_validation

app = typer.Typer(no_args_is_help=True)


@app.command(
    help="Train a model with the given parameters and save it to the given path."
)
def train(
    out_model: Annotated[
        str | None,
        typer.Option("--out-model", "-o", help="Model path in ONNX format"),
    ] = "model.onnx",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of train epochs"),
    ] = config.EPOCHS,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size"),
    ] = config.BATCH_SIZE,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Run in debug mode"),
    ] = config.FAST_DEV_RUN,
    metrics_filename: Annotated[
        str | None,
        typer.Option(
            "--metrics-filename", "-m", help="Metrics filename without extension"
        ),
    ] = None,
    use_logger: Annotated[
        bool,
        typer.Option("--use-logger", "-l", help="Use a logger"),
    ] = False,
):
    if metrics_filename:
        Path(config.METRICS_FOLDER).mkdir(parents=True, exist_ok=True)
        metrics_folder = Path(config.METRICS_FOLDER) / f"{metrics_filename}.csv"

    cm_img_name = (
        Path(config.METRICS_FOLDER) / f"{metrics_filename}_cm.png"
        if metrics_filename
        else None
    )

    dm = LseDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
        batch_size=batch_size,
    )

    model = Lse2TextModel(
        model=CnnV1(input_channel=config.IMG_CHANNELS, out_channels=config.CLASSES),
        num_classes=config.CLASSES,
        cm_img_path=str(cm_img_name),
    )

    if config.state["verbose"] and not debug:
        print(f"IMG SIZE: ({config.IMG_WIDTH}, {config.IMG_HEIGHT})")
        print(f"IMG CHANNELS: {config.IMG_CHANNELS}")
        print(f"CLASSES: {config.CLASSES}")
        print(f"EPOCHS: {epochs}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"MONITORING METRIC: {config.MONITORING_METRIC}")
        print(f"USE LOGGER: {use_logger}")
        print(f"MODEL:\n{model}\n")

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

    train_logger = (
        MLFlowLogger(experiment_name="lse2text", tracking_uri="file:./mlruns")
        if use_logger
        else None
    )

    trainer = L.Trainer(
        logger=train_logger,
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

    if not debug and metrics_folder:
        df = pd.DataFrame(metrics, index=["value"]).T
        df.to_csv(metrics_folder)

    if out_model:
        model.to_onnx(file_path=out_model, export_params=True)

        if config.state["verbose"]:
            print(f"MODEL SAVED TO {out_model}")

    # if save_config:
    #     Path(config.CONFIG_FOLDER).mkdir(parents=True, exist_ok=True)
    #     with open(
    #         f"{config.CONFIG_FOLDER}/{model.model.__class__.__name__}.json", "w"
    #     ) as f:
    #         json.dump(model.config, f, indent=4)


@app.command(help="Runs inference with the given model.")
def predict(
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Model path"),
    ] = "model.onnx",
    max_preds: Annotated[
        int,
        typer.Option("--max-preds", "-p", help="Max number of predictions"),
    ] = config.MAX_PREDS,
):
    dm = LseDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
        max_preds=max_preds,
    )

    dm.setup("predict")
    classes = dm.dataset.classes

    preds = []

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(model_path)
    inp_name = ort_session.get_inputs()[0].name

    for batch in dm.predict_dataloader():
        x, _ = batch
        arr = x.cpu().numpy().astype(np.float32)
        ort_out = ort_session.run(None, {inp_name: arr})[0]
        preds.append(np.argmax(np.array(ort_out), axis=1).tolist())

    assert preds, "There are no predictions"
    assert len(preds) == len(dm.predict_dataset), (
        "Predictions and dataset sizes are different"
    )

    for i in range(max_preds):
        _, label = dm.predict_dataset[i]
        print(f"Prediction: {classes[preds[i][0]]} | Label: {classes[label]}")


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
    dm = LseDataModule(
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

    pd.DataFrame(results).mean().to_csv(f"{config.METRICS_FOLDER}/cv_results.csv")
