from pathlib import Path
import lightning as L
from fastapi import APIRouter
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import pandas as pd

from app.core.config import settings
from app.core.lse_dm import LseDataModule
from app.core.engine import Lse2TextModel
from app.core.model import CnnV1
from app.models.schemas import TrainRequest

router = APIRouter()


@router.post("/train")
async def train(request: TrainRequest):
    metrics_folder = None
    if request.metrics_filename:
        Path(settings.metrics_folder).mkdir(parents=True, exist_ok=True)
        metrics_folder = (
            Path(settings.metrics_folder) / f"{request.metrics_filename}.csv"
        )

    cm_img_name = (
        Path(settings.metrics_folder) / f"{request.metrics_filename}_cm.png"
        if request.metrics_filename
        else None
    )

    dm = LseDataModule(
        root_dir=settings.dataset_dir,
        image_size=(settings.img_width, settings.img_height),
        batch_size=request.batch_size,
    )

    model_config = {
        "input_channel": settings.img_channels,
        "out_channels": settings.classes,
        "hidden_units": [128, 64, 32],
        "adapt_size": (4, 4),
        "p": 0.5,
    }

    model = Lse2TextModel(
        model=CnnV1,
        config=model_config,
        cm_img_path=cm_img_name,
    )

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

    train_logger = (
        MLFlowLogger(experiment_name="lse2text", tracking_uri="file:./mlruns")
        if request.use_logger
        else None
    )

    trainer = L.Trainer(
        logger=train_logger,
        max_epochs=request.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint],
        fast_dev_run=request.debug,
        enable_model_summary=settings.verbose,
    )

    trainer.fit(model=model, datamodule=dm)
    metrics = trainer.test(model=model, datamodule=dm)

    if not request.debug and metrics_folder:
        df = pd.DataFrame(metrics, index=["value"]).T
        df.to_csv(metrics_folder)

    if request.out_model and not request.debug:
        Path(settings.models_folder).mkdir(parents=True, exist_ok=True)
        file_path = Path(settings.models_folder) / request.out_model

        if file_path.suffix != ".onnx":
            file_path = file_path.with_suffix(".onnx")

        if settings.verbose:
            print(f"Saving model to {file_path}")

        model.to_onnx(file_path=file_path, export_params=True)

    return {
        "metrics": metrics,
        "model_path": file_path,
        "metrics_file": str(metrics_folder) if request.metrics_filename else None,
    }
