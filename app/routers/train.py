from pathlib import Path
import lightning as L
from fastapi import APIRouter
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import pandas as pd
from datetime import datetime

from app.core.config import settings
from app.core.lse_dm import LseDataModule
from app.core.model import LseTrasnlator, ModelConfig
from app.models.schemas import TrainRequest

router = APIRouter()


@router.post("/train")
async def train(request: TrainRequest):
    model_config = ModelConfig()

    dm = LseDataModule(
        root_dir=settings.dataset_dir,
        image_size=(settings.img_width, settings.img_height),
        batch_size=request.batch_size,
    )

    dm.setup("fit")

    model = LseTrasnlator(config=model_config, num_classes=len(dm.dataset.classes))

    out_model = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    if request.debug:
        return {"message": "Training completed in debug mode"}

    metrics = trainer.test(model=model, datamodule=dm)

    metrics_folder = None
    file_path = None

    if request.save_metrics:
        Path(settings.metrics_folder).mkdir(parents=True, exist_ok=True)
        metrics_folder = Path(settings.metrics_folder) / f"{out_model}.csv"
        df = pd.DataFrame(metrics, index=["value"]).T
        df.to_csv(metrics_folder)

    if request.save_model:
        Path(settings.models_folder).mkdir(parents=True, exist_ok=True)
        file_path = Path(settings.models_folder) / f"{out_model}.onnx"
        print(f"Saving model to {file_path}")
        model.to_onnx(file_path=file_path, export_params=True)

    return {
        "metrics": metrics,
        "model_path": file_path if file_path else None,
        "metrics_file": metrics_folder if metrics_folder else None,
    }
