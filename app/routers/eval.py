from fastapi import APIRouter
import pandas as pd

from app.core.config import settings
from app.core.lse_dm import LseDataModule
from app.core.evaluation import cross_validation
from app.models.schemas import EvalRequest
from app.core.model import ModelConfig

router = APIRouter()


@router.post("/eval")
async def eval(request: EvalRequest):
    dm = LseDataModule(
        root_dir=settings.dataset_dir,
        image_size=(settings.img_width, settings.img_height),
    )

    model_config = ModelConfig()

    results = cross_validation(
        model_config=model_config,
        dm=dm,
        k=request.k,
        batch_size=request.batch_size,
        epochs=request.epochs,
    )

    res = pd.DataFrame(results).mean()
    res.to_csv(f"{settings.metrics_folder}/cv_results.csv")
    return {"results": res.to_dict()}
