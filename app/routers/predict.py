from fastapi import APIRouter, HTTPException
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
# import json

from app.core.config import settings
from app.core.lse_dm import LseDataModule
from app.models.schemas import PredictRequest

router = APIRouter()


@router.post("/predict")
def predict(request: PredictRequest):
    if request.max_preds <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"max_preds must be greater than 0 (got {request.max_preds})",
        )

    model_path = Path(settings.models_folder) / request.model_name
    print(f"Loading model from {model_path}")

    if not model_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Model path {model_path} not found"
        )

    dm = LseDataModule(
        root_dir=settings.dataset_dir,
        image_size=(settings.img_width, settings.img_height),
        max_preds=request.max_preds,
    )

    dm.setup("predict")
    classes = dm.dataset.classes

    preds = {}

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(model_path)
    inp_name = ort_session.get_inputs()[0].name

    for batch in dm.predict_dataloader():
        x, y = batch
        arr = x.cpu().numpy().astype(np.float32)

        logits = ort_session.run(None, {inp_name: arr})[0]
        pred = np.argmax(np.array(logits), axis=1).tolist()
        label = classes[y.item()]
        preds[label] = classes[pred[0]]

    assert preds, "There are no predictions"

    return preds
