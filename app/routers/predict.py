from fastapi import APIRouter, HTTPException, File, UploadFile
from pathlib import Path
from fastapi.responses import JSONResponse
from torchvision import transforms
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

from app.core.config import settings
from app.core.hands import select_hand
from app.core.lse_dm import LseDataModule

router = APIRouter()


def get_last_model(models_folder: str) -> Path:
    saving_models_folder = Path(models_folder)

    if not saving_models_folder.exists():
        raise HTTPException(
            status_code=404, detail=f"There are no models in {settings.models_folder}"
        )

    models = [f for f in saving_models_folder.iterdir() if f.is_file()]

    if len(models) == 0:
        raise HTTPException(
            status_code=404, detail=f"There are no models in {settings.models_folder}"
        )

    model_path = Path(settings.models_folder) / max(models).name

    if not model_path.suffix == ".onnx":
        raise HTTPException(
            status_code=400, detail=f"Model {model_path} is not an ONNX model"
        )

    return model_path


@router.post("/predict")
async def predict(img: UploadFile = File(..., alias="file")) -> JSONResponse:
    if img is None:
        raise HTTPException(
            status_code=400, detail="Image file is required for prediction"
        )

    model_path = get_last_model(settings.models_folder)

    print(f"Loading model from {model_path}")

    img_transform = transforms.Compose(
        [
            transforms.Resize((settings.img_width, settings.img_height)),
            transforms.CenterCrop((settings.img_width, settings.img_height)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    img_content = await img.read()

    img_bgr = cv2.imdecode(np.frombuffer(img_content, np.uint8), cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    crop_img = select_hand(img_bgr, (settings.img_width, settings.img_height))

    cv2.imwrite("tmp.jpg", crop_img)  # For debugging

    image = Image.fromarray(crop_img).convert("RGB")

    tensor_image = img_transform(image).unsqueeze(0)

    if not model_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Model path {model_path} not found"
        )

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(model_path)
    inp_name = ort_session.get_inputs()[0].name

    x = tensor_image.cpu().numpy().astype(np.float32)

    logits = np.array(ort_session.run(None, {inp_name: x})[0])
    pred = np.argmax(logits, axis=1).tolist()

    dm = LseDataModule(
        root_dir=settings.dataset_dir,
        image_size=(settings.img_width, settings.img_height),
    )

    dm.setup("predict")

    classes = dm.dataset.classes

    assert pred, "There are no predictions"

    return JSONResponse({"pred": classes[pred[0]], "logits": logits.tolist()})
