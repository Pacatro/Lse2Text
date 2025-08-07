from pydantic import BaseModel

from app.core.config import settings


class TrainRequest(BaseModel):
    """Training request parameters.
    Defines the expected structure of the request body for the training endpoint.
    It includes optional parameters for model output path, number of epochs, batch size, debug mode, metrics filename, and logger usage.
    """

    out_model: str = "model.onnx"
    epochs: int = settings.epochs
    batch_size: int = settings.batch_size
    debug: bool = settings.fast_dev_run
    metrics_filename: str | None = None
    use_logger: bool = False


class PredictRequest(BaseModel):
    """Inference request parameters.
    Defines the expected structure of the request body for the inference endpoint.
    It includes the model's name and the maximum number of predictions to return.
    """

    model_name: str = "model.onnx"
    max_preds: int = settings.max_preds


class EvalRequest(BaseModel):
    """Evaluation request parameters.
    Defines the expected structure of the request body for the evaluation endpoint.
    It includes optional parameters for the number of folds, batch size, and number of epochs.
    """

    k: int = settings.k
    batch_size: int = settings.batch_size
    epochs: int = settings.epochs
