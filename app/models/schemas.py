# Define a Pydantic model for the request body
from pydantic import BaseModel


class TrainRequest(BaseModel):
    """Training request parameters.
    Defines the expected structure of the request body for the training endpoint.
    It includes optional parameters for model output path, number of epochs, batch size, debug mode, metrics filename, and logger usage.
    """

    out_model: str = "model.onnx"
    epochs: int = 10
    batch_size: int = 32
    debug: bool = False
    metrics_filename: str | None = None
    use_logger: bool = False


class InferenceRequest(BaseModel):
    """Inference request parameters.
    Defines the expected structure of the request body for the inference endpoint.
    It includes the model's name and the maximum number of predictions to return.
    """

    model_name: str = "model.onnx"
    max_preds: int = 20


class EvalRequest(BaseModel):
    """Evaluation request parameters.
    Defines the expected structure of the request body for the evaluation endpoint.
    It includes optional parameters for the number of folds, batch size, and number of epochs.
    """

    k: int = 5
    batch_size: int = 32
    epochs: int = 10
