from pydantic import BaseModel

from app.core.config import settings


class TrainRequest(BaseModel):
    """Training request parameters.
    Defines the expected structure of the request body for the training endpoint.
    It includes optional parameters for model output path, number of epochs, batch size, debug mode, metrics filename, and logger usage.
    """

    debug: bool = settings.fast_dev_run
    epochs: int = settings.epochs
    batch_size: int = settings.batch_size
    use_logger: bool = False
    save_metrics: bool = True
    save_model: bool = True


class EvalRequest(BaseModel):
    """Evaluation request parameters.
    Defines the expected structure of the request body for the evaluation endpoint.
    It includes optional parameters for the number of folds, batch size, and number of epochs.
    """

    k: int = settings.k
    batch_size: int = settings.batch_size
    epochs: int = settings.epochs
