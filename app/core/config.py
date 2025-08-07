from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Global
    verbose: bool = False
    fast_dev_run: bool = False
    models_folder: str = "saving_models"

    # Folders
    config_folder: str = "config"
    metrics_folder: str = "metrics"

    # Dataset
    dataset_dir: str = "./data"
    zip_file: str = "./data/spanish-sign-language-alphabet-static.zip"
    kaggle_url: str = "https://www.kaggle.com/api/v1/datasets/download/kirlelea/spanish-sign-language-alphabet-static"

    # Image Dimensions
    img_width: int = 224
    img_height: int = 224
    img_channels: int = 1

    # Training
    epochs: int = 50
    batch_size: int = 32
    classes: int = 19
    monitoring_metric: str = "val/loss"

    # Inference
    max_preds: int = 20

    # Evaluation
    k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
