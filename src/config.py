# Global state
state = {"verbose": False}

# Debugging
FAST_DEV_RUN = False

# Folders
CONFIG_FOLDER = "config"
METRICS_FOLDER = "metrics"

# Dataset
DATASET_DIR = "./data"
ZIP_FILE = DATASET_DIR + "/spanish-sign-language-alphabet-static.zip"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/kirlelea/spanish-sign-language-alphabet-static"

# Dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 1

# Training
EPOCHS = 50
BATCH_SIZE = 32
CLASSES = 19
MONITORING_METRIC = "val/loss"

# Inference
MAX_PREDS = 20

# Evaluation
K = 5
