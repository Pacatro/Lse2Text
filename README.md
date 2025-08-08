# LSE to Text API

LSE to Text is a REST API for translating [Spanish Sign Language (`LSE`)](https://en.wikipedia.org/wiki/Spanish_Sign_Language) images to text using deep learning. It provides endpoints for training, prediction, and model evaluation, along with a simple web interface for real-time inference.

---

## Features

- **RESTful API**: A robust API built with FastAPI for LSE translation tasks.
- **Train**: Train a custom model on the Spanish Sign Language alphabet dataset via the `/train` endpoint.
- **Predict**: Predict text from sign language images using a trained model via the `/predict` endpoint.
- **Evaluate**: Evaluate model performance using K-Fold Cross-Validation via the `/eval` endpoint.
- **Web Interface**: A simple frontend to interact with the API, capture images from a webcam, and get real-time predictions.
- **Configurable**: Easily configure model and training parameters.

---

## Requirements

- [`Python`](https://www.python.org/) 3.12+
- [`uv`](https://docs.astral.sh/uv/) (for dependency management)
- [`onnxruntime`](https://onnxruntime.ai/) and a trained model for prediction.

## Installation

1. Clone this repository:

   ```terminal
   git clone https://github.com/Pacatro/Lse2Text
   cd Lse2Text
   ```

2. Install dependencies and create a virtual environment:

   ```terminal
   uv sync
   ```

3. Run the application:

   ```terminal
   uv run uvicorn app.main:app --reload
   ```

The API will be available at `http://127.0.0.1:8000`.

---

## API Endpoints

The API provides the following endpoints:

### `GET /`

Serves the main web interface, which allows you to use your webcam to capture an image and send it for prediction.

### `POST /train`

Trains a new model with the given parameters and saves it.

**Request Body:** `application/json`

| Parameter      | Type    | Description                             | Default |
| -------------- | ------- | --------------------------------------- | ------- |
| `debug`        | boolean | Run in debug mode (fast dev run).       | `false` |
| `epochs`       | integer | Number of training epochs.              | `50`    |
| `batch_size`   | integer | Batch size for training.                | `32`    |
| `use_logger`   | boolean | Use MLFlow logger.                      | `false` |
| `save_metrics` | boolean | Save performance metrics to a CSV file. | `true`  |
| `save_model`   | boolean | Save the trained model in ONNX format.  | `true`  |

**Example Request:**

```bash
curl -X POST "http://127.0.0.1:8000/train" -H "Content-Type: application/json" -d '{
  "epochs": 25,
  "batch_size": 64
}'
```

**Example Response:**

```json
{
  "metrics": [
    {
      "test/acc": 0.98,
      "test/precision": 0.98,
      "test/recall": 0.98,
      "test/f1": 0.98
    }
  ],
  "model_path": "saving_models/LseTrasnlator_20250808_120000.onnx",
  "metrics_file": "metrics/LseTrasnlator_20250808_120000.csv"
}
```

### `POST /predict`

Runs inference on an uploaded image file using the latest trained model.

**Request Body:** `multipart/form-data`

- `file`: The image file to be processed.

**Example Request:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

**Example Response:**

```json
{
  "model": "LseTrasnlator_20250808_120000.onnx",
  "pred": "A",
  "logits": [[...]]
}
```

### `POST /eval`

Runs a K-Fold Cross-Validation evaluation.

**Request Body:** `application/json`

| Parameter    | Type    | Description                         | Default |
| ------------ | ------- | ----------------------------------- | ------- |
| `k`          | integer | The number of folds for CV.         | `5`     |
| `batch_size` | integer | Batch size for evaluation.          | `32`    |
| `epochs`     | integer | Number of training epochs per fold. | `50`    |

**Example Request:**

```bash
curl -X POST "http://127.0.0.1:8000/eval" -H "Content-Type: application/json" -d '{
  "k": 10,
  "epochs": 20
}'
```

**Example Response:**

```json
{
  "results": {
    "val/loss": 0.1,
    "val/acc": 0.97,
    "val/precision": 0.97,
    "val/recall": 0.97,
    "val/f1": 0.97
  }
}
```

---

## Author

Created by [**Paco Algar Muñoz**](https://github.com/Pacatro).
