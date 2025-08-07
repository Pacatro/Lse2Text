# LSE to Text

LSE to Text is a tool for translating [Spanish Sign Language (`LSE`)](https://en.wikipedia.org/wiki/Spanish_Sign_Language) images to text using deep learning. It provides a simple command line interface (`CLI`) for training and evaluating models.

---

## Features

- Train a custom model on Spanish Sign Language alphabet images
- Predict text from sign language images using a trained model
- Evaluate model performance using K-Fold Cross Validation
- Easy CLI for training and inference
- Configurable training parameters

---

## Requirements

- [`Python`](https://www.python.org/) 3.12+
- [`uv`](https://docs.astral.sh/uv/) (for dependency management)

## Installation

1. Clone this repository:

   ```terminal
   git clone https://github.com/Pacatro/Lse2Text
   cd Lse2Text
   ```

2. Run the program (this will install dependencies and create a virtual environment):

   ```terminal
   uv run src/main.py
   ```

---

## Usage

The CLI offers three main commands: `train`, `predict` and `eval`.

### General CLI

```terminal
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --verbose             -v        Verbose mode
  --install-completion            Install completion for the current shell.
  --show-completion               Show completion for the current shell, to copy it or customize the
                                  installation.
  --help                -h        Show this message and exit.

Commands:
  train     Train a model with the given parameters and save it to the given path.
  predict   Runs inference with the given model.
  eval      Runs a K-Fold Cross Validation evaluation.
```

---

### Train Command

Train a model with the given parameters and save it in [`ONNX`](https://onnx.ai/) format.

```terminal
Usage: main.py train [OPTIONS]

Options:
  --out-model         -o      TEXT     Model path in ONNX format [default: model.onnx]
  --epochs            -e      INTEGER  Number of train epochs [default: 50]
  --batch-size        -b      INTEGER  Batch size [default: 32]
  --debug             -d               Run in debug mode
  --metrics-filename  -m      TEXT     Metrics filename without extension [default: None]
  --use-logger        -l               Use a logger
  --help              -h               Show this message and exit.
```

**Example:**

```terminal
uv run src/main.py train -o model.onnx -e 20 -b 64
```

---

### Predict Command

Run inference with the given model.

```terminal
Usage: main.py predict [OPTIONS]

Options:
  --model-path  -m      TEXT     Model path [default: model.onnx]
  --max-preds   -p      INTEGER  Max number of predictions [default: 20]
  --help        -h               Show this message and exit.
```

**Example:**

```terminal
uv run src/main.py predict -m model.onnx -p 10
```

### Evaluate Command

Run a K-Fold Cross Validation evaluation.

```terminal
Usage: main.py eval [OPTIONS]

Options:
  --folds       -k      INTEGER  The number of folds for CV [default: 5]
  --batch-size  -b      INTEGER  Batch size [default: 32]
  --epochs      -e      INTEGER  Number of train epochs [default: 50]
  --help        -h               Show this message and exit.
```

**Example:**

```terminal
uv run src/main.py eval -k 10 -b 64
```

## Author

Created by [**Paco Algar Muñoz**](https://github.com/Pacatro).
