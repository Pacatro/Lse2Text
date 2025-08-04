# LSE to Text

Translate Spanish Sign Language (LSE) to text.

## Installation

>[!NOTE] You need to have [`uv`](https://docs.astral.sh/uv/) installed

Follow the following steps to install the project:

1. Clone this repository:

  ```bash
  git clone https://github.com/Pacatro/Lse2Text
  cd Lse2Text
  ```

2. Run the program, this will install the dependencies and create a virtual environment:

  ```bash
  uv run src/main.py
  ```

## Usage

The program offers a command line interface (CLI) to train and predict LSE to text.

```bash
 Usage: main.py [OPTIONS] COMMAND [ARGS]...

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --verbose             -v                                                                                   │
│ --install-completion            Install completion for the current shell.                                  │
│ --show-completion               Show completion for the current shell, to copy it or customize the         │
│                                 installation.                                                              │
│ --help                -h        Show this message and exit.                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────╮
│ train                                                                                                      │
│ predict                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Train command

Train a model with the given parameters and save it to the given path.

```bash
 Usage: main.py train [OPTIONS]

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --out-model   -o      TEXT     [default: model.pt]                                                         │
│ --epochs      -e      INTEGER  [default: 10]                                                               │
│ --batch-size  -b      INTEGER  [default: 32]                                                               │
│ --debug       -d                                                                                           │
│ --help        -h               Show this message and exit.                                                 │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Predict command

Run inference with the given model.

```bash
 Usage: main.py predict [OPTIONS]

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --model-path  -m      TEXT     [default: model.pt]                                                         │
│ --max                 INTEGER  [default: 20]                                                               │
│ --help        -h               Show this message and exit.                                                 │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Author/s

Created by [**Paco Algar Muñoz**](https://github.com/Pacatro).
