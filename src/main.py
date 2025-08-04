import typer
from typing_extensions import Annotated
import torch
import matplotlib.pyplot as plt

import config
from commands import app as cmd_app

app = typer.Typer(
    no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]}
)
app.add_typer(cmd_app)


def save_image(img: torch.Tensor, label: str, path: str):
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(label)
    plt.savefig(path)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False):
    if verbose:
        config.state["verbose"] = True


if __name__ == "__main__":
    app()
