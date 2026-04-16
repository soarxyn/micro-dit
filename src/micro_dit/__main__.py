import torch
from lightning.pytorch.cli import LightningCLI

from micro_dit.lit import LitDiT


def cli():
    torch.set_float32_matmul_precision("medium")

    LightningCLI(model_class=LitDiT, save_config_callback=None)


if __name__ == "__main__":
    cli()
