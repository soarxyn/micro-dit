import lightning as L
import torch
import wandb
from torchvision.utils import make_grid
from lightning.pytorch.callbacks import WeightAveraging
from lightning.pytorch.loggers import WandbLogger
from torch.optim.swa_utils import get_ema_avg_fn

from micro_dit.lit import LitDiT


class SampleCallback(L.Callback):
    def __init__(self, every_n_steps: int = 10000, num_samples: int = 8):
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps
        self.fixed_noise: torch.Tensor | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not isinstance(pl_module, LitDiT):
            return

        self.fixed_noise = torch.randn(
            (
                self.num_samples,
                pl_module.in_channels,
                pl_module.image_size,
                pl_module.image_size,
            ),
            device=pl_module.device,
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        if (
            trainer.global_step == 0
            or trainer.global_step % self.every_n_steps != 0
            or not isinstance(pl_module, LitDiT)
            or self.fixed_noise is None
        ):
            return

        latents = pl_module.p_sample_loop(
            self.fixed_noise.shape, device=pl_module.device, noise=self.fixed_noise
        )

        images = (pl_module.decode(latents) * 0.5 + 0.5).clamp(0, 1)

        grid = make_grid(images, nrow=self.num_samples)

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(
                {"samples": wandb.Image(grid)},
                step=trainer.global_step,
            )


class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay: float = 0.9999):
        super().__init__(avg_fn=get_ema_avg_fn(decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= 100)
