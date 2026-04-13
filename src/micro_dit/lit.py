from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from micro_dit.model import DiTModel
from micro_dit.scheduler import DiffusionProcess, extract


class LitDiT(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        num_layers: int,
        d_model: int,
        d_time: int,
        num_heads: int,
        d_head: int,
        d_ff: int,
        ffn_type: Literal["simple", "swiglu"],
        timesteps: int = 1000,
        beta0: float = 1e-4,
        betaT: float = 2e-2,
        s: float = 0.008,
        scheduler: Literal["linear", "cosine"] = "linear",
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = DiTModel(
            in_channels,
            image_size,
            patch_size,
            num_layers,
            d_model,
            d_time,
            num_heads,
            d_head,
            d_ff,
            ffn_type,
        )

        self.timesteps = timesteps

        self.diffusion = DiffusionProcess(
            timesteps=timesteps, beta0=beta0, betaT=betaT, s=s, scheduler=scheduler
        )

        self.example_input_array = torch.zeros(1, in_channels, image_size, image_size)

        self.lr = lr

    @torch.inference_mode()
    def p_sample(self, x: torch.Tensor, t_idx: int, clip: bool = True) -> torch.Tensor:
        t: torch.Tensor = torch.full(
            (x.shape[0],), t_idx, device=x.device, dtype=torch.long
        )

        v = self.model(x, t)

        x0 = (
            extract(self.diffusion.sqrt_alpha_cumprod, t, x.shape) * x
            - extract(self.diffusion.sqrt_one_minus_alpha_cumprod, t, x.shape) * v
        )

        if clip:
            x0 = x0.clamp(-1, 1)

        model_mean = (
            extract(self.diffusion.posterior_mean_coef1, t, x.shape) * x0
            + extract(self.diffusion.posterior_mean_coef2, t, x.shape) * x
        )

        if t_idx == 0:
            return model_mean

        posterior_variance_t = extract(self.diffusion.posterior_variance, t, x.shape)

        return model_mean + posterior_variance_t.sqrt() * torch.randn_like(x)

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape: tuple[int, ...],
        device: torch.device | str = "cuda",
        clip: bool = True,
    ):
        image: torch.Tensor = torch.rand(shape, device=device)

        for t in reversed(range(0, self.timesteps)):
            image = self.p_sample(image, t, clip)

        return image

    @torch.inference_mode()
    def p_sample_ddim(
        self,
        shape: tuple[int, ...],
        device: torch.device | str = "cuda",
        clip: bool = True,
        num_timesteps: int = 50,
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    # TODO: Add DDIM sample for DiT
    # x₀_pred = √ᾱₜ · xₜ − √(1−ᾱₜ) · v
    # ε_pred  = √ᾱₜ · v  + √(1−ᾱₜ) · xₜ

    def p_losses(self, x0: torch.Tensor, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise

        x = self.diffusion.q_sample(x0=x0, t=t, noise=noise)
        v = self.diffusion.q_sample_v(x0=x0, t=t, noise=noise)

        y = self.model(x, t)

        return F.mse_loss(v, y)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self): ...

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)
