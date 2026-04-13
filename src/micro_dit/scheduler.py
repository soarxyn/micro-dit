from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# From: https://huggingface.co/blog/annotated-diffusion
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def linear_scheduler(timesteps: int, beta0: float, betaT: float) -> torch.Tensor:
    return torch.linspace(beta0, betaT, timesteps, dtype=torch.float64)


def cosine_scheduler(timesteps: int, s: float) -> torch.Tensor:
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    alpha_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    return torch.clip(1.0 - (alpha_cumprod[1:] / alpha_cumprod[:-1]), 0.0001, 0.9999)


class DiffusionProcess(nn.Module):
    def __init__(
        self,
        *,
        timesteps: int = 1000,
        beta0: float = 1e-4,
        betaT: float = 2e-2,
        s: float = 0.008,
        scheduler: Literal["linear", "cosine"] = "linear",
    ):
        super().__init__()
        self.timesteps = timesteps

        if scheduler == "linear":
            betas = linear_scheduler(timesteps, beta0, betaT)
        elif scheduler == "cosine":
            betas = cosine_scheduler(timesteps, s)

        alpha_cumprod: torch.Tensor = torch.cumprod(1.0 - betas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alpha_cumprod", alpha_cumprod.float())

        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt().float())
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", (1.0 - alpha_cumprod).sqrt().float()
        )
        self.register_buffer("sqrt_recip_alphas", (1.0 / (1.0 - betas)).sqrt().float())

        alpha_cumprod_prev: torch.Tensor = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer(
            "posterior_variance",
            (betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).float(),
        )

        self.register_buffer(
            "posterior_mean_coef1",
            (alpha_cumprod_prev.sqrt() * betas / (1.0 - alpha_cumprod)).float(),
        )

        self.register_buffer(
            "posterior_mean_coef2",
            (
                (1.0 - alpha_cumprod_prev)
                * (1.0 - betas).sqrt()
                / (1.0 - alpha_cumprod)
            ).float(),
        )

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        return (
            extract(self.sqrt_alpha_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape) * noise
        )

    def q_sample_v(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise

        return (
            extract(self.sqrt_alpha_cumprod, t, x0.shape) * noise
            - extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape) * x0
        )
