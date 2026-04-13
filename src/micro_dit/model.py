import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Try to match HF's diffusers implementation (https://github.com/huggingface/diffusers/blob/a08c274c3326579c69b065701f0a7e1982be632a/src/diffusers/models/embeddings.py#L26)
def get_timestep_embedding(
    timesteps: torch.Tensor, d_emb: int = 128, theta: int = 10000
) -> torch.Tensor:
    half_dim: int = d_emb // 2

    freqs: torch.Tensor = -math.log(theta) * torch.arange(
        0, half_dim, dtype=torch.float32, device=timesteps.device
    )
    freqs = freqs / (half_dim - 1)

    embedding: torch.Tensor = freqs.exp()
    embedding = torch.outer(timesteps, embedding)

    embedding = rearrange([embedding.sin(), embedding.cos()], "b ... d -> ... (b d)")

    if d_emb % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))

    return embedding


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_head: int):
        super().__init__()

        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * num_heads * d_head)
        self.out_proj = nn.Linear(num_heads * d_head, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        QKV = self.qkv_proj(x)

        Q, K, V = rearrange(QKV, "b n (r h d) -> r b h n d", r=3, h=self.num_heads)

        A = F.scaled_dot_product_attention(Q, K, V)
        A = rearrange(A, "b h n d -> b n (h d)")

        return self.out_proj(A)


def SimpleFFN(d_model: int, d_ff: int):
    return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        d_ff: int,
        ffn_type: Literal["simple", "swiglu"],
    ):
        super().__init__()

        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

        self.attention = Attention(d_model, num_heads, d_head)
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False, eps=1e-6)

        FFN = {"simple": SimpleFFN, "swiglu": SwiGLUFFN}
        self.ffn = FFN[ffn_type](d_model, d_ff)

        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        g1, b1, a1, g2, b2, a2 = rearrange(self.adaln_mlp(t), "b (r d) -> r b 1 d", r=6)

        x = x + a1 * self.attention(g1 * self.norm1(x) + b1)
        x = x + a2 * self.ffn(g2 * self.norm2(x) + b2)

        return x


class FinalHead(nn.Module):
    def __init__(self, d_model: int, patch_size: int, out_channels: int):
        super().__init__()

        self.norm = nn.RMSNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(d_model, patch_size * patch_size * out_channels)

        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale, shift = rearrange(self.adaln_mlp(t), "b (r d) -> r b 1 d", r=2)
        x = self.out_proj(scale * self.norm(x) + shift)

        return x


class Patchify(nn.Module):
    def __init__(
        self, in_channels: int, patch_size: int, image_size: int, d_model: int
    ):
        super().__init__()

        self.patch_size = patch_size

        num_patches = (image_size // patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, d_model))

        self.out_proj = nn.Linear(in_channels * patch_size * patch_size, d_model)

    def forward(self, x: torch.Tensor):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return self.out_proj(x) + self.pos_emb


class DiTModel(nn.Module):
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
    ):
        super().__init__()

        self.patchify = Patchify(in_channels, patch_size, image_size, d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        self.layers = nn.ModuleList(
            [
                DiTBlock(d_model, num_heads, d_head, d_ff, ffn_type)
                for _ in range(num_layers)
            ]
        )

        self.out = FinalHead(d_model, patch_size, in_channels)

        self.d_time = d_time

        self.h = image_size // patch_size
        self.patch_size = patch_size

        self._init_weights()

    def _init_weights(self):
        def _xavier_init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_xavier_init)

        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

        for block in self.layers:
            nn.init.zeros_(block.adaln_mlp[-1].weight)  # type: ignore
            nn.init.zeros_(block.adaln_mlp[-1].bias)  # type: ignore

        nn.init.zeros_(self.out.adaln_mlp[-1].weight)  # type: ignore
        nn.init.zeros_(self.out.adaln_mlp[-1].bias)  # type: ignore
        nn.init.zeros_(self.out.out_proj.weight)
        nn.init.zeros_(self.out.out_proj.bias)

        nn.init.trunc_normal_(self.patchify.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time = get_timestep_embedding(time, d_emb=self.d_time)
        t = self.time_mlp(time)

        x = self.patchify(x)

        for layer in self.layers:
            x = layer(x, t)

        v = self.out(x, t)

        return rearrange(
            v,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            h=self.h,
        )
