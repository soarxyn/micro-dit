import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import lightning as L
from einops.layers.torch import Rearrange


class Codebook(nn.Module):
    def __init__(self, size: int, emb_dim: int, beta: float = 0.25):
        super().__init__()

        self.embedding = nn.Embedding(size, emb_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)

        self.hidden_dim = emb_dim
        self.beta = beta

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        x_q = self.embedding(indices)
        return rearrange(x_q, "... h w c -> ... c h w").contiguous()


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class Attention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, head_channels: int):
        super().__init__()

        self.num_heads = num_heads

        self.qkv_proj = nn.Conv2d(
            in_channels, 3 * num_heads * head_channels, kernel_size=1, bias=False
        )
        self.out_proj = nn.Conv2d(
            num_heads * head_channels, in_channels, kernel_size=1, bias=False
        )

        self.norm = RMSNorm(in_channels)

        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]

        res = x
        x = self.norm(x)

        QKV = self.qkv_proj(x)

        Q, K, V = rearrange(
            QKV, "b (r heads d) h w -> r b heads (h w) d", r=3, heads=self.num_heads
        )

        A = F.scaled_dot_product_attention(Q, K, V)
        A = rearrange(A, "b heads (h w) d -> b (heads d) h w", h=h, w=w)

        return self.out_proj(A) + res


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = RMSNorm(out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)

        return self.dropout(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.block_1 = Block(in_channels, out_channels, dropout=dropout)
        self.block_2 = Block(out_channels, out_channels)

        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.block_1(x)
        h = self.block_2(h)

        return h + self.res_conv(x)


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_channels: int,
        z_channels: int,
        num_heads: int = 4,
        head_channels: int = 64,
        multipliers: tuple[int, ...] = (1, 2, 4, 8),
        attention_levels: tuple[int, ...] = (-1,),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)

        dimensions = [hidden_dim, *map(lambda mult: hidden_dim * mult, multipliers)]
        in_out = list(zip(dimensions[:-1], dimensions[1:]))

        self.hidden_dim = hidden_dim

        self.down_blocks: nn.ModuleList = nn.ModuleList([])

        n_levels = len(in_out)
        attn_levels = {i % n_levels for i in attention_levels}

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (n_levels - 1)

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResNetBlock(dim_in, dim_in, dropout),
                        ResNetBlock(dim_in, dim_in, dropout),
                        Attention(dim_in, num_heads, head_channels)
                        if idx in attn_levels
                        else nn.Identity(),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
                    ]
                )
            )

        mid_dim = dimensions[-1]

        self.mid_block_1 = ResNetBlock(mid_dim, mid_dim, dropout)
        self.mid_attn = Attention(mid_dim, num_heads, head_channels)
        self.mid_block_2 = ResNetBlock(mid_dim, mid_dim, dropout)

        self.norm_out = RMSNorm(mid_dim)
        self.conv_out = nn.Conv2d(mid_dim, z_channels, kernel_size=3, padding=1)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        z_channels: int,
        num_heads: int = 4,
        head_channels: int = 64,
        multipliers: tuple[int, ...] = (1, 2, 4, 8),
        attention_levels: tuple[int, ...] = (-1,),
        dropout: float = 0.0,
    ):
        super().__init__()

        dimensions = [hidden_dim, *map(lambda mult: hidden_dim * mult, multipliers)]
        in_out = list(zip(dimensions[:-1], dimensions[1:]))

        self.hidden_dim = hidden_dim

        self.up_blocks: nn.ModuleList = nn.ModuleList([])

        n_levels = len(in_out)
        attn_levels = {i % n_levels for i in attention_levels}

        mid_dim = dimensions[-1]

        self.conv_in = nn.Conv2d(z_channels, mid_dim, kernel_size=3, padding=1)
        self.mid_block_1 = ResNetBlock(mid_dim, mid_dim, dropout)
        self.mid_attn = Attention(mid_dim, num_heads, head_channels)
        self.mid_block_2 = ResNetBlock(mid_dim, mid_dim, dropout)

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == n_levels - 1
            down_idx = n_levels - 1 - idx

            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResNetBlock(dim_out, dim_out, dropout),
                        ResNetBlock(dim_out, dim_out, dropout),
                        Attention(dim_out, num_heads, head_channels)
                        if down_idx in attn_levels
                        else nn.Identity(),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, kernel_size=3, padding=1),
                    ]
                )
            )

        self.final_res_block = ResNetBlock(hidden_dim, hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block_1(x)
        x = self.mid_attn(x)
        x = self.mid_block_2(x)

        for block1, block2, attn, upsample in self.up_blocks:  # type: ignore
            x = block1(x)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = self.final_res_block(x)
        return self.final_conv(x).tanh()


class VQGAN(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        z_channels: int,
        emb_dim: int,
        codebook_size: int,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        head_channels: int = 64,
        multipliers: tuple[int, ...] = (1, 2, 4, 8),
        attention_levels: tuple[int, ...] = (-1,),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.encoder = Encoder(
            hidden_dim,
            in_channels,
            z_channels,
            num_heads,
            head_channels,
            multipliers,
            attention_levels,
            dropout,
        )

        self.pre_quant_conv = nn.Conv2d(z_channels, emb_dim, 1)
        self.codebook = Codebook(codebook_size, emb_dim)
        self.post_quant_conv = nn.Conv2d(emb_dim, z_channels, 1)

        self.decoder = Decoder(
            hidden_dim,
            out_channels,
            z_channels,
            num_heads,
            head_channels,
            multipliers,
            attention_levels,
            dropout,
        )

    def decode(self, x_q: torch.Tensor) -> torch.Tensor:
        h = self.post_quant_conv(x_q)
        return self.decoder(h)
