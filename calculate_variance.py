import torch
from safetensors import safe_open
from micro_dit.vqgan import VQGAN

vqgan = VQGAN.load_from_checkpoint("weights/vqgan.ckpt", map_location="cpu", strict=False)
vqgan.eval()

data = safe_open("data/training_indices.safetensors", framework="pt", device="cpu")
indices = data.get_tensor("indices")

with torch.inference_mode():
    latents = vqgan.codebook.lookup(indices.long())

print(f"{latents.std()=}")
