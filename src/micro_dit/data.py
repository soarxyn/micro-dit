import lightning as L
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

_latent_transforms = v2.Compose([v2.RandomHorizontalFlip()])


class LatentDataset(Dataset):
    def __init__(self, path: str):
        self.data = safe_open(path, framework="pt", device="cpu")
        self.num_samples = self.data.get_tensor("indices").shape[0]

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        indices = self.data.get_slice("indices")[index]
        indices = _latent_transforms(indices)
        return {"indices": indices}

    def __len__(self):
        return self.num_samples


class LatentDataModule(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()

        self.path = path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage: str = ""):
        self.train_dataset = LatentDataset(self.path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 1,
            shuffle=True,
            drop_last=self.drop_last,
        )
