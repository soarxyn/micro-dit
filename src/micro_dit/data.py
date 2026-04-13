import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

_latent_transforms = v2.Compose([v2.RandomHorizontalFlip()])


class LatentDataset(Dataset):
    def __init__(self, npy_file: str):
        # TODO: Solve Pickle reading problem in NPY
        self.data = np.load(npy_file, mmap_mode="r", allow_pickle=True)

    def __getitem__(self, index) -> torch.Tensor:
        item = torch.from_numpy(self.data[index].copy())
        return _latent_transforms(item)

    def __len__(self):
        return len(self.data)


class LatentDataModule(L.LightningDataModule):
    def __init__(
        self,
        npy_file: str,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()

        self.npy_file = npy_file

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage: str = ""):
        self.train_dataset = LatentDataset(self.npy_file)

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
