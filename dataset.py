
from torch.utils.data import Dataset
import torch

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, 256, 256)
        return image, image
