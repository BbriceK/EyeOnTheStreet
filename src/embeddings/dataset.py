import torch
from torch.utils.data import Dataset
from PIL import Image
import unicodedata


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        path = unicodedata.normalize("NFC", path)
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx] if self.labels is not None else None
        return img, label, path

