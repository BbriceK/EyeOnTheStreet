import torch
from torch.utils.data import Dataset
from PIL import Image
import unicodedata


class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images (and optional labels).
    
    Returns:
        img   : transformed image tensor
        label : corresponding label (or None if labels are not provided)
        path  : original image path (useful for debugging or tracking)
    """
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Normalize Unicode to avoid file path issues
        path = unicodedata.normalize("NFC", path)
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx] if self.labels is not None else None
        return img, label, path
