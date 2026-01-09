import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import unicodedata


class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images (and optional labels).
    
    Returns:
        img   : transformed image tensor
        label : corresponding label (or None if labels are not provided)
        path (str): final resolved image path
    """
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        original_path = self.image_paths[idx]

        # Try original path first
        path = Path(original_path)
        if not path.exists():
            # Try NFC
            path_nfc = Path(unicodedata.normalize("NFC", original_path))
            if path_nfc.exists():
                path = path_nfc
            else:
                # Try NFD
                path_nfd = Path(unicodedata.normalize("NFD", original_path))
                if path_nfd.exists():
                    path = path_nfd
                else:
                    # Still not found → raise error
                    raise FileNotFoundError(f"Cannot find image: {original_path} (tried NFC & NFD)")

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx] if self.labels is not None else None
        return img, label, str(path)
