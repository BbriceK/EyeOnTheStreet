import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from embeddings.embed_utils import seed_everything
from embeddings.dataset import ImageDataset
from embeddings.create_embeddings import compute_embeddings
from embeddings.dinov2_loader import load_dinov2

def get_image_files(folder):
    image_files = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if not fname.startswith(".") and not fname.startswith("._"):
                image_files.append(os.path.join(root, fname))
    return image_files


class InferenceEmbeddings(Dataset):
    """
    Store embeddings and labels as tensors.
    Args:
        cls_embeddings, reg_embeddings, patch_embeddings: dict of arrays {image_name: embedding}
        image_names: list of image identifiers
    """
    def __init__(self, cls_embeddings, reg_embeddings, patch_embeddings, image_names):
        # Convert dict values to stacked torch tensors
        self.cls_embeddings = torch.from_numpy(
            np.stack(list(cls_embeddings.values()))
        ).float()

        self.reg_embeddings = torch.from_numpy(
            np.stack(list(reg_embeddings.values()))
        ).float()

        self.patch_embeddings = torch.from_numpy(
            np.stack(list(patch_embeddings.values()))
        ).float()

        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        return self.reg_embeddings[idx], self.cls_embeddings[idx], self.patch_embeddings[idx], self.image_names[idx]


def create_infer_emb(data_pth, dino_pth, save_dir):
    seed_everything()

    # ---- Enforce GPU-only execution ----
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image_paths = get_image_files(data_pth)

    # ---- Load model on GPU ----
    model = load_dinov2(dino_pth, device)
    model = model.to(device)
    model.eval()

    dataset = ImageDataset(image_paths=image_paths, transform=transform, labels=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    cls, reg, patch, img_names = compute_embeddings(model, dataloader, device, mode="inference")
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"infer_cls_embeddings.npy"), cls)
    np.save(os.path.join(save_dir, f"infer_reg_embeddings.npy"), reg)
    np.save(os.path.join(save_dir, f"infer_patch_embeddings.npy"), patch)
    np.save(os.path.join(save_dir, f"image_names.json"), img_names)

    return cls, reg, patch, img_names
