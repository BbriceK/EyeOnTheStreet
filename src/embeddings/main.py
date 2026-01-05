import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms

from .embed_utils import seed_everything, load_labels_from_json, build_paths, save_embeddings
from .dataset import ImageDataset
from .create_embeddings import compute_embeddings
from .dinov2_loader import load_dinov2

def main(args):
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

    # Map old 8-label indices to new 4-label indices
    KEEP_LABELS = [0, 1, 2, 7]
    labels_dict = load_labels_from_json(args.json_pth, keep_labels=KEEP_LABELS)
    # Dataset splits
    splits = ["train", "val", "test"]
    # Build image paths for each split
    all_paths = {split: build_paths(args.data_pth, labels_dict, split) for split in splits}

    # ---- Load model on GPU ----
    model = load_dinov2(args.dino_pth, device)
    model = model.to(device)
    model.eval()

    for split in splits:
        paths = all_paths[split]
        labels = [labels_dict[p.replace(args.data_pth, "")[1:]] for p in paths]

        dataset = ImageDataset(paths, labels, transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        cls, reg, patch = compute_embeddings(model, dataloader, device)
        save_embeddings(args.emb_pth, split, cls, reg, patch, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_pth", type=str, help='Path to the image folder')
    parser.add_argument("json_pth", type=str, help='Path to the label JSON file')
    parser.add_argument("emb_pth", type=str, help='Path to output embeddings')
    parser.add_argument("dino_pth", type=str, help='Path to the pretrained DINOV2 weight')
    args = parser.parse_args()
    main(args)
