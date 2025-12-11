import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms

from .embed_utils import seed_everything, load_labels_from_json, save_embeddings
from .dataset import ImageDataset
from .create_embeddings import compute_embeddings
from .dinov2_loader import load_dinov2


def build_paths(data_root, labels_dict, split):
    return [os.path.join(data_root, k) for k in labels_dict if k.startswith(f"{split}/")]


def main(args):
    seed_everything()

    rank = int(os.environ["SLURM_LOCALID"])
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=rank
    )
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    labels_dict = load_labels_from_json(args.json_pth)
    splits = ["train", "val", "test"]
    all_paths = {split: build_paths(args.data_pth, labels_dict, split) for split in splits}

    model = load_dinov2(args.dino_pth, device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    for split in splits:
        paths = all_paths[split]
        labels = [labels_dict[p.replace(args.data_pth, "")[1:]] for p in paths]

        dataset = ImageDataset(paths, labels, transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        cls, reg, patch = compute_embeddings(model, dataloader, device)

        if rank == 0:
            save_embeddings(args.emb_pth, split, cls, reg, patch, labels)

    if rank == 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_pth", type=str, help='Path to the image folder')
    parser.add_argument("json_pth", type=str, help='Path to the label json file')
    parser.add_argument("emb_pth", type=str, help='Path to output embeddings')
    parser.add_argument("dino_pth", type=str, help='Path to the model pretrained weight')
    parser.add_argument("--world_size", type=int, help='Total number of processes (GPUs)')
    parser.add_argument("--dist_url", help='URL for the distributed environment')
    args = parser.parse_args()

    main(args)
