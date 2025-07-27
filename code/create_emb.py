import argparse
import os
from dinov2.models.vision_transformer import vit_small, vit_large, vit_base, vit_giant2
import numpy as np
import random
from torchvision import transforms
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def seed_everything(seed=1111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_image(img_path: str, transform=None) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    if transform:
        img = transform(img).unsqueeze(0)
    return img

def compute_embeddings(model, files: list, device, transform=None) -> dict:
    cls_embeddings = {}
    reg_embeddings = {}
    patch_embeddings = {}

    with torch.no_grad():
        for file in tqdm(files):
            image_tensor = load_image(file, transform).to(device)
            embeddings = model.module.forward_features(image_tensor)
            cls_embeddings[file] = np.array(embeddings['x_norm_clstoken'].cpu().numpy()).tolist()
            reg_embeddings[file] = np.array(embeddings['x_norm_regtokens'].cpu().numpy()).tolist()
            patch_embeddings[file] = np.array(embeddings['x_norm_patchtokens'].cpu().numpy()).tolist()
            
    return cls_embeddings, reg_embeddings, patch_embeddings


def add_root_directory(out_path, img_path: str) -> str:
    return os.path.join(out_path, img_path)

def load_labels_from_json(json_file: str) -> dict:
    with open(json_file, "r") as f:
        labels_dict = json.load(f)
    return labels_dict

def get_relative_path(image_path: str, base_dir: str) -> str:
    return image_path.replace(base_dir, '')


def get_location_from_path(image_name):
    location = image_name[-29:-11]
    return location

def assign_location_ids(labels_dict):
    locations = {}
    location_to_id = {}
    current_id = 0

    for image_path in labels_dict.keys():
        location = get_location_from_path(image_path)
        if location not in location_to_id:
            location_to_id[location] = current_id
            current_id += 1
    return location_to_id


def main(data_path, json_path, emb_dir, dino_path, rank, world_size, dist_url):
    seed_everything()

    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    
    transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    labels_dict = load_labels_from_json(json_path)
    train_paths = [add_root_directory(data_path, img) for img in labels_dict.keys() if img.startswith("train/")]
    val_paths = [add_root_directory(data_path, img) for img in labels_dict.keys() if img.startswith("val/")]
    test_paths = [add_root_directory(data_path, img) for img in labels_dict.keys() if img.startswith("test/")]


    dinov2_vits14 = vit_small(img_size=518, patch_size=14, init_values=1.0, block_chunks=0, num_register_tokens=4)
    state_dict = torch.load(dino_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dinov2_vits14.load_state_dict(state_dict)
    dinov2_vits14.to(device)
    dinov2_vits14 = torch.nn.parallel.DistributedDataParallel(dinov2_vits14, device_ids=[rank], find_unused_parameters=True)

    y_train = []
    for img in train_paths:
        train_key = get_relative_path(img, data_path)[1:]
        y_train.append(labels_dict[train_key])
    y_train = np.array(y_train)


    y_val = []
    for img in val_paths:
        val_key = get_relative_path(img, data_path)[1:]
        y_val.append(labels_dict[val_key])
    y_val = np.array(y_val)


    y_test = []
    for img in test_paths:
        test_key = get_relative_path(img, data_path)[1:]
        y_test.append(labels_dict[test_key])
    y_test = np.array(y_test)


    train_cls_embeddings, train_reg_embeddings, train_patch_embeddings = compute_embeddings(dinov2_vits14, train_paths, device, transform_image) 
    if rank == 0:
        np.save(os.path.join(emb_dir, "train_cls_embeddings.npy"), train_cls_embeddings)
        np.save(os.path.join(emb_dir, "train_reg_embeddings.npy"), train_reg_embeddings)
        np.save(os.path.join(emb_dir, "train_patch_embeddings.npy"), train_patch_embeddings)
        np.save(os.path.join(emb_dir, "y_train.npy"), y_train)

    val_cls_embeddings, val_reg_embeddings, val_patch_embeddings = compute_embeddings(dinov2_vits14, val_paths, device, transform_image)
    if rank == 0:
        np.save(os.path.join(emb_dir, "val_cls_embeddings.npy"), val_cls_embeddings)
        np.save(os.path.join(emb_dir, "val_reg_embeddings.npy"), val_reg_embeddings)
        np.save(os.path.join(emb_dir, "val_patch_embeddings.npy"), val_patch_embeddings)
        np.save(os.path.join(emb_dir, "y_val.npy"), y_val)

    test_cls_embeddings, test_reg_embeddings, test_patch_embeddings = compute_embeddings(dinov2_vits14, test_paths, device, transform_image)
    if rank == 0:
        np.save(os.path.join(emb_dir, "test_cls_embeddings.npy"), test_cls_embeddings)
        np.save(os.path.join(emb_dir, "test_reg_embeddings.npy"), test_reg_embeddings)
        np.save(os.path.join(emb_dir, "test_patch_embeddings.npy"), test_patch_embeddings)
        np.save(os.path.join(emb_dir, "y_test.npy"), y_test)

        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_pth', type=str, help='Path to the data folder')
    parser.add_argument('json_pth', type=str, help='Path to the label file')
    parser.add_argument('emb_pth', type=str, help='Path to output embeddings')
    parser.add_argument('dino_pth', type=str, help='Path to the model pretrained weight')
    parser.add_argument('--world_size', type=int, help='Total number of processes (GPUs)')
    parser.add_argument('--dist_url', help='URL for the distributed environment')
    args = parser.parse_args()
    rank = int(os.environ['SLURM_LOCALID'])
    main(args.data_pth, args.json_pth, args.emb_pth, args.dino_pth, rank, args.world_size, args.dist_url)
