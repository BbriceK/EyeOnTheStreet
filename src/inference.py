import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.model_utils import seed_everything
from models.full_model import FullModel
from inference_utils import create_infer_emb, InferenceEmbeddings


def main(data_path, emb_path, dino_path, save_path, out_path):
    seed_everything()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda")

    inference_cls_embeddings, inference_reg_embeddings, inference_patch_embeddings, inference_image_names = create_infer_emb(data_path, dino_path, emb_path)
    inference_dataset = InferenceEmbeddings(inference_cls_embeddings, inference_reg_embeddings, inference_patch_embeddings, inference_image_names)

    batch_size = 32
    num_classes = 4
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    model = FullModel(num_classes).to(device)
    model.load_state_dict(torch.load(save_path))
    print(f"Loading the best model from {save_path}")

    model.eval()
    y_pred = []
    infer_image_names = []

    with torch.no_grad():
        for batch_idx, (reg_embeddings, cls_embeddings, patch_embeddings, names) in enumerate(inference_loader):
            cls_embeddings = cls_embeddings.to(device)
            reg_embeddings = reg_embeddings.to(device)
            patch_embeddings = patch_embeddings.to(device)

            outputs = model((reg_embeddings, cls_embeddings, patch_embeddings))
            probs = torch.sigmoid(outputs)
            y_pred.append(probs.cpu().detach().numpy())
            infer_image_names.append(names)

    y_pred_full = np.concatenate(y_pred, axis=0)
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, "y_pred_full.npy"), y_pred_full)
    np.save(os.path.join(out_path, "y_name_full.npy"), infer_image_names)
    print("Predictions and image names saved as 'y_pred_full.npy' and 'y_name_full.npy'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_pth", type=str, help='Path to the image folder')
    parser.add_argument("emb_pth", type=str, help='Path to output embeddings')
    parser.add_argument("dino_pth", type=str, help='Path to the pretrained DINOV2 weight')
    parser.add_argument('save_pth', type=str, help='Path to saving the weight')
    parser.add_argument('out_pth', type=str, help='Path to output files')
    args = parser.parse_args()
    main(args.data_pth, args.emb_pth, args.dino_pth, args.save_pth, args.out_pth)
