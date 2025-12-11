import torch
from tqdm import tqdm

def compute_embeddings(model, dataloader, device):
    cls_emb, reg_emb, patch_emb = {}, {}, {}

    model.eval()
    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader):
            imgs = imgs.to(device)
            out = model.module.forward_features(imgs)

            for i, path in enumerate(paths):
                cls_emb[path] = out["x_norm_clstoken"][i].cpu().numpy()
                reg_emb[path] = out["x_norm_regtokens"][i].cpu().numpy()
                patch_emb[path] = out["x_norm_patchtokens"][i].cpu().numpy()

    return cls_emb, reg_emb, patch_emb
