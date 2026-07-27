import torch
from tqdm import tqdm

def compute_embeddings(model, dataloader, device, mode):
    """
    Compute DINOv2 embeddings on a single GPU.

    Args:
        model: DINOv2 model (plain nn.Module)
        dataloader: DataLoader yielding (imgs, labels, paths)
        device: device to run the model on
        mode: 'training' or 'inference'

    Returns:
        cls_emb: dict[path -> cls embedding]
        reg_emb: dict[path -> reg token embedding]
        patch_emb: dict[path -> patch token embedding]
    """
    cls_emb, reg_emb, patch_emb = {}, {}, {}
    img_names = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Handle different return formats
            if mode == "training":
                imgs, labels, paths = batch
            else:  # inference
                imgs, paths = batch
            
            imgs = imgs.to(device, non_blocking=True)
            out = model.forward_features(imgs)
            
            for i, path in enumerate(paths):
                cls_emb[path] = out["x_norm_clstoken"][i].detach().cpu().numpy()
                reg_emb[path] = out["x_norm_regtokens"][i].detach().cpu().numpy()
                patch_emb[path] = out["x_norm_patchtokens"][i].detach().cpu().numpy()
                img_names.append(path)  # FIXED: was appending 'paths' (the whole list)

    if mode == "training":
        return cls_emb, reg_emb, patch_emb
    else:  # inference
        return cls_emb, reg_emb, patch_emb, img_names
