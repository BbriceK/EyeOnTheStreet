import torch
from dinov2.models.vision_transformer import vit_small

def load_dinov2(dino_path, device):
    model = vit_small(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        block_chunks=0,
        num_register_tokens=4
    )
    state_dict = torch.load(dino_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
