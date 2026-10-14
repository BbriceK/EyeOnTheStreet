import torch
from dinov2.models.vision_transformer import vit_small

def load_dinov2(dino_path, device):
    """
    Loads a pre-trained DINOv2 model.

    Args:
        dino_path (str): Path to the saved DINOv2 weights.
        device (str or torch.device): device to load the model on.

    Returns:
        model (torch.nn.Module): DINOv2 model loaded with pre-trained weights on the specified device.
    """
    # Initialize the model with specific hyperparameters
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
