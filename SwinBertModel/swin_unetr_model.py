### swin_unetr_model.py
import torch
from monai.networks.nets import SwinUNETR

def load_swin_unetr(device):
    swin_unetr = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True
    ).to(device)
    # swin_unetr.load_state_dict(torch.load("swin_unetr_checkpoint.pth", map_location=device))
    return swin_unetr
