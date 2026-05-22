import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.splat_encoder import SplatEncoder
from src.differentiable_renderer import DifferentiableSplatRenderer


from src.config import DEVICE, BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME, SPLAT_RENDERER_SAVE_NAME

def load_encoder_state():
    try:
        encoder = SplatEncoder().to(DEVICE)
        encoder_weights_path = os.join(BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME)

        print(f"\n{'-'*10} Loading GS Encoder Model Weights from {encoder_weights_path} {'-'*10}")

        if os.path.exists(encoder_weights_path):
            encoder.load_state_dict(torch.load("splat_encoder_v2_sharp.pth", map_location=DEVICE))

            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
            print(f"\n{'-'*10} GS Encoder loaded and frozen for Diffusion training {'-'*10}")
            return encoder 
        else:
            return None
    except Exception as e:
        print(f"\n{'-'*10} Encountered Error during loading weights for GS Encoder : {e}")
        return None
        