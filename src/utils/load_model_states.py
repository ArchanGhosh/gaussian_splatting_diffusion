import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.splat_encoder import SplatEncoder
from src.differentiable_renderer import DifferentiableSplatRenderer


def load_encoder_state(BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME, DEVICE):
    try:
        encoder = SplatEncoder().to(DEVICE)
        encoder_weights_path = os.path.join(BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME)

        print(f"\n{'-'*10} Loading GS Encoder Model Weights from {encoder_weights_path} {'-'*10}")

        if os.path.exists(encoder_weights_path):
            encoder.load_state_dict(torch.load(encoder_weights_path, map_location=DEVICE))

            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
            print(f"\n{'-'*10} GS Encoder loaded and frozen for Diffusion training {'-'*10}")
            return encoder 
        else:
            print(f"\n{'-'*10} Path for weights for GS Encoder is Invalid {'-'*10}")
            return None
        
    except Exception as e:
        print(f"\n{'-'*10} Encountered Error during loading weights for GS Encoder : {e}")
        return None


def load_renderer_state(BASE_CHKPNT_DIR, SPLAT_RENDERER_SAVE_NAME, DEVICE, IMG_SIZE, GRID_SIZE):
    try:
        renderer = DifferentiableSplatRenderer(img_size=IMG_SIZE, grid_size=GRID_SIZE).to(DEVICE)

        renderer_weights_path = os.path.join(BASE_CHKPNT_DIR, SPLAT_RENDERER_SAVE_NAME)

        print(f"\n{'-'*10} Loading GS Renderer Model Weights from {renderer_weights_path} {'-'*10}")

        if os.path.exists(renderer_weights_path):

            renderer.load_state_dict(torch.load(renderer_weights_path, map_location=DEVICE))
            print(f"\n{'-'*10} GS Renderer loaded and frozen for Diffusion training {'-'*10}")
            return renderer
        else:
            print(f"\n{'-'*10} Path for weights for GS Renderer is Invalid {'-'*10}")
            return None


    except Exception as e:
        print(f"\n{'-'*10} Encountered Error during loading weights for GS Renderer : {e}")
        return None
    

def load_unet_diffusion_state(diffusion_model, opt_diff, BASE_CHKPNT_DIR, UNET_DIFF_MODEL_SAVE_NAME, DEVICE, start_epoch):

    diffu_chkpnt_path = os.path.join(BASE_CHKPNT_DIR, UNET_DIFF_MODEL_SAVE_NAME+start_epoch+".pth")

    print(f"\n{'-'*10} Loading GS Encoder Model Weights from {diffu_chkpnt_path} {'-'*10}")

    if os.path.exists(diffu_chkpnt_path):
        checkpoint = torch.load(diffu_chkpnt_path, map_location=DEVICE)


        diffusion_model.load_state_dict(checkpoint['model_state'])
        opt_diff.load_state_dict(checkpoint['optimizer_state'])


        GLOBAL_MIN = checkpoint['global_min'].to(DEVICE)
        GLOBAL_MAX = checkpoint['global_max'].to(DEVICE)

        return diffusion_model, opt_diff, GLOBAL_MIN, GLOBAL_MAX
    else:
        print(f"\n{'-'*10} Checkpoint not present at : {diffu_chkpnt_path} {'-'*10}")
        return None, None, None, None
