import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.splat_encoder import SplatEncoder
from src.differentiable_renderer import DifferentiableSplatRenderer
from src.stl10_loader import get_stl10_dataloader

from src.utils.load_model_states import load_encoder_state, load_renderer_state

from src.utils.create_data_latents import create_latents
from src.utils.generate_samples import generate_samples_hq


from src.config import DEVICE, BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME, SPLAT_RENDERER_SAVE_NAME, IMG_SIZE, SPLAT_ENCODER_BASE_BATCH_SIZE, TARGET_CLASS

try:
    encoder = load_encoder_state()
    if encoder is None:
        raise ValueError("Encoder Failed to Load")
    
    renderer = load_renderer_state()
    if renderer is None:
        raise ValueError("Renderer Failed to Load")
    

    train_data = get_stl10_dataloader(batch_size=SPLAT_ENCODER_BASE_BATCH_SIZE, image_size=IMG_SIZE, target_class=TARGET_CLASS)
    
    GLOBAL_MIN, GLOBAL_MAX = create_latents(encoder, train_data)

    if GLOBAL_MIN is None or GLOBAL_MAX is None:
        raise ValueError("Couldn't Calculate GLOBAL_MIN or GLOBAL_MAX")
    


    


except Exception as e:
    print(f"\n{'-'*10} Error Occured during Training of GS Diffusion : {e} {'-'*10}")