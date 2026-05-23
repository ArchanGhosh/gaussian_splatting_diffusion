import os
import gc
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.splat_encoder import SplatEncoder
from src.differentiable_renderer import DifferentiableSplatRenderer
from src.stl10_loader import get_stl10_dataloader

from src.utils.load_model_states import load_encoder_state, load_renderer_state

from src.utils.create_data_latents import create_latents
from src.utils.generate_samples import generate_samples_hq


from src.config import DEVICE, BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME, SPLAT_RENDERER_SAVE_NAME, IMG_SIZE, SPLAT_ENCODER_BASE_BATCH_SIZE, TARGET_CLASS, LONG_RUN_EPOCHS, LOG_INTERVAL, SAVE_INTERVAL

def run_splat_diff_training(run_flag, start_long_epochs, end_long_epochs, save_intr, log_intr):
    try:
        if run_flag not in ["start", "resume"]:
            raise ValueError("Run Flag is incorrect, should be 'start' or'resume' ")
        if run_flag == 'start':


        elif run_flag == 'resume':

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--run_flag", choices=["start", "resume"])

    parser.add_argument("--start_long_run_epochs", type=int)

    parser.add_argument("--end_long_run_epochs", type=int)

    parser.add_argument("--save_interval", type=int)

    parser.add_argument("--log_interval", type=int)

    args = parser.parse_args()

    run_splat_diff_training(args.run_flag, args.start_long_run_epochs, args.end_long_run_epochs, args.save_interval, args.log_interval)