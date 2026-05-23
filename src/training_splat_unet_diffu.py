import os
import gc
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from src.unet_diffusion import SplatUNetDiffu

from src.stl10_loader import get_stl10_dataloader
from src.utils.add_noise import noise_images

from src.utils.load_model_states import load_encoder_state, load_renderer_state, load_unet_diffusion_state

from src.utils.create_data_latents import create_latents
from src.utils.generate_samples import generate_samples_hq


from src.config import DEVICE, BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME, SPLAT_RENDERER_SAVE_NAME, UNET_DIFF_MODEL_SAVE_NAME, IMG_SIZE, GRID_SIZE, SPLAT_ENCODER_BASE_BATCH_SIZE, TARGET_CLASS, LONG_RUN_EPOCHS, LOG_INTERVAL, SAVE_INTERVAL, SPLATS_PER_CELL, PARAMS_PER_SPLAT, DIFF_LR, DIFFUSION_BATCH_SIZE, DIFFUSION_STEPS, SPLAT_DIFFUSION_TRAINING_IMG_SAVE_DIR


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def run_splat_diff_training(run_flag, start_long_epochs, end_long_epochs, save_intr, log_intr):
    try:

        print(f" Training Params Provided: \n Starting Epochs : {start_long_epochs}, \n Ending Epochs : {end_long_epochs}, \n Save Interval : {save_intr}, \n Log Interval : {log_intr}")

        train_data = get_stl10_dataloader(batch_size=SPLAT_ENCODER_BASE_BATCH_SIZE, image_size=IMG_SIZE, target_class=TARGET_CLASS)

        if start_long_epochs < 0 or start_long_epochs is None:
            print(f"{'-'*10} No / Invalid Starting Epoch Number provided defaulting to 0 {'-'*10}")
            start_long_epochs = 0
        
        if end_long_epochs <= 0 or end_long_epochs is None:
            print(f"{'-'*10} No / Invalid Ending Epoch Number provided defaulting to {LONG_RUN_EPOCHS} {'-'*10}")
            end_long_epochs = LONG_RUN_EPOCHS

        if start_long_epochs>end_long_epochs:
            raise ValueError(" Starting Epochs cannot be greater than Ending Epochs ")

        encoder = load_encoder_state(os.path.join(BASE_DIR, BASE_CHKPNT_DIR), SPLAT_ENCODER_SAVE_NAME, DEVICE)
        if encoder is None:
            raise ValueError("Encoder Failed to Load")
        
        renderer = load_renderer_state(BASE_CHKPNT_DIR, SPLAT_RENDERER_SAVE_NAME, DEVICE, IMG_SIZE, GRID_SIZE)
        if renderer is None:
            raise ValueError("Renderer Failed to Load")
        

        diffusion_model = SplatUNetDiffu(in_channels=SPLATS_PER_CELL*PARAMS_PER_SPLAT).to(DEVICE)
        opt_diff = optim.AdamW(diffusion_model.parameters(), lr=DIFF_LR)
        loss_fn = nn.HuberLoss() 

        if run_flag == 'start':
            print(f"{'-'*10} Run Flag given as 'start', setting model weights to default and Start Epochs to 0 and End Epochs to {end_long_epochs} {'-'*10}")
            start_long_epochs = 0

            latents_dataset, GLOBAL_MIN, GLOBAL_MAX = create_latents(encoder, train_data)
            if GLOBAL_MIN is None or GLOBAL_MAX is None:
                raise ValueError("Couldn't Calculate GLOBAL_MIN or GLOBAL_MAX")
            
        elif run_flag == 'resume':
            print(f"{'-'*10} Run Flag given as 'resume', setting model weights to last training and Start Epochs to {start_long_epochs} and End Epochs to {end_long_epochs} {'-'*10}")

            diffusion_model, opt_diff, GLOBAL_MIN, GLOBAL_MAX = load_unet_diffusion_state(diffusion_model, opt_diff, BASE_CHKPNT_DIR, UNET_DIFF_MODEL_SAVE_NAME, DEVICE, start_long_epochs)

            if diffusion_model is None:
                raise ValueError("Couldn't Resolve path for checkpoint")
            latents_dataset, _, _ = create_latents(encoder, train_data)
        
        elif run_flag is None or run_flag == "":
            print(f"{'-'*10} Run Flag is empty, setting model weights to default and Start Epochs to 0 and End Epochs to {end_long_epochs} {'-'*10}")
            start_long_epochs = 0

            latents_dataset, GLOBAL_MIN, GLOBAL_MAX = create_latents(encoder, train_data)
            if GLOBAL_MIN is None or GLOBAL_MAX is None:
                raise ValueError("Couldn't Calculate GLOBAL_MIN or GLOBAL_MAX")
        
        else:
            raise ValueError("Unknown Run Flag")
        
        if save_intr is None or save_intr == 0:
            print(f"{'-'*10} Save Interval defaulting to {SAVE_INTERVAL} {'-'*10}") 
            save_intr = SAVE_INTERVAL
        
        if log_intr is None or log_intr == 0:
            print(f"{'-'*10} Log Interval defaulting to {LOG_INTERVAL} {'-'*10}") 
            log_intr = LOG_INTERVAL
        
        latent_loader = torch.utils.data.DataLoader(latents_dataset, batch_size=DIFFUSION_BATCH_SIZE, shuffle=True)

        print(f"{'-'*10} Starting Training {end_long_epochs-start_long_epochs} Epochs {'-'*10}")

        os.makedirs(BASE_CHKPNT_DIR, exist_ok=True)
        os.makedirs(SPLAT_DIFFUSION_TRAINING_IMG_SAVE_DIR, exist_ok=True)
        loss_curve = []

        for epoch in range(start_long_epochs+1, end_long_epochs + 1):
            epoch_loss = 0
            for raw_latents in latent_loader:
                raw_latents = raw_latents.to(DEVICE)

                clean_latents = (raw_latents - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
                clean_latents = torch.clamp(clean_latents, 0.0, 1.0)
                clean_latents = (clean_latents * 2) - 1


                t = torch.randint(low=1, high=DIFFUSION_STEPS, size=(raw_latents.shape[0],)).to(DEVICE)
                noisy_latents, noise = noise_images(clean_latents, t)


                predicted_clean = diffusion_model(noisy_latents, t)
                loss = loss_fn(predicted_clean, clean_latents)

                opt_diff.zero_grad()
                loss.backward()
                opt_diff.step()
                epoch_loss += loss.item()


            if epoch % log_intr == 0:
                avg_loss = epoch_loss / len(latent_loader)
                print(f"{'-'*10} Epoch {epoch}: Loss {avg_loss:.6f} {'-'*10}")
                loss_curve.append({"Epoch": epoch, "Average_Loss": avg_loss})


                diffusion_model.eval()
                with torch.no_grad():
                    sample_z = torch.randn((1, 36, 32, 32)).to(DEVICE)

                    for i in reversed(range(1, DIFFUSION_STEPS, 2)):
                        t_idx = (torch.ones(1) * i).long().to(DEVICE)
                        pred = diffusion_model(sample_z, t_idx)

                        sample_z = pred


                    final_lat = generate_samples_hq(diffusion_model, GLOBAL_MIN, GLOBAL_MAX, n=1).to(DEVICE)
                    preview_img = renderer(final_lat).cpu()

                    plt.figure(figsize=(2,2))
                    plt.imshow(preview_img[0].permute(1,2,0).clip(0,1).numpy())
                    plt.title(f"Ep {epoch}")
                    plt.axis('off')
                    img_sv_path = os.path.join(SPLAT_DIFFUSION_TRAINING_IMG_SAVE_DIR, f"epoch_{epoch:06d}.png")
                    plt.savefig(img_sv_path, bbox_inches='tight', pad_inches=0)

                    plt.close()
                diffusion_model.train()

            if epoch % save_intr == 0:
                path = os.path.join(BASE_CHKPNT_DIR, f"Splat_Diffusion_{epoch}.pth")
                torch.save(diffusion_model.state_dict(), path)
                print(f"{'-'*10} Checkpoint saved: {path} {'-'*10}")

        
        print(f"{'-'*10} Training Complete Proceeding to Save Final Model {'-'*10}")


        print(f"Saving full training state for restoration")


        state_dict = {
            'final_epoch': loss_curve[-1]["Epoch"],
            'model_state': diffusion_model.state_dict(),
            'optimizer_state': opt_diff.state_dict(),
            'global_min': GLOBAL_MIN,
            'global_max': GLOBAL_MAX,
            'loss_curve': loss_curve,
            'final_epoch_loss': loss_curve[-1]["Average_Loss"]
        }

        for key, value in state_dict:
            print(f"{key} : {value}")
        
        diffusion_chk_pth = os.path.join(BASE_CHKPNT_DIR, UNET_DIFF_MODEL_SAVE_NAME+loss_curve[-1]["Epoch"])

        torch.save(state_dict, diffusion_chk_pth)
        print(f"{'-'*10} State saved to: {diffusion_chk_pth} {'-'*10}")
        

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