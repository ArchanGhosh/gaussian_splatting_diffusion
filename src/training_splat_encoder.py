import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from src.stl10_loader import get_stl10_dataloader
from src.splat_encoder import SplatEncoder
from src.differentiable_renderer import DifferentiableSplatRenderer
from src.vgg_loss import VGGLoss

from src.config import DEVICE, TARGET_CLASS, IMG_SIZE, GRID_SIZE, SPLAT_ENCODER_BASE_BATCH_SIZE, SPLAT_ENCODER_ACCUM_STEPS, SPLAT_ENCODER_WARMUP_EPOCHS, SPLAT_ENCODER_POLISH_EPOCHS, SPLAT_ENCODER_STARTING_LR, SPLAT_ENCODER_POLISH_LR, SPLAT_ENCODER_VGG_LOSS_RAT, SPLAT_ENCODER_MSE_LOSS_RAT, SPLAT_ENCODER_TRAINING_IMG_SAVE_DIR

from src.config import BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME, SPLAT_RENDERER_SAVE_NAME


os.makedirs(SPLAT_ENCODER_TRAINING_IMG_SAVE_DIR, exist_ok=True)

torch.cuda.empty_cache()
gc.collect() #Cleanup any memory before the start of training


print(f"\n{'-'*10} Starting SPLAT ENCODER TRAINING {'-'*10}") 
print(f"\n{'-'*10} Base Batch Size : {SPLAT_ENCODER_BASE_BATCH_SIZE} {'-'*10}")
print(f"\n{'-'*10} Gradient Accumulation Steps : {SPLAT_ENCODER_ACCUM_STEPS}, Effective Batch Size : {SPLAT_ENCODER_ACCUM_STEPS * SPLAT_ENCODER_BASE_BATCH_SIZE} {'-'*10}")

# 1. Setup Models
splat_encoder = SplatEncoder().to(DEVICE)
splat_renderer = DifferentiableSplatRenderer(img_size=IMG_SIZE, grid_size=GRID_SIZE).to(DEVICE)
vgg_loss_fn = VGGLoss().to(DEVICE)
mse_loss_fn = nn.MSELoss()


# 2. Setup Loader (SINGLE CLASS - AIRPLANES)

spt_enc_loader = get_stl10_dataloader(batch_size=SPLAT_ENCODER_BASE_BATCH_SIZE, image_size=IMG_SIZE, target_class=TARGET_CLASS)

print(f"\n{'-'*10} Training on {len(spt_enc_loader)} images {'-'*10}")

# Start with Higher LR for coarse shapes
print(f"\n{'-'*10} Starting LR for Optimizer : {SPLAT_ENCODER_STARTING_LR} {'-'*10}")
opt_ae = optim.Adam(splat_encoder.parameters(), lr=SPLAT_ENCODER_STARTING_LR)


TOTAL_EPOCHS = SPLAT_ENCODER_WARMUP_EPOCHS + SPLAT_ENCODER_POLISH_EPOCHS
print(f"\n{'-'*10} Warmup Epochs : {SPLAT_ENCODER_WARMUP_EPOCHS}, Polish Epochs: {SPLAT_ENCODER_POLISH_EPOCHS}, Total Epochs: {TOTAL_EPOCHS} {'-'*10}")


# 3. Training Loop
for epoch in range(TOTAL_EPOCHS + 1):

    # --- LR SCHEDULE ---
    # After Warmup Epoch Reduce LR 
    if epoch == SPLAT_ENCODER_WARMUP_EPOCHS:
        print(f"\n{'-'*10} SWITCHING TO POLISHING PHASE LR : {SPLAT_ENCODER_POLISH_LR} {'-'*10}")
        for param_group in opt_ae.param_groups:
            param_group['lr'] = SPLAT_ENCODER_POLISH_LR

    epoch_loss = 0
    opt_ae.zero_grad()

    for i, (images, _) in enumerate(spt_enc_loader):
        images = images.to(DEVICE)
        images_resized = F.interpolate(images, size=(128, 128), mode='bilinear', align_corners=False)

        # Forward
        splat_grid = splat_encoder(images_resized)
        reconstruction = splat_renderer(splat_grid)

        # Loss (Base Ration : 0.8 VGG + 0.2 MSE)
        l_vgg = vgg_loss_fn(reconstruction, images_resized)
        l_mse = mse_loss_fn(reconstruction, images_resized)
        loss = (SPLAT_ENCODER_VGG_LOSS_RAT * l_vgg + SPLAT_ENCODER_MSE_LOSS_RAT * l_mse)

        # Gradient Accumulation
        loss = loss / SPLAT_ENCODER_ACCUM_STEPS
        loss.backward()

        if (i + 1) % SPLAT_ENCODER_ACCUM_STEPS == 0:
            opt_ae.step()
            opt_ae.zero_grad()

        epoch_loss += loss.item() * SPLAT_ENCODER_ACCUM_STEPS

    print(f"\n{'-'*10} Epoch {epoch}: Loss {epoch_loss/len(spt_enc_loader):.4f} {'-'*10}")
    if epoch % 10 == 0:
        avg_loss = epoch_loss / len(spt_enc_loader)
        #print(f"\n{'-'*10} Epoch {epoch}: Loss {avg_loss:.4f} {'-'*10}")

        # Visual Check
        with torch.no_grad():
            save_path = os.path.join(SPLAT_ENCODER_TRAINING_IMG_SAVE_DIR, f"epoch_{epoch:03d}.png")
            print(f"\n{'-'*10} SAVING IMG for Epoch-{epoch} at {save_path} {'-'*10}")
            fig, ax = plt.subplots(1, 2, figsize=(6,3))

            ax[0].imshow(images_resized[0].permute(1,2,0).cpu().numpy())
            ax[0].set_title("Original")
            #ax[0].axis("off")

            ax[1].imshow(reconstruction[0].permute(1,2,0).cpu().clip(0,1).numpy())
            ax[1].set_title(f"Reconstruction (Loss {avg_loss:.2f})")
            #ax[1].axis("off")

            save_path = os.path.join(SPLAT_ENCODER_TRAINING_IMG_SAVE_DIR, f"epoch_{epoch:03d}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

print(f"\n{'-'*10} Splat Encoder Training Complete {'-'*10}")

print(f"\n{'-'*10} Saving Weights of Encoder and Renderer {'-'*10}")

os.makedirs(BASE_CHKPNT_DIR, exist_ok=True)

ENCODER_PATH = os.join(BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME)
RENDERER_PATH = os.join(BASE_CHKPNT_DIR, SPLAT_RENDERER_SAVE_NAME)

torch.save(splat_encoder.state_dict(), ENCODER_PATH)
print(f"\n{'-'*10} Encoder saved to: {ENCODER_PATH} {'-'*10}")

torch.save(splat_renderer.state_dict(), RENDERER_PATH)
print(f"\n{'-'*10} Renderer saved to: {RENDERER_PATH} {'-'*10}")