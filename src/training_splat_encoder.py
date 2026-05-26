import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from src.stl10_loader import get_stl10_dataloader
from src.splat_encoder import SplatEncoder
from src.differentiable_renderer import DifferentiableSplatRenderer
from src.vgg_loss import VGGLoss

from src.utils.create_loss_graph import save_loss_curve

from src.config import DEVICE, TARGET_CLASS, IMG_SIZE, GRID_SIZE, SPLAT_ENCODER_BASE_BATCH_SIZE, SPLAT_ENCODER_ACCUM_STEPS, SPLAT_ENCODER_WARMUP_EPOCHS, SPLAT_ENCODER_POLISH_EPOCHS, SPLAT_ENCODER_STARTING_LR, SPLAT_ENCODER_POLISH_LR, SPLAT_ENCODER_VGG_LOSS_RAT, SPLAT_ENCODER_MSE_LOSS_RAT, SPLAT_ENCODER_TRAINING_IMG_SAVE_DIR

from src.config import BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME, SPLAT_RENDERER_SAVE_NAME, SAVE_METRICS_DIR


os.makedirs(SPLAT_ENCODER_TRAINING_IMG_SAVE_DIR, exist_ok=True)

torch.cuda.empty_cache()
gc.collect() #Cleanup any memory before the start of training


print(f"\n{'-'*10} Starting SPLAT ENCODER TRAINING {'-'*10}") 
print(f"\n{'-'*10} Base Batch Size : {SPLAT_ENCODER_BASE_BATCH_SIZE} {'-'*10}")
print(f"\n{'-'*10} Gradient Accumulation Steps : {SPLAT_ENCODER_ACCUM_STEPS}, Effective Batch Size : {SPLAT_ENCODER_ACCUM_STEPS * SPLAT_ENCODER_BASE_BATCH_SIZE} {'-'*10} \n ")

# 1. Setup Models
splat_encoder = SplatEncoder().to(DEVICE)
splat_renderer = DifferentiableSplatRenderer(img_size=IMG_SIZE, grid_size=GRID_SIZE).to(DEVICE)
vgg_loss_fn = VGGLoss().to(DEVICE)
mse_loss_fn = nn.MSELoss()


# 2. Setup Loader (SINGLE CLASS - AIRPLANES)

spt_enc_loader = get_stl10_dataloader(batch_size=SPLAT_ENCODER_BASE_BATCH_SIZE, image_size=IMG_SIZE, target_class=TARGET_CLASS)

print(f"\n{'-'*10} Training on {len(spt_enc_loader)*SPLAT_ENCODER_BASE_BATCH_SIZE} images {'-'*10}")

# Start with Higher LR for coarse shapes
print(f"\n{'-'*10} Starting LR for Optimizer : {SPLAT_ENCODER_STARTING_LR} {'-'*10}")
opt_ae = optim.Adam(splat_encoder.parameters(), lr=SPLAT_ENCODER_STARTING_LR)


TOTAL_EPOCHS = SPLAT_ENCODER_WARMUP_EPOCHS + SPLAT_ENCODER_POLISH_EPOCHS
print(f"\n{'-'*10} Warmup Epochs : {SPLAT_ENCODER_WARMUP_EPOCHS}, Polish Epochs: {SPLAT_ENCODER_POLISH_EPOCHS}, Total Epochs: {TOTAL_EPOCHS} {'-'*10}")

loss_curve = []
# 3. Training Loop
for epoch in tqdm(range(TOTAL_EPOCHS + 1), desc="Epochs"):

    # --- LR SCHEDULE ---
    # After Warmup Epoch Reduce LR 
    if epoch == SPLAT_ENCODER_WARMUP_EPOCHS:
        print(f"\n{'-'*10} SWITCHING TO POLISHING PHASE LR : {SPLAT_ENCODER_POLISH_LR} {'-'*10}")
        for param_group in opt_ae.param_groups:
            param_group['lr'] = SPLAT_ENCODER_POLISH_LR

    epoch_loss = 0
    opt_ae.zero_grad()

    for i, (images, _) in enumerate(tqdm(spt_enc_loader, desc="Batches", leave=False)):
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

    print(f"{'-'*10} Epoch {epoch}: Loss {epoch_loss/len(spt_enc_loader):.4f} {'-'*10}")
    loss_curve.append({"Epoch": epoch, "Loss": epoch_loss})
    if epoch % 10 == 0:
        avg_loss = epoch_loss / len(spt_enc_loader)
        #print(f"\n{'-'*10} Epoch {epoch}: Loss {avg_loss:.4f} {'-'*10}")

        # Visual Check
        with torch.no_grad():
            save_path = os.path.join(SPLAT_ENCODER_TRAINING_IMG_SAVE_DIR, f"epoch_{epoch:03d}.png")
            print(f"{'-'*10} SAVING IMG for Epoch-{epoch} at {save_path} {'-'*10}")
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


epochs_4_plt = [item["Epoch"] for item in loss_curve]
losses_4_plt = [item["Loss"] for item in loss_curve]

os.makedirs(SAVE_METRICS_DIR, exist_ok=True)
loss_graph_save_path = os.path.join(SAVE_METRICS_DIR, SPLAT_ENCODER_SAVE_NAME+str(loss_curve[-1]["Epoch"])+ '_loss_curve,png')

save_loss_curve(epochs_4_plt, losses_4_plt, tilte="Encoder Loss", x_label="Epochs", y_label="Huber Loss", output_path= loss_graph_save_path)


print(f"\n{'-'*10} Splat Encoder Training Complete {'-'*10}")

print(f"\n{'-'*10} Saving Weights of Encoder and Renderer {'-'*10}")

os.makedirs(BASE_CHKPNT_DIR, exist_ok=True)

ENCODER_PATH = os.path.join(BASE_CHKPNT_DIR, SPLAT_ENCODER_SAVE_NAME)
RENDERER_PATH = os.path.join(BASE_CHKPNT_DIR, SPLAT_RENDERER_SAVE_NAME)

torch.save(splat_encoder.state_dict(), ENCODER_PATH)
print(f"\n{'-'*10} Encoder saved to: {ENCODER_PATH} {'-'*10}")

torch.save(splat_renderer.state_dict(), RENDERER_PATH)
print(f"\n{'-'*10} Renderer saved to: {RENDERER_PATH} {'-'*10}")