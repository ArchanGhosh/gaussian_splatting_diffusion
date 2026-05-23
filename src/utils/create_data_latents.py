import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DEVICE, TOTAL_CHANNELS, IMG_SIZE



def create_latents(encoder, data_loader):
    try:
        print(f"\n{'-'*10}Extracting Latents for {len(data_loader)} Images {'-'*10}")
        print(f"Latent Channels: {TOTAL_CHANNELS}")


        all_latents = []

        encoder.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                images = images.to(DEVICE)


                images_resized = F.interpolate(images, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)


                latents = encoder(images_resized)


                all_latents.append(latents.cpu())


        latents_dataset = torch.cat(all_latents, dim=0)

        print(f"\n{'-'*10} Extraction Complete. Shape: {latents_dataset.shape} {'-'*10}")



        print(f"\n {'-'*10} Calculating New Percentile Stats {'-'*10}")
        p_low = []
        p_high = []

        for c in range(TOTAL_CHANNELS):

            data = latents_dataset[:, c, :, :].flatten()


            k_low = int(0.01 * len(data))
            k_high = int(0.99 * len(data))
            sorted_data, _ = torch.sort(data)

            p_low.append(sorted_data[k_low])
            p_high.append(sorted_data[k_high])


        GLOBAL_MIN = torch.tensor(p_low).view(1, -1, 1, 1).to(DEVICE)
        GLOBAL_MAX = torch.tensor(p_high).view(1, -1, 1, 1).to(DEVICE)

        print(f"\n{'-'*10} Latents Ready. Stats Calculated {'-'*10}")
        print(f"\n{'-'*10} Range: {GLOBAL_MIN[0,0,0,0]:.4f} to {GLOBAL_MAX[0,0,0,0]:.4f} {'-'*10}")

        return latents_dataset, GLOBAL_MIN, GLOBAL_MAX
    
    except Exception as e:
        print(f"\n{'-'*10} Error Calculating Latents : {e}")
        return None, None, None