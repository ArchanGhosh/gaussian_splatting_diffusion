import torch
from src.config import DEVICE, DIFFUSION_STEPS, _ALPHA, _ALPHA_HAT, _BETA
from tqdm import tqdm

def generate_samples_hq(model, GLOBAL_MIN, GLOBAL_MAX, n=8):
    tqdm.write(f"\n{'-'*10} Generating {n} High-Quality Samples {'-'*10}")
    model.eval()
    with torch.no_grad():

        x = torch.randn((n, 36, 32, 32)).to(DEVICE)


        for i in reversed(range(1, DIFFUSION_STEPS)):
            t = (torch.ones(n) * i).long().to(DEVICE)


            pred_x0 = model(x, t)


            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)


            alpha_t = _ALPHA[t][:, None, None, None]
            alpha_hat_t = _ALPHA_HAT[t][:, None, None, None]
            beta_t = _BETA[t][:, None, None, None]
            alpha_hat_prev = _ALPHA_HAT[t-1][:, None, None, None] if i > 1 else torch.tensor(1.0).to(DEVICE)

            mean_pred = (beta_t * torch.sqrt(alpha_hat_prev) / (1 - alpha_hat_t)) * pred_x0 + \
                        ((1 - alpha_hat_prev) * torch.sqrt(alpha_t) / (1 - alpha_hat_t)) * x

            if i > 1:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean_pred + sigma * noise
            else:
                x = mean_pred


        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)


        final_latents = x * (GLOBAL_MAX - GLOBAL_MIN) + GLOBAL_MIN

        return final_latents