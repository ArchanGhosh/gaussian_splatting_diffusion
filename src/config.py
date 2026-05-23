import torch

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cpu"
)

TARGET_CLASS = 0 

# Model Params
IMG_SIZE = 128
GRID_SIZE = 32
SPLATS_PER_CELL = 4
PARAMS_PER_SPLAT = 9
TOTAL_CHANNELS = SPLATS_PER_CELL * PARAMS_PER_SPLAT


# ENCODER TRAINING PARAMS

SPLAT_ENCODER_BASE_BATCH_SIZE = 4
SPLAT_ENCODER_ACCUM_STEPS = 4  #steps over which gradient will be accumulated so effective batch size is 4*4=16
SPLAT_ENCODER_WARMUP_EPOCHS = 50
SPLAT_ENCODER_POLISH_EPOCHS = 100
SPLAT_ENCODER_STARTING_LR = 1e-3
SPLAT_ENCODER_POLISH_LR = 2e-4
SPLAT_ENCODER_VGG_LOSS_RAT = 0.8
SPLAT_ENCODER_MSE_LOSS_RAT = 0.2
SPLAT_ENCODER_TRAINING_IMG_SAVE_DIR = "splat_encoder_training_imgs" # If Changed, Remember to change in gitignore


# DIFFUSION TRAINING PARAMS
DIFFUSION_STEPS = 500
DIFF_LR = 1e-4
DIFF_EPOCHS = 2000
DIFFUSION_BATCH_SIZE = 32

BETA_START = 1e-4
BETA_END = 0.02
_BETA = torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS).to(DEVICE)
_ALPHA = 1. - _BETA
_ALPHA_HAT = torch.cumprod(_ALPHA, dim=0)

LONG_RUN_EPOCHS = 10000
SAVE_INTERVAL = 2000
LOG_INTERVAL = 100


# CHECKPOINT DIR and Names
BASE_CHKPNT_DIR = "CHECKPOINTS" # If Changed, Remember to change in gitignore
SPLAT_ENCODER_SAVE_NAME = "Splat_Encoder.pth"
SPLAT_RENDERER_SAVE_NAME = "Splat_Renderer.pth"

UNET_DIFF_MODEL_SAVE_NAME = "Splat_Diffuion_final"
SPLAT_DIFFUSION_TRAINING_IMG_SAVE_DIR = "splat_diffusion_training_imgs"