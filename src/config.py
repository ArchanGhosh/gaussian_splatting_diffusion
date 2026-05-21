import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 128
GRID_SIZE = 32
SPLATS_PER_CELL = 4
PARAMS_PER_SPLAT = 9
TOTAL_CHANNELS = SPLATS_PER_CELL * PARAMS_PER_SPLAT