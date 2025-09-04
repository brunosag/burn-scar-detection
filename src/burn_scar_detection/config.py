import os

import torch

# --- Data Configuration ---
DATA_DIR = 'data/processed/'
T1_DIR = os.path.join(DATA_DIR, 't1')
T2_DIR = os.path.join(DATA_DIR, 't2')
MASK_DIR = os.path.join(DATA_DIR, 'mask')

# --- Training Configuration ---
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
EPOCHS = 32
MODEL_PATH = 'models/attention_unet_best.pth'

# --- Band Indices ---
B_RED = 0
B_NIR = 1
B_SWIR1 = 2
B_SWIR2 = 3

# --- Model Configuration ---
IN_CHANNELS = 11
CLASSES = 1
ENCODER_NAME = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
