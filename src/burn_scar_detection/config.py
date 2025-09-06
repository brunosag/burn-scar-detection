import os

import torch

# --- Data Configuration ---
RAW_DATA_DIR = 'data/dataset/'
PROCESSED_DATA_DIR = 'data/processed/'

T1_DIR = os.path.join(RAW_DATA_DIR, 't1')
T2_DIR = os.path.join(RAW_DATA_DIR, 't2')
MASK_DIR = os.path.join(RAW_DATA_DIR, 'mask')

GLCM_T1_DIR = os.path.join(PROCESSED_DATA_DIR, 'glcm_t1')
GLCM_T2_DIR = os.path.join(PROCESSED_DATA_DIR, 'glcm_t2')


# --- Training Configuration ---
BATCH_SIZE = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
EPOCHS = 200
MODEL_PATH = 'models/cbam_unet_best.pth'

# --- Early Stopping Configuration ---
# Number of epochs to wait for improvement before stopping training.
EARLY_STOPPING_PATIENCE = 15
# Minimum change in the monitored quantity to qualify as an improvement.
EARLY_STOPPING_MIN_DELTA = 0.0001

# --- Band Indices ---
B_RED = 0
B_NIR = 1
B_SWIR1 = 2
B_SWIR2 = 3

# --- Model Configuration ---
IN_CHANNELS = 9  # 4 raw bands + 2 spectral indices + 3 GLCM features
CLASSES = 1
ENCODER_NAME = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
