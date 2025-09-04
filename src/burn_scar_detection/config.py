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
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
EPOCHS = 32
MODEL_PATH = 'models/cbam_unet_best.pth'

# --- Band Indices ---
# These are indices for the 4-channel raw input
B_RED = 0
B_NIR = 1
B_SWIR1 = 2
B_SWIR2 = 3

# --- Model Configuration ---
# 4 raw bands + 2 spectral indices + 3 GLCM features
IN_CHANNELS = 9
CLASSES = 1
ENCODER_NAME = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
