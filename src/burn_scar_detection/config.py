import os

import torch

RAW_DATA_DIR = 'data/dataset/'
PROCESSED_DATA_DIR = 'data/processed'

RAW_T1_DIR = os.path.join(RAW_DATA_DIR, 't1')
RAW_T2_DIR = os.path.join(RAW_DATA_DIR, 't2')
RAW_MASK_DIR = os.path.join(RAW_DATA_DIR, 'mask')

PROCESSED_FEATURES_T1_DIR = os.path.join(PROCESSED_DATA_DIR, 'features_t1')
PROCESSED_FEATURES_T2_DIR = os.path.join(PROCESSED_DATA_DIR, 'features_t2')
PROCESSED_MASK_DIR = os.path.join(PROCESSED_DATA_DIR, 'mask')

BATCH_SIZE = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
EPOCHS = 200
MODEL_PATH = 'models/cbam_unet_best.pth'

EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.0001

B_RED = 0
B_NIR = 1
B_SWIR1 = 2
B_SWIR2 = 3

IN_CHANNELS = 9
CLASSES = 1
ENCODER_NAME = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
