import os

import torch

RAW_DATA_DIR = 'data/dataset/'
RAW_T1_DIR = os.path.join(RAW_DATA_DIR, 't1')
RAW_T2_DIR = os.path.join(RAW_DATA_DIR, 't2')
RAW_MASK_DIR = os.path.join(RAW_DATA_DIR, 'mask')

PROCESSED_DATA_DIR = 'data/processed'
PROCESSED_FEATURES_T1_DIR = os.path.join(PROCESSED_DATA_DIR, 'features_t1')
PROCESSED_FEATURES_T2_DIR = os.path.join(PROCESSED_DATA_DIR, 'features_t2')
PROCESSED_MASK_DIR = os.path.join(PROCESSED_DATA_DIR, 'mask')

TEST_RAW_DATA_DIR = 'data/avaliacao'

MODEL_CHECKPOINT_DIR = 'models/'

B_RED = 0
B_NIR = 1
B_SWIR1 = 2
B_SWIR2 = 3

IN_CHANNELS = 9
CLASSES = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
