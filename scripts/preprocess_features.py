import os

import numpy as np
import rasterio
import spyndex
from skimage.feature import graycomatrix, graycoprops

# from skimage.util.shape import view_as_windows # No longer required for global GLCM
from tqdm import tqdm

from burn_scar_detection import config

# --- Configuration ---
RAW_DATA_DIR = config.RAW_DATA_DIR
PROCESSED_DATA_DIR = 'data/processed'

T1_FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features_t1')
T2_FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features_t2')
MASK_PROC_DIR = os.path.join(PROCESSED_DATA_DIR, 'mask')

B_RED = config.B_RED
B_NIR = config.B_NIR
B_SWIR1 = config.B_SWIR1
B_SWIR2 = config.B_SWIR2

# --- Feature Calculation Functions ---


def _compute_spectral_features(raw_patch):
    """Computes NBR and NBRSWIR."""
    params = {
        'N': raw_patch[B_NIR],
        'R': raw_patch[B_RED],
        'S1': raw_patch[B_SWIR1],
        'S2': raw_patch[B_SWIR2],
    }
    for key, value in params.items():
        params[key] = value + 1e-8

    indices = spyndex.computeIndex(index=['NBR', 'NBRSWIR'], params=params)
    indices = np.nan_to_num(np.array(indices), nan=0.0)
    return indices.astype(np.float32)


def _compute_glcm_features(
    patch,
    distances=[1],
    angles=[0],
    properties=['contrast', 'homogeneity', 'correlation'],
):
    """
    Computes global GLCM features for the NIR band of a given patch.
    This replaces the slow, windowed calculation.
    """
    nir_band = patch[B_NIR, :, :]

    # 1. Normalize NIR band to 0-255 range for GLCM calculation
    nir_min, nir_max = nir_band.min(), nir_band.max()
    if nir_max - nir_min > 1e-6:
        nir_band_uint8 = ((nir_band - nir_min) / (nir_max - nir_min) * 255).astype(
            np.uint8
        )
    else:
        nir_band_uint8 = np.zeros_like(nir_band, dtype=np.uint8)

    # 2. Calculate GLCM once for the entire patch (global approach)
    glcm = graycomatrix(
        nir_band_uint8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    # 3. Extract properties from the global GLCM
    glcm_features_vector = []
    for prop in properties:
        glcm_features_vector.append(graycoprops(glcm, prop)[0, 0])

    glcm_features_vector = np.array(glcm_features_vector, dtype=np.float32)

    # 4. Broadcast the global features across the spatial dimensions of the patch
    h_patch, w_patch = patch.shape[1:]
    glcm_features_broadcasted = np.broadcast_to(
        glcm_features_vector[:, np.newaxis, np.newaxis],
        (len(properties), h_patch, w_patch),
    )

    return glcm_features_broadcasted


def _normalize_stack(patch_stack):
    """Applies per-channel percentile clipping and standardization."""
    normalized_stack = np.zeros_like(patch_stack, dtype=np.float32)
    for i in range(patch_stack.shape[0]):
        channel = patch_stack[i, :, :]
        p1, p99 = np.percentile(channel, [1, 99])
        clipped_channel = np.clip(channel, p1, p99)
        mean, std = clipped_channel.mean(), clipped_channel.std()
        normalized_stack[i, :, :] = (clipped_channel - mean) / (std + 1e-8)
    return normalized_stack


# --- Main Processing Loop ---


def process_and_save_features():
    print('Starting full feature pre-processing...')
    os.makedirs(T1_FEATURES_DIR, exist_ok=True)
    os.makedirs(T2_FEATURES_DIR, exist_ok=True)
    os.makedirs(MASK_PROC_DIR, exist_ok=True)

    ids = sorted(
        [f.split('_')[-1].replace('.tif', '') for f in os.listdir(config.RAW_T1_DIR)]
    )

    for time_step, in_dir, out_dir in [
        ('t1', config.RAW_T1_DIR, T1_FEATURES_DIR),
        ('t2', config.RAW_T2_DIR, T2_FEATURES_DIR),
    ]:
        print(f'\nProcessing timeframe: {time_step}')
        for id_ in tqdm(ids, desc=f'Calculating features for {time_step}'):
            fname_tif = f'recorte_{id_}.tif'
            fname_npy = f'recorte_{id_}.npy'
            raw_path = os.path.join(in_dir, fname_tif)

            with rasterio.open(raw_path) as src:
                raw_patch = src.read().astype(np.float32)

            spectral_feats = _compute_spectral_features(raw_patch)
            # Call the updated (global) GLCM function
            glcm_feats = _compute_glcm_features(raw_patch)

            full_feature_stack = np.vstack((raw_patch, spectral_feats, glcm_feats))
            normalized_stack = _normalize_stack(full_feature_stack)
            np.save(os.path.join(out_dir, fname_npy), normalized_stack)

    print('\nProcessing masks...')
    for id_ in tqdm(ids, desc='Processing masks'):
        fname_tif = f'recorte_{id_}.tif'
        fname_npy = f'recorte_{id_}.npy'
        mask_path = os.path.join(config.RAW_MASK_DIR, fname_tif)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        mask_binary = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
        np.save(os.path.join(MASK_PROC_DIR, fname_npy), mask_binary)

    print('\nPre-processing complete.')


if __name__ == '__main__':
    process_and_save_features()
