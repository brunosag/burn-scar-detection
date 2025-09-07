import json
import os

import numpy as np
import rasterio
import spyndex
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from burn_scar_detection import config

RAW_DATA_DIR = config.RAW_DATA_DIR
PROCESSED_DATA_DIR = 'data/processed'
SPLIT_FILE = os.path.join(PROCESSED_DATA_DIR, 'splits.json')

T1_FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features_t1')
T2_FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features_t2')
MASK_PROC_DIR = os.path.join(PROCESSED_DATA_DIR, 'mask')

B_RED = config.B_RED
B_NIR = config.B_NIR
B_SWIR1 = config.B_SWIR1
B_SWIR2 = config.B_SWIR2


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
    patch, properties=['contrast', 'homogeneity', 'correlation']
):
    """Computes global GLCM features for the NIR band."""
    nir_band = patch[B_NIR, :, :]
    nir_min, nir_max = nir_band.min(), nir_band.max()
    if nir_max - nir_min > 1e-6:
        nir_band_uint8 = ((nir_band - nir_min) / (nir_max - nir_min) * 255).astype(
            np.uint8
        )
    else:
        nir_band_uint8 = np.zeros_like(nir_band, dtype=np.uint8)

    glcm = graycomatrix(
        nir_band_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    glcm_features_vector = [graycoprops(glcm, prop)[0, 0] for prop in properties]
    glcm_features_vector = np.array(glcm_features_vector, dtype=np.float32)
    h_patch, w_patch = patch.shape[1:]
    glcm_features_broadcasted = np.broadcast_to(
        glcm_features_vector[:, np.newaxis, np.newaxis],
        (len(properties), h_patch, w_patch),
    )
    return glcm_features_broadcasted


def _get_feature_stack(patch_id, time_step, in_dir):
    """Helper function to load raw data and compute features before normalization."""
    raw_path = os.path.join(in_dir, f'{patch_id}.tif')
    with rasterio.open(raw_path) as src:
        raw_patch = src.read().astype(np.float32)

    spectral_feats = _compute_spectral_features(raw_patch)
    glcm_feats = _compute_glcm_features(raw_patch)
    full_feature_stack = np.vstack((raw_patch, spectral_feats, glcm_feats))
    return full_feature_stack


def calculate_global_statistics(train_ids):
    """Calculates normalization statistics based *only* on the training set."""
    print(f'Calculating global statistics from {len(train_ids)} training samples...')
    all_t1_data = []
    all_t2_data = []

    for id_ in tqdm(train_ids, desc='Loading training data for stats'):
        t1_stack = _get_feature_stack(id_, 't1', config.RAW_T1_DIR)
        t2_stack = _get_feature_stack(id_, 't2', config.RAW_T2_DIR)

        all_t1_data.append(t1_stack.reshape(t1_stack.shape[0], -1))
        all_t2_data.append(t2_stack.reshape(t2_stack.shape[0], -1))

    t1_full = np.concatenate(all_t1_data, axis=1)
    t2_full = np.concatenate(all_t2_data, axis=1)

    stats = {'t1': {}, 't2': {}}
    for full_data, time_key in [(t1_full, 't1'), (t2_full, 't2')]:
        p1 = np.percentile(full_data, 1, axis=1)
        p99 = np.percentile(full_data, 99, axis=1)

        clipped_data = np.clip(full_data, p1[:, np.newaxis], p99[:, np.newaxis])

        mean = np.mean(clipped_data, axis=1)
        std = np.std(clipped_data, axis=1)

        stats[time_key]['p1'] = p1
        stats[time_key]['p99'] = p99
        stats[time_key]['mean'] = mean
        stats[time_key]['std'] = std

    print('Global statistics calculation complete.')
    return stats


def apply_global_normalization(patch_stack, stats):
    """Applies pre-calculated global statistics to normalize a patch."""
    normalized_stack = np.zeros_like(patch_stack, dtype=np.float32)
    num_channels = patch_stack.shape[0]

    for i in range(num_channels):
        p1 = stats['p1'][i]
        p99 = stats['p99'][i]
        mean = stats['mean'][i]
        std = stats['std'][i]

        clipped_channel = np.clip(patch_stack[i, :, :], p1, p99)
        normalized_stack[i, :, :] = (clipped_channel - mean) / (std + 1e-8)

    return normalized_stack


def process_and_save_features(global_stats, all_ids):
    """Processes all images using the calculated global statistics."""
    os.makedirs(T1_FEATURES_DIR, exist_ok=True)
    os.makedirs(T2_FEATURES_DIR, exist_ok=True)
    os.makedirs(MASK_PROC_DIR, exist_ok=True)

    for time_step, in_dir, out_dir in [
        ('t1', config.RAW_T1_DIR, T1_FEATURES_DIR),
        ('t2', config.RAW_T2_DIR, T2_FEATURES_DIR),
    ]:
        print(f'\nProcessing timeframe: {time_step}')
        time_specific_stats = global_stats[time_step]

        for id_ in tqdm(all_ids, desc=f'Applying normalization for {time_step}'):
            feature_stack = _get_feature_stack(id_, time_step, in_dir)

            normalized_stack = apply_global_normalization(
                feature_stack, time_specific_stats
            )

            np.save(os.path.join(out_dir, f'{id_}.npy'), normalized_stack)

    print('\nProcessing masks...')
    for id_ in tqdm(all_ids, desc='Processing masks'):
        mask_path = os.path.join(config.RAW_MASK_DIR, f'{id_}.tif')
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        mask_binary = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
        np.save(os.path.join(MASK_PROC_DIR, f'{id_}.npy'), mask_binary)


def main():
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
    train_ids = splits['train']
    all_ids = splits['train'] + splits['validation']

    global_stats = calculate_global_statistics(train_ids)

    process_and_save_features(global_stats, all_ids)
    print('\nPre-processing complete.')


if __name__ == '__main__':
    main()
