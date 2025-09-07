import json
import os

import numpy as np
import rasterio
import spyndex
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from burn_scar_detection import config

RAW_T1_DIR = config.RAW_T1_DIR
RAW_T2_DIR = config.RAW_T2_DIR
RAW_MASK_DIR = config.RAW_MASK_DIR

PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
SPLIT_FILE = os.path.join(PROCESSED_DATA_DIR, 'splits.json')
STATS_FILE = os.path.join(PROCESSED_DATA_DIR, 'global_stats.npz')

TEST_RAW_DATA_DIR = config.TEST_RAW_DATA_DIR

B_RED = config.B_RED
B_NIR = config.B_NIR
B_SWIR1 = config.B_SWIR1
B_SWIR2 = config.B_SWIR2


def _compute_spectral_features(raw_patch):
    """Computes spectral indices used in training."""
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


def _get_feature_stack(patch_id, time_step, raw_data_dir):
    """Helper function to load raw data and compute features before normalization."""
    raw_path = os.path.join(raw_data_dir, f'{patch_id}.tif')
    with rasterio.open(raw_path) as src:
        raw_patch = src.read().astype(np.float32)

    spectral_feats = _compute_spectral_features(raw_patch)
    glcm_feats = _compute_glcm_features(raw_patch)
    full_feature_stack = np.vstack((raw_patch, spectral_feats, glcm_feats))
    return full_feature_stack


def calculate_and_save_statistics(train_ids):
    """Calculates normalization statistics based *only* on the training set."""
    print(f'Calculating global statistics from {len(train_ids)} training samples...')
    all_t1_data = []
    all_t2_data = []

    for id_ in tqdm(train_ids, desc='Loading training data for stats'):
        t1_stack = _get_feature_stack(id_, 't1', RAW_T1_DIR)
        t2_stack = _get_feature_stack(id_, 't2', RAW_T2_DIR)
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

    np.savez_compressed(STATS_FILE, t1=stats['t1'], t2=stats['t2'])
    print(f'Global statistics saved to {STATS_FILE}')
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


def process_split_data(ids, global_stats, raw_dirs, output_dirs, process_mask=True):
    """Processes features and masks for a given set of file IDs."""
    os.makedirs(output_dirs['t1'], exist_ok=True)
    os.makedirs(output_dirs['t2'], exist_ok=True)
    if process_mask:
        os.makedirs(output_dirs['mask'], exist_ok=True)

    for time_step, in_dir_key, out_dir_key in [('t1', 't1', 't1'), ('t2', 't2', 't2')]:
        print(f'\nProcessing timeframe: {time_step}')
        time_specific_stats = global_stats[time_step]

        for id_ in tqdm(ids, desc=f'Applying normalization for {time_step}'):
            feature_stack = _get_feature_stack(id_, time_step, raw_dirs[in_dir_key])
            normalized_stack = apply_global_normalization(
                feature_stack, time_specific_stats
            )
            np.save(
                os.path.join(output_dirs[out_dir_key], f'{id_}.npy'), normalized_stack
            )

    if process_mask:
        print('\nProcessing masks...')
        for id_ in tqdm(ids, desc='Processing masks'):
            mask_path = os.path.join(raw_dirs['mask'], f'{id_}.tif')
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            mask_binary = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
            np.save(os.path.join(output_dirs['mask'], f'{id_}.npy'), mask_binary)


def main():
    print('--- Processing Training and Validation Data ---')
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
    train_ids = splits['train']
    val_ids = splits['validation']

    stats = calculate_and_save_statistics(train_ids)

    train_val_raw_dirs = {'t1': RAW_T1_DIR, 't2': RAW_T2_DIR, 'mask': RAW_MASK_DIR}
    train_val_output_dirs = {
        't1': config.PROCESSED_FEATURES_T1_DIR,
        't2': config.PROCESSED_FEATURES_T2_DIR,
        'mask': config.PROCESSED_MASK_DIR,
    }
    process_split_data(
        train_ids + val_ids,
        stats,
        train_val_raw_dirs,
        train_val_output_dirs,
        process_mask=True,
    )

    print('\n--- Processing Test Data ---')
    test_raw_t1_dir = os.path.join(TEST_RAW_DATA_DIR, 't1')
    if not os.path.exists(test_raw_t1_dir):
        print(f'Error: Test directory not found at {test_raw_t1_dir}')
        print('Skipping test data processing.')
        return

    test_ids = sorted(
        [
            f.replace('.tif', '')
            for f in os.listdir(test_raw_t1_dir)
            if f.endswith('.tif')
        ]
    )

    test_raw_dirs = {
        't1': os.path.join(TEST_RAW_DATA_DIR, 't1'),
        't2': os.path.join(TEST_RAW_DATA_DIR, 't2'),
    }
    test_output_dirs = {
        't1': os.path.join(PROCESSED_DATA_DIR, 'test_features_t1'),
        't2': os.path.join(PROCESSED_DATA_DIR, 'test_features_t2'),
    }

    process_split_data(
        test_ids, stats, test_raw_dirs, test_output_dirs, process_mask=False
    )

    print('\nFull preprocessing complete for train, validation, and test sets.')


if __name__ == '__main__':
    main()
