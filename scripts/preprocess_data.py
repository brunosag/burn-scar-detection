import os
import shutil

import numpy as np
import rasterio
import spyndex
from skimage.feature import graycomatrix, graycoprops
from skimage.util.shape import view_as_windows
from tqdm import tqdm

B_RED = 0
B_NIR = 1
B_SWIR1 = 2
B_SWIR2 = 3

RAW_DATA_DIR = 'data/'
PROCESSED_DATA_DIR = 'data/processed/'


def _compute_spectral_features(raw_patch):
    params = {
        'N': raw_patch[B_NIR],
        'R': raw_patch[B_RED],
        'S1': raw_patch[B_SWIR1],
        'S2': raw_patch[B_SWIR2],
    }
    indices = spyndex.computeIndex(index=['NBR', 'NBRSWIR'], params=params)
    return np.array(indices).astype(np.float32)


def _compute_glcm_features(
    patch,
    window_size=8,
    step=8,
    distances=[1],
    angles=[0],
    properties=['contrast', 'homogeneity', 'correlation'],
):
    nir_band = patch[B_NIR, :, :]
    nir_min, nir_max = nir_band.min(), nir_band.max()
    if nir_max - nir_min > 1e-6:
        nir_band = ((nir_band - nir_min) / (nir_max - nir_min) * 255).astype(np.uint8)
    else:
        nir_band = np.zeros_like(nir_band, dtype=np.uint8)

    windows = view_as_windows(nir_band, (window_size, window_size), step=step)
    h_windows, w_windows, _, _ = windows.shape
    glcm_features = np.zeros((len(properties), h_windows, w_windows), dtype=np.float32)

    for r in range(h_windows):
        for c in range(w_windows):
            window = windows[r, c, :, :]
            glcm = graycomatrix(
                window,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True,
            )
            for i, prop in enumerate(properties):
                glcm_features[i, r, c] = graycoprops(glcm, prop)[0, 0]

    h_patch, w_patch = patch.shape[1:]
    full_size_features = np.zeros((len(properties), h_patch, w_patch), dtype=np.float32)
    for r in range(h_windows):
        for c in range(w_windows):
            full_size_features[
                :, r * step : r * step + window_size, c * step : c * step + window_size
            ] = glcm_features[:, r, c, np.newaxis, np.newaxis]

    return full_size_features


def process_and_save():
    print('Starting pre-processing of features...')

    t1_raw_dir = os.path.join(RAW_DATA_DIR, 't1')
    t2_raw_dir = os.path.join(RAW_DATA_DIR, 't2')

    t1_processed_dir = os.path.join(PROCESSED_DATA_DIR, 't1')
    t2_processed_dir = os.path.join(PROCESSED_DATA_DIR, 't2')

    # Create destination directories
    os.makedirs(t1_processed_dir, exist_ok=True)
    os.makedirs(t2_processed_dir, exist_ok=True)

    # Also copy the masks over for convenience
    shutil.copytree(
        os.path.join(RAW_DATA_DIR, 'mask'),
        os.path.join(PROCESSED_DATA_DIR, 'mask'),
        dirs_exist_ok=True,
    )

    ids = sorted([f.split('_')[-1].replace('.tif', '') for f in os.listdir(t1_raw_dir)])

    for id_ in tqdm(ids, desc='Processing Images'):
        fname = f'recorte_{id_}.tif'

        with rasterio.open(os.path.join(t1_raw_dir, fname)) as src:
            t1_raw = src.read().astype(np.float32)
        with rasterio.open(os.path.join(t2_raw_dir, fname)) as src:
            t2_raw = src.read().astype(np.float32)

        # Compute all features
        t1_spectral = _compute_spectral_features(t1_raw)
        t2_spectral = _compute_spectral_features(t2_raw)
        t1_glcm = _compute_glcm_features(t1_raw)
        t2_glcm = _compute_glcm_features(t2_raw)
        d_spectral = t1_spectral - t2_spectral

        # Stack into final 11-channel arrays
        t1_full = np.vstack((t1_raw, t1_spectral, d_spectral, t1_glcm))
        t2_full = np.vstack((t2_raw, t2_spectral, d_spectral, t2_glcm))

        # Save as .npy files
        np.save(os.path.join(t1_processed_dir, f'recorte_{id_}.npy'), t1_full)
        np.save(os.path.join(t2_processed_dir, f'recorte_{id_}.npy'), t2_full)

    print('Pre-processing complete.')


if __name__ == '__main__':
    process_and_save()
