import os

import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops
from skimage.util.shape import view_as_windows
from tqdm import tqdm

B_NIR = 1  # NIR is the second band (index 1)

RAW_DATA_DIR = 'data/dataset/'
PROCESSED_DATA_DIR = 'data/processed/'


def _compute_glcm_features(
    patch,
    window_size=8,
    step=8,
    distances=[1],
    angles=[0],
    properties=['contrast', 'homogeneity', 'correlation'],
):
    """Computes GLCM features for the NIR band of a given patch."""
    nir_band = patch[B_NIR, :, :]

    # Normalize NIR band to 0-255 for GLCM calculation
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
    # Fill the larger feature map by broadcasting the window value
    for r in range(h_windows):
        for c in range(w_windows):
            full_size_features[
                :, r * step : r * step + window_size, c * step : c * step + window_size
            ] = glcm_features[:, r, c, np.newaxis, np.newaxis]

    return full_size_features


def process_and_save_glcm():
    print('Starting pre-processing of GLCM features...')

    for time_step in ['t1', 't2']:
        raw_dir = os.path.join(RAW_DATA_DIR, time_step)
        processed_dir = os.path.join(PROCESSED_DATA_DIR, f'glcm_{time_step}')
        os.makedirs(processed_dir, exist_ok=True)

        ids = sorted([f.split('_')[-1] for f in os.listdir(raw_dir)])

        for id_ in tqdm(ids, desc=f'Processing {time_step} GLCM'):
            fname = f'recorte_{id_}'
            with rasterio.open(os.path.join(raw_dir, fname)) as src:
                raw_patch = src.read().astype(np.float32)

            glcm_feats = _compute_glcm_features(raw_patch)
            np.save(
                os.path.join(processed_dir, fname.replace('.tif', '.npy')), glcm_feats
            )

    print('GLCM pre-processing complete.')


if __name__ == '__main__':
    process_and_save_glcm()
