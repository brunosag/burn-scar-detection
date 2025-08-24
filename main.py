import argparse
import os
import sys

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

PRE_FIRE_PATHS = ['bands/pre_B8A.jp2', 'bands/pre_B12.jp2']
POST_FIRE_PATHS = ['bands/post_B8A.jp2', 'bands/post_B12.jp2']
MODEL_FILENAME = 'burn_scar_rf_model.joblib'
DNBR_THRESHOLD = 0.2


def calculate_nbr(nir_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
    """Calculates the Normalized Burn Ratio (NBR)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        nbr = (nir_band.astype(float) - swir_band.astype(float)) / (
            nir_band + swir_band
        )
    return np.nan_to_num(nbr)


def load_bands(
    pre_fire_paths: list[str], post_fire_paths: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads all necessary Sentinel-2 bands from disk."""
    print('Loading Sentinel-2 bands...')
    for path in pre_fire_paths + post_fire_paths:
        if not os.path.exists(path):
            print(f"Error: Band file not found at '{path}'", file=sys.stderr)
            sys.exit(1)

    pre_nir, pre_swir = [rasterio.open(p).read(1) for p in pre_fire_paths]
    post_nir, post_swir = [rasterio.open(p).read(1) for p in post_fire_paths]
    return pre_nir, pre_swir, post_nir, post_swir


def prepare_data_for_model(bands: tuple[np.ndarray, ...], dnbr_threshold: float):
    """Prepares the feature matrix (X) and label vector (y)."""
    print('Preparing data for the model...')
    pre_nir, pre_swir, post_nir, post_swir = bands

    pre_nbr = calculate_nbr(pre_nir, pre_swir)
    post_nbr = calculate_nbr(post_nir, post_swir)
    dnbr = pre_nbr - post_nbr

    y_labels = (dnbr > dnbr_threshold).ravel()

    X_features = np.stack(
        [
            pre_nir.ravel(),
            pre_swir.ravel(),
            post_nir.ravel(),
            post_swir.ravel(),
        ],
        axis=1,
    )

    return X_features, y_labels, dnbr.shape


def visualize_mask(mask: np.ndarray, title: str) -> None:
    """Displays the final burn mask using Matplotlib."""
    print('Visualizing the final mask...')
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask, cmap='gray')

    unburned_patch = mpatches.Patch(color='black', label='Unburned')
    burned_patch = mpatches.Patch(color='white', label='Burned')

    ax.legend(handles=[unburned_patch, burned_patch])
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def train() -> None:
    """Trains the Random Forest model and saves it to disk."""
    bands = load_bands(PRE_FIRE_PATHS, POST_FIRE_PATHS)
    X, y, _ = prepare_data_for_model(bands, DNBR_THRESHOLD)

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, verbose=2
    )

    print('\nTraining the Random Forest model...')
    model.fit(X, y)

    joblib.dump(model, MODEL_FILENAME)
    print(f"\nModel trained and saved to '{MODEL_FILENAME}'")


def predict() -> None:
    """Loads a pre-trained model and generates a burn scar prediction."""
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file not found at '{MODEL_FILENAME}'", file=sys.stderr)
        print(
            "Please run the 'train' command first to create the model.", file=sys.stderr
        )
        sys.exit(1)

    bands = load_bands(PRE_FIRE_PATHS, POST_FIRE_PATHS)
    X, _, original_shape = prepare_data_for_model(bands, DNBR_THRESHOLD)

    print(f"Loading model from '{MODEL_FILENAME}'...")
    model = joblib.load(MODEL_FILENAME)

    print('Generating predictions...')
    y_prediction = model.predict(X)

    prediction_mask = y_prediction.reshape(original_shape)

    visualize_mask(prediction_mask, 'Burn Scar Mask (Random Forest Prediction)')


def main() -> None:
    """
    Main entry point for the script.
    Parses command-line arguments to decide whether to train or predict.
    """
    parser = argparse.ArgumentParser(
        description='Burn scar detection using dNBR and a Random Forest model.'
    )
    parser.add_argument(
        'action',
        choices=['train', 'predict'],
        help="Action to perform: 'train' a new model or 'predict' using an existing one.",
    )
    args = parser.parse_args()

    if args.action == 'train':
        train()
    elif args.action == 'predict':
        predict()


if __name__ == '__main__':
    main()
