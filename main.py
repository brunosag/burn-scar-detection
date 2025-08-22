import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

THRESHOLD = 0.2
PRE_FIRE_PATHS = ['bands/pre_B8A.jp2', 'bands/pre_B12.jp2']
POST_FIRE_PATHS = ['bands/post_B8A.jp2', 'bands/post_B12.jp2']


def calculate_nbr(nir_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
    """Calculates the Normalized Burn Ratio (NBR)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        nbr = (nir_band.astype(float) - swir_band.astype(float)) / (
            nir_band + swir_band
        )
    return nbr


def main():
    """Main function to run the burn scar analysis workflow."""
    pre_nir, pre_swir = [rasterio.open(path).read(1) for path in PRE_FIRE_PATHS]
    post_nir, post_swir = [rasterio.open(path).read(1) for path in POST_FIRE_PATHS]

    pre_nbr = calculate_nbr(pre_nir, pre_swir)
    post_nbr = calculate_nbr(post_nir, post_swir)

    dnbr = np.nan_to_num(pre_nbr - post_nbr)
    burn_mask = dnbr > THRESHOLD

    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(burn_mask, cmap='gray')

    unburned_patch = mpatches.Patch(color='black', label='Unburned')
    burned_patch = mpatches.Patch(color='white', label='Burned')

    ax.legend(handles=[unburned_patch, burned_patch], loc='upper right')
    ax.set_title(f'Final Burn Mask (dNBR > {THRESHOLD})')
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
