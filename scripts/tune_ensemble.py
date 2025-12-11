import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from burn_scar_detection import config as common_config
from burn_scar_detection.data_loading import BurnScarDataset
from burn_scar_detection.inference import (
    load_model_for_inference,
    predict_batch_with_tta,
)


def get_tta_predictions(models, loader, device):
    """Run ensemble on validation set with TTA and collect averaged predictions/ground truths."""
    for model in models:
        model.eval()
    all_probs = []
    all_masks = []
    with torch.no_grad():
        for t1, t2, mask in tqdm(
            loader, desc='Generating validation predictions with ensemble TTA'
        ):
            t1, t2 = t1.to(device), t2.to(device)

            # Average across ensemble (each with TTA)
            ensemble_probs_batch = []
            for model in models:
                probs_batch = predict_batch_with_tta(model, t1, t2)
                ensemble_probs_batch.append(probs_batch)
            avg_probs_batch = torch.mean(torch.stack(ensemble_probs_batch), dim=0)

            all_probs.append(avg_probs_batch.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0).astype(int)
    return all_probs, all_masks


def find_optimal_threshold(all_probs, all_masks):
    """Sweeps thresholds to find the best F1 score."""
    best_f1 = 0.0
    best_threshold = 0.5
    thresholds = np.linspace(0.1, 0.9, num=81)
    masks_flat = all_masks.ravel()

    for threshold in tqdm(thresholds, desc='Tuning threshold'):
        preds_binary_flat = (all_probs > threshold).ravel()
        current_f1 = f1_score(masks_flat, preds_binary_flat)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


def main():
    parser = argparse.ArgumentParser(
        description='Tune decision threshold using validation data for ensemble.'
    )
    parser.add_argument('--batch_size', type=int, default=38)
    args = parser.parse_args()

    split_file_path = os.path.join(common_config.PROCESSED_DATA_DIR, 'splits.json')
    with open(split_file_path, 'r') as f:
        splits = json.load(f)
    val_dataset = BurnScarDataset(
        t1_feature_dir=common_config.PROCESSED_FEATURES_T1_DIR,
        t2_feature_dir=common_config.PROCESSED_FEATURES_T2_DIR,
        mask_dir=common_config.PROCESSED_MASK_DIR,
        file_ids=splits['validation'],
        augmentations=None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Define ensemble: list of (model_id, checkpoint_path) tuples
    ensemble = [
        (
            'smp_siamese',
            'models/best_model_smp_siamese_bce_lovasz_2stage.pth',
        ),  # GLCM run (F1 0.8964)
        (
            'smp_unetpp',
            'models/best_model_smp_unetpp_bce_lovasz_2stage.pth',
        ),  # U-Net++ run (F1 0.8864)
        (
            'custom_unet',
            'models/best_model_custom_unet_bce_lovasz_2stage.pth',
        ),  # Original from README (F1 0.8889)
    ]

    # Load all models
    models = []
    for mid, path in ensemble:
        model = load_model_for_inference(mid, path)
        model.load_state_dict(
            torch.load(path, map_location=common_config.DEVICE)
        )  # Load specific weights
        model.to(common_config.DEVICE)
        models.append(model)

    print('Calculating ensemble TTA predictions for threshold tuning...')
    all_probs, all_masks = get_tta_predictions(models, val_loader, common_config.DEVICE)
    best_threshold, best_f1 = find_optimal_threshold(all_probs, all_masks)

    print('\n--- Ensemble Tuning Complete ---')
    print(f'Best F1 Score on Validation Set (with ensemble TTA): {best_f1:.4f}')
    print(f'Optimal Threshold: {best_threshold:.4f}')

    results = {'best_threshold': best_threshold, 'validation_f1_tta': best_f1}
    output_filename = os.path.join(
        common_config.MODEL_CHECKPOINT_DIR,
        'best_threshold_ensemble.json',
    )
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'Best ensemble threshold saved to: {output_filename}')


if __name__ == '__main__':
    main()
