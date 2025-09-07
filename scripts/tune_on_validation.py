import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from burn_scar_detection import config as common_config
from burn_scar_detection.data_loading import BurnScarDataset
from burn_scar_detection.models import get_model


def get_tta_predictions(model, loader, device):
    """Run model on validation set with TTA and collect predictions/ground truths."""
    model.eval()
    all_probs = []
    all_masks = []
    with torch.no_grad():
        for t1, t2, mask in tqdm(
            loader, desc='Generating validation predictions with TTA'
        ):
            t1, t2 = t1.to(device), t2.to(device)

            pred_identity = torch.sigmoid(model(t1, t2))

            pred_hflip_aug = torch.sigmoid(model(TF.hflip(t1), TF.hflip(t2)))
            pred_hflip = TF.hflip(pred_hflip_aug)

            pred_vflip_aug = torch.sigmoid(model(TF.vflip(t1), TF.vflip(t2)))
            pred_vflip = TF.vflip(pred_vflip_aug)

            avg_prob = torch.mean(
                torch.stack([pred_identity, pred_hflip, pred_vflip]), dim=0
            )

            all_probs.append(avg_prob.cpu().numpy())
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
        description='Tune decision threshold using validation data.'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path to model checkpoint.'
    )
    parser.add_argument(
        '--model_architecture',
        type=str,
        required=True,
        choices=['smp_siamese', 'custom_unet'],
    )
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b0')
    parser.add_argument('--batch_size', type=int, default=32)
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

    model_params = {
        'in_channels': common_config.IN_CHANNELS,
        'classes': common_config.CLASSES,
        'encoder_name': args.encoder_name,
    }
    model = get_model(args.model_architecture, model_params)
    model.load_state_dict(
        torch.load(args.model_path, map_location=common_config.DEVICE)
    )
    model.to(common_config.DEVICE)

    print('Calculating TTA predictions for threshold tuning...')
    all_probs, all_masks = get_tta_predictions(model, val_loader, common_config.DEVICE)
    best_threshold, best_f1 = find_optimal_threshold(all_probs, all_masks)

    print('\n--- Tuning Complete ---')
    print(f'Model: {args.model_path}')
    print(f'Best F1 Score on Validation Set (with TTA): {best_f1:.4f}')
    print(f'Optimal Threshold: {best_threshold:.4f}')

    results = {'best_threshold': best_threshold, 'validation_f1_tta': best_f1}
    output_filename = os.path.join(
        os.path.dirname(args.model_path),
        f'best_threshold_{args.model_architecture}.json',
    )
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'Best threshold saved to: {output_filename}')


if __name__ == '__main__':
    main()
