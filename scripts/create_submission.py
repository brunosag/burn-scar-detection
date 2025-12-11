import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from burn_scar_detection import config as common_config
from burn_scar_detection.data_loading import BurnScarDataset
from burn_scar_detection.inference import (
    load_model_for_inference,
    predict_batch_with_tta,
)


def load_preprocessed_test_sample(t1_path, t2_path):
    """Loads a single preprocessed test sample."""
    t1_features = np.load(t1_path)
    t2_features = np.load(t2_path)
    return torch.from_numpy(t1_features), torch.from_numpy(t2_features)


def format_submission_row(image_id, binary_mask):
    """Flattens the mask into a dictionary for the submission CSV."""
    pixels = binary_mask.flatten()
    row_data = {f'pixel_{i}': pixels[i] for i in range(len(pixels))}
    row_data['id'] = image_id
    return row_data


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission file.')
    parser.add_argument(
        '--single_model',
        type=str,
        default=None,
        help='If set, use only this model_id (e.g., "smp_siamese") instead of ensemble.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        required=True,
        help='Optimal decision threshold found during tuning.',
    )
    parser.add_argument(
        '--batch_size', type=int, default=38, help='Training batch size'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='submission.csv',
        help='Path to save the submission CSV file',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    t1_test_dir = os.path.join(common_config.PROCESSED_DATA_DIR, 'test_features_t1')
    t2_test_dir = os.path.join(common_config.PROCESSED_DATA_DIR, 'test_features_t2')

    num_test_samples = 315
    test_ids = [f'recorte_{i}' for i in range(1, num_test_samples + 1)]

    actual_t1_files = sorted(
        [f.replace('.npy', '') for f in os.listdir(t1_test_dir) if f.endswith('.npy')]
    )
    if set(test_ids) != set(actual_t1_files):
        raise ValueError('Generated test_ids do not match actual .npy files in t1 dir')

    test_dataset = BurnScarDataset(
        t1_feature_dir=t1_test_dir,
        t2_feature_dir=t2_test_dir,
        file_ids=test_ids,
        test_mode=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count() or 2,
        pin_memory=True,
    )

    if len(test_dataset) != num_test_samples:
        raise ValueError(
            f'Test dataset has {len(test_dataset)} samples, but expected {num_test_samples}'
        )

    if args.single_model:
        model_path = f'models/best_model_{args.single_model}_bce_lovasz_2stage.pth'
        model = load_model_for_inference(args.single_model, model_path)
        models = [model]
    else:
        ensemble = [
            (
                'smp_siamese',
                'models/best_model_smp_siamese_bce_lovasz_2stage.pth',
            ),
            ('smp_unetpp', 'models/best_model_smp_unetpp_bce_lovasz_2stage.pth'),
            (
                'custom_unet',
                'models/best_model_custom_unet_bce_lovasz_2stage.pth',
            ),
        ]
        models = []
        for mid, path in ensemble:
            model = load_model_for_inference(
                mid, path
            )  # Assumes load_model_for_inference can handle mid
            model.load_state_dict(
                torch.load(path, map_location=common_config.DEVICE)
            )  # Load specific weights
            models.append(model)

    submission_rows = {}
    sample_index = 0

    with torch.no_grad():
        for t1_batch, t2_batch, id_batch in tqdm(
            test_loader, desc='Generating predictions'
        ):
            t1_batch = t1_batch.to(common_config.DEVICE)
            t2_batch = t2_batch.to(common_config.DEVICE)

            probabilities_batch = predict_batch_with_tta(
                model, t1_batch, t2_batch
            )  # Or ensemble version

            for i in range(probabilities_batch.shape[0]):
                probabilities = probabilities_batch[i].cpu().numpy()
                image_id = id_batch[i]  # From loader, e.g., 'recorte_1'
                binary_mask = (probabilities > args.threshold).astype(np.uint8)
                submission_id = image_id + '.tif'

                if submission_id in submission_rows:
                    print(
                        f'Warning: Duplicate id {submission_id} detected—overwriting.'
                    )
                submission_rows[submission_id] = format_submission_row(
                    submission_id, binary_mask
                )
                sample_index += 1

    print('Creating submission file...')
    # Convert dict to list for DataFrame
    submission_list = list(submission_rows.values())
    submission_df = pd.DataFrame(submission_list)

    # Reorder columns and ensure no dups
    cols = ['id'] + [col for col in submission_df.columns if col != 'id']
    submission_df = submission_df[cols].drop_duplicates(subset=['id'])  # Extra safety

    # Final check
    if len(submission_df) != 315:
        raise ValueError(f'Expected 315 unique rows, but got {len(submission_df)}')

    submission_df.to_csv(args.output_csv, index=False)
    print(f'Submission saved successfully to {args.output_csv}')


if __name__ == '__main__':
    main()
