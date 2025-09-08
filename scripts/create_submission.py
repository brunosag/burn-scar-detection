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
        '--model',
        type=str,
        required=True,
        choices=['smp_siamese', 'custom_unet'],
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
        required=True,
        help='Path to save the submission CSV file',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model_for_inference(args.model)

    t1_test_dir = os.path.join(common_config.PROCESSED_DATA_DIR, 'test_features_t1')
    t2_test_dir = os.path.join(common_config.PROCESSED_DATA_DIR, 'test_features_t2')
    test_ids = sorted(
        [f.replace('.npy', '') for f in os.listdir(t1_test_dir) if f.endswith('.npy')]
    )

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

    submission_rows = []
    with torch.no_grad():
        for t1_batch, t2_batch, id_batch in tqdm(
            test_loader, desc='Generating predictions'
        ):
            t1_batch = t1_batch.to(common_config.DEVICE)
            t2_batch = t2_batch.to(common_config.DEVICE)

            probabilities_batch = predict_batch_with_tta(model, t1_batch, t2_batch)

            for i in range(probabilities_batch.shape[0]):
                probabilities = probabilities_batch[i].cpu().numpy()
                image_id = id_batch[i]
                binary_mask = (probabilities > args.threshold).astype(np.uint8)
                submission_id = image_id.replace('_', '') + '.tif'
                submission_rows.append(
                    format_submission_row(submission_id, binary_mask)
                )

    print('Creating submission file...')
    submission_df = pd.DataFrame(submission_rows)
    cols = ['id'] + [col for col in submission_df.columns if col != 'id']
    submission_df = submission_df[cols]
    submission_df.to_csv(args.output_csv, index=False)
    print(f'Submission saved successfully to {args.output_csv}')


if __name__ == '__main__':
    main()
