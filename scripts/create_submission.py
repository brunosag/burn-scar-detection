import argparse
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from burn_scar_detection import config as common_config
from burn_scar_detection.models import get_model


def load_preprocessed_test_sample(t1_path, t2_path):
    """Loads a single preprocessed test sample."""
    t1_features = np.load(t1_path)
    t2_features = np.load(t2_path)
    return torch.from_numpy(t1_features), torch.from_numpy(t2_features)


def predict_with_tta(model, t1_tensor, t2_tensor, device):
    """Performs inference with horizontal and vertical flip augmentations."""
    model.eval()
    t1 = t1_tensor.to(device).unsqueeze(0)
    t2 = t2_tensor.to(device).unsqueeze(0)

    with torch.no_grad():
        pred_identity = torch.sigmoid(model(t1, t2))

        pred_hflip_aug = torch.sigmoid(model(TF.hflip(t1), TF.hflip(t2)))
        pred_hflip = TF.hflip(pred_hflip_aug)

        pred_vflip_aug = torch.sigmoid(model(TF.vflip(t1), TF.vflip(t2)))
        pred_vflip = TF.vflip(pred_vflip_aug)

        avg_prob = torch.mean(
            torch.stack([pred_identity, pred_hflip, pred_vflip]), dim=0
        )

    return avg_prob.squeeze().cpu().numpy()


def format_submission_row(image_id, binary_mask):
    """Flattens the mask into a dictionary for the submission CSV."""
    pixels = binary_mask.flatten()
    row_data = {f'pixel_{i}': pixels[i] for i in range(len(pixels))}
    row_data['id'] = image_id
    return row_data


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission file.')
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path to best model checkpoint.'
    )
    parser.add_argument(
        '--model_architecture',
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
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b0')
    parser.add_argument('--output_csv', type=str, default='submission.csv')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f'Loading model: {args.model_path}')
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

    t1_test_dir = os.path.join(common_config.PROCESSED_DATA_DIR, 'test_features_t1')
    t2_test_dir = os.path.join(common_config.PROCESSED_DATA_DIR, 'test_features_t2')
    test_ids = sorted(
        [f.replace('.npy', '') for f in os.listdir(t1_test_dir) if f.endswith('.npy')]
    )
    print(f'Found {len(test_ids)} preprocessed test images.')

    submission_rows = []
    for image_id in tqdm(test_ids, desc='Generating predictions'):
        t1_path = os.path.join(t1_test_dir, f'{image_id}.npy')
        t2_path = os.path.join(t2_test_dir, f'{image_id}.npy')

        t1_tensor, t2_tensor = load_preprocessed_test_sample(t1_path, t2_path)
        probabilities = predict_with_tta(
            model, t1_tensor, t2_tensor, common_config.DEVICE
        )
        binary_mask = (probabilities > args.threshold).astype(np.uint8)
        submission_id = image_id.replace('_', '') + '.tif'
        submission_rows.append(format_submission_row(submission_id, binary_mask))

    print('Creating submission file...')
    submission_df = pd.DataFrame(submission_rows)
    cols = ['id'] + [col for col in submission_df.columns if col != 'id']
    submission_df = submission_df[cols]
    submission_df.to_csv(args.output_csv, index=False)
    print(f'Submission saved successfully to {args.output_csv}')


if __name__ == '__main__':
    main()
