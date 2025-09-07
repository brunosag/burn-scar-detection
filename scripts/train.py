import argparse
import csv
import json
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from burn_scar_detection import config as common_config
from burn_scar_detection.data_loading import BurnScarDataset, JointTransform
from burn_scar_detection.engine import evaluate, train_one_epoch
from burn_scar_detection.models import get_model


def parse_args():
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train Burn Scar Segmentation Model')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['smp_siamese', 'custom_unet'],
        help='Model architecture to train.',
    )
    parser.add_argument(
        '--epochs', type=int, default=200, help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=24, help='Training batch size'
    )
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=15,
        help='Patience for early stopping (epochs without improvement)',
    )
    parser.add_argument(
        '--early_stopping_delta',
        type=float,
        default=0.0001,
        help='Minimum improvement in F1 score to reset patience',
    )
    parser.add_argument(
        '--scheduler_patience',
        type=int,
        default=5,
        help='Patience for learning rate scheduler (epochs without improvement)',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print('--- Starting Training Run ---')
    print(f'Selected Model Architecture: {args.model}')
    print(f'Using device: {common_config.DEVICE}')

    split_file_path = os.path.join(common_config.PROCESSED_DATA_DIR, 'splits.json')
    with open(split_file_path, 'r') as f:
        splits = json.load(f)

    train_ids = splits['train']
    val_ids = splits['validation']

    train_dataset = BurnScarDataset(
        t1_feature_dir=common_config.PROCESSED_FEATURES_T1_DIR,
        t2_feature_dir=common_config.PROCESSED_FEATURES_T2_DIR,
        mask_dir=common_config.PROCESSED_MASK_DIR,
        file_ids=train_ids,
        augmentations=JointTransform(p_flip=0.5, p_photometric=0.2),
    )
    val_dataset = BurnScarDataset(
        t1_feature_dir=common_config.PROCESSED_FEATURES_T1_DIR,
        t2_feature_dir=common_config.PROCESSED_FEATURES_T2_DIR,
        mask_dir=common_config.PROCESSED_MASK_DIR,
        file_ids=val_ids,
        augmentations=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() or 2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count() or 2,
        pin_memory=True,
    )
    print(
        f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}'
    )

    model = get_model(
        args.model,
        n_channels=common_config.N_CHANNELS,
        n_classes=common_config.N_CLASSES,
    ).to(common_config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=args.scheduler_patience,
    )

    best_f1_score = -1.0
    epochs_no_improve = 0

    run_name = args.model
    model_save_path = os.path.join(
        common_config.MODEL_CHECKPOINT_DIR, f'best_model_{run_name}.pth'
    )
    log_file_path = os.path.join(
        common_config.MODEL_CHECKPOINT_DIR, f'training_log_{run_name}.csv'
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    csv_header = [
        'epoch',
        'train_loss',
        'val_loss',
        'val_f1',
        'val_iou',
        'val_auc',
        'learning_rate',
    ]
    print(f'Logging training progress to: {log_file_path}')

    with open(log_file_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(csv_header)

        for epoch in range(1, args.epochs + 1):
            print(f'\n--- Epoch {epoch}/{args.epochs} ---')
            train_loss = train_one_epoch(
                model, train_loader, optimizer, common_config.DEVICE
            )
            val_loss, val_f1, val_iou, val_auc = evaluate(
                model, val_loader, common_config.DEVICE
            )

            print(
                f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                f'Val IoU: {val_iou:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}'
            )

            scheduler.step(val_f1)

            current_lr = optimizer.param_groups[0]['lr']
            log_data = [
                epoch,
                f'{current_lr:.8f}',
                f'{val_f1:.6f}',
                f'{val_auc:.6f}',
                f'{val_iou:.6f}',
                f'{val_loss:.6f}',
                f'{train_loss:.6f}',
            ]
            writer.writerow(log_data)

            improvement_delta = val_f1 - best_f1_score

            if improvement_delta > args.early_stopping_delta:
                print(
                    f'Validation F1 improved from {best_f1_score:.4f} to {val_f1:.4f}'
                )
                best_f1_score = val_f1
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
                print(f'✅ New best model saved to {model_save_path}')
            else:
                epochs_no_improve += 1
                print(
                    f'No significant improvement in F1 score for {epochs_no_improve} epoch(s).'
                )

            if epochs_no_improve >= args.early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch} epochs.')
                print(f'Best validation F1 achieved: {best_f1_score:.4f}')
                break

    print('\n--- Training Finished ---')


if __name__ == '__main__':
    main()
