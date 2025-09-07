import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from burn_scar_detection import config as common_config
from burn_scar_detection import losses
from burn_scar_detection.data_loading import BurnScarDataset, JointTransform
from burn_scar_detection.engine import evaluate, train_one_epoch
from burn_scar_detection.models import get_model


def compute_pos_weight(mask_dir, file_ids):
    """Calculates pos_weight to counter class imbalance in training data."""
    print(f'Calculating pos_weight from {len(file_ids)} training masks...')
    pos_pixels = 0
    neg_pixels = 0
    for id_ in tqdm(file_ids, desc='Analyzing mask imbalance'):
        mask_path = os.path.join(mask_dir, f'{id_}.npy')
        if not os.path.exists(mask_path):
            print(f'Warning: Mask file not found {mask_path}, skipping.')
            continue

        try:
            m = np.load(mask_path)
            pos_pixels += (m > 0.5).sum()
            neg_pixels += (m <= 0.5).sum()
        except Exception as e:
            print(f'Error loading mask {id_}: {e}')

    if pos_pixels == 0:
        print(
            'Warning: No positive pixels found in training set for weight calculation.'
        )
        return torch.tensor(1.0, device=common_config.DEVICE)

    weight = float(neg_pixels / pos_pixels)
    print(f'Negative/Positive pixel ratio (pos_weight): {weight:.2f}')
    return torch.tensor([weight], device=common_config.DEVICE)


LOSS_FUNCTION_REGISTRY = {
    'bce_lovasz': losses.bce_lovasz_loss,
    'bce_dice': losses.bce_dice_loss,
}


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
        '--loss_type',
        type=str,
        default='bce_lovasz',
        choices=LOSS_FUNCTION_REGISTRY.keys(),
        help='Loss function to use. Recommended: bce_lovasz.',
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='val_f1',
        choices=['val_loss', 'val_f1', 'val_auc', 'val_iou'],
        help='Metric to monitor for early stopping. Recommended: val_f1 or val_iou.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs',
    )
    parser.add_argument(
        '--batch_size', type=int, default=24, help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate (max_lr for OneCycleLR)',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-2, help='AdamW weight decay parameter.'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=20,
        help='Patience for early stopping (epochs without improvement)',
    )
    parser.add_argument(
        '--early_stopping_delta',
        type=float,
        default=0.001,
        help='Minimum improvement in monitored metric to reset patience',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print('--- Starting Training Run ---')
    print(f'Selected Model Architecture: {args.model}')
    print(f'Loss Function: {args.loss_type}')
    print(f'Monitoring Metric for Early Stopping: {args.metric}')
    print(f'Using device: {common_config.DEVICE}')

    if 'loss' in args.metric:
        scheduler_mode = 'min'
        best_val_metric = np.inf
    else:
        scheduler_mode = 'max'
        best_val_metric = -np.inf

    split_file_path = os.path.join(common_config.PROCESSED_DATA_DIR, 'splits.json')
    with open(split_file_path, 'r') as f:
        splits = json.load(f)

    train_ids = splits['train']
    val_ids = splits['validation']

    pos_weight = compute_pos_weight(common_config.PROCESSED_MASK_DIR, train_ids)

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

    criterion = LOSS_FUNCTION_REGISTRY[args.loss_type]

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4,
    )

    epochs_no_improve = 0
    run_name = f'{args.model}_{args.loss_type}'
    model_save_path = os.path.join(
        common_config.MODEL_CHECKPOINT_DIR, f'best_model_{run_name}.pth'
    )
    log_file_path = os.path.join(
        common_config.MODEL_CHECKPOINT_DIR, f'training_log_{run_name}.csv'
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print(f'Logging training progress to: {log_file_path}')

    for epoch in range(1, args.epochs + 1):
        print(f'\n--- Epoch {epoch}/{args.epochs} ---')

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            common_config.DEVICE,
            scheduler,
            criterion,
            pos_weight,
        )
        val_loss, val_f1, val_iou, val_auc = evaluate(
            model, val_loader, common_config.DEVICE, criterion, pos_weight
        )

        print(
            f'AUC-ROC: {val_auc:.4f} | F1: {val_f1:.4f} | IoU: {val_iou:.4f} | Val. Loss: {val_loss:.4f} | Train Loss: {train_loss:.4f}'
        )

        metrics_epoch = {
            'val_auc': val_auc,
            'val_f1': val_f1,
            'val_iou': val_iou,
            'val_loss': val_loss,
        }
        current_metric_value = metrics_epoch[args.metric]
        current_lr = optimizer.param_groups[0]['lr']

        log_metrics = {
            'epoch': epoch,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'val_iou': val_iou,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'learning_rate': current_lr,
        }
        log_df = pd.DataFrame([log_metrics])
        log_df.to_csv(
            log_file_path,
            mode='w' if epoch == 1 else 'a',
            header=epoch == 1,
            index=False,
            float_format='%.6f',
        )

        improvement = False
        if scheduler_mode == 'max':
            improvement_margin = current_metric_value - best_val_metric
            if improvement_margin > args.early_stopping_delta:
                improvement = True
        else:
            improvement_margin = best_val_metric - current_metric_value
            if improvement_margin > args.early_stopping_delta:
                improvement = True

        if improvement:
            print(
                f'Validation metric ({args.metric}) improved from {best_val_metric:.4f} to {current_metric_value:.4f}'
            )
            best_val_metric = current_metric_value
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved to '{model_save_path}'")
        else:
            epochs_no_improve += 1
            print(
                f'No significant improvement in {args.metric} for {epochs_no_improve} epoch(s).'
            )

        if epochs_no_improve >= args.early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch} epochs.')
            print(f'Best validation {args.metric} achieved: {best_val_metric:.4f}')
            break

    print('\n--- Training Finished ---')


if __name__ == '__main__':
    main()
