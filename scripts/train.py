import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from burn_scar_detection import config
from burn_scar_detection.data_loading import BurnScarDataset, JointTransform
from burn_scar_detection.engine import evaluate, train_one_epoch
from burn_scar_detection.model import SiameseAttentionUNet


def main():
    print(f'--- Using device: {config.DEVICE} ---')

    # 1. Setup Augmentations and Datasets
    augmentations = JointTransform(p_flip=0.5, p_photometric=0.2)

    full_dataset = BurnScarDataset(
        t1_dir=config.T1_DIR,
        t2_dir=config.T2_DIR,
        glcm_t1_dir=config.GLCM_T1_DIR,
        glcm_t2_dir=config.GLCM_T2_DIR,
        mask_dir=config.MASK_DIR,
        augmentations=augmentations,
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset_aug = random_split(
        full_dataset, [train_size, val_size], generator
    )

    val_dataset_no_aug_instance = BurnScarDataset(
        t1_dir=config.T1_DIR,
        t2_dir=config.T2_DIR,
        glcm_t1_dir=config.GLCM_T1_DIR,
        glcm_t2_dir=config.GLCM_T2_DIR,
        mask_dir=config.MASK_DIR,
        augmentations=None,
    )
    val_dataset = torch.utils.data.Subset(
        val_dataset_no_aug_instance, val_dataset_aug.indices
    )

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    print(f'Total samples: {len(full_dataset)}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    # 3. Initialize Model, Optimizer, and Scheduler
    model = SiameseAttentionUNet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        classes=config.CLASSES,
    ).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 4. Training Loop Setup for Early Stopping
    best_val_iou = -1.0
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    for epoch in range(1, config.EPOCHS + 1):
        print(f'\n--- Epoch {epoch}/{config.EPOCHS} ---')
        train_loss = train_one_epoch(model, train_loader, optimizer, config.DEVICE)
        val_loss, val_f1, val_iou, val_auc = evaluate(model, val_loader, config.DEVICE)

        print(f'Train Loss: {train_loss:.4f}')
        print(
            f'Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val IoU: {val_iou:.4f} | Val AUC: {val_auc:.4f}'
        )

        scheduler.step(val_loss)

        improvement_delta = val_iou - best_val_iou

        if improvement_delta > config.EARLY_STOPPING_MIN_DELTA:
            print(f'Validation IoU improved from {best_val_iou:.4f} to {val_iou:.4f}')
            best_val_iou = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f'✅ New best model saved with IoU: {best_val_iou:.4f}')
        else:
            epochs_no_improve += 1
            print(f'No significant improvement for {epochs_no_improve} epoch(s).')

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f'\nEarly stopping triggered after {epoch} epochs.')
            print(f'Best validation IoU achieved: {best_val_iou:.4f}')
            break


if __name__ == '__main__':
    main()
