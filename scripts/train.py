import os

import torch
import torch.optim as optim
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
        mask_dir=config.MASK_DIR,
        augmentations=augmentations,
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset_aug = random_split(
        full_dataset, [train_size, val_size], generator
    )

    val_dataset = torch.utils.data.Subset(
        BurnScarDataset(
            config.T1_DIR, config.T2_DIR, config.MASK_DIR, augmentations=None
        ),
        val_dataset_aug.indices,
    )

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    print(f'Total samples: {len(full_dataset)}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    # 3. Initialize Model, Optimizer
    model = SiameseAttentionUNet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        classes=config.CLASSES,
    ).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    best_val_iou = -1.0
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    for epoch in range(1, config.EPOCHS + 1):
        print(f'\n--- Epoch {epoch}/{config.EPOCHS} ---')
        train_loss = train_one_epoch(model, train_loader, optimizer, config.DEVICE)
        val_loss, val_f1, val_iou, val_auc = evaluate(model, val_loader, config.DEVICE)

        print(f'Train Loss: {train_loss:.4f}')
        print(
            f'Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val IoU: {val_iou:.4f} | Val AUC: {val_auc:.4f}'
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f'✅ New best model saved with IoU: {best_val_iou:.4f}')


if __name__ == '__main__':
    main()
