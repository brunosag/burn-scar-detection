import argparse
import functools
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from tqdm import tqdm

from burn_scar_detection import config as common_config
from burn_scar_detection import losses as loss_module
from burn_scar_detection.data_loading import BurnScarDataset, JointTransform
from burn_scar_detection.models import get_model

LOSS_FUNCTION_REGISTRY = {
    'bce_lovasz': loss_module.bce_lovasz_loss,
    'bce_dice': loss_module.bce_dice_loss,
}


def compute_pos_weight(mask_dir, file_ids):
    """Calculates pos_weight to counter class imbalance in training data."""
    print(f'Calculating pos_weight from {len(file_ids)} training masks...')
    pos_pixels = 0
    neg_pixels = 0
    for id_ in tqdm(file_ids, desc='Analyzing mask imbalance'):
        mask_path = os.path.join(mask_dir, f'{id_}.npy')
        if not os.path.exists(mask_path):
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

    weight = float(neg_pixels / max(1, pos_pixels))
    print(f'Negative/Positive pixel ratio (pos_weight): {weight:.2f}')
    return torch.tensor([weight], device=common_config.DEVICE)


def collate_for_lr_finder(batch):
    """
    Reformats the batch from (t1, t2, mask) per sample to ([stacked_t1, stacked_t2], stacked_mask) for the whole batch.
    This allows LRFinder to correctly unpack inputs = [stacked_t1, stacked_t2] and call model(*inputs).
    """
    t1_list = [item[0] for item in batch]
    t2_list = [item[1] for item in batch]
    mask_list = [item[2] for item in batch]

    stacked_t1 = torch.stack(t1_list)
    stacked_t2 = torch.stack(t2_list)
    stacked_mask = torch.stack(mask_list)

    inputs = [stacked_t1, stacked_t2]
    targets = stacked_mask

    return inputs, targets


class SiameseModelWrapper(torch.nn.Module):
    """Wraps the Siamese model to handle LRFinder's input format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs[0], inputs[1])


def parse_args():
    parser = argparse.ArgumentParser(description='Learning Rate Range Test')
    parser.add_argument(
        '--model', type=str, required=True, choices=['smp_siamese', 'custom_unet']
    )
    parser.add_argument(
        '--stage_to_test',
        type=int,
        required=True,
        choices=[1, 2],
        help='Stage 1 tests from scratch. Stage 2 requires loading a pretrained model.',
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (required for testing Stage 2).',
    )
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--end_lr', type=float, default=1.0, help='Maximum LR to test.')
    parser.add_argument(
        '--num_iter', type=int, default=100, help='Number of batches to iterate over.'
    )
    return parser.parse_args()


def run_lr_test(args, model, train_loader, criterion, optimizer):
    lr_finder = LRFinder(model, optimizer, criterion, device=common_config.DEVICE)
    print(f'Running LR range test for {args.num_iter} iterations...')
    lr_finder.range_test(
        train_loader, end_lr=args.end_lr, num_iter=args.num_iter, step_mode='exp'
    )

    output_filename = f'lr_finder_stage_{args.stage_to_test}_{args.model}.png'
    print(f'Saving plot to {output_filename}')
    ax, suggested_lr = lr_finder.plot()
    fig = ax.get_figure()
    fig.savefig(output_filename)

    print(f'LR range test complete. Suggested max LR: {suggested_lr:.2E}')
    print(
        "Inspect the plot 'lr_finder_stage_{args.stage_to_test}_{args.model}.png' to select the best LR."
    )
    print('Rule of thumb: pick the LR where the loss decline is steepest.')

    lr_finder.reset()


def main():
    args = parse_args()

    split_file_path = os.path.join(common_config.PROCESSED_DATA_DIR, 'splits.json')
    with open(split_file_path, 'r') as f:
        splits = json.load(f)
    train_ids = splits['train']

    train_dataset = BurnScarDataset(
        t1_feature_dir=common_config.PROCESSED_FEATURES_T1_DIR,
        t2_feature_dir=common_config.PROCESSED_FEATURES_T2_DIR,
        mask_dir=common_config.PROCESSED_MASK_DIR,
        file_ids=train_ids,
        augmentations=JointTransform(p_flip=0.5, p_photometric=0.2),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_for_lr_finder,
    )

    pos_weight = compute_pos_weight(common_config.PROCESSED_MASK_DIR, train_ids)

    model = get_model(args.model, common_config.N_CHANNELS, common_config.N_CLASSES)

    if args.stage_to_test == 1:
        print('Testing Stage 1 configuration (from scratch model).')
        criterion_base = LOSS_FUNCTION_REGISTRY['bce_dice']
        criterion = functools.partial(
            criterion_base, bce_weight=0.5, pos_weight=pos_weight
        )
    else:
        print('Testing Stage 2 configuration.')
        if not args.model_checkpoint:
            raise ValueError('--model_checkpoint must be provided for Stage 2 test.')
        print(f'Loading weights from {args.model_checkpoint} for fine-tuning test.')
        model.load_state_dict(
            torch.load(args.model_checkpoint, map_location=common_config.DEVICE)
        )
        criterion_base = LOSS_FUNCTION_REGISTRY['bce_lovasz']
        criterion = functools.partial(
            criterion_base, bce_weight=0.3, pos_weight=pos_weight
        )

    wrapped_model = SiameseModelWrapper(model).to(common_config.DEVICE)
    optimizer = optim.AdamW(wrapped_model.parameters(), lr=1e-8, weight_decay=1e-2)

    run_lr_test(args, wrapped_model, train_loader, criterion, optimizer)


if __name__ == '__main__':
    main()
