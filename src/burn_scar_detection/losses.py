import segmentation_models_pytorch as smp
import torch.nn.functional as F

lovasz_loss = smp.losses.LovaszLoss(mode='binary', from_logits=True)
dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)


def calculate_bce_loss(logits, y_true, pos_weight=None):
    """Calculates Binary Cross Entropy loss, optionally applying pos_weight."""
    return F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pos_weight)


def bce_lovasz_loss(logits, y_true, pos_weight=None, bce_weight=0.5):
    """
    Combines BCE loss (for good probability calibration) and Lovász loss
    (for high IoU/F1 score). Recommended primary loss function.

    Args:
        logits (torch.Tensor): Model predictions before sigmoid activation.
        y_true (torch.Tensor): Ground truth labels (0 or 1).
        pos_weight (torch.Tensor, optional): Weight for positive class in BCE.
        bce_weight (float): The weight assigned to the BCE portion of the loss.
    """
    bce = calculate_bce_loss(logits, y_true, pos_weight)
    lovasz = lovasz_loss(logits, y_true)
    return bce_weight * bce + (1.0 - bce_weight) * lovasz


def bce_dice_loss(logits, y_true, pos_weight=None, bce_weight=0.5):
    """
    Combines BCE loss and Dice loss. A strong baseline alternative to BCE+Lovász,
    often used as Stage 1 for two-stage training.

    Args:
        logits (torch.Tensor): Model predictions before sigmoid activation.
        y_true (torch.Tensor): Ground truth labels (0 or 1).
        pos_weight (torch.Tensor, optional): Weight for positive class in BCE.
        bce_weight (float): The weight assigned to the BCE portion of the loss.
    """
    bce = calculate_bce_loss(logits, y_true, pos_weight)
    dice = dice_loss(logits, y_true)
    return bce_weight * bce + (1.0 - bce_weight) * dice
