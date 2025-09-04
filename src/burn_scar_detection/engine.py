import numpy as np
import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from tqdm import tqdm

dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
focal_loss = smp.losses.FocalLoss(mode='binary')


def composite_loss(y_pred_logits, y_true):
    return 0.5 * dice_loss(y_pred_logits, y_true) + 0.5 * focal_loss(
        y_pred_logits, y_true
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loop = tqdm(loader, desc='Training')
    total_loss = 0
    for t1, t2, mask in loop:
        t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
        optimizer.zero_grad()
        predictions_logits = model(t1, t2)
        loss = composite_loss(predictions_logits, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_probs_flat, all_masks_flat = [], []
    all_preds_bin, all_masks_bin = [], []
    with torch.no_grad():
        loop = tqdm(loader, desc='Evaluating')
        for t1, t2, mask in loop:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
            predictions_logits = model(t1, t2)
            loss = composite_loss(predictions_logits, mask)
            total_loss += loss.item()

            predictions_probs = torch.sigmoid(predictions_logits)
            all_probs_flat.append(predictions_probs.cpu().numpy().ravel())
            all_masks_flat.append(mask.cpu().numpy().ravel())

            all_preds_bin.append(predictions_probs > 0.5)
            all_masks_bin.append(mask)

    all_preds_bin = torch.cat(all_preds_bin)
    all_masks_bin = torch.cat(all_masks_bin).int()
    f1 = binary_f1_score(all_preds_bin, all_masks_bin)
    iou = binary_jaccard_index(all_preds_bin, all_masks_bin)

    all_probs_flat = np.concatenate(all_probs_flat)
    all_masks_flat = np.concatenate(all_masks_flat)
    auc = roc_auc_score(all_masks_flat, all_probs_flat)

    return total_loss / len(loader), f1.item(), iou.item(), auc
