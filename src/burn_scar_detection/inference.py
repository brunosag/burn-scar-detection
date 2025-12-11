import torch
import torchvision.transforms.functional as TF

from . import config as common_config
from .models import get_model


def load_model_for_inference(model_id: str, model_path: str):
    """Initializes a model and loads its weights from a checkpoint file."""
    print(f'Loading model: {model_path}')

    model = get_model(
        model_id, n_channels=common_config.N_CHANNELS, n_classes=common_config.N_CLASSES
    )
    model.load_state_dict(torch.load(model_path, map_location=common_config.DEVICE))
    model.to(common_config.DEVICE)
    model.eval()

    return model


def predict_batch_with_tta(model, t1_batch, t2_batch):
    """Applies TTA (identity, hflip, vflip, hvflip) to a batch of inputs."""
    pred_identity = torch.sigmoid(model(t1_batch, t2_batch))

    pred_hflip_aug = torch.sigmoid(model(TF.hflip(t1_batch), TF.hflip(t2_batch)))
    pred_hflip = TF.hflip(pred_hflip_aug)

    pred_vflip_aug = torch.sigmoid(model(TF.vflip(t1_batch), TF.vflip(t2_batch)))
    pred_vflip = TF.vflip(pred_vflip_aug)

    pred_hvflip_aug = torch.sigmoid(
        model(TF.hflip(TF.vflip(t1_batch)), TF.hflip(TF.vflip(t2_batch)))
    )
    pred_hvflip = TF.vflip(TF.hflip(pred_hvflip_aug))

    avg_prob_batch = torch.mean(
        torch.stack([pred_identity, pred_hflip, pred_vflip, pred_hvflip]), dim=0
    )
    return avg_prob_batch
