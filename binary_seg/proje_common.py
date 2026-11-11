"""Shared helpers for the multi-seed PraNet-V1/V2 study.

Everything here is written to reproduce the *official* behaviour of the
PraNet-V2 repository (MyTrain_med.py, MyTest_med.py, eval.py) exactly, while
making the pieces reusable from our own training / evaluation scripts.
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

MODEL_TYPES = ['PraNet-V1', 'PVT-PraNet-V1', 'PraNet-V2', 'PVT-PraNet-V2']

# Default test sets. CVC-ColonDB is missing from the official eval.py list
# (eval.py:220, MyTest_med.py:82) but is part of the standard PraNet TestDataset
# and is the main "unseen" set in our study (S3).
TEST_SETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def get_device():
    """CUDA when available, otherwise CPU (used by the CPU test notebook)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def device_name(device):
    return torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed, deterministic=False):
    """Seed python, numpy and torch (CPU + CUDA).

    Bit-exact repetition is NOT expected even with deterministic=True: CUDA's
    bilinear-upsampling backward pass is non-deterministic. The goal of the study
    is to measure seed variance, not to reproduce runs bit by bit.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader worker_init_fn (the training pipeline has no random
    augmentation, but this keeps workers reproducible if one is added)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_model(model_type):
    """Build one of the four models exactly as the official scripts do.

    Must be called from the binary_seg directory (imports use the `lib` package).
    """
    if model_type == 'PraNet-V1':
        from lib.PraNet_Res2Net import PraNet
        return PraNet()
    if model_type == 'PVT-PraNet-V1':
        from lib.PraNet_Res2Net import PVT_PraNet
        return PVT_PraNet()
    if model_type == 'PraNet-V2':
        from lib.pranet import PraNet_V2
        return PraNet_V2(num_class=1)
    if model_type == 'PVT-PraNet-V2':
        from lib.pranet import PVT_PraNet_V2
        return PVT_PraNet_V2(num_class=1)
    raise ValueError('Unknown model_type {!r}; choose from {}'.format(model_type, MODEL_TYPES))


def is_v2(model_type):
    return model_type in ('PraNet-V2', 'PVT-PraNet-V2')


def split_outputs(model_type, outputs):
    """Return (fg_maps, bg_maps) with fg_maps ordered [map2, map3, map4, map5].

    map2 is the final (finest) prediction, map5 the coarse decoder output.
    V1 models return (map5, map4, map3, map2) and have no background maps.
    V2 models return (fg2, fg3, fg4, fg5, bg2, bg3, bg4, bg5).
    """
    if is_v2(model_type):
        fg = list(outputs[:4])
        bg = list(outputs[4:])
    else:
        m5, m4, m3, m2 = outputs
        fg = [m2, m3, m4, m5]
        bg = None
    return fg, bg


def load_weights(model, path, allowed_missing_prefixes=('conv.',)):
    """Load a state_dict, reporting key mismatches instead of hiding them.

    The official MyTest_med.py loads V2 weights with strict=False, which would
    silently leave mismatched layers randomly initialised. Here we load
    non-strictly but raise if any *used* key is missing. The only tolerated
    missing keys are those of the unused 1->3 channel `conv` adapter of the
    V2 models (only applied to single-channel inputs).
    """
    # weights_only=False: our resumable checkpoints also contain RNG states
    # (numpy arrays), which the safe loader of torch>=2.6 rejects.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state and 'optimizer' in state:
        state = state['model']  # our own resumable checkpoint format
    if state and all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}  # saved from DataParallel
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(tuple(allowed_missing_prefixes))]
    report = {
        'missing_keys': list(result.missing_keys),
        'unexpected_keys': list(result.unexpected_keys),
    }
    if missing:
        raise RuntimeError('Checkpoint {} is missing {} used keys, e.g. {}'.format(
            path, len(missing), missing[:5]))
    return report


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #
def _fg_terms(pred, mask):
    """Weighted BCE + weighted IoU of the foreground map, plus the boundary
    weight map. Identical to the foreground part of MyTrain_med.structure_loss
    and to the original PraNet structure_loss."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    p = torch.sigmoid(pred)
    inter = ((p * mask) * weit).sum(dim=(2, 3))
    union = ((p + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wbce, wiou, weit


def loss_v1(pred, mask):
    """Original PraNet (V1) structure loss: wBCE + wIoU."""
    wbce, wiou, _ = _fg_terms(pred, mask)
    return (wbce + wiou).mean()


def loss_v2(pred, pred_bg, mask, bg_weight=0.8):
    """PraNet-V2 structure loss (MyTrain_med.py:19-38).

    bg_weight=0.8 is the official value. bg_weight=0 is our ablation: the
    background term is not computed at all, so the background heads receive no
    gradient.
    """
    wbce, wiou, weit = _fg_terms(pred, mask)
    loss = wbce + wiou
    if bg_weight != 0:
        mask_bg = 1 - mask
        wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
        wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        loss = loss + bg_weight * wbce2
    return loss.mean()


def total_loss(model_type, outputs, gts, bg_weight=0.8):
    """Sum of the four deeply-supervised side-output losses."""
    fg, bg = split_outputs(model_type, outputs)
    if is_v2(model_type):
        return sum(loss_v2(f, b, gts, bg_weight) for f, b in zip(fg, bg))
    return sum(loss_v1(f, gts) for f in fg)


# --------------------------------------------------------------------------- #
# Inference post-processing
# --------------------------------------------------------------------------- #
def predict_uint8(logits, gt_shape):
    """Official post-processing (MyTest_med.py:100-111): resize logits to the GT
    size, sigmoid, per-image min-max normalisation, quantise to uint8 (the
    official pipeline writes PNGs and eval.py re-reads them)."""
    out = F.interpolate(logits, size=gt_shape, mode='bilinear', align_corners=False)
    out = out.sigmoid().data.cpu().numpy().squeeze()
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return (out * 255).astype(np.uint8)


def protocol_logits(fg_maps):
    """The two inference protocols found in MyTest_med.py:
    - 'last': only the final map (map2)  -> official protocol for V1
    - 'sum' : map2 + map3 + map4 + map5  -> official protocol for V2
    """
    return {'last': fg_maps[0], 'sum': fg_maps[0] + fg_maps[1] + fg_maps[2] + fg_maps[3]}


def official_protocol(model_type):
    return 'sum' if is_v2(model_type) else 'last'


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
_THRESHOLDS = np.linspace(1, 0, 256)
_LEVELS = np.arange(256).astype(np.float64) / 255
# _ABOVE[j, k] is True when grey level k is >= threshold j. The comparison uses
# exactly the same float64 values as eval.py (uint8 -> float64 / 255 versus
# np.linspace(1, 0, 256)), so the counts below are identical to the per-pixel
# loop in eval.py, just computed from histograms.
_ABOVE = (_LEVELS[None, :] >= _THRESHOLDS[:, None]).astype(np.float64)


def dice_iou_curves(pred_uint8, gt_bin):
    """Dice and IoU at the 256 thresholds of eval.py (Fmeasure_calu)."""
    pred_uint8 = pred_uint8.ravel()
    gt_bin = gt_bin.ravel().astype(bool)
    hist_all = np.bincount(pred_uint8, minlength=256).astype(np.float64)
    hist_fg = np.bincount(pred_uint8[gt_bin], minlength=256).astype(np.float64)
    num_rec = _ABOVE @ hist_all     # predicted positives per threshold
    num_and = _ABOVE @ hist_fg      # true positives per threshold
    num_obj = float(gt_bin.sum())
    fn = num_obj - num_and
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = np.where(num_and == 0, 0.0, 2 * num_and / (num_obj + num_rec))
        iou = np.where(num_and == 0, 0.0, num_and / (fn + num_rec))
    return dice, iou


def image_metrics(pred_uint8, gt_raw, full=False):
    """Per-image metrics with eval.py's definitions.

    gt_raw: the mask as read from disk (uint8). eval.py binarises the *raw*
    values with > 0.5, i.e. any non-zero pixel counts as polyp.
    meanDic / meanIoU are the per-image means over 256 thresholds; their mean
    over images equals the dataset-level numbers printed by eval.py.
    """
    gt_bin = (gt_raw.astype(np.float64) > 0.5).astype(np.float64)
    dice, iou = dice_iou_curves(pred_uint8, gt_bin)
    out = {'meanDic': float(dice.mean()), 'meanIoU': float(iou.mean())}
    if full:
        from utils.eval_functions import StructureMeasure, original_WFb, EnhancedMeasure
        pred = pred_uint8.astype(np.float64) / 255
        out['Sm'] = float(StructureMeasure(pred, gt_bin))
        out['wFm'] = float(original_WFb(pred, gt_bin))
        out['mae'] = float(np.mean(np.abs(gt_bin - pred)))
        em = np.zeros(len(_THRESHOLDS))
        for j, t in enumerate(_THRESHOLDS):
            em[j] = EnhancedMeasure((pred >= t).astype(np.float64), gt_bin)
        out['meanEm'] = float(em.mean())
    return out
