"""Render-and-compare losses: part masks and gripper tip constraints."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.pose_geometry import PART_CLASSES, PARTS, part_transforms, pose_errors, rotation_geodesic_deg


ARM_NAMES = ("left", "right")
CHANNEL_NAMES = tuple(f"{arm}_{name}" for arm in ARM_NAMES for name in ("shaft", "wrist", "grippers"))
DEFAULT_WEIGHTS = {"mask": 1., "tips": 1., "tips_gap": .5}


def soft_mask_loss(pred, target):
    """Per-sample multi-scale soft Dice + part-balanced BCE. Returns [B]."""
    losses = []
    for scale in (1, 2, 4):
        p = pred if scale == 1 else F.avg_pool2d(pred, scale)
        t = target if scale == 1 else F.avg_pool2d(target, scale)
        dims = (-2, -1)
        dice = 1 - (2 * (p * t).sum(dims) + 1) / (p.sum(dims) + t.sum(dims) + 1)
        bce = F.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), t, reduction="none")
        foreground = (bce * t).sum(dims) / t.sum(dims).clamp_min(1)
        background = (bce * (1 - t)).sum(dims) / (1 - t).sum(dims).clamp_min(1)
        losses.append(dice.mean(-1) + .25 * (foreground + background).mean(-1))
    return torch.stack(losses).mean(0)


def _tip_inputs(pred, target, confidence):
    pred, target, confidence = pred.reshape(-1, 2, 2), target.reshape(-1, 2, 2), confidence.reshape(-1, 2)
    confidence = torch.nan_to_num(confidence.detach(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
    confidence = confidence * torch.isfinite(target).all(-1)
    return pred.float(), torch.nan_to_num(target.detach().float(), nan=0., posinf=0., neginf=0.), confidence.float()


def endpoint_loss(pred, target, confidence, scale):
    """Permutation-invariant Smooth L1 on jaw tips. Returns [B]."""
    b, arms = pred.shape[0], pred.shape[1]
    pred, target, confidence = _tip_inputs(pred, target, confidence)
    scale = scale.detach().reshape(-1, 1, 1).clamp_min(10.)

    def cost(order):
        error = F.smooth_l1_loss(pred[:, order] / scale, target / scale, beta=.05, reduction="none").sum(-1)
        return (error * confidence).sum(-1) / confidence.sum(-1).clamp_min(1e-8)

    sample_weight = confidence.amax(-1)
    costs = torch.minimum(cost([0, 1]), cost([1, 0]))
    weighted = costs * sample_weight
    return weighted.reshape(b, arms).sum(-1) / sample_weight.reshape(b, arms).sum(-1).clamp_min(1e-8)


def tip_gap_loss(pred, target, confidence, scale):
    """Projected jaw-opening error, locally normalized. Returns [B]."""
    b, arms = pred.shape[0], pred.shape[1]
    pred, target, confidence = _tip_inputs(pred, target, confidence)
    predicted_gap = torch.linalg.vector_norm(pred[:, 0] - pred[:, 1], dim=-1)
    target_gap = torch.linalg.vector_norm(target[:, 0] - target[:, 1], dim=-1)
    error = (predicted_gap - target_gap) / scale.detach().reshape(-1).clamp_min(10.)
    weight = confidence.amin(-1)
    cost = F.smooth_l1_loss(error, torch.zeros_like(error), beta=.05, reduction="none")
    weighted = cost * weight
    return weighted.reshape(b, arms).sum(-1) / weight.reshape(b, arms).sum(-1).clamp_min(1e-8)


def total_loss(result, batch, weights=None):
    """Scalar per candidate [B] plus named terms (mask + tip constraints)."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    scale = batch["tip_scale"]
    terms = {
        "mask": soft_mask_loss(result["mask"], batch["mask"]),
        "tips": endpoint_loss(result["tips"], batch["tips"], batch["tip_confidence"], scale),
        "tips_gap": tip_gap_loss(result["tips"], batch["tips"], batch["tip_confidence"], scale),
    }
    total = sum(weights[name] * terms[name] for name in DEFAULT_WEIGHTS)
    return total, terms


@torch.no_grad()
def part_dice(pred, target):
    """Per-sample, per-part hard Dice; absent in both images is undefined. [B, C]."""
    p, t = pred > .5, target > .5
    denominator = p.sum((-2, -1)) + t.sum((-2, -1))
    return torch.where(denominator > 0, 2 * (p & t).sum((-2, -1)) / denominator.clamp_min(1), torch.nan)


def dice_dict(pred, target):
    values = part_dice(pred, target)
    if values.ndim == 1:
        values = values[None]
    row = values[0].tolist()
    named = {name: (None if not np_isfinite(v) else float(v)) for name, v in zip(CHANNEL_NAMES, row)}
    finite = [v for v in named.values() if v is not None]
    named["mean"] = float(sum(finite) / len(finite)) if finite else None
    for arm, prefix in enumerate(ARM_NAMES):
        keys = [f"{prefix}_{part}" for part in ("shaft", "wrist", "grippers")]
        vals = [named[k] for k in keys if named[k] is not None]
        named[f"{prefix}_mean"] = float(sum(vals) / len(vals)) if vals else None
        named[f"{prefix}_pixels"] = int((pred[0, 3 * arm:3 * arm + PART_CLASSES] > .5).any(0).sum())
    return named


def np_isfinite(value):
    return value == value and abs(value) != float("inf")


@torch.no_grad()
def part_pose_errors(pred, gt, pivot, shaft_offset):
    """Wrist errors plus per-rigid-body translation/rotation errors. Shapes [A]."""
    wrist = pose_errors(pred, gt)
    pred_t, gt_t = part_transforms(pred, pivot, shaft_offset), part_transforms(gt, pivot, shaft_offset)
    parts = {}
    for name in PARTS:
        dt_mm = (pred_t[name][..., :3, 3] - gt_t[name][..., :3, 3]) * 1000
        parts[name] = {
            "translation_error_mm": dt_mm.norm(dim=-1)[0],
            "rotation_error_deg": rotation_geodesic_deg(pred_t[name][..., :3, :3], gt_t[name][..., :3, :3])[0],
        }
    return wrist, parts


def pose_error_dict(pred, gt, pivot, shaft_offset):
    wrist, parts = part_pose_errors(pred, gt, pivot, shaft_offset)
    out = {}
    for arm, name in enumerate(ARM_NAMES):
        out[f"{name}_translation_error_mm"] = float(wrist["translation_error_mm"][0, arm])
        out[f"{name}_rotation_error_deg"] = float(wrist["rotation_error_deg"][0, arm])
        out[f"{name}_joints_mae_deg"] = float(wrist["joints_mae_deg"][0, arm])
        out[f"{name}_joints_error_deg"] = [float(v) for v in wrist["joints_error_deg"][0, arm]]
        for part, values in parts.items():
            out[f"{name}_{part}_translation_error_mm"] = float(values["translation_error_mm"][arm])
            out[f"{name}_{part}_rotation_error_deg"] = float(values["rotation_error_deg"][arm])
    return out
