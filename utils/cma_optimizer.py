"""CMA-ES and Adam render-and-compare optimizers for dual-instrument poses."""
from __future__ import annotations

import time

import numpy as np
import torch

from utils.losses import dice_dict, total_loss
from utils.parameterization import (ARM_DIM, N_ARMS, POSE_DIM, cma_bounds, cma_stds,
                                    matrix_to_quaternion, pose_to_vector, quaternion_to_matrix,
                                    vector_to_pose)


def _expand_frame(frame, batch, device):
    rgb = frame.rgb.to(device)[None].expand(batch, -1, -1, -1).contiguous()
    mask = frame.mask.to(device)[None].expand(batch, -1, -1, -1).contiguous()
    tips = frame.tips.to(device)[None].expand(batch, -1, -1, -1).contiguous()
    confidence = frame.tip_confidence.to(device)[None].expand(batch, -1, -1).contiguous()
    scale = frame.tip_scale.to(device)[None].expand(batch, -1).contiguous()
    k = frame.K.to(device)
    return {"rgb": rgb, "mask": mask, "tips": tips, "tip_confidence": confidence, "tip_scale": scale, "K": k}


def _concat(parts, key):
    return torch.cat([row[key] for row in parts], 0)


def _pack_best(packed, vector):
    return {
        "loss": float(packed["loss"][0]),
        "x": np.asarray(vector, np.float64),
        "dice": packed["dice"][0],
        "terms": {name: float(value[0]) for name, value in packed["terms"].items()},
        "result": {key: packed["result"][key][:1] for key in packed["result"]},
        "pose": {key: packed["pose"][key][:1] for key in packed["pose"]},
    }


@torch.no_grad()
def evaluate_population(vectors, renderer, frame, weights, alpha_limit, jaw_limit, chunk=8, return_render=True):
    device = renderer.mesh.vertices.device
    vectors = np.asarray(vectors, np.float64)
    losses, dice, term_rows = [], [], []
    results, poses = [], []
    for start in range(0, len(vectors), max(1, int(chunk))):
        pose = vector_to_pose(vectors[start:start + chunk], device, alpha_limit=alpha_limit, jaw_limit=jaw_limit)
        batch = _expand_frame(frame, pose["R"].shape[0], device)
        result = renderer(pose, batch["K"])
        loss, terms = total_loss(result, batch, weights)
        losses.append(loss.detach())
        term_rows.append({name: value.detach() for name, value in terms.items()})
        dice.extend(dice_dict(result["mask"][i:i + 1], batch["mask"][i:i + 1]) for i in range(pose["R"].shape[0]))
        if return_render:
            results.append(result)
            poses.append(pose)
    packed = {
        "loss": torch.cat(losses).float().cpu().numpy(),
        "terms": {name: torch.cat([row[name] for row in term_rows]).float().cpu().numpy() for name in term_rows[0]},
        "dice": dice,
    }
    if return_render:
        packed["result"] = {key: _concat(results, key) for key in results[0]}
        packed["pose"] = {key: _concat(poses, key) for key in poses[0]}
    return packed


def optimize_frame_cmaes(renderer, frame, x0, weights, translation_min, translation_max,
                         alpha_limit, jaw_limit, sigma=1.0, popsize=12, maxiter=60,
                         seed=2026, rotation_std=0.25, translation_xy_std=0.003, translation_z_std=0.005,
                         joint_std=0.15, translation_xy_m=0.02, translation_z_m=0.03,
                         chunk=8, early_stop_tol=0.001, early_stop_loss=0.1):
    start = time.perf_counter()
    x0 = np.asarray(x0, np.float64)
    if maxiter <= 0:
        packed = evaluate_population([x0], renderer, frame, weights, alpha_limit, jaw_limit, chunk)
        best = _pack_best(packed, x0)
        history = [{"generation": 0, "loss": best["loss"], "x": best["x"].tolist(),
                    "dice": best["dice"], "terms": best["terms"]}]
        return best, history, time.perf_counter() - start, {
            "optimizer": "cmaes", "evaluations": 1, "generations": 0, "stop": {"maxiter": 0}}

    try:
        import cma
    except ImportError as exc:
        raise RuntimeError("CMA-ES requires the `cma` package: pip install cma") from exc

    low, high = cma_bounds(
        x0, alpha_limit, jaw_limit,
        translation_xy_m=translation_xy_m, translation_z_m=translation_z_m,
        translation_min=translation_min, translation_max=translation_max)
    stds = cma_stds(rotation_std, translation_xy_std, translation_z_std, joint_std)
    options = {
        "bounds": [low.tolist(), high.tolist()],
        "popsize": int(popsize),
        "maxiter": int(maxiter),
        "seed": int(seed),
        "CMA_stds": stds.tolist(),
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    }

    es = cma.CMAEvolutionStrategy(x0, float(sigma), options)
    history = []
    best = {"loss": float("inf"), "x": x0.copy(), "dice": None, "terms": None}
    prev_loss = None
    early_stopped = False
    generation = 0
    while not es.stop():
        xs = es.ask()
        packed = evaluate_population(xs, renderer, frame, weights, alpha_limit, jaw_limit, chunk, return_render=False)
        losses = packed["loss"].tolist()
        es.tell(xs, losses)
        generation += 1
        index = int(np.argmin(packed["loss"]))
        row = {
            "generation": generation,
            "loss": float(packed["loss"][index]),
            "x": np.asarray(xs[index], np.float64).tolist(),
            "dice": packed["dice"][index],
            "terms": {name: float(value[index]) for name, value in packed["terms"].items()},
            "population_best": float(min(losses)),
            "population_mean": float(np.mean(losses)),
        }
        history.append(row)
        if row["loss"] < best["loss"]:
            best = {"loss": row["loss"], "x": np.asarray(xs[index], np.float64),
                    "dice": row["dice"], "terms": row["terms"]}
        if early_stop_tol is not None and early_stop_tol > 0 and prev_loss is not None:
            loss_ok = early_stop_loss is None or early_stop_loss <= 0 or row["loss"] < early_stop_loss
            if abs(row["loss"] - prev_loss) < early_stop_tol and loss_ok:
                early_stopped = True
                break
        prev_loss = row["loss"]

    elapsed = time.perf_counter() - start
    xbest = np.asarray(best["x"] if early_stopped else es.result.xbest, np.float64)
    packed = evaluate_population([xbest], renderer, frame, weights, alpha_limit, jaw_limit, chunk)
    best = _pack_best(packed, xbest)
    stop = {str(k): (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in es.stop().items()}
    if early_stopped:
        stop["early_stop_tol"] = float(early_stop_tol)
        if early_stop_loss is not None and early_stop_loss > 0:
            stop["early_stop_loss"] = float(early_stop_loss)
    stats = {
        "optimizer": "cmaes",
        "evaluations": int(es.result.evaluations),
        "generations": generation,
        "early_stopped": early_stopped,
        "translation_search_m": {"xy": float(translation_xy_m), "z": float(translation_z_m)},
        "cma_stds": {
            "rotation": float(rotation_std),
            "translation_xy": float(translation_xy_std),
            "translation_z": float(translation_z_std),
            "joints": float(joint_std),
        },
        "stop": stop,
    }
    return best, history, elapsed, stats


def _dice_lr_multiplier(dice_mean, low=0.80, high=0.95, coarse=3.0, fine=0.3):
    """Instrument-Splatting style: boost LR while overlap is poor, shrink when nearly done."""
    if dice_mean != dice_mean:
        return float(coarse)
    if dice_mean < low:
        return float(coarse)
    if dice_mean < high:
        return 1.0
    return float(fine)


def optimize_frame_adam(renderer, frame, x0, weights, translation_min, translation_max,
                        alpha_limit, jaw_limit, maxiter=300, lr_translation=0.1, lr_rotation=0.001,
                        lr_joints=None, lr_step=None, lr_gamma=0.5, translation_xy_m=0.02,
                        translation_z_m=0.03, early_stop_tol=0.001, early_stop_loss=0.1,
                        grad_clip=1.0, dice_lr=True, dice_lr_low=0.80, dice_lr_high=0.95,
                        dice_lr_coarse=3.0, dice_lr_fine=0.3):
    """Adam on quaternion + local millimetre translation (Instrument-Splatting style).

    ``lr_translation`` is for ``delta_t`` in millimetres (default 0.1).
    ``lr_rotation`` default 0.001 — 0.01 oscillates and stalls around Dice ~0.5.
    With ``dice_lr``, learning rates are scaled by overlap quality (×3 when Dice<0.8).
    Prefer ``maxiter`` around 300; 60 steps is usually not enough from a 10° init.
    """
    if lr_joints is None:
        lr_joints = 0.01
    device = renderer.mesh.vertices.device
    start = time.perf_counter()
    x0 = np.asarray(x0, np.float64).reshape(-1)
    if x0.shape[0] != POSE_DIM:
        raise ValueError(f"Expected {POSE_DIM}-d pose vector, got {x0.shape}")

    if maxiter <= 0:
        packed = evaluate_population([x0], renderer, frame, weights, alpha_limit, jaw_limit, chunk=1)
        best = _pack_best(packed, x0)
        history = [{"generation": 0, "loss": best["loss"], "x": best["x"].tolist(),
                    "dice": best["dice"], "terms": best["terms"]}]
        return best, history, time.perf_counter() - start, {
            "optimizer": "adam", "evaluations": 1, "generations": 0, "stop": {"maxiter": 0}}

    init_pose = vector_to_pose(x0, device, alpha_limit=alpha_limit, jaw_limit=jaw_limit)
    t0 = init_pose["t"][0].detach().clone()
    quat = torch.nn.Parameter(matrix_to_quaternion(init_pose["R"][0].detach()))
    delta_mm = torch.nn.Parameter(torch.zeros_like(t0))
    joints = torch.nn.Parameter(init_pose["joints"][0].detach().clone())
    param_groups = [
        {"params": [quat], "lr": float(lr_rotation), "lr_base": float(lr_rotation)},
        {"params": [delta_mm], "lr": float(lr_translation), "lr_base": float(lr_translation)},
        {"params": [joints], "lr": float(lr_joints), "lr_base": float(lr_joints)},
    ]
    optimizer = torch.optim.Adam(param_groups, eps=1e-15)
    use_step_lr = lr_step is not None and int(lr_step) > 0
    delta_limit = torch.tensor([translation_xy_m, translation_xy_m, translation_z_m],
                               device=device, dtype=torch.float32) * 1000
    workspace_min = None if translation_min is None else torch.as_tensor(translation_min, device=device, dtype=torch.float32)
    workspace_max = None if translation_max is None else torch.as_tensor(translation_max, device=device, dtype=torch.float32)
    batch = _expand_frame(frame, 1, device)
    history = []
    best = {"loss": float("inf"), "x": x0.copy(), "dice": None, "terms": None, "dice_mean": -1.}
    prev_loss = None
    early_stopped = False

    def build_pose():
        rotation = quaternion_to_matrix(quat)
        translation = t0 + delta_mm / 1000
        return {"R": rotation[None], "t": translation[None], "joints": joints[None]}

    for step in range(1, int(maxiter) + 1):
        with torch.no_grad():
            preview = build_pose()
            preview_dice = dice_dict(renderer(preview, batch["K"])["mask"][:1], batch["mask"][:1])
            preview_mean = preview_dice.get("mean")
            preview_mean = float(preview_mean) if preview_mean is not None else float("nan")
            lr_mult = _dice_lr_multiplier(preview_mean, dice_lr_low, dice_lr_high,
                                          dice_lr_coarse, dice_lr_fine) if dice_lr else 1.0
            schedule = 1.0
            if use_step_lr and step > 1:
                schedule = float(lr_gamma) ** ((step - 1) // int(lr_step))
            for group in optimizer.param_groups:
                group["lr"] = group["lr_base"] * schedule * lr_mult

        optimizer.zero_grad(set_to_none=True)
        pose = build_pose()
        result = renderer(pose, batch["K"])
        loss_vec, terms = total_loss(result, batch, weights)
        loss = loss_vec.mean()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([quat, delta_mm, joints], float(grad_clip))
        optimizer.step()
        with torch.no_grad():
            quat.copy_(torch.nn.functional.normalize(quat, dim=-1))
            delta_mm.clamp_(-delta_limit, delta_limit)
            translation = t0 + delta_mm / 1000
            if workspace_min is not None:
                translation = torch.maximum(translation, workspace_min)
            if workspace_max is not None:
                translation = torch.minimum(translation, workspace_max)
            delta_mm.copy_((translation - t0) * 1000)
            joints[..., 0].clamp_(-alpha_limit, alpha_limit)
            joints[..., 1].clamp_(-jaw_limit, jaw_limit)
            joints[..., 2].clamp_(-jaw_limit, jaw_limit)
            pose = build_pose()
            result = renderer(pose, batch["K"])
            loss_vec, terms = total_loss(result, batch, weights)
            dice = dice_dict(result["mask"][:1], batch["mask"][:1])
            term_cpu = {name: float(value[0]) for name, value in terms.items()}
            loss_value = float(loss_vec.mean())
            vector = pose_to_vector({k: v.detach() for k, v in pose.items()})
            dice_mean = dice.get("mean")
            dice_mean = float(dice_mean) if dice_mean is not None else float("nan")

        row = {
            "generation": step,
            "loss": loss_value,
            "x": vector.tolist(),
            "dice": dice,
            "terms": term_cpu,
            "lr_translation_mm": float(optimizer.param_groups[1]["lr"]),
            "lr_rotation": float(optimizer.param_groups[0]["lr"]),
            "lr_joints": float(optimizer.param_groups[2]["lr"]),
            "lr_mult": float(lr_mult),
            "delta_t_mm": delta_mm.detach().cpu().tolist(),
        }
        history.append(row)
        better = (dice_mean == dice_mean and dice_mean > best["dice_mean"] + 1e-6) or (
            (not (best["dice_mean"] == best["dice_mean"]) or abs(dice_mean - best["dice_mean"]) <= 1e-6)
            and loss_value < best["loss"])
        if better:
            best = {"loss": loss_value, "x": vector.copy(), "dice": dice, "terms": term_cpu,
                    "dice_mean": dice_mean if dice_mean == dice_mean else best["dice_mean"]}
        if early_stop_tol is not None and early_stop_tol > 0 and prev_loss is not None:
            loss_ok = early_stop_loss is None or early_stop_loss <= 0 or loss_value < early_stop_loss
            if abs(loss_value - prev_loss) < early_stop_tol and loss_ok:
                early_stopped = True
                break
        prev_loss = loss_value

    elapsed = time.perf_counter() - start
    packed = evaluate_population([best["x"]], renderer, frame, weights, alpha_limit, jaw_limit, chunk=1)
    best = _pack_best(packed, best["x"])
    stop = {"maxiter": int(maxiter)}
    if early_stopped:
        stop = {"early_stop_tol": float(early_stop_tol)}
        if early_stop_loss is not None and early_stop_loss > 0:
            stop["early_stop_loss"] = float(early_stop_loss)
    stats = {
        "optimizer": "adam",
        "parameterization": "quaternion + local_translation_mm",
        "evaluations": int(len(history)),
        "generations": int(len(history)),
        "early_stopped": early_stopped,
        "dice_lr": bool(dice_lr),
        "lr": {"translation_mm": float(lr_translation), "rotation": float(lr_rotation),
               "joints": float(lr_joints), "step": None if lr_step is None else int(lr_step),
               "gamma": float(lr_gamma)},
        "translation_search_m": {"xy": float(translation_xy_m), "z": float(translation_z_m)},
        "stop": stop,
    }
    return best, history, elapsed, stats
