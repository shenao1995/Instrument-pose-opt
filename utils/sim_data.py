"""Load Isaac dual-instrument runs and convert link poses to the canonical FK state."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.pose_geometry import PARTS, get_convention, matrix_to_rotation_6d, vector_pose
from utils.tips import extract_arm_tips

CONVENTION = "simulated_dual_arm_v1"
SOURCES = {"left": ("endo/left_endo_rgb.mp4", "mask"), "right": ("endo/right_endo_rgb.mp4", "mask_right")}
ARM_PREFIXES = ("", "right_")
ARM_NAMES = ("left", "right")
DEFAULT_ALBEDO = {
    "shaft": [.55, .55, .58],
    "wrist": [.62, .62, .65],
    "gripper_left": [.50, .50, .52],
    "gripper_right": [.50, .50, .52],
}


def quaternion_matrix(q):
    q = np.asarray(q, np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotation_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def list_runs(data_root):
    root = Path(data_root)
    runs = sorted(p for p in root.glob("run_*") if (p / "poses.json").exists())
    if not runs:
        raise FileNotFoundError(f"No run_* directories with poses.json under {root}")
    return runs


def load_intrinsics(run, instrument, camera, size):
    """Camera matrix at the working size with the principal point at the image centre."""
    path = Path(run) / "camera_intrinsics.json"
    if not path.exists():
        path = Path(instrument) / "camera_intrinsics.json"
    entry = json.loads(path.read_text(encoding="utf-8"))[camera]
    k = np.asarray(entry["K"], np.float64)
    source_w, source_h = entry["resolution"]
    k[0, 2], k[1, 2] = (source_w - 1) / 2, (source_h - 1) / 2
    sx, sy = size[1] / source_w, size[0] / source_h
    k[0] *= sx
    k[1] *= sy
    k[0, 2] += (sx - 1) / 2
    k[1, 2] += (sy - 1) / 2
    return k.astype(np.float32), (source_h, source_w)


class PoseConverter:
    """Simulated link poses -> canonical (R, t, alpha, theta_left, theta_right) per arm."""

    def __init__(self, convention, camera, baseline):
        self.convention = get_convention(convention)
        self.rotations = {part: np.asarray(self.convention.rotations[part], np.float64) for part in PARTS}
        shift = np.array([baseline, 0., 0.])
        self.shifts = (np.zeros(3), shift) if camera == "left" else (-shift, np.zeros(3))

    def canonical_rotation(self, poses, prefix, part):
        link = self.convention.links[part]
        q = poses[prefix + link]
        return quaternion_matrix(q[:4]) @ self.rotations[part].T, np.asarray(q[4:7], np.float64)

    def convert(self, poses):
        blocks = []
        for arm, prefix in enumerate(ARM_PREFIXES):
            r_wrist, t_wrist = self.canonical_rotation(poses, prefix, "wrist")
            r_shaft, _ = self.canonical_rotation(poses, prefix, "shaft")
            r_left, _ = self.canonical_rotation(poses, prefix, "gripper_left")
            r_right, _ = self.canonical_rotation(poses, prefix, "gripper_right")
            m = r_wrist.T @ r_shaft
            alpha = -np.arctan2(m[0, 2], m[0, 0])
            ml, mr = r_wrist.T @ r_left, r_wrist.T @ r_right
            theta_left = np.arctan2(ml[1, 0], ml[0, 0])
            theta_right = -np.arctan2(mr[1, 0], mr[0, 0])
            r6 = matrix_to_rotation_6d(torch.from_numpy(r_wrist)).numpy()
            blocks.append(np.concatenate((r6, t_wrist + self.shifts[arm], [alpha, theta_left, theta_right])))
        return np.asarray(blocks, np.float32)


def decode_label(label, classes=6):
    return np.stack([label == c for c in range(1, classes + 1)]).astype(np.float32)


def estimate_albedo(runs, camera, count_runs=1, count_frames=4):
    sums, counts = np.zeros((3, 3)), np.zeros(3)
    video_name, mask_dir = SOURCES[camera]
    for run in runs[:count_runs]:
        n_poses = len(json.loads((run / "poses.json").read_text(encoding="utf-8"))["frames"])
        picks = set(np.linspace(0, n_poses - 1, min(count_frames, n_poses)).round().astype(int).tolist())
        capture = cv2.VideoCapture(str(run / video_name))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in picks:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
                path = run / mask_dir / f"{index:04d}.png"
                label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if label is not None:
                    for part in range(3):
                        select = (label == part + 1) | (label == part + 4)
                        if select.any():
                            sums[part] += rgb[select].sum(0)
                            counts[part] += select.sum()
            index += 1
        capture.release()
    if (counts == 0).any():
        return dict(DEFAULT_ALBEDO)
    colours = sums / np.maximum(counts, 1)[:, None]
    return {"shaft": colours[0].tolist(), "wrist": colours[1].tolist(),
            "gripper_left": colours[2].tolist(), "gripper_right": colours[2].tolist()}


@dataclass
class FrameSample:
    run: str
    index: int
    rgb: torch.Tensor
    mask: torch.Tensor
    K: torch.Tensor
    pose: dict
    pose_vector: np.ndarray
    tips: torch.Tensor
    tip_confidence: torch.Tensor
    tip_scale: torch.Tensor
    source_size: tuple


def evenly_spaced_indices(n, k):
    """Pick ``k`` indices evenly spanning ``[0, n)`` (includes ends when ``k>1``)."""
    n, k = int(n), int(k)
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    raw = np.round(np.linspace(0, n - 1, k)).astype(np.int64)
    # Rounding can collide for small n; refill gaps while keeping order.
    chosen, used = [], set()
    for value in raw.tolist():
        value = int(np.clip(value, 0, n - 1))
        if value not in used:
            chosen.append(value)
            used.add(value)
    cursor = 0
    while len(chosen) < k and cursor < n:
        if cursor not in used:
            chosen.append(cursor)
            used.add(cursor)
        cursor += 1
    return sorted(chosen)


def iter_frames(run, converter, camera, size, instrument, frame_stride=1, limit=0, start=0):
    """Yield frames from a run.

    When ``limit > 0``, selects that many frames **evenly spaced** across the video
    (after applying ``start`` / ``frame_stride``), instead of taking the first ``limit`` frames.
    """
    run = Path(run)
    records = json.loads((run / "poses.json").read_text(encoding="utf-8"))["frames"]
    k, source_size = load_intrinsics(run, instrument, camera, size)
    k_tensor = torch.from_numpy(k)
    video_name, mask_dir = SOURCES[camera]
    capture = cv2.VideoCapture(str(run / video_name))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video {run / video_name}")

    candidates = [index for index in range(start, len(records), max(1, int(frame_stride)))]
    if limit and limit > 0:
        selected = set(candidates[i] for i in evenly_spaced_indices(len(candidates), limit))
    else:
        selected = set(candidates)

    try:
        for index, record in enumerate(records):
            ok, bgr = capture.read()
            if not ok:
                break
            if index not in selected:
                continue
            label = cv2.imread(str(run / mask_dir / f"{index:04d}.png"), cv2.IMREAD_UNCHANGED)
            if label is None:
                raise FileNotFoundError(f"Missing mask {run / mask_dir / f'{index:04d}.png'}")
            if tuple(label.shape[:2]) != source_size:
                raise ValueError(f"{run.name} frame {index}: mask size {label.shape[:2]} != {source_size}")
            label = cv2.resize(label, size[::-1], interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(cv2.resize(bgr, size[::-1], interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            mask = decode_label(label)
            # Premultiply RGB by the GT instrument mask so photometric terms never
            # see endoscopic background outside the observation support.
            rgb = rgb.transpose(2, 0, 1) * (mask.max(0, keepdims=True) > 0)
            pose_vector = converter.convert(record["poses"])
            pose = vector_pose(torch.from_numpy(pose_vector.reshape(1, -1)))
            tips, confidence, scales = extract_arm_tips(mask)
            yield FrameSample(
                run=run.name, index=index,
                rgb=torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255,
                mask=torch.from_numpy(mask),
                K=k_tensor.clone(),
                pose={key: value.cpu() for key, value in pose.items()},
                pose_vector=pose_vector,
                tips=torch.from_numpy(tips),
                tip_confidence=torch.from_numpy(confidence),
                tip_scale=torch.from_numpy(scales),
                source_size=source_size,
            )
    finally:
        capture.release()
