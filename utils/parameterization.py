"""Identity / centroid / GT / perturbed-GT initialization with an in-FOV check."""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from utils.pose_geometry import axis_rotation

ARM_DIM = 9
N_ARMS = 2
POSE_DIM = ARM_DIM * N_ARMS


def axis_angle_to_matrix(rvec):
    theta = torch.linalg.norm(rvec, dim=-1, keepdim=True)
    axis = rvec / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*rvec.shape[:-1], 3, 3)
    identity = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(skew)
    angle = theta[..., None]
    return identity + angle.sin() * skew + (1 - angle.cos()) * (skew @ skew)


def matrix_to_quaternion(matrix):
    """Rotation matrix [...,3,3] -> unit quaternion [...,4] as (w,x,y,z)."""
    flat = matrix.detach().cpu().numpy().reshape(-1, 3, 3)
    # SciPy returns (x,y,z,w).
    q_xyzw = Rotation.from_matrix(flat).as_quat()
    q = np.concatenate((q_xyzw[:, 3:4], q_xyzw[:, :3]), axis=1)
    return torch.as_tensor(q, device=matrix.device, dtype=matrix.dtype).reshape(*matrix.shape[:-2], 4)


def quaternion_to_matrix(quaternions):
    """Unit quaternion [...,4] (w,x,y,z) -> rotation matrix [...,3,3]."""
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack((
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ), -1).reshape(*q.shape[:-1], 3, 3)


def identity_vector(depth=0.06, x_split=0.04):
    """Identity rotations, joints at 0, wrists split left/right in front of the camera."""
    x = np.zeros(POSE_DIM, np.float64)
    x[3], x[5] = -abs(x_split), depth
    x[ARM_DIM + 3], x[ARM_DIM + 5] = abs(x_split), depth
    return x


def pose_to_vector(pose):
    r = pose["R"].detach().cpu().numpy()
    t = pose["t"].detach().cpu().numpy()
    joints = pose["joints"].detach().cpu().numpy()
    if r.ndim == 4:
        r, t, joints = r[0], t[0], joints[0]
    x = np.zeros(len(r) * ARM_DIM, np.float64)
    for arm in range(len(r)):
        rotvec = Rotation.from_matrix(r[arm]).as_rotvec()
        offset = arm * ARM_DIM
        x[offset:offset + 3] = rotvec
        x[offset + 3:offset + 6] = t[arm]
        x[offset + 6:offset + 9] = joints[arm]
    return x


def vector_to_pose(vector, device, dtype=torch.float32, alpha_limit=np.pi / 2, jaw_limit=np.deg2rad(125.)):
    x = torch.as_tensor(np.asarray(vector, np.float64), device=device, dtype=dtype)
    if x.ndim == 1:
        x = x[None]
    if x.shape[-1] != POSE_DIM:
        raise ValueError(f"Expected {POSE_DIM}-d pose vectors, got {tuple(x.shape)}")
    blocks = x.reshape(x.shape[0], N_ARMS, ARM_DIM)
    joints = blocks[..., 6:9].clone()
    joints[..., 0].clamp_(-alpha_limit, alpha_limit)
    joints[..., 1].clamp_(-jaw_limit, jaw_limit)
    joints[..., 2].clamp_(-jaw_limit, jaw_limit)
    return {"R": axis_angle_to_matrix(blocks[..., :3]), "t": blocks[..., 3:6], "joints": joints}


def cma_bounds(x0, alpha_limit, jaw_limit, translation_xy_m=0.02, translation_z_m=0.03,
               translation_min=None, translation_max=None):
    """Per-arm CMA box bounds. Translation is relative to ``x0`` (±xy / ±z metres)."""
    x0 = np.asarray(x0, np.float64).reshape(-1)
    if x0.shape[0] != POSE_DIM:
        raise ValueError(f"Expected {POSE_DIM}-d pose vector, got {x0.shape}")
    low, high = [], []
    workspace_min = None if translation_min is None else np.asarray(translation_min, np.float64)
    workspace_max = None if translation_max is None else np.asarray(translation_max, np.float64)
    delta = np.array([translation_xy_m, translation_xy_m, translation_z_m], np.float64)
    for arm in range(N_ARMS):
        offset = arm * ARM_DIM
        t0 = x0[offset + 3:offset + 6]
        t_low, t_high = t0 - delta, t0 + delta
        if workspace_min is not None:
            t_low = np.maximum(t_low, workspace_min)
        if workspace_max is not None:
            t_high = np.minimum(t_high, workspace_max)
        if np.any(t_low >= t_high):
            raise ValueError(f"Empty CMA translation bounds for arm {arm}: low={t_low}, high={t_high}")
        low.extend([-np.pi, -np.pi, -np.pi, *t_low.tolist(), -alpha_limit, -jaw_limit, -jaw_limit])
        high.extend([np.pi, np.pi, np.pi, *t_high.tolist(), alpha_limit, jaw_limit, jaw_limit])
    return np.asarray(low, np.float64), np.asarray(high, np.float64)


def cma_stds(rotation=0.25, translation_xy=0.003, translation_z=0.005, joints=0.15):
    """Initial CMA coordinate-wise stds (metres / radians)."""
    std = []
    for _ in range(N_ARMS):
        std.extend([rotation, rotation, rotation,
                    translation_xy, translation_xy, translation_z,
                    joints, joints, joints])
    return np.asarray(std, np.float64)


def arm_in_fov(mask, min_pixels=80):
    if mask.ndim == 3:
        mask = mask[None]
    b, c, h, w = mask.shape
    if c % 3:
        raise ValueError("Semantic masks have 3 channels per instrument")
    arms = c // 3
    counts = (mask.reshape(b, arms, 3, h, w) > .5).any(2).sum((-2, -1))
    return counts, counts >= min_pixels


def _randn(shape, dtype, device, generator):
    if generator is None:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=torch.float32, generator=generator).to(device=device, dtype=dtype)


def perturb_gt_pose(pose, rotation_degrees=10., translation_mm=5., joints_degrees=8.,
                    translation_min=(-.08, -.08, .025), translation_max=(.08, .08, .12),
                    alpha_limit=np.pi / 2, jaw_limit=np.deg2rad(125.), generator=None):
    """Small independent noise around a ground-truth wrist pose, clamped to decoder bounds."""
    p = {key: value.clone() for key, value in pose.items()}
    device, dtype = p["t"].device, p["t"].dtype
    low = torch.as_tensor(translation_min, dtype=dtype, device=device)
    high = torch.as_tensor(translation_max, dtype=dtype, device=device)
    margin = torch.tensor([.01, .01, .005], dtype=dtype, device=device)
    random_angles = _randn(p["joints"].shape, dtype, device, generator) * np.deg2rad(rotation_degrees)
    delta = (axis_rotation(random_angles[..., 0], "x")
             @ axis_rotation(random_angles[..., 1], "y")
             @ axis_rotation(random_angles[..., 2], "z"))
    p["R"] = delta @ p["R"]
    shift = _randn(p["t"].shape, dtype, device, generator) * (translation_mm / 1000)
    p["t"] = torch.maximum(torch.minimum(p["t"] + shift, high - margin), low + margin)
    joints = p["joints"] + _randn(p["joints"].shape, dtype, device, generator) * np.deg2rad(joints_degrees)
    joints[..., 0].clamp_(-alpha_limit, alpha_limit)
    half = (joints[..., 1:].sum(-1) / 2).clamp(0, jaw_limit)
    yaw = (joints[..., 1] - joints[..., 2]) / 2
    yaw = torch.minimum(torch.maximum(yaw, -(jaw_limit - half)), jaw_limit - half)
    p["joints"] = torch.stack((joints[..., 0], half + yaw, half - yaw), -1)
    return p


def centroid_translation(mask, k, depth):
    fg = mask.sum(0) > .5
    if not fg.any():
        return None
    ys, xs = torch.where(fg)
    u, v = xs.float().mean(), ys.float().mean()
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    return torch.stack(((u - cx) * depth / fx, (v - cy) * depth / fy, mask.new_tensor(float(depth))))


@torch.no_grad()
def initialize_pose(init, frame, renderer, depth, x_split, min_pixels, alpha_limit, jaw_limit,
                    translation_min=(-.08, -.08, .025), translation_max=(.08, .08, .12),
                    rotation_degrees=10., translation_mm=5., joints_degrees=8., seed=0,
                    previous_vector=None, first_frame=False):
    """Build a starting pose. ``perturb-gt`` uses noisy first-frame GT; later frames may reuse the previous result."""
    device = renderer.mesh.vertices.device
    k = frame.K.to(device)
    extra = {}
    if init == "gt":
        pose = {key: value.to(device) for key, value in frame.pose.items()}
        note = "gt"
    elif init == "perturb-gt":
        if first_frame or previous_vector is None:
            pose = {key: value.to(device) for key, value in frame.pose.items()}
            generator = torch.Generator()
            generator.manual_seed(int(seed) & 0x7FFFFFFF)
            pose = perturb_gt_pose(
                pose, rotation_degrees, translation_mm, joints_degrees,
                translation_min, translation_max, alpha_limit, jaw_limit, generator)
            note = "perturb-gt"
            extra = {"rotation_deg": float(rotation_degrees), "translation_mm": float(translation_mm),
                     "joints_deg": float(joints_degrees), "seed": int(seed)}
        else:
            pose = vector_to_pose(previous_vector, device, alpha_limit=alpha_limit, jaw_limit=jaw_limit)
            note = "previous"
    elif init == "centroid":
        pose = vector_to_pose(identity_vector(depth, x_split), device,
                              alpha_limit=alpha_limit, jaw_limit=jaw_limit)
        for arm in range(N_ARMS):
            xyz = centroid_translation(frame.mask[3 * arm:3 * arm + 3].to(device), k, depth)
            if xyz is None:
                continue
            pose["t"][0, arm] = xyz
        note = "centroid"
    else:
        pose = vector_to_pose(identity_vector(depth, x_split), device,
                              alpha_limit=alpha_limit, jaw_limit=jaw_limit)
        note = "identity"
        rendered = renderer(pose, k)["mask"][0]
        _, flags = arm_in_fov(rendered, min_pixels)
        if not bool(flags.all()):
            for arm in range(N_ARMS):
                xyz = centroid_translation(frame.mask[3 * arm:3 * arm + 3].to(device), k, depth)
                if xyz is None:
                    continue
                pose["t"][0, arm] = xyz
            note = "identity+centroid"
    rendered = renderer(pose, k)["mask"][0]
    counts, flags = arm_in_fov(rendered, min_pixels)
    return pose, pose_to_vector(pose), {"visible": bool(flags.all()), "init": note,
                                        "pixels": [int(v) for v in counts[0].tolist()], **extra}
