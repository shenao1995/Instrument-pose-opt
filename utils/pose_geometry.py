"""Batched articulated kinematics and semantic mesh rendering for one or two instruments.

Coordinates: metres, radians, OpenCV camera (+x right, +y down, +z forward).
R/t describe WRIST to CAMERA, not camera to world. No Gaussian dependencies.

Pose tensors carry an explicit ARM axis: R [B,A,3,3], t [B,A,3], joints [B,A,3].
A flat pose vector holds A consecutive 12-d blocks (6-d rotation, translation,
alpha, theta_left, theta_right). A=1 reproduces the original single-instrument
behaviour; A=2 renders two instruments of the same model into 3*A semantic
channels ordered (arm0 shaft, arm0 wrist, arm0 grippers, arm1 shaft, ...).

Canonical part frames (shared by every mesh convention):
  wrist:   origin at the wrist joint, +x distal, wrist pitch about y, jaw yaw about z.
  shaft:   rigidly attached to the wrist frame at alpha=0, rotated by Ry(-alpha).
  jaws:    origin at the jaw pivot (+x * pivot from the wrist), Rz(+theta_left) / Rz(-theta_right),
           theta_left + theta_right is the jaw opening and is nonnegative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("shaft", "wrist", "gripper_left", "gripper_right")
LABELS = (0, 1, 2, 2)
CLASSES_PER_ARM = PART_CLASSES = 3
POSE_DIM = 12  # 6-d rotation + translation + (alpha, theta_left, theta_right) per arm
CLASS_NAMES = ("shaft", "wrist", "grippers")


def mask_channel_names(arms):
    if arms == 1:
        return list(CLASS_NAMES)
    return [f"arm{a}_{name}" for a in range(arms) for name in CLASS_NAMES]


# --------------------------------------------------------------------------- #
# Mesh conventions: how OBJ-local coordinates map to the canonical part frames.
# --------------------------------------------------------------------------- #
_BIAS = ((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))
# Simulated wrist OBJ: instrument axis +x, jaw axis +y, wrist-pitch axis +z.
# C maps canonical wrist coordinates to simulated ones (v_sim = C @ v_canonical),
# i.e. canonical y = -sim z and canonical z = sim y (a 90 degree roll about x).
_SIM_C = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
# Simulated shaft OBJ extends along its local +z and pitches about its local +y.
# B_shaft maps shaft-local to canonical: local z -> -x (proximal), local y -> +y.
_SIM_B_SHAFT = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])


@dataclass(frozen=True)
class MeshConvention:
    """OBJ file names plus the rigid transform from OBJ-local to canonical coordinates.

    canonical = local @ rotation.T + translation, per part. `pivot` is the jaw
    pivot offset along canonical +x; `shaft_offset` is the x-shift baked into the
    canonical shaft vertices (the original Instrument-Splatting shaft frame sits
    215.9 mm behind the wrist; the simulated shaft shares the wrist origin).
    `links` names the source rigid bodies for label files (informational).
    """
    name: str
    files: dict
    rotations: dict
    translations: dict
    pivot: float
    shaft_offset: float
    links: dict = field(default_factory=dict)
    notes: str = ""

    def canonical_vertices(self, part, vertices):
        return np.asarray(vertices, dtype=np.float64) @ np.asarray(self.rotations[part]).T + np.asarray(self.translations[part])


CONVENTIONS = {
    "instrument_splatting": MeshConvention(
        name="instrument_splatting",
        files={part: f"transformed_{part}.obj" for part in PARTS},
        rotations={"shaft": _BIAS, "wrist": tuple(tuple(s*v for v in row) for s, row in zip((1, -1, -1), _BIAS)),
                   "gripper_left": _BIAS, "gripper_right": _BIAS},
        translations={"shaft": (0.2159, 0., 0.), "wrist": (0., 0., 0.),
                      "gripper_left": (-0.009, 0., 0.), "gripper_right": (-0.009, 0., 0.)},
        pivot=0.009, shaft_offset=0.2159,
        links={part: part for part in PARTS},
        notes="Original Instrument-Splatting OBJs; wrist flipped by diag(1,-1,-1)."),
    "simulated_dual_arm_v1": MeshConvention(
        name="simulated_dual_arm_v1",
        # The simulated 'left'/'right' jaws are named from the robot's side and are
        # mirrored relative to the canonical +z jaw ordering; swapping the slots keeps
        # theta_left + theta_right (the opening) nonnegative as the decoder requires.
        files={"shaft": "shaft.obj", "wrist": "wrist.obj",
               "gripper_left": "right_gripper.obj", "gripper_right": "left_gripper.obj"},
        rotations={"shaft": tuple(map(tuple, _SIM_B_SHAFT)), "wrist": tuple(map(tuple, _SIM_C.T)),
                   "gripper_left": tuple(map(tuple, np.eye(3))), "gripper_right": tuple(map(tuple, np.eye(3)))},
        translations={part: (0., 0., 0.) for part in PARTS},
        pivot=0.0095, shaft_offset=0.,
        links={"shaft": "shaft", "wrist": "wrist", "gripper_left": "right", "gripper_right": "left"},
        notes="Isaac simulation OBJs (assimp export, no MTL). Jaw pivot 9.5 mm along wrist +x."),
}


# Metadata written by create_data.py names the original convention this way.
CONVENTION_ALIASES = {"instrument_splatting_wrist_opencv_v1": "instrument_splatting"}


def get_convention(convention):
    if isinstance(convention, MeshConvention):
        return convention
    convention = CONVENTION_ALIASES.get(convention, convention)
    if convention not in CONVENTIONS:
        raise ValueError(f"Unknown mesh convention {convention!r}; choose from {sorted(CONVENTIONS)}")
    return CONVENTIONS[convention]


def mesh_from_metadata(metadata, mesh_dir=None, load_appearance=True):
    """Build the InstrumentMesh a dataset or checkpoint was generated with."""
    return InstrumentMesh(mesh_dir or metadata["mesh_dir"], metadata.get("faces_per_part", 0),
                          load_appearance=load_appearance,
                          convention=metadata.get("geometry_convention", "instrument_splatting"),
                          arms=int(metadata.get("arms", 1)), part_albedo=metadata.get("part_albedo"))


def rotation_6d_to_matrix(d):
    """First two COLUMNS; Gram-Schmidt with a finite orthogonal fallback."""
    a, b = d[..., :3], d[..., 3:6]
    unit_x = torch.zeros_like(a)
    unit_x[..., 0] = 1
    x = F.normalize(torch.where(a.norm(dim=-1, keepdim=True) > 1e-6, a, unit_x), dim=-1)
    b = b - (x * b).sum(-1, keepdim=True) * x
    fallback = F.one_hot(x.abs().argmin(-1), 3).to(x)
    fallback = fallback - (fallback * x).sum(-1, keepdim=True) * x
    y = F.normalize(torch.where(b.norm(dim=-1, keepdim=True) > 1e-6, b, fallback), dim=-1)
    return torch.stack((x, y, torch.linalg.cross(x, y)), dim=-1)


def matrix_to_rotation_6d(r):
    return torch.cat((r[..., :, 0], r[..., :, 1]), dim=-1)


def axis_rotation(angle, axis):
    c, s = angle.cos(), angle.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    if axis == "y":
        values = (c, z, s, z, o, z, -s, z, c)
    elif axis == "z":
        values = (c, -s, z, s, c, z, z, z, o)
    else:
        values = (o, z, z, z, c, -s, z, s, c)
    return torch.stack(values, -1).reshape(*angle.shape, 3, 3)


def pose_vector(pose):
    """[..., A, 12] blocks flattened to [..., A*12]."""
    blocks = torch.cat((matrix_to_rotation_6d(pose["R"]), pose["t"], pose["joints"]), -1)
    return blocks.flatten(-2)


def vector_pose(vector):
    """Flat [..., A*12] vector to an arm-indexed pose dict (A inferred from the length)."""
    if vector.shape[-1] % 12:
        raise ValueError(f"Pose vectors hold 12 values per arm, got {vector.shape[-1]}")
    blocks = vector.reshape(*vector.shape[:-1], vector.shape[-1] // 12, 12)
    return {"R": rotation_6d_to_matrix(blocks[..., :6]), "t": blocks[..., 6:9], "joints": blocks[..., 9:12]}


def pose_arms(pose):
    return pose["R"].shape[-3]


def instruments_json(pose, index=0):
    """JSON-serializable wrist pose of every instrument in sample ``index``."""
    r, t, joints = pose["R"][index], pose["t"][index], pose["joints"][index]
    rows = []
    for arm in range(len(r)):
        transform = torch.eye(4, device=r.device, dtype=r.dtype)
        transform[:3, :3], transform[:3, 3] = r[arm], t[arm]
        rows.append({"T_wrist_camera": transform.tolist(), "translation_m": t[arm].tolist(),
                     "joints_rad": joints[arm].tolist(), "joint_order": ["alpha", "theta_left", "theta_right"]})
    return rows


def detach_pose(pose):
    return {k: v.detach() for k, v in pose.items()}


def rotation_geodesic_deg(pred_r, gt_r):
    """Geodesic SO(3) angle in degrees between rotations shaped [..., 3, 3]."""
    relative = pred_r.transpose(-1, -2) @ gt_r
    cos = ((relative.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) - 1) * .5).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def pose_errors(pred, gt):
    """Per-sample, per-arm pose errors against ground truth.

    Returns a dict of tensors with leading shape [B, A]:
      translation_error_mm  Euclidean ||t_pred - t_gt|| in millimetres
      translation_mse_mm2   mean of squared XYZ errors in mm^2
      rotation_error_deg    geodesic wrist orientation error
      joints_mae_deg        mean |joint| error over alpha/theta_left/theta_right
      joints_mse_deg2       mean squared joint error in degrees^2
      joints_error_deg      [B, A, 3] absolute joint errors in degrees
    """
    dt_mm = (pred["t"] - gt["t"]) * 1000
    dj_deg = torch.rad2deg(pred["joints"] - gt["joints"]).abs()
    return {
        "translation_error_mm": dt_mm.norm(dim=-1),
        "translation_mse_mm2": dt_mm.square().mean(-1),
        "rotation_error_deg": rotation_geodesic_deg(pred["R"], gt["R"]),
        "joints_mae_deg": dj_deg.mean(-1),
        "joints_mse_deg2": dj_deg.square().mean(-1),
        "joints_error_deg": dj_deg,
    }


def select_arm(pose, arm):
    """Single-arm view [B,3,3]/[B,3]/[B,3] of an arm-indexed pose."""
    return {k: v[:, arm] for k, v in pose.items()}


def part_rotations(pose):
    """Camera rotations of the four rigid bodies, each [B,A,3,3]."""
    r, a = pose["R"], pose["joints"]
    return {"shaft": r @ axis_rotation(-a[..., 0], "y"), "wrist": r,
            "gripper_left": r @ axis_rotation(a[..., 1], "z"), "gripper_right": r @ axis_rotation(-a[..., 2], "z")}


def part_transforms(pose, pivot=0.009, shaft_offset=0.2159):
    """Canonical part-local to OpenCV camera transforms [B,A,4,4], matching InstrumentMesh.

    Three semantic classes contain four rigid bodies per arm: the two jaws rotate
    independently. Canonical coordinates include the OBJ alignment of the convention.
    """
    r, t = pose["R"], pose["t"]
    rotations = part_rotations(pose)
    pivot_point = t + rotations["wrist"] @ t.new_tensor([pivot, 0, 0])
    translations = {"shaft": t - rotations["shaft"] @ t.new_tensor([shaft_offset, 0, 0]), "wrist": t,
                    "gripper_left": pivot_point, "gripper_right": pivot_point}
    result = {}
    for name in PARTS:
        transform = torch.eye(4, device=r.device, dtype=r.dtype).expand(*r.shape[:-2], 4, 4).clone()
        transform[..., :3, :3], transform[..., :3, 3] = rotations[name], translations[name]
        result[name] = transform
    return result


def project(points, k):
    p = points @ k.transpose(-1, -2)
    return p[..., :2] / p[..., 2:].clamp_min(1e-5)


@torch.no_grad()
def visible_tips(vertices, faces, tips, near=.001, tolerance=.00075):
    """Ray/triangle visibility, including occluders whose vertices miss the ray.

    vertices [B,N,3], faces [F,3], tips [B,T,3] or [B,A,2,3]; returns booleans of tips' leading shape.
    """
    shape = tips.shape[:-1]
    tips = tips.reshape(tips.shape[0], -1, 3)
    tri = vertices[:, faces]
    a, b, c = tri.unbind(-2)
    e1, e2 = b-a, c-a
    rays = tips / tips[..., 2:].clamp_min(near)
    h = torch.linalg.cross(rays[:,:,None],e2[:,None],dim=-1)
    determinant = (e1[:,None]*h).sum(-1)
    valid_det = determinant.abs() > 1e-12
    inv = torch.where(valid_det,1/determinant.masked_fill(~valid_det,1),0)
    u = (-a[:,None]*h).sum(-1)*inv
    q = torch.linalg.cross(-a,e1,dim=-1)
    v = (rays[:,:,None]*q[:,None]).sum(-1)*inv
    depth = (e2*q).sum(-1)[:,None]*inv
    hit = valid_det & (u >= 0) & (v >= 0) & (u+v <= 1) & (depth >= near)
    nearest = depth.masked_fill(~hit,float("inf")).amin(-1)
    return ((tips[...,2] > near) & (nearest >= tips[...,2]-tolerance)).reshape(shape)


def load_calibration(path=ROOT / "data/surgpose_sample/transforms.json", image_size=(256, 320)):
    from PIL import Image
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    frame = meta["frames"][0]
    first = path.parent / frame["file_path"]
    if not first.suffix:
        first = first.with_suffix(".png")
    with Image.open(first) as im:
        w, h = im.size
    k = np.asarray(meta["intrinsic_matrix"], dtype=np.float32)
    # Pixel-centre resize convention, consistent with OpenCV/PIL resizing.
    sx, sy = image_size[1] / w, image_size[0] / h
    k[0] *= sx
    k[1] *= sy
    k[0, 2] += (sx - 1) / 2
    k[1, 2] += (sy - 1) / 2
    transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
    # Rounded reference rotations are projected back to SO(3).
    u, _, vt = np.linalg.svd(transform[:3, :3])
    u[:, -1] *= np.linalg.det(u @ vt)
    r = torch.from_numpy(u @ vt)
    base = torch.cat((matrix_to_rotation_6d(r), torch.from_numpy(transform[:3, 3]),
                      torch.tensor([0., np.deg2rad(5), np.deg2rad(5)], dtype=torch.float32)))
    return torch.from_numpy(k), base


class InstrumentMesh(nn.Module):
    """Canonical part meshes plus forward kinematics for `arms` copies of one instrument.

    `vertices` holds ONE canonical copy (all parts); `faces`/`attributes`/`face_albedo`
    are replicated per arm so the renderer sees a single 3*arms-class scene.
    `part_albedo` (per part RGB in [0,1]) provides flat diffuse colours for OBJs
    without MTL materials; otherwise MTL Kd colours are used.
    """
    def __init__(self, mesh_dir=ROOT / "data/instrument_mesh", faces_per_part=0, load_appearance=False,
                 convention="instrument_splatting", arms=1, part_albedo=None):
        super().__init__()
        import trimesh
        self.convention = get_convention(convention)
        self.mesh_dir = str(Path(mesh_dir).resolve())
        self.faces_per_part = faces_per_part
        self.load_appearance = load_appearance
        if not isinstance(arms, int) or arms < 1:
            raise ValueError("arms must be a positive integer")
        self.arms = arms
        self.classes = CLASSES_PER_ARM*arms
        self.part_names = mask_channel_names(arms)
        self.appearance_hashes = {}
        if load_appearance and faces_per_part:
            raise ValueError("Material RGB rendering currently requires --faces-per-part 0")
        if part_albedo is not None:
            albedo_map = {part: [float(c) for c in part_albedo[part]] for part in PARTS}
            self.part_albedo = albedo_map
            self.appearance_hashes["part_albedo"] = hashlib.sha256(json.dumps(albedo_map, sort_keys=True).encode()).hexdigest()
        else:
            self.part_albedo = None
        all_normals,all_albedo = [],[]
        self.counts = []
        all_v, all_f, all_labels, tips = [], [], [], []
        offset = 0
        hashes = {}
        for i, part in enumerate(PARTS):
            path = Path(mesh_dir) / self.convention.files[part]
            hashes[part] = hashlib.sha256(path.read_bytes()).hexdigest()
            if load_appearance and self.part_albedo is None:
                from utils.pose_appearance import load_material_mesh
                mesh,material_hashes = load_material_mesh(path)
                self.appearance_hashes.update(material_hashes)
            else:
                mesh = trimesh.load(path, force="mesh", process=True)
            # OBJ normal/UV seams duplicate positions. Weld them before geometry
            # decimation, otherwise disconnected triangles collapse independently.
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            mesh.update_faces(mesh.unique_faces())
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
            v = self.convention.canonical_vertices(part, mesh.vertices)
            if part.startswith("gripper"):
                # Distal cap centre on ORIGINAL mesh, independent of simplification.
                tips.append(v[v[:, 0] >= v[:, 0].max() - 0.0003].mean(0))
            mesh.vertices = v
            if faces_per_part and len(mesh.faces) > faces_per_part:
                try:
                    mesh = mesh.simplify_quadric_decimation(face_count=faces_per_part)
                except ImportError as exc:
                    raise RuntimeError("Mesh simplification requires: pip install fast-simplification") from exc
            v = np.asarray(mesh.vertices, dtype=np.float32)
            if load_appearance:
                all_normals.append(np.asarray(mesh.vertex_normals,dtype=np.float32))
                if self.part_albedo is not None:
                    all_albedo.append(np.tile(np.asarray(self.part_albedo[part],np.float32).clip(0,1),(len(mesh.faces),1)))
                else:
                    all_albedo.append(np.asarray(mesh.visual.face_colors[:,:3],dtype=np.float32)/255)
            all_v.append(v)
            all_f.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
            all_labels.extend([LABELS[i]] * len(v))
            self.counts.append(len(v))
            offset += len(v)
        self.hashes = hashes
        vertices = torch.from_numpy(np.concatenate(all_v))
        faces = torch.from_numpy(np.concatenate(all_f))
        labels = torch.tensor(all_labels)
        self.register_buffer("vertices", vertices)
        self.register_buffer("part_faces", faces)
        # Per-arm replicas: arm a occupies vertices [a*N,(a+1)*N) and classes [3a,3a+3).
        self.register_buffer("faces", torch.cat([faces + a*len(vertices) for a in range(arms)]))
        arm_labels = torch.cat([labels + CLASSES_PER_ARM*a for a in range(arms)])
        self.register_buffer("attributes", F.one_hot(arm_labels, self.classes).float())
        self.register_buffer("local_tips", torch.tensor(np.asarray(tips), dtype=torch.float32))
        if load_appearance:
            self.register_buffer("normals",torch.from_numpy(np.concatenate(all_normals)))
            self.register_buffer("face_albedo",torch.from_numpy(np.concatenate(all_albedo)).repeat(arms,1))

    def _check_arms(self, pose):
        if pose_arms(pose) != self.arms:
            raise ValueError(f"Pose has {pose_arms(pose)} arm(s) but the mesh was built for {self.arms}")

    def camera_normals(self,pose):
        """[B, arms*N, 3] normals in camera coordinates, arm-major like forward()."""
        self._check_arms(pose)
        n = self.normals.split(self.counts)
        rotations = part_rotations(pose)
        result = []
        for arm in range(self.arms):
            result.extend(v @ rotations[name][:, arm].transpose(-1,-2) for v,name in zip(n,PARTS))
        return torch.cat(result,1)

    def forward(self, pose):
        """Returns vertices [B, arms*N, 3] (arm-major) and jaw tips [B, arms, 2, 3] in camera coordinates."""
        self._check_arms(pose)
        r, t, angles = pose["R"], pose["t"], pose["joints"]
        canonical = self.vertices.split(self.counts)
        pivot = t.new_tensor([self.convention.pivot, 0, 0])
        shaft_canonical = canonical[0] - canonical[0].new_tensor([self.convention.shaft_offset, 0, 0])
        transformed, tip_positions = [], []
        for arm in range(self.arms):
            ra, ta, aa = r[:, arm], t[:, arm, None], angles[:, arm]
            # T_shaft_camera = T_wrist_camera @ inverse(T_wrist_shaft).
            shaft = shaft_canonical @ axis_rotation(-aa[:, 0], "y").transpose(-1, -2)
            transformed.append(shaft @ ra.transpose(-1, -2) + ta)
            transformed.append(canonical[1] @ ra.transpose(-1, -2) + ta)
            tips = []
            for idx, sign in ((2, 1), (3, -1)):
                gr = axis_rotation(sign * aa[:, idx - 1], "z")
                v = canonical[idx] @ gr.transpose(-1, -2) + pivot
                transformed.append(v @ ra.transpose(-1, -2) + ta)
                tip = self.local_tips[idx - 2][None, None] @ gr.transpose(-1, -2) + pivot
                tips.append((tip @ ra.transpose(-1, -2) + ta)[:, 0])
            tip_positions.append(torch.stack(tips, 1))
        return torch.cat(transformed, 1), torch.stack(tip_positions, 1)


def clip_triangles_near(tri, attrs, near):
    """Differentiable near-plane clipping; topology selection is discrete."""
    inside = tri[..., 2] >= near
    count = inside.sum(-1)
    out, labels = [tri[count == 3]], [attrs[count == 3]]
    for n in (1, 2):
        group = tri[count == n]
        if not len(group):
            continue
        flags = inside[count == n]
        # Put the exceptional vertex first, preserving cyclic winding.
        first = (flags if n == 1 else ~flags).long().argmax(-1)
        order = (first[:, None] + torch.arange(3, device=tri.device)) % 3
        g = group.gather(1, order[..., None].expand(-1, -1, 3))
        a, b, c = g.unbind(1)
        def intersect(v):
            return a + ((near - a[:, 2]) / (v[:, 2] - a[:, 2]))[:, None] * (v - a)
        ab, ac = intersect(b), intersect(c)
        label = attrs[count == n]
        if n == 1:
            out.append(torch.stack((a, ab, ac), 1))
            labels.append(label)
        else:
            out.extend((torch.stack((ab, b, c), 1), torch.stack((ab, c, ac), 1)))
            labels.extend((label, label))
    return torch.cat(out), torch.cat(labels)


class SemanticRenderer(nn.Module):
    """Opaque semantic z-buffer with differentiable silhouette antialiasing.

    Visibility/topology choices are discrete. Coverage at selected visible edges
    is an explicit differentiable function of the projected mesh, NOT a detached
    blurred mask or a straight-through gradient. Internal edges of the same label
    do not alter coverage. Supersampling resolves subpixel jaws and silhouettes.
    Output masks have mesh.classes channels (3 per arm); both arms share one z-buffer,
    so inter-instrument occlusion is rendered exactly.
    """
    VERSION = 3
    RGB_VERSION = 4

    def __init__(self, mesh, image_size=(256, 320), backend="nvdiffrast", tile_size=32,
                 supersample=2, edge_width=1.0,render_rgb=False):
        super().__init__()
        self.mesh = mesh
        self.classes = mesh.classes
        self.render_rgb = render_rgb
        if render_rgb and not getattr(mesh,"load_appearance",False):
            raise ValueError("RGB rendering requires InstrumentMesh(load_appearance=True)")
        self.image_size = tuple(image_size)
        if not isinstance(supersample, int) or supersample < 1:
            raise ValueError("supersample must be a positive integer")
        if edge_width <= 0 or tile_size < 1:
            raise ValueError("edge_width and tile_size must be positive")
        self.tile_size, self.supersample, self.edge_width = tile_size, supersample, edge_width
        self.raster_size = tuple(supersample*s for s in self.image_size)
        self.near, self.far = .001, 2.
        if backend == "auto":
            # Keep the old CLI spelling, but never silently train on a different
            # renderer when the requested production dependency is missing.
            backend = "nvdiffrast"
        self.backend = backend
        if render_rgb and backend != "nvdiffrast":
            raise ValueError("Differentiable material RGB rendering requires --renderer nvdiffrast")
        self.ctx = None
        if backend == "nvdiffrast":
            if not mesh.vertices.is_cuda:
                raise ValueError("nvdiffrast requires --device cuda")
            try:
                import nvdiffrast.torch as dr
            except ImportError as exc:
                raise RuntimeError("nvdiffrast is required. See README.md for installation; "
                                   "on this Windows machine use .venv/Scripts/python.exe. "
                                   "Select --renderer torch explicitly only for reference/CPU checks.") from exc
            self.dr = dr
            self.ctx = dr.RasterizeCudaContext(device=mesh.vertices.device)
            self.register_buffer("raster_faces",mesh.faces.to(torch.int32).contiguous())
            self.topology_hash = dr.antialias_construct_topology_hash(self.raster_faces)
            # No per-frame face conversions or topology reconstruction.
            return
        elif backend != "torch":
            raise ValueError(f"Unknown renderer: {backend}")
        else:
            warnings.warn("Using portable torch triangle renderer; use nvdiffrast for throughput.", stacklevel=2)
        # Cache topology once. Silhouette edges separate front/back facing faces,
        # or lie on an open mesh boundary. Nonmanifold edges are retained safely;
        # the semantic visibility check below rejects occluded/internal edges.
        faces = mesh.faces.detach().cpu().numpy()
        face_edges = faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2)
        edges, inverse, counts = np.unique(np.sort(face_edges,axis=1),axis=0,return_inverse=True,return_counts=True)
        device = mesh.vertices.device
        self.register_buffer("edges",torch.as_tensor(edges,device=device))
        self.register_buffer("edge_inverse",torch.as_tensor(inverse,device=device))
        self.register_buffer("edge_counts",torch.as_tensor(counts,device=device))

    def configuration(self):
        config = {"version":self.VERSION,"backend":self.backend,"supersample":self.supersample,
                  "edge_width":self.edge_width,"near":self.near,"far":self.far}
        if self.mesh.arms != 1:
            config.update(arms=self.mesh.arms,classes=self.classes)
        if self.render_rgb:
            appearance = "flat_albedo_smooth_headlight_v1" if self.mesh.part_albedo is not None else "mtl_kd_smooth_headlight_v1"
            config.update(version=self.RGB_VERSION,appearance=appearance,
                          material_sha256=self.mesh.appearance_hashes)
        return config

    def forward(self, pose, k):
        vertices, tips = self.mesh(pose)
        if k.ndim == 2:
            k = k[None].expand(len(vertices), -1, -1)
        raster_k = k.clone()
        raster_k[:, :2] *= self.supersample
        raster_k[:, :2, 2] += (self.supersample-1)/2
        if self.backend == "nvdiffrast":
            mask = self._nvdiffrast(vertices, raster_k,pose)
        else:
            mask = torch.stack([self._torch(v, camera) for v, camera in zip(vertices, raster_k)])
        if self.supersample > 1:
            mask = F.avg_pool2d(mask,self.supersample)
        c = self.classes
        # tips: [B, arms, 2, 2] pixels; tips_camera: [B, arms, 2, 3] metres.
        output = {"mask":mask[:,:c],"tips":project(tips,k[:,None]),"tips_camera":tips}
        if self.render_rgb:
            # Already premultiplied by pixel coverage, including soft edges.
            output["rgb"] = mask[:,c:c+3]
        return output

    def _nvdiffrast(self, vertices, k,pose=None):
        h, w = self.raster_size
        p = vertices @ k.transpose(-1, -2)
        z = vertices[..., 2]
        # nvdiffrast rows run bottom-to-top: flip image on output.
        clip = torch.stack((2 * (p[..., 0] + .5 * z) / w - z,
                            z - 2 * (p[..., 1] + .5 * z) / h,
                            (self.far + self.near) / (self.far - self.near) * z -
                            2 * self.far * self.near / (self.far - self.near), z), -1).contiguous()
        faces = self.raster_faces
        rast, _ = self.dr.rasterize(self.ctx, clip, faces, resolution=[h, w],grad_db=False)
        # Attributes are constant semantic one-hots, broadcast across poses.
        attrs = self.mesh.attributes[None].contiguous()
        color, _ = self.dr.interpolate(attrs, rast, faces)
        if self.render_rgb:
            normals,_ = self.dr.interpolate(self.mesh.camera_normals(pose).contiguous(),rast,faces)
            positions,_ = self.dr.interpolate(vertices.contiguous(),rast,faces)
            normals = F.normalize(normals,dim=-1,eps=1e-6)
            view = F.normalize(-positions,dim=-1,eps=1e-6)
            diffuse = (normals*view).sum(-1,keepdim=True).abs().clamp(0,1)
            face_id = (rast[...,3].long()-1).clamp_min(0)
            albedo = self.mesh.face_albedo[face_id]
            # A camera-aligned, two-sided headlight gives smooth shape cues.
            # These are CAD material colors, not recovered endoscopic textures.
            rgb = (albedo*(.35+.65*diffuse)+.12*diffuse.pow(32)).clamp(0,1)
            rgb = rgb*(rast[...,3:] > 0)
            color = torch.cat((color,rgb),-1)
        if torch.is_grad_enabled() and clip.requires_grad and len(vertices) > 1:
            # nvdiffrast 0.4.0 AA backward coalesces by (triangle, edge), without
            # the instance ID. A warp processing two instances can mix their
            # vertex gradients. Keep rasterization/interpolation batched, but
            # isolate differentiable AA per sample. The native library remains
            # unmodified; inference/data generation can use batched AA safely.
            color = torch.cat([self.dr.antialias(color[i:i+1].contiguous(),rast[i:i+1],
                               clip[i:i+1],faces,topology_hash=self.topology_hash)
                               for i in range(len(vertices))],dim=0)
        else:
            color = self.dr.antialias(color.contiguous(),rast,clip,faces,topology_hash=self.topology_hash)
        return color.flip(1).permute(0, 3, 1, 2).clamp(0, 1)

    @torch.no_grad()
    def _hard_visibility(self, vertices, k):
        """Perspective-correct opaque depth/label rasterization; no soft faces."""
        h, w = self.raster_size
        background = self.classes
        tri = vertices[self.mesh.faces]
        attrs = self.mesh.attributes[self.mesh.faces[:, 0]]
        tri, attrs = clip_triangles_near(tri, attrs, self.near)
        if not len(tri):
            return torch.full((h,w),background,device=vertices.device,dtype=torch.long), vertices.new_full((h,w),float("inf"))
        uv = project(tri, k)
        bounds_min, bounds_max = uv.amin(1), uv.amax(1)
        labels = torch.full((h,w),background,device=vertices.device,dtype=torch.long)
        zbuffer = vertices.new_full((h,w),float("inf"))
        part = attrs.argmax(-1)
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                th, tw = min(self.tile_size, h-y), min(self.tile_size, w-x)
                keep = ((bounds_max[:, 0] >= x) & (bounds_min[:, 0] < x+tw) &
                        (bounds_max[:, 1] >= y) & (bounds_min[:, 1] < y+th))
                if not keep.any():
                    continue
                a, b, c = uv[keep].unbind(1)
                yy, xx = torch.meshgrid(torch.arange(y,y+th,device=vertices.device),
                                        torch.arange(x,x+tw,device=vertices.device), indexing="ij")
                pixels = torch.stack((xx, yy), -1).reshape(-1, 2).to(vertices)
                def cross(u, v):
                    return u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]
                area = cross(b-a, c-a)
                valid_area = area.abs() > 1e-8
                safe_area = area.masked_fill(~valid_area,1)
                wb = cross(pixels[None]-a[:, None], (c-a)[:, None]) / safe_area[:, None]
                wc = cross((b-a)[:, None], pixels[None]-a[:, None]) / safe_area[:, None]
                bary = torch.stack((1-wb-wc, wb, wc), -1)
                inside = (bary >= -1e-6).all(-1) & valid_area[:,None]
                depth = 1 / (bary / tri[keep, :, 2][:, None]).sum(-1).clamp_min(1e-6)
                depth = depth.masked_fill(~inside | (depth < self.near) | (depth > self.far),float("inf"))
                front, index = depth.min(0)
                selected = torch.where(torch.isfinite(front),part[keep][index],background)
                labels[y:y+th,x:x+tw] = selected.reshape(th,tw)
                zbuffer[y:y+th,x:x+tw] = front.reshape(th,tw)
        return labels, zbuffer

    def _silhouette_edges(self, vertices):
        with torch.no_grad():
            tri = vertices[self.mesh.faces]
            normal = torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1)
            facing = ((normal*tri[:,0]).sum(-1) >= 0).long().repeat_interleave(3)
            lo = torch.ones(len(self.edges),device=vertices.device,dtype=torch.long)
            hi = torch.zeros_like(lo)
            lo.scatter_reduce_(0,self.edge_inverse,facing,reduce="amin")
            hi.scatter_reduce_(0,self.edge_inverse,facing,reduce="amax")
            keep = (lo != hi) | (self.edge_counts != 2)
        indices = self.edges[keep]
        edges = vertices[indices]
        valid = (edges[...,2] >= self.near).any(-1)
        edges, indices = edges[valid], indices[valid]
        a,b = edges.unbind(1)
        # Clip edges intersecting the near plane while retaining geometric grads.
        crossing = (a[:,2] < self.near) ^ (b[:,2] < self.near)
        denominator = (b[:,2]-a[:,2]).masked_fill(~crossing,1)
        hit = a + ((self.near-a[:,2])/denominator)[:,None]*(b-a)
        a = torch.where((a[:,2] < self.near)[:,None],hit,a)
        b = torch.where((b[:,2] < self.near)[:,None],hit,b)
        return torch.stack((a,b),1), self.mesh.attributes[indices[:,0]].argmax(-1)

    def _torch(self, vertices, k):
        c = self.classes
        labels,zbuffer = self._hard_visibility(vertices,k)
        edges,edge_labels = self._silhouette_edges(vertices)
        h,w = self.raster_size
        hard = F.one_hot(labels,c+1)[...,:c].permute(2,0,1).to(vertices)
        if not len(edges):
            return hard + vertices.sum()*0
        uv = project(edges,k)
        radius = self.edge_width/2  # in HIGH-resolution pixels
        bounds_min = uv.detach().amin(1)-radius
        bounds_max = uv.detach().amax(1)+radius
        palette = torch.cat((torch.eye(c,device=vertices.device),vertices.new_zeros(1,c)),0)
        rows = []
        for y in range(0,h,self.tile_size):
            tiles = []
            for x in range(0,w,self.tile_size):
                th,tw = min(self.tile_size,h-y),min(self.tile_size,w-x)
                keep = ((bounds_max[:,0] >= x)&(bounds_min[:,0] < x+tw)&
                        (bounds_max[:,1] >= y)&(bounds_min[:,1] < y+th))
                original = hard[:,y:y+th,x:x+tw]
                if not keep.any():
                    tiles.append(original+vertices.sum()*0)
                    continue
                a,b = uv[keep].unbind(1)
                direction = b-a
                normal = F.normalize(torch.stack((-direction[:,1],direction[:,0]),-1),dim=-1)
                yy,xx = torch.meshgrid(torch.arange(y,y+th,device=vertices.device),
                                      torch.arange(x,x+tw,device=vertices.device),indexing="ij")
                pixels = torch.stack((xx,yy),-1).reshape(-1,2).to(vertices)
                delta = pixels[None]-a[:,None]
                fraction = ((delta*direction[:,None]).sum(-1)/direction.square().sum(-1)[:,None].clamp_min(1e-12)).clamp(0,1)
                closest = a[:,None]+fraction[...,None]*direction[:,None]
                displacement = pixels[None]-closest
                distance = displacement.square().sum(-1).clamp_min(1e-12).sqrt()
                with torch.no_grad():
                    def sample(offset):
                        location = (closest+offset*normal[:,None]).round().long()
                        sx,sy = location[...,0],location[...,1]
                        inside = (sx >= 0)&(sx < w)&(sy >= 0)&(sy < h)
                        sx,sy = sx.clamp(0,w-1),sy.clamp(0,h-1)
                        return labels[sy,sx].masked_fill(~inside,c),zbuffer[sy,sx].masked_fill(~inside,float("inf"))
                    plus,zplus = sample(max(.75,radius+.25))
                    minus,zminus = sample(-max(.75,radius+.25))
                    own = edge_labels[keep,None]
                    edge_depth = 1/((1-fraction)/edges[keep,0,2,None]+fraction/edges[keep,1,2,None])
                    # Reject rear silhouettes and internal edges with identical
                    # visible semantics on both sides. The tolerance permits the
                    # pixel-centre depth change across a sloped face.
                    own_depth = torch.minimum(zplus.masked_fill(plus != own,float("inf")),
                                              zminus.masked_fill(minus != own,float("inf")))
                    current = labels[y:y+th,x:x+tw].reshape(1,-1)
                    valid = ((plus != minus)&((plus == own)|(minus == own))&
                             (edge_depth <= own_depth+.00075)&
                             ((current == plus)|(current == minus))&(distance <= radius))
                    best_distance,best = distance.masked_fill(~valid,float("inf")).min(0)
                    found = torch.isfinite(best_distance)
                selected_distance = distance.gather(0,best[None])[0]
                signed = (displacement*normal[:,None]).sum(-1).gather(0,best[None])[0]
                signed_distance = torch.where(signed >= 0,selected_distance,-selected_distance)
                weight = (.5+signed_distance/self.edge_width).clamp(0,1)
                plus_label = plus.gather(0,best[None])[0]
                minus_label = minus.gather(0,best[None])[0]
                antialiased = palette[plus_label]*weight[:,None]+palette[minus_label]*(1-weight[:,None])
                color = torch.where(found[:,None],antialiased,original.reshape(c,-1).T)
                tiles.append(color.T.reshape(c,th,tw)+vertices.sum()*0)
            rows.append(torch.cat(tiles,-1))
        return torch.cat(rows,-2).clamp(0,1)
