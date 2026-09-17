"""Mask-derived distal gripper tips. Same geodesic procedure as instrument-tracking."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def _numpy_mask(mask):
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.shape[0] != 3:
        raise ValueError("Expected semantic mask [3,H,W]")
    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255
    return mask


def gripper_scale(mask):
    ys, xs = np.nonzero(_numpy_mask(mask)[2] > .5)
    if not len(xs):
        return np.float32(10.)
    return np.float32(max(10., np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))


def _geodesic(component, wrist_distance):
    y, x = np.nonzero(component)
    ids = np.full(component.shape, -1, np.int32)
    ids[y, x] = np.arange(len(x))
    rows, cols, values = [], [], []
    h, w = component.shape
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        ny, nx = y + dy, x + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        src = np.flatnonzero(inside)
        dst = ids[ny[inside], nx[inside]]
        keep = dst >= 0
        rows.append(src[keep])
        cols.append(dst[keep])
        values.append(np.full(keep.sum(), np.hypot(dy, dx)))
    graph = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(x), len(x)))
    root_distance = wrist_distance[y, x]
    roots = np.flatnonzero(root_distance <= root_distance.min() + 1.)
    distances = dijkstra(graph, indices=roots, min_only=True, directed=False)
    image = np.zeros(component.shape, np.float64)
    image[y, x] = distances
    return image


def extract_gripper_tips(mask, min_area=8):
    """Wrist-rooted geodesic branches; return XY pseudo-tips and confidence."""
    mask = _numpy_mask(mask)
    grip, wrist = mask[2] > .5, mask[1] > .5
    points, confidence = np.zeros((2, 2), np.float32), np.zeros(2, np.float32)
    if not wrist.any() or grip.sum() < min_area:
        return points, confidence
    h, w = grip.shape
    ys, xs = np.nonzero(grip | wrist)
    y0, y1 = max(0, ys.min() - 2), min(h, ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(w, xs.max() + 3)
    grip, wrist = grip[y0:y1, x0:x1], wrist[y0:y1, x0:x1]
    wrist_distance = ndimage.distance_transform_edt(~wrist)
    components, _ = ndimage.label(grip, np.ones((3, 3)))
    sizes = np.bincount(components.ravel())
    candidates = [i for i in range(1, len(sizes)) if sizes[i] >= max(min_area, grip.sum() * .025)]
    candidates.sort(key=lambda i: (wrist_distance[components == i].min(), -sizes[i]))
    endpoints = []
    for cid in candidates[:2]:
        component = components == cid
        if wrist_distance[component].min() > max(4., .12 * np.hypot(*grip.shape)):
            continue
        distance = _geodesic(component, wrist_distance)
        maximum = distance[component].max()
        if maximum < 4 or not np.isfinite(maximum):
            continue
        chosen = None
        for fraction in (.35, .45, .55, .65, .75):
            branches, nb = ndimage.label(component & (distance > fraction * maximum), np.ones((3, 3)))
            parts = []
            for j in range(1, nb + 1):
                coords = np.argwhere(branches == j)
                d = distance[coords[:, 0], coords[:, 1]]
                if len(coords) >= max(3, min_area // 2) and d.max() >= .4 * maximum and np.ptp(d) >= max(2., .12 * d.max()):
                    parts.append(coords)
            if len(parts) == 2:
                chosen = parts
                break
        if chosen is None:
            chosen = [np.argwhere(component)]
        for coords in chosen:
            d = distance[coords[:, 0], coords[:, 1]]
            cap = coords[d >= d.max() - 1.5]
            cap_mask = np.zeros_like(component)
            cap_mask[cap[:, 0], cap[:, 1]] = True
            cap_labels, cap_count = ndimage.label(cap_mask, np.ones((3, 3)))
            if cap_count > 1:
                cap_id = 1 + np.argmax(np.bincount(cap_labels.ravel())[1:])
                cap = np.argwhere(cap_labels == cap_id)
            point = cap.mean(0)[::-1] + np.array([x0, y0])
            clipped = ((cap[:, 0] + y0 <= 1) | (cap[:, 0] + y0 >= h - 2) |
                       (cap[:, 1] + x0 <= 1) | (cap[:, 1] + x0 >= w - 2)).any()
            endpoints.append((point, 0. if clipped else 1., float(d.max())))
    endpoints.sort(key=lambda item: item[2], reverse=True)
    selected = []
    for point, conf, _ in endpoints:
        if all(np.linalg.norm(point - other[0]) > 3 for other in selected):
            selected.append((point, conf))
        if len(selected) == 2:
            break
    for i, (point, conf) in enumerate(selected):
        points[i], confidence[i] = point, conf
    if len(selected) == 1:
        confidence[0] *= .4
    return points, confidence


def split_arm_masks(mask):
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.shape[0] % 3:
        raise ValueError("Expected semantic mask [A*3,H,W]")
    return [mask[3 * a:3 * a + 3] for a in range(mask.shape[0] // 3)]


def extract_arm_tips(mask):
    """Per-instrument mask tips and scales. ``mask`` is [A*3,H,W] in [0,1]."""
    arms = split_arm_masks(mask)
    tips, confidence, scales = [], [], []
    for part in arms:
        xy, conf = extract_gripper_tips(part)
        tips.append(xy)
        confidence.append(conf)
        scales.append(gripper_scale(part))
    return (np.stack(tips).astype(np.float32),
            np.stack(confidence).astype(np.float32),
            np.asarray(scales, np.float32))
