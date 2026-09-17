"""OBJ/MTL material loading for differentiable shaded RGB (no invented texture)."""
from pathlib import Path
import hashlib
import warnings
import numpy as np


def load_material_mesh(path):
    import trimesh
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    libraries = [line.split(maxsplit=1)[1] for line in lines if line.startswith("mtllib ")]
    materials, hashes = {}, {}
    for name in libraries:
        material_path = path.parent/name
        if not material_path.exists():
            material_path = path.with_suffix(".mtl")
            if not material_path.exists():
                raise FileNotFoundError(f"Missing MTL: {path.parent/name}")
            warnings.warn(f"{path.name}: using {material_path.name} for missing {name}",stacklevel=2)
        hashes[material_path.name] = hashlib.sha256(material_path.read_bytes()).hexdigest()
        current = None
        for line in material_path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "newmtl":
                current = fields[1]
                materials[current] = [.7,.7,.7]
            elif fields[0] == "Kd" and current is not None:
                materials[current] = [float(v) for v in fields[1:4]]
            elif fields[0] == "map_Kd":
                raise ValueError("This RGB mode uses MTL diffuse colors and smooth normals. "
                                 "A map_Kd texture needs a UV texture shader; it must not be silently ignored.")
    vertices,faces,colors = [],[],[]
    material = None
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "usemtl":
            material = fields[1]
        elif fields[0] == "f":
            indices = [int(f.split("/")[0]) for f in fields[1:]]
            indices = [i-1 if i > 0 else len(vertices)+i for i in indices]
            if material not in materials:
                raise ValueError(f"{path}: missing diffuse material {material!r}")
            for i in range(1,len(indices)-1):
                faces.append([indices[0],indices[i],indices[i+1]])
                colors.append(materials[material])
    colors = np.asarray(colors,dtype=np.float32).clip(0,1)
    # Trimesh propagates face colors through face cleanup; the renderer samples
    # face colors directly so material boundaries never smear across vertices.
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),
                           face_colors=np.round(colors*255).astype(np.uint8),process=False)
    return mesh,hashes
