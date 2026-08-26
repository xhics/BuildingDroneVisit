"""Physical geometry contracts for facade openings and fine architecture."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Opening:
    opening_id: str
    contour_uz_m: tuple[tuple[float, float], ...]
    reveal_depth_m: float | None
    material: str
    provenance: str

    def contains(self, u_m: float, z_m: float) -> bool:
        inside = False
        points = self.contour_uz_m
        for i, (x1, y1) in enumerate(points):
            x0, y0 = points[i - 1]
            if (y1 > z_m) != (y0 > z_m):
                crossing = (x0 - x1) * (z_m - y1) / (y0 - y1) + x1
                if u_m < crossing:
                    inside = not inside
        return inside


def wall_hit_is_solid(u_m: float, z_m: float, openings: list[Opening]) -> bool:
    """A wall ray hit is empty exactly where a topological opening exists."""
    return not any(opening.contains(u_m, z_m) for opening in openings)


def box_mesh(center: tuple[float, float, float], size: tuple[float, float, float]) -> dict:
    cx, cy, cz = center; sx, sy, sz = (v * 0.5 for v in size)
    vertices = [[cx+x, cy+y, cz+z] for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)]
    faces = [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]
    return {"vertices": vertices, "faces": faces}


def cylinder_mesh(center: tuple[float, float, float], radius_m: float, height_m: float, segments: int = 16) -> dict:
    cx, cy, cz = center
    vertices = []
    for z in (cz - height_m / 2, cz + height_m / 2):
        vertices.extend([[cx + radius_m * math.cos(2*math.pi*i/segments), cy + radius_m * math.sin(2*math.pi*i/segments), z] for i in range(segments)])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments+j, segments+i])
    faces.extend([list(reversed(range(segments))), list(range(segments, 2*segments))])
    return {"vertices": vertices, "faces": faces}


def primitive_mesh(kind: str, **parameters) -> dict:
    """Class-specific solid primitives used by collision and rendering."""
    if kind in {"balcony", "canopy", "beam"}:
        return box_mesh(parameters["center"], parameters["size"])
    if kind == "column":
        return cylinder_mesh(parameters["center"], parameters["radius_m"], parameters["height_m"], parameters.get("segments", 16))
    raise ValueError(f"unsupported architectural primitive: {kind}")


def tube_along_polyline(points: list[tuple[float, float, float]], radius_m: float, segments: int = 10) -> dict:
    """Tube whose ring centres remain exactly on a measured roof polyline."""
    if len(points) < 2:
        raise ValueError("a gutter requires at least two roof-edge points")
    centres = np.asarray(points, float)
    vertices = []
    for i, centre in enumerate(centres):
        tangent = centres[min(i + 1, len(centres)-1)] - centres[max(i - 1, 0)]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        reference = np.array([0., 0., 1.]) if abs(tangent[2]) < .9 else np.array([0., 1., 0.])
        axis1 = np.cross(tangent, reference); axis1 /= np.linalg.norm(axis1)
        axis2 = np.cross(tangent, axis1)
        for j in range(segments):
            angle = 2 * math.pi * j / segments
            vertices.append((centre + radius_m * (math.cos(angle)*axis1 + math.sin(angle)*axis2)).tolist())
    faces = []
    for ring in range(len(centres)-1):
        for j in range(segments):
            k = (j+1) % segments; a = ring*segments; b = (ring+1)*segments
            faces.append([a+j, a+k, b+k, b+j])
    return {"vertices": vertices, "faces": faces, "centreline": centres.tolist(), "kind": "gutter_tube"}


def railing_mesh(start: tuple[float,float,float], end: tuple[float,float,float], height_m: float, spacing_m: float = .12, bar_m: float = .025) -> dict:
    """Open railing made of thin bars, never an opaque plane."""
    start, end = np.asarray(start,float), np.asarray(end,float)
    length = float(np.linalg.norm(end-start)); count = max(2, int(math.ceil(length/spacing_m))+1)
    vertices, faces = [], []
    for t in np.linspace(0,1,count):
        p = start + t*(end-start)
        mesh = box_mesh((p[0],p[1],p[2]+height_m/2), (bar_m,bar_m,height_m))
        offset=len(vertices); vertices.extend(mesh["vertices"]); faces.extend([[offset+i for i in f] for f in mesh["faces"]])
    return {"vertices": vertices, "faces": faces, "coverage": min(1.0, count*bar_m/max(length,1e-9)), "kind": "open_railing"}


def classify_sign(depth_offset_m: float, threshold_m: float = .08) -> str:
    return "surface_sign" if abs(depth_offset_m) <= threshold_m else "projecting_sign"


__all__ = ["Opening", "box_mesh", "classify_sign", "cylinder_mesh", "primitive_mesh", "railing_mesh", "tube_along_polyline", "wall_hit_is_solid"]
