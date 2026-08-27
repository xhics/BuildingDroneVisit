"""Prévisualisation raster reproductible de la grammaire du viewer.

Ce petit rastériseur n'est pas un second viewer : il produit des planches de
contrôle indépendantes du navigateur afin de vérifier les détails procéduraux
sous plusieurs azimuts. Il reprend la même projection et le même tri peintre
que le canvas du livrable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

PALETTE = {
    "background_top": "#152334",
    "background_bottom": "#101722",
    "target": "#70493e",
    "other": "#74818b",
    "roof": "#4f5b56",
    "grass": "#66835c",
    "road": "#424b52",
    "window": "#26363d",
    "door": "#172b34",
    "band": "#3b3534",
    "canopy": "#67463c",
    "pier": "#70493e",
    "gable": "#7d4e40",
    "entrance_tower": "#7d4e40",
    "arched_window": "#26363d",
    "sign": "#20334f",
    "sign_post": "#202a33",
    "veg": "#426d51",
    "pole": "#9aa7b3",
}


def _normalise(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


def _basis(focus: list[float], azimuth_deg: float, altitude_deg: float, distance: float):
    azimuth, altitude = math.radians(azimuth_deg), math.radians(altitude_deg)
    ca, sa = math.cos(altitude), math.sin(altitude)
    eye = [
        focus[0] + math.cos(azimuth) * ca * distance,
        focus[1] + math.sin(azimuth) * ca * distance,
        focus[2] + sa * distance,
    ]
    forward = _normalise([focus[i] - eye[i] for i in range(3)])
    right = _normalise([forward[1], -forward[0], 0.0])
    up = [
        right[1] * forward[2],
        -right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    return eye, forward, right, up


def _faces(payload: dict, *, include_vegetation: bool = False) -> list[dict]:
    faces: list[dict] = []
    textures = payload.get("facade_textures") or []
    for ground in payload.get("ground") or []:
        ring = [[float(point[0]), float(point[1]), 0.0] for point in ground.get("ring") or []]
        if len(ring) >= 3:
            faces.append({"vertices": ring, "kind": "grass" if ground.get("kind") == "vegetal" else "road"})
    for volume in payload.get("volumes") or []:
        footprint = volume.get("fp") or []
        heights = volume.get("wh") or []
        signed_area = 0.5 * sum(
            footprint[index][0] * footprint[(index + 1) % len(footprint)][1]
            - footprint[(index + 1) % len(footprint)][0] * footprint[index][1]
            for index in range(len(footprint))
        )
        side = 1.0 if signed_area >= 0 else -1.0
        for index, point in enumerate(footprint):
            following = (index + 1) % len(footprint)
            neighbour = footprint[following]
            h0 = float(heights[index] if index < len(heights) else volume.get("h") or 8.0)
            h1 = float(heights[following] if following < len(heights) else volume.get("h") or 8.0)
            faces.append({
                "vertices": [[point[0], point[1], 0.0], [neighbour[0], neighbour[1], 0.0], [neighbour[0], neighbour[1], h1], [point[0], point[1], h0]],
                "kind": "target" if volume.get("target") else "other",
                "normal": [side * float(neighbour[1] - point[1]), side * float(point[0] - neighbour[0]), 0.0],
            })
        vertices, triangles = volume.get("rv") or [], volume.get("rf") or []
        for triangle in triangles:
            faces.append({"vertices": [vertices[index] for index in triangle], "kind": "roof"})
    for feature in payload.get("facade_features") or []:
        vertices = feature.get("vertices") or []
        if len(vertices) >= 3:
            faces.append({"vertices": vertices, "kind": str(feature.get("kind") or "target")})
    for texture in textures:
        for triangle in texture.get("render_triangles") or []:
            faces.append({
                "vertices": triangle["vertices"],
                "kind": "target",
                "texture": texture,
                "uv_px": triangle["uv_px"],
            })
    for vegetation in ((payload.get("vegetation") or []) if include_vegetation else []):
        rings = vegetation.get("rings") or []
        for level in range(max(0, len(rings) - 1)):
            lower, upper = rings[level], rings[level + 1]
            count = min(len(lower), len(upper))
            for index in range(count):
                faces.append(
                    {
                        "vertices": [
                            lower[index],
                            lower[(index + 1) % count],
                            upper[(index + 1) % count],
                            upper[index],
                        ],
                        "kind": "veg",
                    }
                )
    for furniture in ((payload.get("furniture") or []) if include_vegetation else []):
        centre = furniture.get("c") or [0.0, 0.0]
        radius = max(float(furniture.get("r") or 0.15), 0.12)
        height_m = float(furniture.get("h") or 3.0)
        x, y = float(centre[0]), float(centre[1])
        faces.append(
            {
                "vertices": [
                    [x - radius, y, 0.0],
                    [x + radius, y, 0.0],
                    [x + radius, y, height_m],
                    [x - radius, y, height_m],
                ],
                "kind": "pole",
            }
        )
    return faces


def render(
    payload: dict,
    *,
    azimuth_deg: float,
    altitude_deg: float = 1.0,
    width: int = 960,
    height: int = 540,
    texture_root: Path | None = None,
) -> Image.Image:
    palette = dict(PALETTE)
    appearance = payload.get("appearance_profile") or {}
    palette.update(
        {
            "target": appearance.get("brick", palette["target"]),
            "entrance_tower": appearance.get("brick", palette["entrance_tower"]),
            "gable": appearance.get("brick", palette["gable"]),
            "pier": appearance.get("brick", palette["pier"]),
            "canopy": appearance.get("brick", palette["canopy"]),
            "roof": appearance.get("roof", palette["roof"]),
            "window": appearance.get("glass", palette["window"]),
            "arched_window": appearance.get("glass", palette["arched_window"]),
        }
    )
    scale = 2
    width_hi, height_hi = width * scale, height * scale
    image = Image.new("RGB", (width_hi, height_hi), PALETTE["background_top"])
    draw = ImageDraw.Draw(image, "RGBA")
    top = tuple(int(PALETTE["background_top"][i : i + 2], 16) for i in (1, 3, 5))
    bottom = tuple(int(PALETTE["background_bottom"][i : i + 2], 16) for i in (1, 3, 5))
    for y in range(height_hi):
        t = y / max(1, height_hi - 1)
        colour = tuple(round(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width_hi, y), fill=colour)

    camera = payload.get("camera") or {}
    focus = [float(value) for value in camera.get("focus") or [0.0, 0.0, 5.0]]
    distance = float(camera.get("target_distance_m") or 120.0)
    eye, forward, right, up = _basis(focus, azimuth_deg, altitude_deg, distance)
    focal = height_hi * 0.5 / math.tan(math.pi / 6)

    def project(point: list[float]):
        delta = [float(point[i]) - eye[i] for i in range(3)]
        depth = sum(delta[i] * forward[i] for i in range(3))
        if depth <= 0.5:
            return None
        x = sum(delta[i] * right[i] for i in range(3))
        y = sum(delta[i] * up[i] for i in range(3))
        return (width_hi * 0.5 + focal * x / depth, height_hi * 0.50 - focal * y / depth, depth)

    projected = []
    for face in _faces(payload):
        polygon = [project(vertex) for vertex in face["vertices"]]
        if any(point is None for point in polygon):
            continue
        projected.append((sum(point[2] for point in polygon) / len(polygon), polygon, face))
    projected.sort(reverse=True, key=lambda item: item[0])
    for _depth, polygon, face in projected:
        kind = face["kind"]
        points = [(round(point[0]), round(point[1])) for point in polygon]
        fill = palette.get(kind, palette["other"])
        outline = "#8aa8b5" if kind == "window" else fill
        polygon_fill = (66, 109, 81, 68) if kind == "veg" else (154, 167, 179, 150) if kind == "pole" else fill
        draw.polygon(points, fill=polygon_fill, outline=outline)
        texture = face.get("texture")
        vertices = face["vertices"]
        centre = [sum(float(vertex[i]) for vertex in vertices) / len(vertices) for i in range(3)]
        normal = face.get("normal")
        facing = normal is None or sum(normal[i] * (eye[i] - centre[i]) for i in range(3)) > 0
        area = abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))) * 0.5
        stable = area >= 25 and all(-width_hi <= x <= width_hi * 2 and -height_hi <= y <= height_hi * 2 for x, y in points)
        if texture and texture_root is not None and len(points) == 3 and facing and stable:
            texture_path = texture_root / texture["path"]
            if texture_path.is_file():
                import cv2
                import numpy as np

                source_rgba = np.asarray(Image.open(texture_path).convert("RGBA"))
                source = source_rgba[:, :, :3]
                src = np.float32(face["uv_px"])
                dst = np.float32(points)
                matrix = cv2.getAffineTransform(src, dst)
                warped = cv2.warpAffine(source, matrix, (width_hi, height_hi))
                mask = cv2.warpAffine(source_rgba[:, :, 3], matrix, (width_hi, height_hi))
                base = np.asarray(image).copy()
                alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
                base = np.clip(base.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                image = Image.fromarray(base, "RGB")
                draw = ImageDraw.Draw(image, "RGBA")

    return image.resize((width, height), Image.Resampling.LANCZOS)


def render_orbit(payload: dict, output_dir: Path, *, texture_root: Path | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    azimuths = [150, 180, 210, 240, 270, 300]
    outputs = []
    frames = []
    for azimuth in azimuths:
        frame = render(payload, azimuth_deg=azimuth, texture_root=texture_root)
        path = output_dir / f"facade_{azimuth:03d}.png"
        frame.save(path)
        outputs.append(path)
        frames.append(frame.resize((480, 270), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (1440, 540), "#101722")
    draw = ImageDraw.Draw(sheet)
    for slot, (azimuth, frame) in enumerate(zip(azimuths, frames)):
        x, y = (slot % 3) * 480, (slot // 3) * 270
        sheet.paste(frame, (x, y))
        draw.rectangle((x + 8, y + 8, x + 74, y + 30), fill="#101822")
        draw.text((x + 15, y + 12), f"{azimuth} deg", fill="white")
    sheet_path = output_dir / "facade_orbit_sheet.png"
    sheet.save(sheet_path)
    outputs.append(sheet_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text("utf-8"))
    for path in render_orbit(payload, args.output_dir, texture_root=args.payload.parent):
        print(path)


if __name__ == "__main__":
    main()
