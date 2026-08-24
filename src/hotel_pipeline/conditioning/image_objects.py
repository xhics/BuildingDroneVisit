"""Candidats architecturaux 2D détectés hors ligne dans les photographies.
Ce détecteur ne prétend pas remplacer un modèle sémantique. Il extrait les
formes que les pixels établissent déjà : quadrilatères rectangulaires pouvant
correspondre à des panneaux ou ouvertures, et membres linéaires longs pouvant
correspondre à des poutres, bordures ou montants. Le registre les conserve
comme candidats `UNKNOWN` jusqu'à validation sémantique et multivue.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


def _image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)[:120]


def detect(path: Path) -> list[dict]:
    """Retourne un petit ensemble classé de formes structurelles 2D."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    height, width = image.shape[:2]
    scale = min(1.0, 1280.0 / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    work_h, work_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 55, 150)

    candidates: list[dict] = []
    image_area = float(work_h * work_w)
    contours, _hierarchy = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    panels: list[dict] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < min(work_w, work_h) * 0.05:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, w, h = cv2.boundingRect(polygon)
        area = abs(float(cv2.contourArea(polygon)))
        fraction = area / image_area
        if not 0.0007 <= fraction <= 0.18 or min(w, h) < 10:
            continue
        rectangularity = area / max(float(w * h), 1.0)
        aspect = max(w, h) / max(min(w, h), 1)
        if rectangularity < 0.68 or aspect > 8.0:
            continue
        points = polygon.reshape(-1, 2).astype(float) / scale
        panels.append(
            {
                "class": "panel_or_opening_candidate",
                "geometry_2d": {
                    "type": "polygon",
                    "points": np.round(points, 1).tolist(),
                },
                "score": round(min(1.0, rectangularity * (1.0 - abs(aspect - 2.0) / 12.0)), 3),
                "area_fraction": round(fraction, 5),
            }
        )
    panels.sort(key=lambda item: (item["score"], item["area_fraction"]), reverse=True)
    candidates.extend(panels[:8])

    minimum = max(24, round(min(work_w, work_h) * 0.08))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(30, minimum // 2),
        minLineLength=minimum,
        maxLineGap=max(8, minimum // 5),
    )
    members: list[dict] = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = (float(value) for value in raw)
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
            axis_gap = min(angle, abs(90.0 - angle), abs(180.0 - angle))
            # Les structures bâties dominantes sont horizontales/verticales.
            # Les diagonales restent portées par le détecteur de toiture.
            if axis_gap > 12.0:
                continue
            members.append(
                {
                    "class": "linear_member_candidate",
                    "geometry_2d": {
                        "type": "segment",
                        "xyxy": [
                            round(x1 / scale, 1),
                            round(y1 / scale, 1),
                            round(x2 / scale, 1),
                            round(y2 / scale, 1),
                        ],
                    },
                    "score": round(min(1.0, length / max(work_w, work_h)), 3),
                    "angle_deg": round(angle, 1),
                }
            )
    members.sort(key=lambda item: item["score"], reverse=True)
    candidates.extend(members[:12])
    return candidates


def detect_cached(path: Path, asset_id: str, cache_dir: Path) -> list[dict]:
    """Mémorise le résultat par empreinte pour garder le build reproductible."""
    digest = _image_digest(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{_safe_id(asset_id)}.json"
    if cache.is_file():
        payload = json.loads(cache.read_text("utf-8"))
        if payload.get("image_sha256") == digest:
            return list(payload.get("candidates", []))
    candidates = detect(path)
    cache.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "asset_id": asset_id,
                "image_sha256": digest,
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        "utf-8",
    )
    return candidates
