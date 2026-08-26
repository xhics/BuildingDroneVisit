"""Dessine les figures de vol stylisées par-dessus l'image satellite."""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from .maneuvers import Maneuver, Waypoint

_LEGEND_MARGIN = 16
_LEGEND_LINE_H = 22


def _pixel(w: Waypoint, cx: float, cy: float, mpp: float) -> tuple[float, float]:
    # Nord vers le haut de l'image : l'axe nord (m) et l'axe y (px) sont opposés.
    return cx + w.east_m / mpp, cy - w.north_m / mpp


def _draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    p_from: tuple[float, float],
    p_to: tuple[float, float],
    color: tuple[int, int, int, int],
    size: float = 9.0,
) -> None:
    angle = math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])
    left = (
        p_to[0] - size * math.cos(angle - math.radians(28)),
        p_to[1] - size * math.sin(angle - math.radians(28)),
    )
    right = (
        p_to[0] - size * math.cos(angle + math.radians(28)),
        p_to[1] - size * math.sin(angle + math.radians(28)),
    )
    draw.polygon([p_to, left, right], fill=color)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _dot(p: tuple[float, float], r: float) -> list[float]:
    return [p[0] - r, p[1] - r, p[0] + r, p[1] + r]


def render(image: Image.Image, mpp: float, maneuvers: list[Maneuver], *, title: str) -> Image.Image:
    """Renvoie une copie de ``image`` annotée des trajectoires et d'une légende."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    cx, cy = canvas.width / 2.0, canvas.height / 2.0

    # Repère de l'adresse géocodée, au centre de l'image.
    draw.ellipse(_dot((cx, cy), 5), outline=(255, 255, 255, 230), width=2)

    for maneuver in maneuvers:
        pixels = [_pixel(w, cx, cy, mpp) for w in maneuver.waypoints]
        color = (*maneuver.color, 235)
        draw.line(pixels, fill=color, width=4, joint="curve")

        draw.ellipse(_dot(pixels[0], 5), fill=(255, 255, 255, 255), outline=color, width=2)
        draw.ellipse(_dot(pixels[-1], 5), fill=color, outline=(255, 255, 255, 255), width=2)

        step = max(1, len(pixels) // 6)
        for i in range(step, len(pixels) - 1, step):
            _draw_arrowhead(draw, pixels[i - 1], pixels[i], color)

    _draw_scale_bar(draw, canvas.width, canvas.height, mpp)
    _draw_north_arrow(draw, canvas.width)
    _draw_legend(draw, maneuvers)
    _draw_title(draw, canvas.width, title)
    return canvas


def _draw_scale_bar(draw: ImageDraw.ImageDraw, width: int, height: int, mpp: float) -> None:
    bar_m = 20.0
    bar_px = bar_m / mpp
    x0, y0 = _LEGEND_MARGIN, height - 28
    draw.line([(x0, y0), (x0 + bar_px, y0)], fill=(255, 255, 255, 255), width=3)
    draw.text((x0, y0 - 16), f"{bar_m:.0f} m", fill=(255, 255, 255, 255), font=_font(13))


def _draw_north_arrow(draw: ImageDraw.ImageDraw, width: int) -> None:
    x, y = width - 30, 34
    draw.line([(x, y + 20), (x, y - 12)], fill=(255, 255, 255, 255), width=3)
    draw.polygon([(x, y - 20), (x - 6, y - 8), (x + 6, y - 8)], fill=(255, 255, 255, 255))
    draw.text((x - 4, y - 36), "N", fill=(255, 255, 255, 255), font=_font(14))


def _draw_legend(draw: ImageDraw.ImageDraw, maneuvers: list[Maneuver]) -> None:
    font = _font(13)
    x, y = _LEGEND_MARGIN, _LEGEND_MARGIN
    draw.rectangle(
        [x - 8, y - 8, x + 250, y + _LEGEND_LINE_H * len(maneuvers) + 4],
        fill=(0, 0, 0, 120),
    )
    for i, maneuver in enumerate(maneuvers):
        row_y = y + i * _LEGEND_LINE_H
        draw.rectangle([x, row_y + 4, x + 16, row_y + 16], fill=(*maneuver.color, 255))
        draw.text((x + 24, row_y), f"{i + 1}. {maneuver.name_fr}", fill=(255, 255, 255, 255), font=font)


def _draw_title(draw: ImageDraw.ImageDraw, width: int, title: str) -> None:
    font = _font(14)
    draw.text((width / 2, 8), title, fill=(255, 255, 255, 255), font=font, anchor="ma")


__all__ = ["render"]
