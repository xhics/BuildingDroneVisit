"""Lot d'images destiné à un modèle de forme, sélectionné sur des preuves.

Les reconstructions existantes du pilote ont tourné sur le corpus brut, avant
que le tri d'identité existe : la meilleure aligne six images pour cinq cents
points. Le tri en retient vingt dont l'identité est établie et la résolution
suffisante — le lot n'est pas plus gros, il est **qualifié**.

Le module ne lance aucun modèle. Il produit la sélection et sa justification,
pour que l'exécution — locale, distante, ou différée — parte de la même base
vérifiable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..logging import get_logger

log = get_logger("shape-input")

#: En deçà, l'image n'apporte pas assez de détail pour contraindre une forme.
MIN_SIDE = 480

#: Deux vues trop proches en azimut n'apportent pas de parallaxe nouvelle.
MIN_ANGULAR_SPACING_DEG = 8.0

#: Un modèle feed-forward sature au-delà : mieux vaut peu d'images bien
#: réparties qu'un lot large et redondant.
MAX_IMAGES = 24


@dataclass
class ShapeImage:
    """Une image retenue, et ce qui justifie sa présence."""

    asset_id: str
    path: Path
    bearing_deg: float | None
    reference_score: float
    min_side: int

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "path": str(self.path),
            "bearing_deg": None if self.bearing_deg is None else round(self.bearing_deg, 1),
            "reference_score": round(self.reference_score, 3),
            "min_side": self.min_side,
        }


@dataclass
class ShapeInput:
    """Le lot soumis au modèle, et ce qu'il couvre."""

    hotel_id: str
    images: list[ShapeImage] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def placed(self) -> list[ShapeImage]:
        return [i for i in self.images if i.bearing_deg is not None]

    def angular_span(self) -> float:
        """Étendue angulaire réellement couverte, en degrés."""
        bearings = sorted(i.bearing_deg for i in self.placed)
        if len(bearings) < 2:
            return 0.0
        gaps = [
            (bearings[(k + 1) % len(bearings)] - bearings[k]) % 360.0
            for k in range(len(bearings))
        ]
        return max(0.0, 360.0 - max(gaps))

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self.images),
            "placed": len(self.placed),
            "angular_span_deg": round(self.angular_span(), 1),
            "rejected": self.rejected,
            "images": [i.as_dict() for i in self.images],
            "caveats": [
                "l'identité de chaque image est établie par le dépistage : une "
                "photographie du bâtiment voisin fausserait la forme reconstruite",
                "une reconstruction feed-forward sort dans un repère arbitraire ; "
                "elle doit être recalée sur l'emprise géoréférencée avant tout "
                "usage métrique",
            ],
        }


def _resolve_bearing(asset_id: str, by_id: dict) -> float | None:
    """Azimut de vue, en résolvant les recadrages vers leur asset source."""
    asset = by_id.get(asset_id)
    if asset is None:
        for token in asset_id.split("_")[1:]:
            if len(token) < 10:
                continue
            for key, candidate in by_id.items():
                if token in key:
                    asset = candidate
                    break
            if asset is not None:
                break
    if asset is None:
        return None
    raw = asset.get("bearing_from_building_deg")
    if raw in (None, "None", ""):
        return None
    try:
        return float(raw) % 360.0
    except (TypeError, ValueError):
        return None


def _portable_image_path(
    raw_path: str,
    asset_id: str,
    *,
    workspace_root: Path,
    by_id: dict,
    by_name: dict[str, Path],
) -> Path:
    """Résout une image après déplacement du workspace vers une autre machine."""
    path = Path(raw_path)
    if path.is_file():
        return path

    parts = path.parts
    if "02_images" in parts:
        relative = Path(*parts[parts.index("02_images") :])
        candidate = workspace_root / relative
        if candidate.is_file():
            return candidate

    asset = by_id.get(asset_id)
    if asset is None:
        for token in asset_id.split("_")[1:]:
            if len(token) < 10:
                continue
            asset = next((item for key, item in by_id.items() if token in key), None)
            if asset is not None:
                break
    if asset and asset.get("local_path"):
        candidate = workspace_root / str(asset["local_path"])
        if candidate.is_file():
            return candidate

    return by_name.get(path.name, path)


def build(
    screening_path: Path,
    asset_manifest_path: Path,
    min_reference_score: float = 0.1,
    max_images: int = MAX_IMAGES,
) -> ShapeInput:
    """Sélectionne les images qui contraindront la forme, et dit ce qui est écarté."""
    from PIL import Image

    screening = json.loads(Path(screening_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(asset_manifest_path).read_text(encoding="utf-8"))
    by_id = {a["id"]: a for a in manifest.get("assets", [])}
    workspace_root = Path(asset_manifest_path).resolve().parent.parent
    images_root = workspace_root / "02_images"
    by_name = {
        path.name: path
        for path in images_root.rglob("*")
        if path.is_file()
    } if images_root.is_dir() else {}

    rejected = {"identite": 0, "score": 0, "resolution": 0, "redondance": 0}
    candidates: list[ShapeImage] = []

    for entry in screening.get("assets", []):
        if entry.get("status") != "match":
            rejected["identite"] += 1
            continue
        score = float(entry.get("reference_score", 0.0))
        if score < min_reference_score:
            rejected["score"] += 1
            continue
        path = _portable_image_path(
            str(entry["path"]),
            str(entry["asset_id"]),
            workspace_root=workspace_root,
            by_id=by_id,
            by_name=by_name,
        )
        try:
            with Image.open(path) as raw:
                side = int(min(raw.size))
        except Exception:
            rejected["resolution"] += 1
            continue
        if side < MIN_SIDE:
            rejected["resolution"] += 1
            continue
        candidates.append(
            ShapeImage(
                asset_id=entry["asset_id"],
                path=path,
                bearing_deg=_resolve_bearing(entry["asset_id"], by_id),
                reference_score=score,
                min_side=side,
            )
        )

    # Les meilleures d'abord, puis on écarte celles qui n'ajoutent pas d'angle :
    # deux vues séparées de trois degrés donnent la même parallaxe, et un
    # modèle feed-forward sature sur un lot redondant.
    candidates.sort(key=lambda i: i.reference_score, reverse=True)
    kept: list[ShapeImage] = []
    for image in candidates:
        if len(kept) >= max_images:
            break
        if image.bearing_deg is None:
            # Une image sans azimut mesuré ne peut ni garantir la répartition
            # angulaire du lot, ni servir d'appui au recalage final : elle est
            # gardée en queue de lot, jamais au détriment d'une vue placée.
            pass
        elif any(
            other.bearing_deg is not None
            and abs((image.bearing_deg - other.bearing_deg + 180) % 360 - 180)
            < MIN_ANGULAR_SPACING_DEG
            for other in kept
        ):
            rejected["redondance"] += 1
            continue
        kept.append(image)

    result = ShapeInput(
        hotel_id=str(manifest.get("hotel_id", "unknown")), images=kept, rejected=rejected
    )
    log.info(
        "lot de forme : %d image(s), %d placée(s), arc %.0f°",
        len(kept),
        len(result.placed),
        result.angular_span(),
    )
    return result
