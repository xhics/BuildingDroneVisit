"""Établir l'orientation d'un bâtiment depuis ses façades (collecte V2).

L'azimut avant venait du centroïde d'un stationnement supposé — association
fondée sur la proximité, démentie par l'inspection : 137,7° pour une façade qui
en vaut 227. Une erreur de quatre-vingt-dix degrés, invisible tant que rien ne
confrontait l'hypothèse à une image.

Ce module tire l'orientation de ce qui la porte réellement : les **segments
d'empreinte** que des vues documentent. Une photo ne dit pas où se trouve une
façade — elle confirme que le segment atteint par son rayon porte bien l'entrée
ou la façade principale. La position vient de la géométrie, la confirmation de
l'image.

```text
rayon caméra → bâtiment   quel segment est vu
normale extérieure        où ce segment regarde
regroupement colinéaire   quel mur ces segments composent
photo                     ce mur porte-t-il l'entrée ?
```

Un mur réel est découpé par les décrochements du relevé : le segment 3 vaut
226,0°, le 5 vaut 229,0°, le 7 vaut 227,7°, le 10 vaut 228,4°. Les traiter
séparément ferait dépendre l'orientation de celui qu'un rayon touche en
premier, non de la façade qu'il documente.

**Aucune moyenne en cas de contradiction.** Deux preuves qui désignent des murs
opposés ne se concilient pas en prenant le milieu : elles disent qu'on ne sait
pas, et l'orientation reste `unresolved`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .logging import get_logger

log = get_logger("orientation")


class OrientationUndetermined(RuntimeError):
    """Les preuves ne convergent pas : rien n'est décidé."""


@dataclass(frozen=True)
class Segment:
    """Un segment d'empreinte et la direction où il regarde."""

    index: int
    length_m: float
    outward_normal_deg: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "length_m": round(self.length_m, 2),
            "outward_normal_deg": round(self.outward_normal_deg, 2),
        }


@dataclass
class FacadeGroup:
    """Des segments colinéaires : un mur, tel que le relevé l'a découpé."""

    segments: list[Segment] = field(default_factory=list)

    @property
    def total_length_m(self) -> float:
        return sum(segment.length_m for segment in self.segments)

    @property
    def normal_deg(self) -> float:
        """Normale commune, **pondérée par la longueur**.

        Un décrochement de deux mètres ne pèse pas autant qu'un mur de
        quarante : la moyenne simple laisserait un détail du relevé déplacer
        l'orientation de la façade.
        """
        x = sum(
            math.cos(math.radians(s.outward_normal_deg)) * s.length_m
            for s in self.segments
        )
        y = sum(
            math.sin(math.radians(s.outward_normal_deg)) * s.length_m
            for s in self.segments
        )
        return math.degrees(math.atan2(y, x)) % 360.0

    def as_dict(self) -> dict:
        return {
            "normal_deg": round(self.normal_deg, 2),
            "total_length_m": round(self.total_length_m, 2),
            "segments": [s.as_dict() for s in self.segments],
        }


@dataclass
class OrientationEvidence:
    """Une vue et ce qu'elle établit du bâtiment."""

    asset_id: str
    checksum: str
    camera_lat: float
    camera_lon: float
    segment_index: int
    segment_normal_deg: float

    #: Ce que l'image **montre**, en clair. La géométrie donne la position ; la
    #: photo dit si ce mur porte l'entrée.
    observation: str = ""

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "sha256": self.checksum,
            "camera_lat": self.camera_lat,
            "camera_lon": self.camera_lon,
            "segment_index": self.segment_index,
            "segment_normal_deg": round(self.segment_normal_deg, 2),
            "observation": self.observation,
        }


@dataclass
class OrientationDecision:
    """Ce qui a été décidé, et de quoi le vérifier.

    Append-only : une orientation révisée laisse la précédente lisible. Sur ce
    site, la première valait 137,7° et venait d'un stationnement dont
    l'association a été démentie — l'effacer rendrait cette erreur invisible.
    """

    hotel_id: str
    building_digest: str
    front_azimuth_deg: float
    method: str

    groups: list[FacadeGroup] = field(default_factory=list)
    evidence: list[OrientationEvidence] = field(default_factory=list)

    decided_by: str = ""
    rationale: str = ""
    decided_at: str = ""

    #: Ce que la décision **ne** prétend pas établir.
    limits: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "building_digest": self.building_digest,
            "front_azimuth_deg": round(self.front_azimuth_deg, 2),
            "method": self.method,
            "facade_groups": [g.as_dict() for g in self.groups],
            "evidence": [e.as_dict() for e in self.evidence],
            "decided_by": self.decided_by,
            "rationale": self.rationale,
            "decided_at": self.decided_at,
            "limits": self.limits or [
                "la photo confirme ce que le segment porte ; elle ne le "
                "localise pas — la position vient de l'empreinte",
                "une voie d'accès ou un stationnement peuvent corroborer, "
                "jamais décider seuls",
            ],
        }


def outward_normal(polygon, first, second) -> float | None:  # noqa: ANN001
    """Azimut de la normale **extérieure** d'un segment.

    Le côté se détermine en sortant du segment : celui des deux qui n'est pas
    dans le polygone. Choisir par convention d'orientation des sommets
    dépendrait du sens dans lequel la source a écrit l'anneau.
    """
    from shapely.geometry import Point

    dx, dy = second[0] - first[0], second[1] - first[1]
    middle = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)

    for nx, ny in ((dy, -dx), (-dy, dx)):
        norm = math.hypot(nx, ny)
        if norm == 0:
            return None
        probe = Point(middle[0] + nx / norm * 0.05, middle[1] + ny / norm * 0.05)
        if not polygon.contains(probe):
            return math.degrees(math.atan2(nx, ny)) % 360.0
    return None


def segments_of(polygon) -> list[Segment]:  # noqa: ANN001
    """Tous les segments de l'anneau extérieur, avec leur normale."""
    from shapely.geometry import LineString

    coords = list(polygon.exterior.coords)
    found: list[Segment] = []
    for index in range(len(coords) - 1):
        normal = outward_normal(polygon, coords[index], coords[index + 1])
        if normal is None:
            continue
        found.append(
            Segment(
                index=index,
                length_m=LineString([coords[index], coords[index + 1]]).length,
                outward_normal_deg=normal,
            )
        )
    return found


def group_collinear(segments: list[Segment], tolerance_deg: float) -> list[FacadeGroup]:
    """Réunit les segments dont les normales se ressemblent.

    Déterministe : les segments sont parcourus dans l'ordre de l'anneau, et
    chacun rejoint le premier groupe compatible. Deux exécutions rendent les
    mêmes murs.
    """
    groups: list[FacadeGroup] = []
    for segment in segments:
        for group in groups:
            gap = abs(
                (segment.outward_normal_deg - group.normal_deg + 180.0) % 360.0 - 180.0
            )
            if gap <= tolerance_deg:
                group.segments.append(segment)
                break
        else:
            groups.append(FacadeGroup(segments=[segment]))
    return groups


def segment_seen_from(polygon, camera, segments: list[Segment]) -> Segment | None:  # noqa: ANN001
    """Segment que la caméra regarde **de face**.

    Le rayon vers le centroïde perce d'abord ce qui se trouve sur son chemin —
    souvent un décrochement latéral, dont la normale n'a rien à voir avec le
    mur observé. Sur ce site, la même façade était ainsi lue à 227,7° depuis une
    vue et 139,3° depuis l'autre, alors que les deux la regardent.

    On retient donc le segment dont la normale pointe le plus directement vers
    la caméra : c'est celui qui lui fait face, donc celui que la photo montre.
    Les égalités se départagent par la distance, puis par l'indice — deux
    exécutions rendent le même segment.
    """
    from shapely.geometry import LineString, Point

    coords = list(polygon.exterior.coords)
    origin = Point(camera)

    facing: list[tuple[float, float, int, Segment]] = []
    for segment in segments:
        edge = LineString([coords[segment.index], coords[segment.index + 1]])
        middle = edge.interpolate(0.5, normalized=True)
        towards_camera = math.degrees(
            math.atan2(origin.x - middle.x, origin.y - middle.y)
        ) % 360.0
        gap = abs(
            (towards_camera - segment.outward_normal_deg + 180.0) % 360.0 - 180.0
        )
        # Au-delà d'un quart de tour, le segment tourne le dos à la caméra :
        # il ne peut pas être ce qu'elle photographie.
        if gap >= 90.0:
            continue
        facing.append((gap, origin.distance(middle), segment.index, segment))

    if not facing:
        return None
    return min(facing, key=lambda row: (row[0], row[1], row[2]))[3]


def decide(
    hotel_id: str,
    building_digest: str,
    polygon,  # noqa: ANN001
    evidence: list[OrientationEvidence],
    tolerance_deg: float,
    decided_by: str,
    rationale: str,
) -> OrientationDecision:
    """Arrête l'orientation, ou refuse de trancher.

    Les preuves doivent désigner **le même mur**. Deux vues qui pointent des
    façades opposées ne se concilient pas par une moyenne : elles établissent
    qu'on ne sait pas.
    """
    if not evidence:
        raise OrientationUndetermined(
            "aucune preuve : une orientation sans vue qui la porte n'est qu'une "
            "hypothèse de plus"
        )

    segments = segments_of(polygon)
    groups = group_collinear(segments, tolerance_deg)

    seen: list[FacadeGroup] = []
    for item in evidence:
        for group in groups:
            if any(s.index == item.segment_index for s in group.segments):
                if group not in seen:
                    seen.append(group)
                break

    if len(seen) != 1:
        normales = sorted(round(g.normal_deg, 1) for g in seen)
        raise OrientationUndetermined(
            f"les preuves désignent {len(seen)} murs différents ({normales}) : "
            "elles ne se concilient pas par une moyenne, et l'orientation reste "
            "indéterminée"
        )

    facade = seen[0]
    decision = OrientationDecision(
        hotel_id=hotel_id,
        building_digest=building_digest,
        front_azimuth_deg=facade.normal_deg,
        method="facade_segments_confirmed_by_imagery",
        groups=groups,
        evidence=evidence,
        decided_by=decided_by,
        rationale=rationale,
        decided_at=datetime.now(timezone.utc).isoformat(),
    )
    log.info(
        "orientation : %.1f° depuis %d segment(s) sur %.1f m, confirmée par "
        "%d vue(s)",
        decision.front_azimuth_deg, len(facade.segments),
        facade.total_length_m, len(evidence),
    )
    return decision


def facade_for(group: "FacadeGroup", front_azimuth_deg: float) -> str | None:
    """Quel côté du bâtiment ce mur constitue, l'avant étant connu.

    Le mur qui regarde dans la direction de l'avant **est** la façade avant :
    c'est la définition même de l'azimut. Les trois autres s'en déduisent par
    quarts de tour, et un mur oblique n'en est aucune — le forcer dans le
    quadrant le plus proche inventerait une façade là où le bâtiment n'en a
    qu'un pan coupé.
    """
    # Les quatre côtés **cardinaux** seulement : `SECTOR_CENTRES` porte aussi
    # les coins, et parcourir la liste entière faisait retenir
    # `front_right_corner` à 45° avant `right` à 90° — deux façades sur quatre
    # restaient alors sans nom.
    cotes = (
        (0.0, "FACADE_PRIMARY"),
        (90.0, "FACADE_RIGHT"),
        (180.0, "FACADE_REAR"),
        (270.0, "FACADE_LEFT"),
    )
    offset = (group.normal_deg - front_azimuth_deg) % 360.0
    # Les quatre quadrants sont **jointifs** : tout mur reçoit un côté, celui
    # dont la direction est la plus proche. Un bâtiment n'a pas de mur qui ne
    # regarde nulle part, et prévoir un cas « aucun côté » créerait une
    # branche que rien n'atteindrait.
    nom, _ecart = min(
        ((nom, abs((offset - centre + 180.0) % 360.0 - 180.0)) for centre, nom in cotes),
        key=lambda row: row[1],
    )
    return nom


def facades_from(polygon, front_azimuth_deg: float, tolerance_deg: float) -> dict:  # noqa: ANN001
    """Découpe l'empreinte en façades nommées, une fois l'avant établi.

    Chaque côté retient le **mur le plus long** de son quadrant : un bâtiment
    présente souvent plusieurs pans vers la même direction, et prendre le
    premier venu ferait dépendre la façade de l'ordre des sommets.
    """
    from shapely.geometry import LineString, MultiLineString

    coords = list(polygon.exterior.coords)
    walls: dict[str, FacadeGroup] = {}

    for group in group_collinear(segments_of(polygon), tolerance_deg):
        name = facade_for(group, front_azimuth_deg)
        if name is None:
            continue
        if name not in walls or group.total_length_m > walls[name].total_length_m:
            walls[name] = group

    shapes: dict[str, dict] = {}
    for name, group in walls.items():
        parts = [
            LineString([coords[s.index], coords[s.index + 1]])
            for s in sorted(group.segments, key=lambda s: s.index)
        ]
        shapes[name] = {
            "normal_deg": round(group.normal_deg, 2),
            "length_m": round(group.total_length_m, 2),
            "segments": [s.index for s in group.segments],
            "wkt": (
                MultiLineString([list(part.coords) for part in parts]).wkt
                if len(parts) > 1 else parts[0].wkt
            ),
        }
    return shapes
