"""Le maillage canonique : une seule géométrie, consommée par tous.

Historiquement, chaque consommateur reconstruisait ses propres triangles :
le rendeur extrudait l'emprise, le textureur relisait le payload et
ré-extrudait à sa manière (`_build_triangles_from_payload`), la collision et
l'export en faisaient autant. Résultat : plusieurs versions différentes du
même bâtiment dans le même pipeline, dont aucune n'était fausse isolément,
mais aucune n'était la même.

Ce module établit l'autorité unique : `CanonicalSceneMesh`. Le viewer, le
z-buffer, les textures, la collision, le raycast et l'export consomment tous
exactement cette instance — et `mesh_digest()` permet de le prouver : le même
digest partout, sinon le pipeline est cassé.

Deux garanties structurelles accompagnent le maillage :

- chaque sommet d'emprise porte une identité stable (`FootprintVertex` :
  ``vertex_id``, ``x``, ``y``, ``ground_z``, ``top_z``). Inverser l'ordre du
  contour CW↔CCW inverse le **record entier**, pas seulement XY : les hauteurs
  restent attachées au bon point du sol. Si une réparation géométrique
  (``buffer(0)``) déplace ou duplique des sommets, la correspondance est
  refaite **géométriquement** (plus proche voisin), jamais par indice.
- le pied de chaque mur suit le terrain : ``ground_z = terrain.height_at(x,y)``
  sommet par sommet. Le niveau structurel du plancher reste un attribut
  distinct (``floor_z``), pour ne pas confondre sol extérieur et dallage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate

from ..schemas.canonical_states import MeasurementState

#: Tolérance de soudure des sommets, en mètres. Assez petite pour ne jamais
#: fusionner deux sommets réellement distincts à la précision des sources,
#: assez grande pour rapprocher mur et toiture construits séparément.
WELD_TOLERANCE_M = 0.02

#: Quantification du digest : un millimètre. Deux maillages qui diffèrent
#: sous cette résolution sont la même géométrie pour tous les consommateurs.
DIGEST_QUANTUM_M = 1e-3


@dataclass
class FootprintVertex:
    """Un sommet d'emprise et ses altitudes, inséparables.

    L'identité stable (``vertex_id``) survit à la réorientation du contour et
    aux réparations géométriques : ce qui identifie le sommet, c'est sa place
    mesurée sur le bâtiment, pas sa position dans une liste.
    """

    vertex_id: str
    x: float
    y: float
    ground_z: float = 0.0
    top_z: float = 0.0
    #: Niveau structurel du plancher, distinct du sol extérieur.
    floor_z: float | None = None

    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


def _drop_closing_point(ring: np.ndarray) -> np.ndarray:
    ring = np.asarray(ring, dtype=np.float64)
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    return ring


def footprint_records(
    footprint: np.ndarray,
    top_heights: np.ndarray | float | None = None,
    bottom_heights: np.ndarray | float | None = None,
) -> list[FootprintVertex]:
    """Normalise une emprise en sommets identifiés, sans perdre les hauteurs.

    L'ordre du contour est ramené au sens trigonométrique ; si le contour
    fourni était horaire, on inverse le **record entier** — x, y mais aussi
    ground_z et top_z, qui restent attachés à leur point. Une emprise
    invalide réparée par ``buffer(0)`` voit ses sommets rematchés par plus
    proche voisin géométrique : les indices d'origine ne signifient plus rien
    après réparation, les distances si.
    """
    ring = _drop_closing_point(footprint)
    if len(ring) < 3:
        raise ValueError("une emprise bâtie requiert au moins trois sommets")

    tops = None
    bottoms = None
    if top_heights is not None:
        tops = np.broadcast_to(
            np.asarray(top_heights, dtype=np.float64), (len(ring),)
        ).copy()
    if bottom_heights is not None:
        bottoms = np.broadcast_to(
            np.asarray(bottom_heights, dtype=np.float64), (len(ring),)
        ).copy()

    polygon = Polygon(ring)
    repaired = False
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        repaired = True
    if polygon.is_empty or polygon.geom_type != "Polygon":
        raise ValueError("emprise invalide : impossible de fermer le bâtiment")
    exterior = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)

    # Correspondance géométrique : chaque sommet réparé hérite des hauteurs
    # du sommet d'origine le plus proche. Jamais par indice : buffer(0) peut
    # réordonner, fusionner ou déplacer les sommets.
    def _nearest(source: np.ndarray) -> np.ndarray:
        distances = np.hypot(
            source[:, None, 0] - ring[None, :, 0],
            source[:, None, 1] - ring[None, :, 1],
        )
        return distances.argmin(axis=1)

    owner = _nearest(exterior) if repaired else np.arange(len(ring))

    records = [
        FootprintVertex(
            vertex_id=f"fpv{index}",
            x=float(point[0]),
            y=float(point[1]),
            ground_z=float(bottoms[owner[index]]) if bottoms is not None else 0.0,
            top_z=float(tops[owner[index]]) if tops is not None else 0.0,
        )
        for index, point in enumerate(exterior)
    ]

    # Orientation canonique CCW. Inverser le contour inverse le record
    # entier : les hauteurs suivent leur sommet, jamais la position.
    if not polygon.exterior.is_ccw:
        records.reverse()
    return records


@dataclass(frozen=True)
class SolidMesh:
    """Représentation minimale conservée pour compatibilité historique."""

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class TriangleRecord:
    triangle_id: int
    surface_id: str
    vertices: tuple[int, int, int]
    measurement_state: MeasurementState
    confidence: float
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class RayHit:
    triangle_id: int
    surface_id: str
    distance: float


@dataclass(frozen=True)
class SurfaceRecord:
    surface_id: str
    building_id: str
    part_id: str
    kind: str
    triangle_ids: tuple[int, ...]
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    bounds: tuple[float, float, float, float, float, float]
    source_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "surface_id": self.surface_id,
            "building_id": self.building_id,
            "part_id": self.part_id,
            "kind": self.kind,
            "triangle_ids": list(self.triangle_ids),
            "centroid": list(self.centroid),
            "normal": list(self.normal),
            "bounds": list(self.bounds),
            "source_ids": list(self.source_ids),
        }


class CanonicalSceneMesh:
    """La géométrie unique d'un bâtiment, et son empreinte vérifiable.

    Un seul objet circule : le renderer le rastérise, le textureur le
    projette, la collision le traverse, l'export le sérialise. Personne ne
    reconstruit rien. ``mesh_digest()`` scelle le contrat : calculé après le
    rendu comme avant l'export, il doit être strictement identique.
    """

    #: Rôles possibles d'une face : paroi verticale, toiture, fond posé sur
    #: le terrain, ou face verticale de décrochement de toiture.
    KINDS = ("wall", "roof", "base", "roof_step")

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_kind: list[str] | np.ndarray | None = None,
        records: list[FootprintVertex] | None = None,
        triangle_ids: np.ndarray | None = None,
        surface_ids: list[str] | None = None,
        material_ids: list[str] | None = None,
        measurement_states: list[MeasurementState | str] | None = None,
        confidence: np.ndarray | None = None,
        provenance: list[dict] | None = None,
        uncertainty: list[dict] | None = None,
        roof_plane_ids: list[str | None] | None = None,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
        self.faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
        if face_kind is None:
            face_kind = ["wall"] * len(self.faces)
        kinds = [str(kind) for kind in face_kind]
        if len(kinds) != len(self.faces):
            raise ValueError("face_kind doit couvrir exactement toutes les faces")
        unknown = set(kinds) - set(self.KINDS)
        if unknown:
            raise ValueError(f"rôle de face inconnu : {sorted(unknown)}")
        self.face_kind = kinds
        count = len(self.faces)
        self.triangle_ids = np.asarray(
            np.arange(count, dtype=np.uint32) if triangle_ids is None else triangle_ids,
            dtype=np.uint32,
        ).reshape(-1)
        self.roof_plane_ids = list(roof_plane_ids or [None] * count)
        auto_assign_surfaces = surface_ids is None
        self.surface_ids = list(surface_ids or ["unassigned"] * count)
        self.material_ids = list(material_ids or ["material/default"] * count)
        self.measurement_states = [
            value if isinstance(value, MeasurementState) else MeasurementState(str(value))
            for value in (measurement_states or [MeasurementState.UNKNOWN] * count)
        ]
        self.confidence = np.asarray(
            np.zeros(count, dtype=np.float64) if confidence is None else confidence,
            dtype=np.float64,
        ).reshape(-1)
        self.provenance = list(provenance or [{} for _ in range(count)])
        self.uncertainty = list(uncertainty or [{} for _ in range(count)])
        self.records = list(records or [])
        self.hole_records: list[list[FootprintVertex]] = []
        self.feature_id: str | None = None
        self.provenance_class: str | None = None
        self.roof_topology: dict | None = None
        self.surface_catalog: dict[str, SurfaceRecord] = {}
        self.validate_triangle_metadata()
        if auto_assign_surfaces:
            self.assign_surface_ids("building", "main")

    def _triangle_normal(self, index: int) -> np.ndarray:
        triangle = self.vertices[self.faces[index]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        return normal / length if length > 1e-12 else np.zeros(3)

    @staticmethod
    def _compass(vector: np.ndarray) -> str:
        x, y = (float(v) for v in vector[:2])
        if np.hypot(x, y) <= 1e-8:
            return "flat"
        angle = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        labels = ("east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast")
        return labels[int((angle + 22.5) // 45.0) % 8]

    def _physical_surface_groups(self) -> list[tuple[str, list[int]]]:
        """Group connected coplanar triangles into physical surfaces."""
        from collections import defaultdict, deque

        by_key: defaultdict[tuple, list[int]] = defaultdict(list)
        normals: dict[int, np.ndarray] = {}
        for index, (face, kind) in enumerate(zip(self.faces, self.face_kind)):
            normal = self._triangle_normal(index)
            normals[index] = normal
            centre = self.vertices[face].mean(axis=0)
            if kind == "wall":
                label = f"facade/{self._compass(normal)}"
                key = (label, round(float(normal @ centre), 2))
            elif kind == "roof":
                aspect = self._compass(normal[:2])
                label = f"roof/plane-{aspect}"
                # Normal + plane offset distinguishes physical roof planes.
                key = (label, *np.round(normal, 3), round(float(normal @ centre), 2))
            elif kind == "roof_step":
                label = "roof-step"
                key = (label, self._compass(normal), round(float(normal @ centre), 2))
            elif kind == "base":
                label = "foundation"
                key = (label, round(float(centre[2]), 2))
            else:
                label = kind.replace("_", "-")
                key = (label, *np.round(normal, 3), round(float(normal @ centre), 2))
            by_key[key].append(index)

        groups: list[tuple[str, list[int]]] = []
        for key, candidates in by_key.items():
            owners: defaultdict[int, list[int]] = defaultdict(list)
            for face_index in candidates:
                for vertex in self.faces[face_index]:
                    owners[int(vertex)].append(face_index)
            unseen = set(candidates)
            while unseen:
                component: list[int] = []
                queue = deque([unseen.pop()])
                while queue:
                    current = queue.popleft()
                    component.append(current)
                    neighbours = {
                        neighbour
                        for vertex in self.faces[current]
                        for neighbour in owners[int(vertex)]
                    }
                    for neighbour in sorted(neighbours & unseen):
                        unseen.remove(neighbour)
                        queue.append(neighbour)
                groups.append((str(key[0]), sorted(component)))
        return groups

    def assign_surface_ids(
        self, building_id: str, part_id: str = "main",
        previous_surfaces: dict[str, dict] | None = None,
    ) -> None:
        """Assign readable IDs from physical geometry, never array ordering."""
        groups = self._physical_surface_groups()
        descriptors: list[dict] = []
        for label, indices in groups:
            points = np.vstack([self.vertices[self.faces[index]] for index in indices])
            normal = np.mean([self._triangle_normal(index) for index in indices], axis=0)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-12:
                normal /= norm
            descriptors.append({
                "label": label, "indices": indices,
                "centroid": points.mean(axis=0), "normal": normal,
                "bounds": np.r_[points.min(axis=0), points.max(axis=0)],
            })
        descriptors.sort(key=lambda row: (
            row["label"], *np.round(row["centroid"], 4), *np.round(row["normal"], 4)
        ))
        counters: dict[str, int] = {}
        used_previous: set[str] = set()
        assigned = [""] * len(self.faces)
        catalog: dict[str, SurfaceRecord] = {}
        for descriptor in descriptors:
            label = descriptor["label"]
            counters[label] = counters.get(label, 0) + 1
            surface_id = f"{building_id}/{part_id}/{label}-{counters[label]:02d}"
            matched = self._match_previous_surface(
                descriptor, previous_surfaces or {}, used_previous
            )
            if matched is not None:
                surface_id = matched
                used_previous.add(matched)
            for index in descriptor["indices"]:
                assigned[index] = surface_id
            source_ids = sorted({
                str(source)
                for index in descriptor["indices"]
                for source in self.provenance[index].get("source_ids", [])
            })
            catalog[surface_id] = SurfaceRecord(
                surface_id, building_id, part_id, label.split("/")[0],
                tuple(int(self.triangle_ids[index]) for index in descriptor["indices"]),
                tuple(float(v) for v in descriptor["centroid"]),
                tuple(float(v) for v in descriptor["normal"]),
                tuple(float(v) for v in descriptor["bounds"]), tuple(source_ids),
            )
        self.surface_ids = assigned
        self.surface_catalog = catalog
        self.validate_triangle_metadata()

    def _rebuild_surface_catalog_from_ids(self) -> None:
        catalog: dict[str, SurfaceRecord] = {}
        for surface_id in sorted(set(self.surface_ids)):
            indices = [i for i, value in enumerate(self.surface_ids) if value == surface_id]
            points = np.vstack([self.vertices[self.faces[index]] for index in indices])
            normal = np.mean([self._triangle_normal(index) for index in indices], axis=0)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-12:
                normal /= norm
            path = surface_id.split("/")
            source_ids = sorted({
                str(source) for index in indices
                for source in self.provenance[index].get("source_ids", [])
            })
            catalog[surface_id] = SurfaceRecord(
                surface_id=surface_id,
                building_id=path[0] if path else "building",
                part_id=path[1] if len(path) > 1 else "main",
                kind=path[2].split("-")[0] if len(path) > 2 else self.face_kind[indices[0]],
                triangle_ids=tuple(int(self.triangle_ids[index]) for index in indices),
                centroid=tuple(float(value) for value in points.mean(axis=0)),
                normal=tuple(float(value) for value in normal),
                bounds=tuple(float(value) for value in np.r_[points.min(axis=0), points.max(axis=0)]),
                source_ids=tuple(source_ids),
            )
        self.surface_catalog = catalog

    @staticmethod
    def _match_previous_surface(descriptor: dict, previous: dict, used: set[str]) -> str | None:
        best_id, best_score = None, 0.0
        for surface_id, raw in previous.items():
            if surface_id in used or raw.get("kind") != descriptor["label"].split("/")[0]:
                continue
            old_normal = np.asarray(raw.get("normal", [0, 0, 0]), dtype=float)
            old_centroid = np.asarray(raw.get("centroid", [np.inf] * 3), dtype=float)
            normal_score = max(0.0, float(np.dot(old_normal, descriptor["normal"])))
            distance = float(np.linalg.norm(old_centroid - descriptor["centroid"]))
            centroid_score = float(np.exp(-distance / 5.0))
            old_bounds = np.asarray(raw.get("bounds", [0] * 6), dtype=float)
            extent = np.maximum(0.0, np.minimum(old_bounds[3:], descriptor["bounds"][3:]) - np.maximum(old_bounds[:3], descriptor["bounds"][:3]))
            union = np.maximum(old_bounds[3:], descriptor["bounds"][3:]) - np.minimum(old_bounds[:3], descriptor["bounds"][:3])
            overlap = float(np.prod(extent + 0.05) / max(np.prod(union + 0.05), 1e-9))
            score = 0.45 * overlap + 0.30 * normal_score + 0.25 * centroid_score
            if score > best_score:
                best_id, best_score = surface_id, score
        return best_id if best_score >= 0.65 else None

    def validate_triangle_metadata(self) -> None:
        count = len(self.faces)
        fields = {
            "triangle_ids": len(self.triangle_ids),
            "surface_ids": len(self.surface_ids),
            "face_kind": len(self.face_kind),
            "material_ids": len(self.material_ids),
            "measurement_states": len(self.measurement_states),
            "confidence": len(self.confidence),
            "provenance": len(self.provenance),
            "uncertainty": len(self.uncertainty),
            "roof_plane_ids": len(self.roof_plane_ids),
        }
        invalid = [name for name, length in fields.items() if length != count]
        if invalid:
            raise ValueError(f"triangle metadata misaligned: {', '.join(invalid)}")
        if len(set(int(value) for value in self.triangle_ids)) != count:
            raise ValueError("triangle_ids must be unique")
        if np.any(~np.isfinite(self.confidence)) or np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("confidence must be finite and within [0, 1]")
        for surface_id, kind in zip(self.surface_ids, self.face_kind):
            if not surface_id:
                raise ValueError("every triangle must reference a physical surface")
            expected = {
                "wall": "/facade/", "roof": "/roof/",
                "roof_step": "/roof-step-", "base": "/foundation-",
            }.get(kind)
            if surface_id != "unassigned" and expected and expected not in surface_id:
                raise ValueError(
                    f"surface kind mismatch: {surface_id!r} cannot own {kind!r}"
                )

    # ------------------------------------------------------------------
    # Consommation
    # ------------------------------------------------------------------
    def triangles(self) -> list[tuple[np.ndarray, int]]:
        """Triangles avec l'index de face — le seul chemin vers la géométrie."""
        return [
            (self.vertices[face], index) for index, face in enumerate(self.faces)
        ]

    def triangle_records(self) -> list[TriangleRecord]:
        return [
            TriangleRecord(
                int(self.triangle_ids[index]), self.surface_ids[index],
                tuple(int(vertex) for vertex in face),
                self.measurement_states[index], float(self.confidence[index]),
                tuple(str(value) for value in self.provenance[index].get("source_ids", [])),
            )
            for index, face in enumerate(self.faces)
        ]

    def triangles_for_surface(self, surface_id: str) -> list[TriangleRecord]:
        return [record for record in self.triangle_records() if record.surface_id == surface_id]

    def surface(self, surface_id: str) -> SurfaceRecord:
        try:
            return self.surface_catalog[surface_id]
        except KeyError as exc:
            raise KeyError(f"unknown physical surface: {surface_id}") from exc

    def surface_audit(self) -> dict:
        counts = {
            surface_id: len(self.triangles_for_surface(surface_id))
            for surface_id in sorted(set(self.surface_ids))
        }
        unknown = sum(not value or value == "unassigned" for value in self.surface_ids)
        return {
            "triangles": len(self.faces),
            "surfaces": len(counts),
            "triangles_without_surface_id": unknown,
            "duplicate_surface_ids": max(0, len(self.surface_catalog) - len(set(self.surface_catalog))),
            "unknown_surface_count": unknown,
            "triangles_by_surface": counts,
            "passed": unknown == 0 and len(counts) == len(self.surface_catalog),
        }

    @classmethod
    def merge(cls, meshes: list[CanonicalSceneMesh]) -> CanonicalSceneMesh:
        """Create the one scene-level mesh without rebuilding any geometry."""
        if not meshes:
            raise ValueError("at least one canonical mesh is required")
        vertices, faces = [], []
        kinds: list[str] = []
        surface_ids: list[str] = []
        materials: list[str] = []
        states: list[MeasurementState] = []
        confidence: list[float] = []
        provenance: list[dict] = []
        uncertainty: list[dict] = []
        roof_plane_ids: list[str | None] = []
        offset = 0
        for mesh in meshes:
            mesh.validate_triangle_metadata()
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + offset)
            offset += len(mesh.vertices)
            kinds.extend(mesh.face_kind)
            surface_ids.extend(mesh.surface_ids)
            materials.extend(mesh.material_ids)
            states.extend(mesh.measurement_states)
            confidence.extend(float(value) for value in mesh.confidence)
            provenance.extend(mesh.provenance)
            uncertainty.extend(mesh.uncertainty)
            roof_plane_ids.extend(mesh.roof_plane_ids)
        merged = cls(
            np.vstack(vertices), np.vstack(faces), kinds,
            triangle_ids=np.arange(sum(len(mesh.faces) for mesh in meshes), dtype=np.uint32),
            surface_ids=surface_ids, material_ids=materials,
            measurement_states=states, confidence=np.asarray(confidence),
            provenance=provenance, uncertainty=uncertainty,
            roof_plane_ids=roof_plane_ids,
        )
        merged._rebuild_surface_catalog_from_ids()
        return merged

    def raycast(
        self, origin: np.ndarray, direction: np.ndarray, max_distance_m: float = 5e3
    ) -> float | None:
        """Distance de la première intersection rayon/maillage, sinon None.

        Möller–Trumbore direct : la collision et le raycast consomment le
        même maillage que le rendu, sans reconstruction parallèle.
        """
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return None
        direction = direction / norm
        hit = self.raycast_hit(origin, direction, max_distance_m)
        return None if hit is None else hit.distance

    def raycast_hit(
        self, origin: np.ndarray, direction: np.ndarray, max_distance_m: float = 5e3
    ) -> RayHit | None:
        """Return both triangle and stable physical surface for the first hit."""
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return None
        direction = direction / norm
        best: tuple[float, int] | None = None
        for face_index, face in enumerate(self.faces):
            a, b, c = self.vertices[face]
            edge1 = b - a
            edge2 = c - a
            pvec = np.cross(direction, edge2)
            det = float(np.dot(edge1, pvec))
            if abs(det) < 1e-12:
                continue
            inv_det = 1.0 / det
            tvec = origin - a
            u = float(np.dot(tvec, pvec)) * inv_det
            if u < -1e-9 or u > 1.0 + 1e-9:
                continue
            qvec = np.cross(tvec, edge1)
            v = float(np.dot(direction, qvec)) * inv_det
            if v < -1e-9 or u + v > 1.0 + 1e-9:
                continue
            t = float(np.dot(edge2, qvec)) * inv_det
            if t < 1e-6 or t > max_distance_m:
                continue
            if best is None or t < best[0]:
                best = (t, face_index)
        if best is None:
            return None
        distance, face_index = best
        return RayHit(
            int(self.triangle_ids[face_index]), self.surface_ids[face_index], distance
        )

    # ------------------------------------------------------------------
    # Topologie
    # ------------------------------------------------------------------
    def boundary_edges(self) -> list[tuple[int, int]]:
        """Arêtes portées par une seule face : les fissures du volume."""
        from collections import Counter

        counts: Counter[tuple[int, int]] = Counter()
        for face in self.faces:
            a, b, c = (int(v) for v in face)
            counts.update(
                {
                    tuple(sorted((a, b))),
                    tuple(sorted((b, c))),
                    tuple(sorted((c, a))),
                }
            )
        return sorted(edge for edge, count in counts.items() if count == 1)

    def weld_vertices(self, tolerance_m: float = WELD_TOLERANCE_M) -> CanonicalSceneMesh:
        """Soude les sommets confondus à la tolérance donnée.

        Mur et toiture sont construits l'un contre la frontière de l'autre ;
        la soudure retire le dernier écart de flottant, et retourne un
        maillage neuf — jamais de mutation silencieuse du maillage publié.
        """
        if len(self.vertices) == 0:
            welded = CanonicalSceneMesh(
                self.vertices.copy(), self.faces.copy(), self.face_kind, self.records,
                triangle_ids=self.triangle_ids.copy(),
                surface_ids=list(self.surface_ids),
                material_ids=list(self.material_ids),
                measurement_states=list(self.measurement_states),
                confidence=self.confidence.copy(),
                provenance=list(self.provenance),
                uncertainty=list(self.uncertainty),
                roof_plane_ids=list(self.roof_plane_ids),
            )
            welded._rebuild_surface_catalog_from_ids()
            return welded
        keys = np.round(self.vertices / tolerance_m).astype(np.int64)
        _, first, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        welded_vertices = self.vertices[first]
        # `inverse` associe chaque ancien sommet à son sommet soudé : ce sont
        # les faces qu'il faut réindexer, pas le tableau lui-même.
        welded_faces = np.asarray(inverse).reshape(-1)[self.faces]
        keep = np.array([len(set(int(v) for v in face)) == 3 for face in welded_faces])
        welded = CanonicalSceneMesh(
            welded_vertices,
            welded_faces[keep] if keep.any() else welded_faces[:0],
            [kind for kind, good in zip(self.face_kind, keep) if good],
            self.records,
            triangle_ids=self.triangle_ids[keep],
            surface_ids=[value for value, good in zip(self.surface_ids, keep) if good],
            material_ids=[value for value, good in zip(self.material_ids, keep) if good],
            measurement_states=[value for value, good in zip(self.measurement_states, keep) if good],
            confidence=self.confidence[keep],
            provenance=[value for value, good in zip(self.provenance, keep) if good],
            uncertainty=[value for value, good in zip(self.uncertainty, keep) if good],
            roof_plane_ids=[value for value, good in zip(self.roof_plane_ids, keep) if good],
        )
        welded._rebuild_surface_catalog_from_ids()
        return welded

    # ------------------------------------------------------------------
    # Empreinte canonique
    # ------------------------------------------------------------------
    def canonical_form(self) -> tuple[np.ndarray, np.ndarray]:
        """Forme canonique : invariante à toute permutation de la mémoire.

        Les sommets sont quantifiés au millimètre puis triés ; les faces sont
        réindexées, leurs coins remis en ordre croissante (arêtes non
        orientées) puis triées entre elles. Deux maillages géométriquement
        identiques produisent donc la même forme, quel que soit leur ordre
        de construction.
        """
        quantized = np.round(self.vertices / DIGEST_QUANTUM_M) * DIGEST_QUANTUM_M
        order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order), dtype=np.int64)
        sorted_vertices = quantized[order]
        faces = np.sort(rank[self.faces], axis=1)
        faces = faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))]
        return sorted_vertices, faces

    def mesh_digest(self) -> str:
        """Empreinte SHA-256 de la géométrie, indépendante de la représentation."""
        self.validate_triangle_metadata()
        vertices, _faces = self.canonical_form()
        digest = hashlib.sha256()
        digest.update(vertices.astype(np.float64).tobytes())
        quantized = np.round(self.vertices / DIGEST_QUANTUM_M) * DIGEST_QUANTUM_M
        records = []
        for index, face in enumerate(self.faces):
            coordinates = sorted(tuple(float(v) for v in quantized[vertex]) for vertex in face)
            records.append((
                coordinates, self.surface_ids[index], self.face_kind[index],
                self.material_ids[index], self.measurement_states[index].value,
                round(float(self.confidence[index]), 8),
                repr(sorted(self.provenance[index].items())),
                repr(sorted(self.uncertainty[index].items())),
                self.roof_plane_ids[index],
            ))
        for record in sorted(records, key=repr):
            digest.update(repr(record).encode("utf-8"))
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------
    def as_dict(self) -> dict:
        payload = {
            "vertices": np.round(self.vertices, 4).tolist(),
            "faces": self.faces.astype(int).tolist(),
            "face_kind": list(self.face_kind),
            "triangle_ids": self.triangle_ids.astype(int).tolist(),
            "surface_ids": list(self.surface_ids),
            "material_ids": list(self.material_ids),
            "measurement_states": [value.value for value in self.measurement_states],
            "confidence": self.confidence.astype(float).tolist(),
            "provenance": list(self.provenance),
            "uncertainty": list(self.uncertainty),
            "roof_plane_ids": list(self.roof_plane_ids),
            "mesh_digest": self.mesh_digest(),
            "surface_catalog": {
                key: value.as_dict() for key, value in sorted(self.surface_catalog.items())
            },
        }
        if self.records:
            payload["footprint_vertices"] = [
                {
                    "vertex_id": record.vertex_id,
                    "x": round(record.x, 4),
                    "y": round(record.y, 4),
                    "ground_z": round(record.ground_z, 4),
                    "top_z": round(record.top_z, 4),
                    "floor_z": (
                        round(record.floor_z, 4) if record.floor_z is not None else None
                    ),
                }
                for record in self.records
            ]
        if self.roof_topology is not None:
            payload["roof_topology"] = self.roof_topology
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> CanonicalSceneMesh:
        mesh = cls(
            np.asarray(payload["vertices"], dtype=np.float64),
            np.asarray(payload["faces"], dtype=np.int64),
            payload.get("face_kind"),
            triangle_ids=payload.get("triangle_ids"),
            surface_ids=payload.get("surface_ids"),
            material_ids=payload.get("material_ids"),
            measurement_states=payload.get("measurement_states"),
            confidence=payload.get("confidence"),
            provenance=payload.get("provenance"),
            uncertainty=payload.get("uncertainty"),
            roof_plane_ids=payload.get("roof_plane_ids"),
        )
        mesh.feature_id = payload.get("feature_id")
        mesh.provenance_class = payload.get("provenance_class")
        mesh.roof_topology = payload.get("roof_topology")
        mesh.surface_catalog = {
            surface_id: SurfaceRecord(
                surface_id=surface_id,
                building_id=str(raw["building_id"]), part_id=str(raw["part_id"]),
                kind=str(raw["kind"]),
                triangle_ids=tuple(int(value) for value in raw.get("triangle_ids", [])),
                centroid=tuple(float(value) for value in raw["centroid"]),
                normal=tuple(float(value) for value in raw["normal"]),
                bounds=tuple(float(value) for value in raw["bounds"]),
                source_ids=tuple(str(value) for value in raw.get("source_ids", [])),
            )
            for surface_id, raw in (payload.get("surface_catalog") or {}).items()
        }
        if not mesh.surface_catalog:
            mesh._rebuild_surface_catalog_from_ids()
        return mesh

    def audit(self) -> dict:
        """Étanchéité, connexité et cohérence des rôles de faces."""
        from collections import Counter, defaultdict, deque

        faces = self.faces
        vertices = self.vertices
        edge_counts: Counter[tuple[int, int]] = Counter()
        owners: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        degenerate = 0
        signed_volume = 0.0
        for face_index, face in enumerate(faces):
            ids = [int(v) for v in face]
            if len(set(ids)) < 3:
                degenerate += 1
                continue
            a, b, c = vertices[ids]
            if bool(np.linalg.norm(np.cross(b - a, c - a)) <= 1e-9):
                degenerate += 1
            signed_volume += float(np.dot(a, np.cross(b, c))) / 6.0
            edges = [
                tuple(sorted((ids[0], ids[1]))),
                tuple(sorted((ids[1], ids[2]))),
                tuple(sorted((ids[2], ids[0]))),
            ]
            edge_counts.update(edges)
            for edge in edges:
                owners[edge].append(face_index)
        face_adjacency: defaultdict[int, set[int]] = defaultdict(set)
        for related in owners.values():
            for face_index in related:
                face_adjacency[face_index].update(
                    i for i in related if i != face_index
                )
        unseen = set(range(len(faces)))
        components = 0
        while unseen:
            components += 1
            queue = deque([unseen.pop()])
            while queue:
                for neighbour in face_adjacency[queue.popleft()]:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        queue.append(neighbour)
        boundary = sum(value == 1 for value in edge_counts.values())
        non_manifold = sum(value > 2 for value in edge_counts.values())
        volume = abs(signed_volume)
        watertight = boundary == 0 and non_manifold == 0 and degenerate == 0
        return {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "boundary_edges": boundary,
            "non_manifold_edges": non_manifold,
            "degenerate_faces": degenerate,
            "connected_components": components,
            "volume_m3": round(volume, 3),
            "watertight": watertight,
            "supported": watertight and components == 1 and volume > 0.0,
        }


def _triangulate_polygon(polygon: Polygon) -> list[tuple[int, int, int]] | None:
    """Triangule un polygone shapely et retourne les triplets de coordonnées."""
    triangles = []
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        coords = np.asarray(triangle.exterior.coords[:-1], dtype=np.float64)
        triangles.append(coords)
    return triangles or None
