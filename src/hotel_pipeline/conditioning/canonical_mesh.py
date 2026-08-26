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
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate

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
        self.records = list(records or [])
        self.feature_id: str | None = None
        self.provenance_class: str | None = None

    # ------------------------------------------------------------------
    # Consommation
    # ------------------------------------------------------------------
    def triangles(self) -> list[tuple[np.ndarray, int]]:
        """Triangles avec l'index de face — le seul chemin vers la géométrie."""
        return [
            (self.vertices[face], index) for index, face in enumerate(self.faces)
        ]

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
        best: float | None = None
        for face in self.faces:
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
            if best is None or t < best:
                best = t
        return best

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
            return CanonicalSceneMesh(
                self.vertices.copy(), self.faces.copy(), self.face_kind, self.records
            )
        keys = np.round(self.vertices / tolerance_m).astype(np.int64)
        _, first, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        welded_vertices = self.vertices[first]
        # `inverse` associe chaque ancien sommet à son sommet soudé : ce sont
        # les faces qu'il faut réindexer, pas le tableau lui-même.
        welded_faces = np.asarray(inverse).reshape(-1)[self.faces]
        keep = np.array([len(set(int(v) for v in face)) == 3 for face in welded_faces])
        return CanonicalSceneMesh(
            welded_vertices,
            welded_faces[keep] if keep.any() else welded_faces[:0],
            [kind for kind, good in zip(self.face_kind, keep) if good],
            self.records,
        )

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
        vertices, faces = self.canonical_form()
        digest = hashlib.sha256()
        digest.update(vertices.astype(np.float64).tobytes())
        digest.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------
    def as_dict(self) -> dict:
        payload = {
            "vertices": np.round(self.vertices, 4).tolist(),
            "faces": self.faces.astype(int).tolist(),
            "face_kind": list(self.face_kind),
            "mesh_digest": self.mesh_digest(),
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
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> CanonicalSceneMesh:
        mesh = cls(
            np.asarray(payload["vertices"], dtype=np.float64),
            np.asarray(payload["faces"], dtype=np.int64),
            payload.get("face_kind"),
        )
        mesh.feature_id = payload.get("feature_id")
        mesh.provenance_class = payload.get("provenance_class")
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
