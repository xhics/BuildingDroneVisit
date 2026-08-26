"""Acceptation d'un point LiDAR comme support d'une façade.

Un nuage de façade ne contient jamais que la façade : une aile qui prolonge,
un auvent, un mur voisin parallèle à deux mètres — tout se présente avec la
même allure locale. Accepter ces points sans examen déplace le mur cible.

Cinq conditions, toutes requises :

1. **distance au plan** : le candidat reste dans une lame mince autour du
   plan de façade ajusté ;
2. **normale compatible** : l'orientation locale du nuage suit celle du plan ;
3. **même composante** : le point appartient à la composante connexe du
   nuage qui porte le mur cible — un plan ajusté sur un mélange de murs est
   déjà suspect, on ne le laisse pas voter deux fois ;
4. **cohérence extérieure** : le point ne dépasse pas du côté extérieur du
   mur au-delà de la tolérance — une façade plaquée devant la vraie est
   rejetée même si elle est parfaitement plane ;
5. **visibilité multiple** : le point est vu depuis plusieurs poses ; un
   point mono-vue ne soutient pas une surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Épaisseur de la lame acceptée autour du plan de façade, en mètres.
MAX_PLANAR_DISTANCE_M = 0.25

#: Alignement minimal entre normale locale et normale du plan (cosinus).
MIN_NORMAL_COS = 0.90

#: Dépassement toléré côté extérieur du mur, en mètres. Une façade parasite
#: plaquée devant la cible dépasse largement cette marge.
MAX_OUTWARD_PROTRUSION_M = 0.35

#: Nombre minimal de poses distinctes devant voir un point de support.
MIN_VIEWS = 2

#: Rayon de voisinage pour la séparation en composantes, en mètres.
COMPONENT_RADIUS_M = 1.0


@dataclass
class FacadeSupportVerdict:
    """Ce que le filtrage a retenu, condition par condition."""

    accepted: np.ndarray
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.accepted.sum())

    def as_dict(self) -> dict:
        return {
            "accepted": self.count,
            "rejected": int(len(self.accepted) - self.count),
            "rejection_reasons": dict(sorted(self.reasons.items())),
        }


def estimate_normals(points: np.ndarray, neighbours: int = 10) -> np.ndarray:
    """Normales locales par PCA de voisinage."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, groups = tree.query(points, k=min(neighbours, len(points)))
    normals = np.zeros((len(points), 3))
    for position, group in enumerate(np.atleast_2d(groups)):
        cloud = points[group] - points[group].mean(axis=0)
        normals[position] = np.linalg.svd(cloud, full_matrices=False)[2][2]
    return normals


def largest_component_labels(points: np.ndarray, radius_m: float = COMPONENT_RADIUS_M):
    """Étiquette les composantes connexes du nuage ; retourne la plus fournie.

    Le rayon de connexion s'adapte à la densité locale : un mur clairsemé
    reste une seule composante là où un rayon figé le hacherait en îlots.
    Une façade parasite, elle, reste séparée tant que son écart dépasse le
    voisinage typique du mur cible.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    count = len(points)
    if count < 2:
        return np.zeros(count, dtype=np.int64)
    spacing = float(np.median(tree.query(points, k=2)[0][:, 1]))
    effective_radius = float(np.clip(3.0 * spacing, radius_m, 5.0))
    labels = np.full(count, -1, dtype=np.int64)
    current = 0
    for seed in range(count):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = current
        while stack:
            node = stack.pop()
            for neighbour in tree.query_ball_point(points[node], effective_radius):
                if labels[neighbour] < 0:
                    labels[neighbour] = current
                    stack.append(neighbour)
        current += 1
    if current <= 1:
        return labels
    sizes = np.bincount(labels)
    dominant = int(sizes.argmax())
    return np.where(labels == dominant, 0, 1)


def gate_facade_support(
    points: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    outward_normal: np.ndarray | None = None,
    view_ids: list | np.ndarray | None = None,
    component_points: np.ndarray | None = None,
    component_reference: np.ndarray | None = None,
    max_planar_distance_m: float = MAX_PLANAR_DISTANCE_M,
    min_normal_cos: float = MIN_NORMAL_COS,
    max_outward_protrusion_m: float = MAX_OUTWARD_PROTRUSION_M,
    min_views: int = MIN_VIEWS,
) -> FacadeSupportVerdict:
    """Décide, point par point, s'il peut soutenir la façade visée.

    Paramètres
    ----------
    points :
        Candidats (N, 3), déjà exprimés dans le repère de la scène.
    plane_point, plane_normal :
        Plan de la façade cible, ajusté indépendamment des candidats.
    outward_normal :
        Direction horizontale vers l'extérieur du bâtiment : elle borne le
        dépassement autorisé côté rue.
    view_ids :
        Identifiant de pose d'origine de chaque point ; absent, la condition
        de visibilité multiple dégénère en exigence de densité locale.
    component_points :
        Nuage complet du bâtiment, servant à séparer les composantes. Sans
        lui, la composante est supposée unique.
    component_reference :
        Un point certain d'appartenir au mur cible (par exemple le centre de
        la façade attendue) : la composante retenue est celle qui le porte.
    """
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    plane_point = np.asarray(plane_point, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)

    reasons: dict[str, int] = {}
    accepted = np.ones(len(points), dtype=bool)

    def reject(mask: np.ndarray, reason: str) -> None:
        fresh = mask & accepted
        if fresh.any():
            reasons[reason] = reasons.get(reason, 0) + int(fresh.sum())
        # Écriture en place : la fermeture ne doit pas relier le nom.
        np.logical_and(accepted, ~mask, out=accepted)

    # 1. Lame mince autour du plan.
    offsets = (points - plane_point) @ normal
    reject(np.abs(offsets) > max_planar_distance_m, "planar_distance")

    # 2. Normale locale compatible.
    if accepted.sum() >= 3:
        normals = estimate_normals(points[accepted])
        alignment = np.abs(normals @ normal)
        weak = np.zeros(len(points), dtype=bool)
        weak[np.where(accepted)[0]] = alignment < min_normal_cos
        reject(weak, "normal_mismatch")

    # 3. Composante connexe : un mur voisin n'hérite pas du vote du cible.
    if component_points is not None and len(component_points) >= 4:
        labels = largest_component_labels(np.asarray(component_points)[:, :3])
        if component_reference is not None:
            reference_label = labels[
                int(
                    np.argmin(
                        np.linalg.norm(
                            np.asarray(component_points)
                            - np.asarray(component_reference, dtype=np.float64),
                            axis=1,
                        )
                    )
                )
            ]
        else:
            reference_label = int(np.bincount(labels).argmax())
        # Projection des étiquettes sur les candidats par plus proche voisin.
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(component_points))
        _d, nearest = tree.query(points)
        foreign = labels[nearest] != reference_label
        reject(foreign, "foreign_component")

    # 4. Cohérence extérieure : pas de dépassement côté rue.
    if outward_normal is not None:
        outward = np.asarray(outward_normal, dtype=np.float64)
        outward = outward / max(float(np.linalg.norm(outward)), 1e-12)
        # Le dépassement s'exprime dans le plan du mur : la projection du
        # vecteur plan→point sur l'extérieur doit rester bornée.
        lateral = (points - plane_point) - offsets[:, None] * normal[None, :]
        protrusion = lateral @ outward
        reject(protrusion > max_outward_protrusion_m, "outside_protrusion")

    # 5. Visibilité depuis plusieurs poses.
    if view_ids is not None and len(view_ids) == len(points):
        views = np.asarray([str(v) for v in view_ids])
        for position in np.where(accepted)[0]:
            same_spot = np.abs(offsets - offsets[position]) <= max_planar_distance_m
            distinct = len(set(views[same_spot & accepted]))
            if distinct < min_views:
                accepted[position] = False
                reasons["single_view"] = reasons.get("single_view", 0) + 1

    return FacadeSupportVerdict(accepted=accepted, reasons=reasons)
