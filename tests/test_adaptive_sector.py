"""Secteur et point de vue dans la recherche adaptative (collecte V2).

Le défaut constaté sur le pilote : les huit besoins avaient exactement 315
candidats éligibles chacun et recommandaient les deux mêmes. Aucun ne
discriminait par secteur, donc le classement était identique partout et le
premier du tri gagnait huit fois — façade avant et façade arrière servies par
la même vue, ce qui est géométriquement impossible.

Deux questions distinctes, que ces tests séparent :

```text
cible → caméra    de quel côté du bâtiment se tient l'observateur
caméra → cible    l'objectif est-il effectivement tourné vers elle
```

Le troisième sujet est le point de vue : deux cadrages d'un même panorama sont
deux acquisitions et une seule observation. Un quota de deux points de vue ne
se remplit pas avec deux cadrages de la même position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest
from shapely.geometry import Point

from hotel_pipeline.adaptive_search import (
    SectorContext,
    SectorFit,
    measure_candidate,
    select_for_demand,
)
from hotel_pipeline.demand_targets import DemandTarget
from hotel_pipeline.schemas import DEFAULT_POLICY

from test_adaptive_search import candidate, demand

SEARCH = DEFAULT_POLICY.adaptive_search

#: Cible à l'origine du plan projeté : les positions se lisent en mètres.
TARGET_SHAPE = Point(0.0, 0.0).buffer(12.0)

#: Convention de `sectors.sector_for` : l'azimut avant du bâtiment.
FRONT_AZIMUTH = 0.0


class FlatProjection:
    """Projection d'essai : degrés convertis en mètres autour de l'origine.

    Suffit ici parce qu'on éprouve la **convention** de secteur, non la
    justesse d'une projection cartographique — celle-ci a ses propres tests.
    """

    def point(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon * 78_000.0, lat * 111_320.0)


def at_bearing(bearing_deg: float, metres: float = 60.0):
    """Un candidat placé à un azimut donné **vu depuis la cible**."""
    east = metres * math.sin(math.radians(bearing_deg))
    north = metres * math.cos(math.radians(bearing_deg))
    return (north / 111_320.0, east / 78_000.0)


def sector_context(demand_id: str, required_bearing: float | None, **overrides):
    target = DemandTarget(
        demand_id=demand_id,
        shape=TARGET_SHAPE,
        required_bearing_deg=required_bearing,
        half_width_deg=DEFAULT_POLICY.geometry.sector_observer_half_width_deg,
    )
    fields = dict(
        targets={demand_id: target},
        projection=FlatProjection(),
        front_azimuth_deg=FRONT_AZIMUTH,
        heading_tolerance_deg=SEARCH.heading_tolerance_deg,
    )
    fields.update(overrides)
    return SectorContext(**fields)


def measure_at(bearing_deg: float, target_ref: str, required: float, **kw):
    """Mesure un candidat placé à `bearing_deg` contre un besoin sectoriel."""
    need = demand("obligation:X", target_ref)
    lat, lon = at_bearing(bearing_deg)
    return measure_candidate(
        candidate("c-1", lat, lon, **kw), need, [], 3,
        target_lat=0.0, target_lon=0.0, policy=SEARCH,
        sector=sector_context("obligation:X", required),
    )


# --- position de l'observateur : de quel côté --------------------------------


def test_the_opposite_side_is_rejected() -> None:
    """Le défaut du pilote : avant et arrière servis par la même vue."""
    measure = measure_at(180.0, "front", required=0.0)

    assert measure.sector_fit is SectorFit.WRONG_SECTOR
    assert measure.rejection_reason is not None
    assert "hors du secteur" in measure.rejection_reason


def test_the_exact_sector_is_a_strong_candidate() -> None:
    measure = measure_at(5.0, "front", required=0.0)

    assert measure.sector_fit is SectorFit.EXACT
    assert measure.observer_sector == "front"
    assert measure.sector_compatible is True
    assert measure.rejection_reason is None


def test_a_corner_view_is_compatible_but_auxiliary() -> None:
    """Le demi-angle autorise le coin ; il ne le rend pas équivalent.

    Sans cette distinction, une vue oblique créditerait une façade principale
    au même titre qu'une vue frontale — et `demands assess`, qui exige
    l'égalité du secteur discret, refuserait ensuite de la compter.
    """
    # 50° : hors du secteur discret `front`, dans le cône de 67,5°.
    measure = measure_at(50.0, "front", required=0.0)

    assert measure.sector_compatible is True
    assert measure.sector_fit is SectorFit.ADJACENT
    assert measure.observer_sector == "front_right_corner"
    assert measure.rejection_reason is None, "compatible : pas un rejet"
    assert "auxiliaire" in measure.sector_reason


def test_an_unconstrained_demand_accepts_any_side() -> None:
    """Un corridor se documente d'où l'on veut."""
    measure = measure_at(200.0, "ACCESS_ROAD_MAIN", required=None)

    assert measure.sector_fit is SectorFit.UNCONSTRAINED
    assert measure.rejection_reason is None


def test_an_unknown_front_azimuth_stays_unknown() -> None:
    """Sans orientation du bâtiment, « de face » ne se décide pas."""
    need = demand("obligation:X", "front")
    lat, lon = at_bearing(5.0)
    measure = measure_candidate(
        candidate("c-1", lat, lon), need, [], 3,
        target_lat=0.0, target_lon=0.0, policy=SEARCH,
        sector=sector_context("obligation:X", 0.0, front_azimuth_deg=None),
    )

    assert measure.observer_sector is None
    assert measure.sector_fit is SectorFit.UNKNOWN
    assert measure.rejection_reason is None, "l'ignorance n'est pas un rejet"


def test_an_unresolved_target_carries_its_reason() -> None:
    need = demand("obligation:PROPERTY_SIGN", "PROPERTY_SIGN")
    lat, lon = at_bearing(5.0)
    measure = measure_candidate(
        candidate("c-1", lat, lon), need, [], 3,
        target_lat=0.0, target_lon=0.0, policy=SEARCH,
        sector=SectorContext(
            targets={}, projection=FlatProjection(),
            unresolved={"obligation:PROPERTY_SIGN": "enseigne non géoréférencée"},
        ),
    )

    assert measure.sector_fit is SectorFit.UNKNOWN
    assert measure.sector_reason == "enseigne non géoréférencée"


# --- orientation de la caméra : question distincte ----------------------------


def test_position_and_orientation_are_measured_separately() -> None:
    """Bien placée, mais tournée à l'opposé de la cible."""
    # Observateur au nord (bearing 0 depuis la cible) : il doit viser le sud.
    measure = measure_at(0.0, "front", required=0.0, original_heading_deg=0.0)

    assert measure.sector_fit is SectorFit.EXACT, "la position est bonne"
    assert measure.heading_targets_object is False, "l'objectif regarde ailleurs"
    assert measure.heading_offset_deg == pytest.approx(180.0, abs=1.0)


def test_a_camera_looking_at_the_target_from_the_right_side_passes_both() -> None:
    measure = measure_at(0.0, "front", required=0.0, original_heading_deg=180.0)

    assert measure.sector_fit is SectorFit.EXACT
    assert measure.heading_targets_object is True
    assert measure.heading_offset_deg == pytest.approx(0.0, abs=1.0)


def test_a_missing_heading_stays_none_not_false() -> None:
    """Un cap absent n'est pas un cap qui vise ailleurs."""
    measure = measure_at(0.0, "front", required=0.0)

    assert measure.heading_targets_object is None
    assert measure.heading_offset_deg is None


def test_without_a_policy_the_verdict_is_withheld() -> None:
    """La mesure est publiée, la préférence ne s'invente pas."""
    need = demand("obligation:X", "front")
    lat, lon = at_bearing(0.0)
    measure = measure_candidate(
        candidate("c-1", lat, lon, original_heading_deg=180.0), need, [], 3,
        target_lat=0.0, target_lon=0.0, policy=SEARCH,
        sector=sector_context("obligation:X", 0.0, heading_tolerance_deg=None),
    )

    assert measure.heading_offset_deg is not None, "la mesure est faite"
    assert measure.heading_targets_object is None, "le verdict ne l'est pas"


# --- populations disjointes : ce que le pilote n'avait pas --------------------


def test_front_and_rear_select_disjoint_populations() -> None:
    """Le résultat attendu du correctif, énoncé comme un test."""
    ring = [
        candidate(f"c-{int(b):03d}", *at_bearing(float(b)))
        for b in range(0, 360, 15)
    ]

    def eligible_for(target_ref: str, required: float) -> set[str]:
        need = demand(f"obligation:{target_ref}", target_ref)
        context = sector_context(need.demand_id, required)
        return {
            m.candidate_id
            for c in ring
            if (
                m := measure_candidate(
                    c, need, [], 3, target_lat=0.0, target_lon=0.0,
                    policy=SEARCH, sector=context,
                )
            ).rejection_reason is None
        }

    front = eligible_for("front", 0.0)
    rear = eligible_for("rear", 180.0)
    left = eligible_for("left", 270.0)
    right = eligible_for("right", 90.0)

    assert front and rear and left and right, "chaque secteur trouve des vues"
    assert not (front & rear), "avant et arrière sont opposés"
    assert not (left & right), "gauche et droite sont opposés"
    # L'intersection avec un côté adjacent reste légitime : ce sont les coins.
    assert front & right, "les vues de coin restent compatibles avec deux faces"


def test_corner_overlap_is_identified_as_auxiliary() -> None:
    """L'intersection légitime doit se voir, pas se confondre avec du principal."""
    corner = candidate("c-coin", *at_bearing(45.0))
    fits = {}
    for ref, required in (("front", 0.0), ("right", 90.0)):
        need = demand(f"obligation:{ref}", ref)
        fits[ref] = measure_candidate(
            corner, need, [], 3, target_lat=0.0, target_lon=0.0,
            policy=SEARCH, sector=sector_context(need.demand_id, required),
        ).sector_fit

    assert set(fits.values()) == {SectorFit.ADJACENT}, (
        "une vue de coin est auxiliaire des deux côtés, principale d'aucun"
    )


# --- points de vue : deux cadrages ne font pas deux observations -------------


@dataclass
class Framing:
    """Un cadrage, avec son panorama d'origine."""

    candidate_id: str
    panorama_id: str
    bearing: float


def test_two_framings_of_one_panorama_count_as_one_viewpoint() -> None:
    """Le cas énoncé : A/bâtiment, A/contexte, B/bâtiment → A + B, jamais A+A."""
    from hotel_pipeline.plan import group_viewpoints

    framings = [
        candidate("pano-A-bati", *at_bearing(0.0), panorama_id="A",
                  requested_heading_deg=180.0),
        candidate("pano-A-ctx", *at_bearing(0.0), panorama_id="A",
                  requested_heading_deg=90.0),
        candidate("pano-B-bati", *at_bearing(20.0), panorama_id="B",
                  requested_heading_deg=200.0),
    ]
    need = demand("obligation:front", "front", viewpoints_required=2)
    context = sector_context(need.demand_id, 0.0)

    measures = [
        measure_candidate(c, need, [], 3, target_lat=0.0, target_lon=0.0,
                          policy=SEARCH, sector=context)
        for c in framings
    ]
    viewpoints = group_viewpoints(
        framings, DEFAULT_POLICY.geometry.viewpoint_separation_m
    )

    retained = select_for_demand(
        measures, {c.candidate_id: c for c in framings}, need,
        0.0, 0.0, wanted=2, policy=SEARCH, viewpoints=viewpoints,
    )

    assert len(retained) == 2
    assert len({viewpoints[c] for c in retained}) == 2, (
        "deux cadrages du même panorama ne font pas deux points de vue"
    )
    assert "pano-B-bati" in retained


def test_a_single_panorama_leaves_the_demand_partially_covered() -> None:
    """Faute de second point de vue, on ne complète pas avec un cadrage."""
    from hotel_pipeline.plan import group_viewpoints

    framings = [
        candidate("pano-A-1", *at_bearing(0.0), panorama_id="A",
                  requested_heading_deg=180.0),
        candidate("pano-A-2", *at_bearing(0.0), panorama_id="A",
                  requested_heading_deg=170.0),
    ]
    need = demand("obligation:front", "front", viewpoints_required=2)
    context = sector_context(need.demand_id, 0.0)

    measures = [
        measure_candidate(c, need, [], 3, target_lat=0.0, target_lon=0.0,
                          policy=SEARCH, sector=context)
        for c in framings
    ]
    viewpoints = group_viewpoints(
        framings, DEFAULT_POLICY.geometry.viewpoint_separation_m
    )

    retained = select_for_demand(
        measures, {c.candidate_id: c for c in framings}, need,
        0.0, 0.0, wanted=2, policy=SEARCH, viewpoints=viewpoints,
    )

    assert len(retained) == 1, (
        "le besoin reste partiellement couvert, ce qui est plus vrai que de "
        "le compléter avec un second cadrage de la même position"
    )


def test_a_corner_view_does_not_fill_a_principal_quota() -> None:
    """Classer une vue de coin ne suffit pas : il faut l'écarter du quota.

    `demands_assess._serves` exige l'égalité du secteur discret. Recommander
    une vue de coin pour remplir un besoin de façade ferait acheter au plan une
    image que l'évaluation refuserait ensuite de compter : la boucle ne se
    fermerait jamais.
    """
    frontal = candidate("c-face", *at_bearing(5.0))
    corner = candidate("c-coin", *at_bearing(50.0))
    need = demand("obligation:front", "front", viewpoints_required=2)
    context = sector_context(need.demand_id, 0.0)

    by_id = {c.candidate_id: c for c in (frontal, corner)}
    measures = [
        measure_candidate(c, need, [], 3, target_lat=0.0, target_lon=0.0,
                          policy=SEARCH, sector=context)
        for c in by_id.values()
    ]

    retained = select_for_demand(
        measures, by_id, need, 0.0, 0.0, wanted=2, policy=SEARCH,
    )

    assert "c-coin" not in retained, (
        "la vue de coin reste au manifeste comme auxiliaire, mais ne crédite "
        "pas un besoin de façade principale"
    )
    assert retained == ["c-face"]
