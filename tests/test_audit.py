"""L'audit chiffre ce qui est mesuré, ce qui est supposé, et ce qui manque."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.conditioning.audit import (
    SOURCE_FIDELITY,
    FidelityAudit,
    FidelityItem,
    audit,
)


def _square(half: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )


class _Prism:
    def __init__(self, feature_id, target, height, assumed, roof=0):
        self.feature_id = feature_id
        self.is_target = target
        self.height_m = height
        self.height_assumed = assumed
        self.height_source = "hypothèse — hors tuile" if assumed else "nuage"
        self.footprint = _square(10.0)
        self.roof_faces = list(range(roof))
        self.roof_vertices = None

    @property
    def roof_measured(self):
        return len(self.roof_faces) > 0


class _Scene:
    def __init__(self, prisms):
        self.prisms = prisms
        self.hotel_id = "t"
        self.centre = (0.0, 0.0)

    @property
    def target(self):
        return next((p for p in self.prisms if p.is_target), None)


# --- pondération ------------------------------------------------------------


def test_le_score_pondere_par_la_surface() -> None:
    """Un voisin supposé pèse plus qu'un arbuste mesuré."""
    grand = FidelityItem("grand", "assumed", 1000.0, "")
    petit = FidelityItem("petit", "lidar_ndsm", 10.0, "")
    result = FidelityAudit(hotel_id="t", items=[grand, petit])

    # Le score doit rester proche de la fidélité du poste dominant.
    assert result.score < 0.3


def test_une_scene_entierement_mesuree_score_haut() -> None:
    item = FidelityItem("cible", "lidar_ndsm", 500.0, "")
    assert FidelityAudit(hotel_id="t", items=[item]).score == pytest.approx(1.0)


def test_une_scene_vide_ne_divise_pas_par_zero() -> None:
    assert FidelityAudit(hotel_id="t").score == 0.0
    assert FidelityAudit(hotel_id="t").by_source() == {}


def test_la_part_par_source_totalise_un() -> None:
    result = FidelityAudit(
        hotel_id="t",
        items=[
            FidelityItem("a", "assumed", 300.0, ""),
            FidelityItem("b", "lidar_cloud", 700.0, ""),
        ],
    )
    assert sum(result.by_source().values()) == pytest.approx(1.0)


# --- classement des écarts --------------------------------------------------


def test_les_ecarts_sont_ranges_par_poids() -> None:
    """Un écart se juge à la surface qu'il touche, non à son existence."""
    leger = FidelityItem("léger", "assumed", 10.0, "", gap="peu de chose")
    lourd = FidelityItem("lourd", "assumed", 5000.0, "", gap="beaucoup")
    sans = FidelityItem("sans", "lidar_ndsm", 900.0, "")

    gaps = FidelityAudit(hotel_id="t", items=[leger, lourd, sans]).gaps()

    assert [g.poste for g in gaps] == ["lourd", "léger"]


def test_un_poste_mesure_sans_ecart_n_est_pas_liste() -> None:
    item = FidelityItem("ok", "lidar_ndsm", 100.0, "")
    assert FidelityAudit(hotel_id="t", items=[item]).gaps() == []


# --- audit d'une scène ------------------------------------------------------


def test_un_volume_suppose_est_declare_comme_tel() -> None:
    scene = _Scene([_Prism("OBST", False, 8.0, assumed=True)])

    result = audit(scene)

    assert result.items[0].source == "assumed"
    assert result.items[0].fidelity == SOURCE_FIDELITY["assumed"]
    assert "hors tuile" in result.items[0].gap


def test_un_toit_mesure_est_credite_au_plus_haut() -> None:
    scene = _Scene([_Prism("TARGET", True, 12.0, assumed=False, roof=800)])

    result = audit(scene)

    assert result.items[0].source == "lidar_ndsm"
    # Même mesuré, il reste un écart : la façade n'est pas relevée.
    assert "façade" in result.items[0].gap


def test_une_hauteur_mesuree_sans_toit_est_intermediaire() -> None:
    scene = _Scene([_Prism("OBST", False, 9.0, assumed=False)])

    result = audit(scene)

    assert result.items[0].source == "lidar_cloud"
    assert SOURCE_FIDELITY["assumed"] < result.items[0].fidelity < 1.0


# --- projection -------------------------------------------------------------


def test_les_volumes_supposes_projettent_un_gain() -> None:
    """C'est l'objet de l'analyse : dire ce que la source suivante rapporte."""
    scene = _Scene(
        [
            _Prism("TARGET", True, 12.0, assumed=False, roof=800),
            _Prism("OBST", False, 8.0, assumed=True),
        ]
    )

    result = audit(scene)
    tuiles = [p for p in result.projections if "tuiles" in p.levier]

    assert tuiles
    assert tuiles[0].gain_points > 0
    assert "OBST" in " ".join(tuiles[0].postes)


def test_une_scene_entierement_mesuree_ne_projette_pas_de_tuiles() -> None:
    scene = _Scene([_Prism("TARGET", True, 12.0, assumed=False, roof=800)])

    result = audit(scene)

    assert [p for p in result.projections if "tuiles" in p.levier] == []


def test_les_projections_sont_rangees_par_gain() -> None:
    scene = _Scene(
        [
            _Prism("TARGET", True, 12.0, assumed=False, roof=800),
            _Prism("A", False, 8.0, assumed=True),
        ]
    )

    result = audit(scene)
    gains = [p.gain_points for p in result.projections]

    assert gains == sorted(gains, reverse=True)


def test_le_rapport_porte_ses_reserves() -> None:
    joined = " ".join(FidelityAudit(hotel_id="t").as_dict()["caveats"])

    assert "surface visible" in joined
    assert "probabilités" in joined
