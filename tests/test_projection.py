"""Une projection annonce un intervalle, enregistre son réalisé, se corrige."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.conditioning.projection import (
    DEFAULT_YIELD,
    Ledger,
    LedgerEntry,
    simulate,
)


def _entry(predicted: float, realised: float | None, source="lidar_cloud"):
    return LedgerEntry(
        hotel_id="h",
        levier="tuiles",
        source=source,
        predicted_median=predicted,
        predicted_p10=predicted * 0.8,
        predicted_p90=predicted * 1.2,
        coverage=0.78,
        score_before=0.465,
        recorded_at="2026-01-01T00:00:00Z",
        realised_points=realised,
        score_after=None if realised is None else 0.9,
    )


# --- simulation -------------------------------------------------------------


def test_la_projection_rend_un_intervalle_et_non_un_point() -> None:
    gain = simulate("t", 100.0, 200.0, 0.15, 0.90, coverage=0.8)

    assert gain.quantile(0.10) < gain.median < gain.quantile(0.90)
    assert gain.samples.size > 100


def test_une_couverture_plus_large_projette_davantage() -> None:
    faible = simulate("t", 100.0, 200.0, 0.15, 0.90, coverage=0.4)
    forte = simulate("t", 100.0, 200.0, 0.15, 0.90, coverage=0.95)

    assert forte.median > faible.median


def test_un_rendement_incertain_elargit_l_intervalle() -> None:
    """L'intervalle doit refléter ce qu'on ignore, pas seulement la moyenne."""
    sur = simulate(
        "t", 100.0, 200.0, 0.15, 0.90, coverage=0.8,
        yields={"lidar_cloud": (0.9, 0.02)},
    )
    incertain = simulate(
        "t", 100.0, 200.0, 0.15, 0.90, coverage=0.8,
        yields={"lidar_cloud": (0.9, 0.25)},
    )

    largeur_sure = sur.quantile(0.90) - sur.quantile(0.10)
    largeur_floue = incertain.quantile(0.90) - incertain.quantile(0.10)
    assert largeur_floue > largeur_sure * 2


def test_une_surface_nulle_ne_projette_rien() -> None:
    assert simulate("t", 0.0, 200.0, 0.15, 0.90, coverage=0.8).median == 0.0


def test_la_simulation_est_reproductible() -> None:
    a = simulate("t", 100.0, 200.0, 0.15, 0.90, coverage=0.8, seed=7)
    b = simulate("t", 100.0, 200.0, 0.15, 0.90, coverage=0.8, seed=7)

    assert a.median == pytest.approx(b.median)


# --- registre ---------------------------------------------------------------


def test_un_realise_dans_l_intervalle_est_reconnu() -> None:
    entry = _entry(predicted=30.0, realised=32.0)

    assert entry.within_interval is True
    assert entry.error == pytest.approx(2.0)


def test_un_realise_hors_intervalle_est_signale() -> None:
    """Cas réel : projeté 26,8 points, réalisé 43,8."""
    entry = _entry(predicted=26.8, realised=43.8)

    assert entry.within_interval is False
    assert entry.error > 15.0


def test_une_projection_ouverte_n_a_pas_d_ecart() -> None:
    entry = _entry(predicted=30.0, realised=None)

    assert entry.error is None
    assert entry.within_interval is None


def test_le_resume_ignore_les_projections_ouvertes() -> None:
    ledger = Ledger([_entry(30.0, None), _entry(30.0, 33.0)])

    resume = ledger.summary()

    assert resume["entries"] == 2
    assert resume["closed"] == 1


def test_un_registre_vide_ne_calibre_rien() -> None:
    ledger = Ledger()

    assert ledger.calibration() == {}
    assert ledger.coverage_bias() == 1.0
    assert ledger.summary()["closed"] == 0


# --- calibration ------------------------------------------------------------


def test_la_calibration_pese_le_nombre_d_observations() -> None:
    """Un site ne fait pas une loi ; dix comptent nettement."""
    une = Ledger([_entry(26.8, 43.8)]).coverage_bias()
    dix = Ledger([_entry(26.8, 43.8)] * 10).coverage_bias()

    assert 1.0 < une < dix


def test_une_sous_estimation_repetee_releve_la_couverture() -> None:
    """Le rendement plafonne à un : c'est la couverture qui était trop basse."""
    ledger = Ledger([_entry(26.8, 43.8)] * 5)

    assert ledger.coverage_bias() > 1.1


def test_une_sur_estimation_abaisse_la_couverture() -> None:
    ledger = Ledger([_entry(40.0, 20.0)] * 5)

    assert ledger.coverage_bias() < 1.0


def test_le_biais_reste_borne() -> None:
    """Une correction ne doit pas promettre plus qu'une emprise ne contient."""
    fou = Ledger([_entry(1.0, 100.0)] * 20)

    assert fou.coverage_bias() <= 2.0


def test_la_calibration_corrige_la_projection_suivante() -> None:
    ledger = Ledger([_entry(26.8, 43.8)] * 5)

    avant = simulate("t", 122000.0, 235190.0, 0.15, 0.90, coverage=0.78)
    apres = simulate(
        "t", 122000.0, 235190.0, 0.15, 0.90,
        coverage=min(0.78 * ledger.coverage_bias(), 0.99),
        yields=ledger.calibration(),
    )

    assert apres.median > avant.median


def test_le_rendement_ne_depasse_jamais_un() -> None:
    calibration = Ledger([_entry(10.0, 90.0)] * 20).calibration()

    assert calibration["lidar_cloud"][0] <= 1.0


def test_le_rapport_porte_ses_reserves() -> None:
    joined = " ".join(Ledger().as_dict()["caveats"])

    assert "plusieurs" in joined
    assert "décevoir" in joined
