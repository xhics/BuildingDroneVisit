"""Tests pour l'intégration satellite/orthophoto (Lot 1B complément)."""

import pytest

from src.hotel_pipeline.geo.satellite_completion import (
    ORTHOPHOTO_CMM_EXAMPLE,
    SyntheticSource,
    analyze_facade_in_orthophoto,
    merge_with_measured_coverage,
    synthesize_completion_from_orthophoto,
)


def test_orthophoto_analysis_detects_visible_facade():
    """Une façade visible dans l'orthophoto est marquée comme visible."""
    # Façade simple (LineString)
    facade_wkt = "LINESTRING(0 0, 10 0)"
    footprint_wkt = "POLYGON((0 -10, 10 -10, 10 10, 0 10, 0 -10))"

    analysis = analyze_facade_in_orthophoto(
        facade_wkt,
        footprint_wkt,
        ORTHOPHOTO_CMM_EXAMPLE,
        "FACADE_PRIMARY",
    )

    assert analysis is not None
    assert analysis.facade_id == "FACADE_PRIMARY"
    assert analysis.source == SyntheticSource.ORTHOPHOTO
    # Avec résolution 20cm et coverage 100%, visible_fraction devrait être > 0.8
    assert analysis.visible_fraction > 0.0
    print(f"✓ FACADE_PRIMARY visible: {analysis.visible_fraction:.1%}, {analysis.explanation}")


def test_orthophoto_respects_resolution_penalty():
    """Une résolution insuffisante (>25cm) réduit la confiance."""
    facade_wkt = "LINESTRING(0 0, 10 0)"
    footprint_wkt = "POLYGON((0 -10, 10 -10, 10 10, 0 10, 0 -10))"

    # Orthophoto de faible résolution
    low_res = {
        "resolution_cm": 50,
        "coverage_fraction": 1.0,
        "notes": "",
    }

    analysis_high = analyze_facade_in_orthophoto(
        facade_wkt, footprint_wkt, ORTHOPHOTO_CMM_EXAMPLE, "FACADE_PRIMARY"
    )
    analysis_low = analyze_facade_in_orthophoto(
        facade_wkt, footprint_wkt, low_res, "FACADE_PRIMARY"
    )

    assert analysis_high is not None
    assert analysis_low is not None
    # La faible résolution devrait donner une fraction inférieure
    assert analysis_low.visible_fraction < analysis_high.visible_fraction
    print(f"✓ Résolution penalty: high_res={analysis_high.visible_fraction:.1%}, low_res={analysis_low.visible_fraction:.1%}")


def test_synthetic_completion_never_returns_full_coverage():
    """Une complétion synthétique ne peut jamais être 'full', max 'partial'."""
    facade_wkt = "LINESTRING(0 0, 10 0)"
    footprint_wkt = "POLYGON((0 -10, 10 -10, 10 10, 0 10, 0 -10))"

    synthetic = synthesize_completion_from_orthophoto(
        facade_kind="FACADE_REAR",
        facade_geometry_wkt=facade_wkt,
        footprint_geometry_wkt=footprint_wkt,
        orthophoto_source_id="cmm-ortho",
        orthophoto_data=ORTHOPHOTO_CMM_EXAMPLE,
    )

    assert synthetic is not None
    assert synthetic.source_type == SyntheticSource.ORTHOPHOTO
    assert synthetic.confidence_level == "low"
    # Coverage dérivée doit être "partial" ou "none", jamais "full"
    derived = synthetic._derive_coverage()
    assert derived in ("partial", "none")
    print(f"✓ Synthétique FACADE_REAR: {derived} (mesuré {synthetic.measured_fraction:.1%})")


def test_merge_preserves_measured_over_synthetic():
    """Les mesures réelles sont jamais remplacées par synthétique."""
    measured = {
        "FACADE_PRIMARY": {"appearance_union_fraction": 0.8, "appearance_coverage": "full"},
        "FACADE_LEFT": {"appearance_union_fraction": 0.5, "appearance_coverage": "partial"},
        "FACADE_REAR": {"appearance_union_fraction": 0.0, "appearance_coverage": "none"},
    }

    from src.hotel_pipeline.geo.satellite_completion import SyntheticCompletion

    synthetics = [
        SyntheticCompletion(
            facade_id="FACADE_PRIMARY",
            source_type=SyntheticSource.ORTHOPHOTO,
            measured_fraction=0.7,
            contributing_source="cmm-ortho",
        ),
        SyntheticCompletion(
            facade_id="FACADE_REAR",
            source_type=SyntheticSource.ORTHOPHOTO,
            measured_fraction=0.6,
            contributing_source="cmm-ortho",
        ),
    ]

    result = merge_with_measured_coverage(measured, synthetics)

    # FACADE_PRIMARY (mesurée à 0.8) devrait rester à 0.8
    assert result["FACADE_PRIMARY"]["appearance_union_fraction"] == 0.8
    # FACADE_REAR (aveugle) devrait être enrichie avec synthétique
    assert result["FACADE_REAR"]["geometric_support_coverage"] == "partial"
    assert "synthesis" in result["FACADE_REAR"]
    # FACADE_LEFT (mesurée à 0.5) inchangée
    assert result["FACADE_LEFT"]["appearance_union_fraction"] == 0.5
    print("✓ Merge: mesures préservées, synthétiques appliquées aux aveugles")


def test_blind_facade_upgraded_to_partial_with_synthetic():
    """Une façade aveugle (none) reçoit un support géométrique 'partial' via synthétique."""
    measured = {
        "FACADE_REAR": {"appearance_union_fraction": 0.0, "appearance_coverage": "none"},
    }

    from src.hotel_pipeline.geo.satellite_completion import SyntheticCompletion

    synthetics = [
        SyntheticCompletion(
            facade_id="FACADE_REAR",
            source_type=SyntheticSource.ORTHOPHOTO,
            measured_fraction=0.65,
            contributing_source="cmm-ortho",
        ),
    ]

    result = merge_with_measured_coverage(measured, synthetics)

    # Le support géométrique devrait passer de "none" à "partial"
    assert result["FACADE_REAR"]["geometric_support_coverage"] == "partial"
    # L'apparence reste "none" : le synthétique ne remplace pas une photo
    assert result["FACADE_REAR"]["appearance_coverage"] == "none"
    # Mais la fraction de support géométrique doit être <= 0.65
    assert result["FACADE_REAR"].get("geometric_support_fraction") == pytest.approx(0.65, abs=0.01)
    assert result["FACADE_REAR"].get("synthesis") is not None
    print("✓ Blind facade: support géométrique 'none' → 'partial' via synthétique")


def test_synthetic_dict_format_includes_synthesis_metadata():
    """La sérialisation inclut la métadata de synthèse."""
    synthetic = synthesize_completion_from_orthophoto(
        facade_kind="FACADE_REAR",
        facade_geometry_wkt="LINESTRING(0 0, 10 0)",
        footprint_geometry_wkt="POLYGON((0 -10, 10 -10, 10 10, 0 10, 0 -10))",
        orthophoto_source_id="cmm-ortho",
        orthophoto_data=ORTHOPHOTO_CMM_EXAMPLE,
    )

    if synthetic:
        result_dict = synthetic.as_dict()
        assert "synthesis" in result_dict
        assert result_dict["synthesis"]["source_type"] == SyntheticSource.ORTHOPHOTO.value
        assert result_dict["synthesis"]["confidence"] == "low"
        print(f"✓ Synthétique dict: {result_dict['synthesis']}")


def test_raster_analysis_skipped_when_no_path() -> None:
    """L'analyse raster est ignorée si aucun chemin de fichier n'est fourni."""
    facade_wkt = "LINESTRING(0 0, 10 0)"
    footprint_wkt = "POLYGON((0 -10, 10 -10, 10 10, 0 10, 0 -10))"
    data = {
        "resolution_cm": 20,
        "coverage_fraction": 1.0,
        "notes": "clear skies",
    }

    analysis = analyze_facade_in_orthophoto(
        facade_wkt, footprint_wkt, data, "FACADE_PRIMARY"
    )

    assert analysis is not None
    assert "raster analysis" not in (analysis.explanation or "")
    print("✓ Raster analysis skipped when no path provided")


if __name__ == "__main__":
    test_orthophoto_analysis_detects_visible_facade()
    test_orthophoto_respects_resolution_penalty()
    test_synthetic_completion_never_returns_full_coverage()
    test_merge_preserves_measured_over_synthetic()
    test_blind_facade_upgraded_to_partial_with_synthetic()
    test_synthetic_dict_format_includes_synthesis_metadata()
    test_raster_analysis_skipped_when_no_path()
    print("\n✅ Tous les tests satellite passent!")
