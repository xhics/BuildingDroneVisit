"""Couverture d'apparence mesurée, façade par façade.

`appearance_coverage` était un littéral : « partial » pour FACADE_PRIMARY,
« none » pour les autres. Vrai par accident sur le pilote, faux sur tout autre
bâtiment. Ces tests fixent le comportement sur une géométrie dont la vérité est
connue d'avance, puis vérifient qu'il n'existe aucune valeur en dur.
"""

from __future__ import annotations

from shapely.geometry import LineString, Polygon

from hotel_pipeline.geo.facade_coverage import (
    FacadeCoverage,
    coverage_from_subjects,
    sample_facade,
    visible_points,
)

#: Un carré de 20 m, murs alignés sur les axes. Le mur sud va de (0,0) à (20,0)
#: et regarde vers le sud : tout observateur en y < 0 le voit de face.
CARRE = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
MUR_SUD = LineString([(0, 0), (20, 0)])
MUR_NORD = LineString([(0, 20), (20, 20)])


def test_the_outward_normal_points_away_from_the_footprint() -> None:
    samples = sample_facade(MUR_SUD, CARRE)

    assert samples, "le mur doit produire des points"
    # Toutes les normales du mur sud regardent vers le sud.
    assert all(sample.normal[1] < 0 for sample in samples)


def test_a_wall_is_seen_from_its_own_side() -> None:
    samples = sample_facade(MUR_SUD, CARRE)

    seen, report = visible_points(
        samples, origin=(10.0, -30.0), footprint=CARRE, obstacles=[],
        heading_deg=0.0, fov_deg=180.0,
    )

    assert report.observed_fraction == 1.0
    assert len(seen) == len(samples)


def test_a_wall_is_never_seen_through_its_own_building() -> None:
    """Le test de face manquait : un mur arrière comptait depuis l'avant."""
    samples = sample_facade(MUR_NORD, CARRE)

    _, report = visible_points(
        samples, origin=(10.0, -30.0), footprint=CARRE, obstacles=[],
        heading_deg=0.0, fov_deg=180.0,
    )

    assert report.observed_fraction == 0.0
    # Rejeté parce qu'il tourne le dos, pas parce qu'il sort du cadre.
    assert report.rejected_back_facing == report.sampled


def test_a_neighbour_building_hides_the_wall() -> None:
    samples = sample_facade(MUR_SUD, CARRE)
    ecran = Polygon([(-5, -12), (25, -12), (25, -8), (-5, -8)])

    _, report = visible_points(
        samples, origin=(10.0, -30.0), footprint=CARRE, obstacles=[ecran],
        heading_deg=0.0, fov_deg=180.0,
    )

    assert report.observed_fraction == 0.0
    assert report.rejected_occluded == report.sampled


def test_a_narrow_field_of_view_crops_the_wall() -> None:
    """Le cadrage retranche : une caméra étroite ne voit qu'une part du mur."""
    samples = sample_facade(MUR_SUD, CARRE)

    _, large = visible_points(
        samples, origin=(10.0, -15.0), footprint=CARRE, obstacles=[],
        heading_deg=0.0, fov_deg=180.0,
    )
    _, etroit = visible_points(
        samples, origin=(10.0, -15.0), footprint=CARRE, obstacles=[],
        heading_deg=0.0, fov_deg=20.0,
    )

    assert large.observed_fraction > etroit.observed_fraction
    assert etroit.rejected_out_of_frame > 0


def test_distance_beyond_the_limit_is_not_exploitable() -> None:
    samples = sample_facade(MUR_SUD, CARRE)

    _, report = visible_points(
        samples, origin=(10.0, -400.0), footprint=CARRE, obstacles=[],
        heading_deg=0.0, fov_deg=180.0, max_distance_m=150.0,
    )

    assert report.observed_fraction == 0.0
    assert report.rejected_too_far == report.sampled


def test_an_unknown_heading_does_not_crop_but_still_requires_facing() -> None:
    """Cap absent = cadrage inconnu, jamais cadrage total sur 360°."""
    samples = sample_facade(MUR_NORD, CARRE)

    _, report = visible_points(
        samples, origin=(10.0, -30.0), footprint=CARRE, obstacles=[],
        heading_deg=None, fov_deg=None,
    )

    # Le mur nord reste invisible depuis le sud, cap connu ou non.
    assert report.observed_fraction == 0.0
    assert report.rejected_back_facing == report.sampled


# --- cumul de plusieurs vues ------------------------------------------------


def _sujets(*origins):
    return [
        (f"vue-{index}", origin, heading, fov)
        for index, (origin, heading, fov) in enumerate(origins)
    ]


def test_two_partial_views_cover_more_than_either_alone() -> None:
    """C'est le cas qui justifie l'union : deux moitiés font un mur."""
    samples = sample_facade(MUR_SUD, CARRE)
    gauche = ((2.0, -12.0), 0.0, 60.0)
    droite = ((18.0, -12.0), 0.0, 60.0)

    seule = coverage_from_subjects(
        "FACADE_SUD", samples, _sujets(gauche), CARRE, []
    )
    deux = coverage_from_subjects(
        "FACADE_SUD", samples, _sujets(gauche, droite), CARRE, []
    )

    assert deux.union_fraction > seule.union_fraction
    # La meilleure vue seule ne progresse pas : le recouvrement n'est pas un
    # recalage, et le rapport ne doit pas laisser croire l'inverse.
    assert deux.best_fraction == seule.best_fraction
    assert len(deux.contributing) == 2


def test_appearance_coverage_is_derived_not_declared() -> None:
    aucune = FacadeCoverage("F", appearance_union_fraction=0.0, sampled=10)
    partielle = FacadeCoverage("F", appearance_union_fraction=0.5, sampled=10)
    complete = FacadeCoverage("F", appearance_union_fraction=0.95, sampled=10)

    assert aucune.appearance_coverage == "none"
    assert partielle.appearance_coverage == "partial"
    assert complete.appearance_coverage == "full"


def test_a_wall_with_no_view_reports_none_and_names_nobody() -> None:
    samples = sample_facade(MUR_NORD, CARRE)

    coverage = coverage_from_subjects(
        "FACADE_NORD", samples, _sujets(((10.0, -30.0), 0.0, 90.0)), CARRE, []
    )

    assert coverage.appearance_coverage == "none"
    assert coverage.contributing == []
    assert coverage.best_subject is None


# --- pondération sectorielle et distance -----------------------------------


def test_weighted_union_fraction_is_zero_when_no_view() -> None:
    samples = sample_facade(MUR_SUD, CARRE)

    coverage = coverage_from_subjects(
        "FACADE_SUD", samples, _sujets(((10.0, 30.0), 0.0, 90.0)), CARRE, []
    )

    assert coverage.weighted_union_fraction == 0.0
    assert coverage.union_fraction == 0.0


def test_weighted_union_fraction_boosts_frontal_views() -> None:
    samples = sample_facade(MUR_SUD, CARRE)
    frontal = ((10.0, -12.0), 0.0, 60.0)
    lateral = ((10.0, -12.0), 90.0, 60.0)

    coverage = coverage_from_subjects(
        "FACADE_SUD", samples,
        _sujets(frontal, lateral),
        CARRE, [],
    )

    assert coverage.union_fraction > 0.0
    assert coverage.weighted_union_fraction <= coverage.union_fraction
    assert coverage.weighted_union_fraction > 0.0


def test_weighted_union_fraction_decays_with_distance() -> None:
    samples = sample_facade(MUR_SUD, CARRE)
    proche = ((10.0, -12.0), 0.0, 60.0)
    loin = ((10.0, -150.0), 0.0, 60.0)

    coverage_proche = coverage_from_subjects(
        "FACADE_SUD", samples, _sujets(proche), CARRE, []
    )
    coverage_loin = coverage_from_subjects(
        "FACADE_SUD", samples, _sujets(loin), CARRE, []
    )

    assert coverage_proche.weighted_union_fraction >= coverage_loin.weighted_union_fraction


def test_appearance_coverage_uses_weighted_union() -> None:
    """Le seuil 0.9 s'applique à weighted_union_fraction, pas union_fraction."""
    complete = FacadeCoverage("F", union_fraction=1.0, weighted_union_fraction=0.95, sampled=10)
    assert complete.appearance_coverage == "full"

    partial = FacadeCoverage("F", union_fraction=1.0, weighted_union_fraction=0.5, sampled=10)
    assert partial.appearance_coverage == "partial"

    aucune = FacadeCoverage("F", union_fraction=0.0, weighted_union_fraction=0.0, sampled=10)
    assert aucune.appearance_coverage == "none"


def test_facade_coverage_as_dict_includes_weighted_fraction() -> None:
    coverage = FacadeCoverage(
        "FACADE_PRIMARY",
        union_fraction=0.8,
        weighted_union_fraction=0.7,
        best_fraction=0.6,
        best_subject="vue-1",
        best_distance_m=25.0,
        contributing=["vue-1", "vue-2"],
        sampled=10,
    )

    d = coverage.as_dict()
    assert "weighted_union_fraction" in d
    assert d["weighted_union_fraction"] == 0.7
    assert d["appearance_coverage"] == "partial"
