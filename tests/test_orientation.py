"""Établir l'orientation depuis les façades (collecte V2).

L'azimut avant venait du centroïde d'un stationnement supposé : 137,7° pour une
façade qui en vaut 227. Une erreur de quatre-vingt-dix degrés, invisible tant
que rien ne confrontait l'hypothèse à une image.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from hotel_pipeline.orientation import (
    OrientationEvidence,
    OrientationUndetermined,
    decide,
    group_collinear,
    segment_seen_from,
    segments_of,
)

#: Un rectangle simple : côtés à 0°, 90°, 180°, 270° de normale extérieure.
CARRÉ = Polygon([(0, 0), (40, 0), (40, 20), (0, 20), (0, 0)])


def _evidence(asset_id="a1", segment_index=0, normal=0.0, **overrides):
    fields = dict(
        asset_id=asset_id, checksum="c" * 64,
        camera_lat=45.5, camera_lon=-73.4,
        segment_index=segment_index, segment_normal_deg=normal,
        observation="enseigne et entrée visibles",
    )
    fields.update(overrides)
    return OrientationEvidence(**fields)


# --- la normale sort du bâtiment ----------------------------------------------


def test_normals_point_outward_whatever_the_ring_order() -> None:
    """Choisir par convention d'orientation des sommets dépendrait du sens dans
    lequel la source a écrit l'anneau."""
    direct = {round(s.outward_normal_deg) for s in segments_of(CARRÉ)}
    inverse = {
        round(s.outward_normal_deg)
        for s in segments_of(Polygon(list(CARRÉ.exterior.coords)[::-1]))
    }

    assert direct == inverse == {0, 90, 180, 270}


# --- un mur, plusieurs segments -----------------------------------------------


def test_collinear_segments_form_one_wall() -> None:
    """Un mur réel est découpé par les décrochements du relevé.

    Sur le pilote : 226,0°, 229,0°, 227,7° et 228,4° sont la même façade.
    """
    murs = group_collinear(segments_of(CARRÉ), tolerance_deg=8.0)

    assert len(murs) == 4, "un rectangle a quatre murs distincts"


def test_the_common_normal_is_weighted_by_length() -> None:
    """Un décrochement de deux mètres ne pèse pas autant qu'un mur de quarante :
    la moyenne simple laisserait un détail du relevé déplacer la façade."""
    from hotel_pipeline.orientation import FacadeGroup, Segment

    mur = FacadeGroup(segments=[
        Segment(index=0, length_m=40.0, outward_normal_deg=228.0),
        Segment(index=1, length_m=2.0, outward_normal_deg=222.0),
    ])

    assert mur.normal_deg == pytest.approx(227.7, abs=0.3), (
        "le long segment domine, comme il le doit"
    )
    assert mur.total_length_m == 42.0


def test_a_tighter_tolerance_splits_what_a_looser_one_joins() -> None:
    """La tolérance vient de la politique : elle décide de ce qu'est un mur."""
    from hotel_pipeline.orientation import Segment

    segments = [
        Segment(index=0, length_m=10.0, outward_normal_deg=226.0),
        Segment(index=1, length_m=10.0, outward_normal_deg=232.0),
    ]

    assert len(group_collinear(segments, tolerance_deg=8.0)) == 1
    assert len(group_collinear(segments, tolerance_deg=4.0)) == 2


# --- le segment vu est celui qui fait face ------------------------------------


def test_the_segment_seen_is_the_one_facing_the_camera() -> None:
    """Le rayon vers le centroïde perce d'abord ce qui se trouve sur son
    chemin — souvent un décrochement latéral, dont la normale n'a rien à voir
    avec le mur observé.

    Sur le pilote, la même façade se lisait ainsi 227,7° depuis une vue et
    139,3° depuis l'autre, alors que les deux la regardent.
    """
    segments = segments_of(CARRÉ)

    # Caméra plein sud : elle regarde le mur du bas, normale 180°.
    vu = segment_seen_from(CARRÉ, (20, -30), segments)
    assert round(vu.outward_normal_deg) == 180

    # Caméra plein nord : le mur du haut, normale 0°.
    vu = segment_seen_from(CARRÉ, (20, 50), segments)
    assert round(vu.outward_normal_deg) == 0


def test_a_segment_turning_its_back_is_never_selected() -> None:
    """Au-delà d'un quart de tour, le segment ne peut pas être ce que la caméra
    photographie."""
    segments = segments_of(CARRÉ)
    vu = segment_seen_from(CARRÉ, (20, -30), segments)

    écart = abs((180 - vu.outward_normal_deg + 180) % 360 - 180)
    assert écart < 90


def test_the_choice_is_deterministic() -> None:
    """Deux exécutions rendent le même segment, sinon l'orientation dériverait
    d'une exécution à l'autre."""
    segments = segments_of(CARRÉ)
    premier = segment_seen_from(CARRÉ, (20, -30), segments)
    second = segment_seen_from(CARRÉ, (20, -30), list(reversed(segments)))

    assert premier.index == second.index


# --- décider, ou refuser de décider -------------------------------------------


def test_two_views_of_the_same_wall_decide() -> None:
    segments = segments_of(CARRÉ)
    bas = next(s for s in segments if round(s.outward_normal_deg) == 180)

    décision = decide(
        "pilote", "d" * 16, CARRÉ,
        [
            _evidence("a1", bas.index, bas.outward_normal_deg),
            _evidence("a2", bas.index, bas.outward_normal_deg),
        ],
        tolerance_deg=8.0, decided_by="opérateur",
        rationale="les deux vues montrent l'entrée sous fronton",
    )

    assert décision.front_azimuth_deg == pytest.approx(180.0, abs=0.1)
    assert décision.method == "facade_segments_confirmed_by_imagery"
    assert len(décision.evidence) == 2


def test_contradictory_evidence_leaves_the_orientation_undetermined() -> None:
    """Deux preuves qui désignent des murs opposés ne se concilient pas en
    prenant le milieu : elles disent qu'on ne sait pas."""
    segments = segments_of(CARRÉ)
    bas = next(s for s in segments if round(s.outward_normal_deg) == 180)
    haut = next(s for s in segments if round(s.outward_normal_deg) == 0)

    with pytest.raises(OrientationUndetermined, match="murs différents"):
        decide(
            "pilote", "d" * 16, CARRÉ,
            [
                _evidence("a1", bas.index, bas.outward_normal_deg),
                _evidence("a2", haut.index, haut.outward_normal_deg),
            ],
            tolerance_deg=8.0, decided_by="opérateur",
            rationale="preuves opposées",
        )


def test_no_evidence_decides_nothing() -> None:
    """Une orientation sans vue qui la porte n'est qu'une hypothèse de plus —
    c'est exactement ce qu'était le stationnement."""
    with pytest.raises(OrientationUndetermined, match="aucune preuve"):
        decide("pilote", "d" * 16, CARRÉ, [], 8.0, "opérateur", "sans preuve")


def test_the_decision_states_what_it_does_not_establish() -> None:
    """La photo confirme ce que le mur porte ; elle ne le localise pas."""
    segments = segments_of(CARRÉ)
    bas = next(s for s in segments if round(s.outward_normal_deg) == 180)

    décision = decide(
        "pilote", "d" * 16, CARRÉ, [_evidence("a1", bas.index, 180.0)],
        8.0, "opérateur", "entrée visible",
    )
    publié = décision.as_dict()

    assert any("ne le localise pas" in limite for limite in publié["limits"])
    assert any("jamais décider seuls" in limite for limite in publié["limits"])
    assert publié["evidence"][0]["sha256"] == "c" * 64


def test_a_nearer_wall_turning_its_back_loses_to_a_farther_facing_one() -> None:
    """Le filtre des 90° n'est pas redondant : sans lui, un mur plus proche
    mais tourné à l'opposé pourrait être choisi.

    Un rectangle simple ne le montre pas — l'écart nul du mur qui fait face
    gagne de toute façon. Il faut une forme où l'écart le plus faible n'est pas
    celui du mur observé.
    """
    from hotel_pipeline.orientation import Segment, segment_seen_from

    # Une forme en L : la caméra est logée dans le creux, où un retour de mur
    # lui tourne le dos tout en étant plus proche qu'elle.
    from shapely.geometry import Polygon

    forme = Polygon([
        (0, 0), (40, 0), (40, 30), (20, 30), (20, 10), (0, 10), (0, 0),
    ])
    segments = segments_of(forme)
    vu = segment_seen_from(forme, (10, -20), segments)

    # Depuis le sud, seul le mur du bas (normale 180°) fait face.
    assert round(vu.outward_normal_deg) == 180, (
        f"segment {vu.index} de normale {vu.outward_normal_deg}° retenu"
    )
    écart = abs((180 - vu.outward_normal_deg + 180) % 360 - 180)
    assert écart < 90


def test_all_walls_turning_their_backs_yield_nothing() -> None:
    """Une caméra à l'intérieur ne photographie aucune façade extérieure :
    rendre un segment quand même inventerait une observation."""
    from hotel_pipeline.orientation import segment_seen_from

    assert segment_seen_from(CARRÉ, (20, 10), segments_of(CARRÉ)) is None


def test_a_long_wall_is_not_outvoted_by_two_short_offsets() -> None:
    """La pondération par la longueur, éprouvée sur un cas qui la discrimine.

    Deux décrochements de deux mètres à 240° contre un mur de quarante à 226° :
    sans pondération, la moyenne partirait vers 235°.
    """
    from hotel_pipeline.orientation import FacadeGroup, Segment

    mur = FacadeGroup(segments=[
        Segment(index=0, length_m=40.0, outward_normal_deg=226.0),
        Segment(index=1, length_m=2.0, outward_normal_deg=240.0),
        Segment(index=2, length_m=2.0, outward_normal_deg=240.0),
    ])

    assert mur.normal_deg == pytest.approx(227.3, abs=0.4), (
        "le mur de quarante mètres l'emporte sur deux décrochements"
    )
    # Sans pondération, la moyenne des trois vaudrait environ 235,3°.
    assert abs(mur.normal_deg - 235.3) > 5.0


# --- propager, non seulement décider ------------------------------------------


def test_the_four_walls_are_named_from_the_front_azimuth() -> None:
    """`SECTOR_CENTRES` porte aussi les coins : parcourir la liste entière
    faisait retenir `front_right_corner` à 45° avant `right` à 90°, et deux
    façades sur quatre restaient sans nom."""
    from hotel_pipeline.orientation import facades_from

    murs = facades_from(CARRÉ, front_azimuth_deg=180.0, tolerance_deg=8.0)

    assert set(murs) == {
        "FACADE_PRIMARY", "FACADE_LEFT", "FACADE_RIGHT", "FACADE_REAR",
    }
    assert murs["FACADE_PRIMARY"]["normal_deg"] == pytest.approx(180.0, abs=0.1)
    assert murs["FACADE_REAR"]["normal_deg"] == pytest.approx(0.0, abs=0.1)


def test_an_oblique_wall_joins_the_nearest_side() -> None:
    """Les quadrants sont jointifs : un bâtiment n'a pas de façade qui ne
    regarde nulle part.

    Un mur à 45° exactement bascule au côté suivant dès qu'il le dépasse — la
    frontière est nette, et aucun mur ne reste sans nom.
    """
    from hotel_pipeline.orientation import FacadeGroup, Segment, facade_for

    def côté(normale):
        return facade_for(
            FacadeGroup(segments=[
                Segment(index=0, length_m=10.0, outward_normal_deg=normale)
            ]),
            front_azimuth_deg=180.0,
        )

    assert côté(180.0) == "FACADE_PRIMARY"
    assert côté(225.0) == "FACADE_PRIMARY", "45° exactement reste du côté avant"
    assert côté(226.0) == "FACADE_RIGHT", "au-delà, il bascule"
    assert côté(270.0) == "FACADE_RIGHT"
    assert côté(0.0) == "FACADE_REAR"


def test_the_longest_wall_wins_its_side() -> None:
    """Un bâtiment présente souvent plusieurs pans vers la même direction :
    prendre le premier venu ferait dépendre la façade de l'ordre des sommets."""
    from shapely.geometry import Polygon

    from hotel_pipeline.orientation import facades_from

    # Deux pans vers le sud : 30 m au niveau y=0, 8 m au niveau y=5. Le plus
    # long doit l'emporter, quel que soit l'ordre des sommets.
    forme = Polygon([
        (0, 0), (30, 0), (30, 5), (38, 5), (38, 25), (0, 25), (0, 0),
    ])
    murs = facades_from(forme, front_azimuth_deg=180.0, tolerance_deg=8.0)

    sud = murs["FACADE_PRIMARY"]
    assert sud["length_m"] == pytest.approx(38.0, abs=0.1), (
        "les deux pans sud sont colinéaires en normale : ils forment un mur"
    )
    assert sorted(sud["segments"]) == [0, 2]


def test_the_demand_activation_sorts_by_time_not_by_name() -> None:
    """Les fichiers sont nommés d'après l'empreinte des besoins, dont l'ordre
    alphabétique n'a rien à voir avec leur chronologie.

    Le tri par nom activait un manifeste à neuf besoins alors que la dernière
    construction en produisait sept.
    """
    import json
    import os
    import time

    from hotel_pipeline.cli import _activate_latest_demands
    from hotel_pipeline.workspace import Workspace

    import tempfile

    with tempfile.TemporaryDirectory() as racine:
        os.environ["HOTEL_PIPELINE_WORK"] = racine
        workspace = Workspace("essai")
        sources = workspace.path("01_sources")
        sources.mkdir(parents=True, exist_ok=True)

        # « zzz » est postérieur par le nom, antérieur par la date.
        (sources / "capture_demands_zzz.json").write_text(
            json.dumps({"hotel_id": "essai", "demands": [{"demand_id": "a"}] * 9}),
            "utf-8",
        )
        time.sleep(0.01)
        (sources / "capture_demands_aaa.json").write_text(
            json.dumps({"hotel_id": "essai", "demands": [{"demand_id": "b"}] * 7}),
            "utf-8",
        )

        résultat = _activate_latest_demands(workspace)

        assert "7 besoin" in résultat, résultat
        canonique = json.loads(
            (sources / "capture_demands.json").read_text("utf-8")
        )
        assert len(canonique["demands"]) == 7
    os.environ.pop("HOTEL_PIPELINE_WORK", None)


def test_the_bearing_accounts_for_meridian_convergence() -> None:
    """Un `atan2` sur des degrés géographiques ignore la latitude.

    À 45°, un degré de longitude vaut environ 0,71 degré de latitude en
    distance : la formule naïve tordait 46 secteurs sur 313 du pilote. Et
    vérifier avec la même formule qu'on applique ne prouve rien — c'est ce qui
    avait fait conclure à « 0/313 » alors que 46 étaient faux.
    """
    import math

    from hotel_pipeline.visibility import bearing_deg

    # Un point au nord-est exact d'un bâtiment situé à 45° de latitude.
    b_lat, b_lon = 45.0, -73.0
    c_lat, c_lon = 45.001, -72.999

    naif = math.degrees(math.atan2(c_lon - b_lon, c_lat - b_lat)) % 360.0
    canonique = bearing_deg(b_lat, b_lon, c_lat, c_lon)

    assert abs(naif - canonique) > 5.0, (
        "les deux formules diffèrent assez pour changer de secteur"
    )
    # Le vrai relèvement penche vers le nord : à cette latitude, un degré de
    # longitude est plus court qu'un degré de latitude.
    assert canonique < naif


def test_the_two_formulas_can_disagree_on_the_sector() -> None:
    """Le cas qui compte : pas seulement un écart d'angle, un **secteur**
    différent.

    À 45° de latitude, une caméra 22 m au nord et 47 m à l'est du bâtiment est
    classée « arrière » par le relèvement canonique et « coin arrière-gauche »
    par la formule naïve. C'est ce genre d'écart qui rendait 46 secteurs faux.
    """
    import math

    from hotel_pipeline.sectors import sector_for
    from hotel_pipeline.visibility import bearing_deg

    b_lat, b_lon, front = 45.0, -73.0, 227.89
    lat, lon = b_lat + 0.0002, b_lon + 0.0006

    canonique = sector_for(bearing_deg(b_lat, b_lon, lat, lon), front)
    naif = sector_for(
        math.degrees(math.atan2(lon - b_lon, lat - b_lat)) % 360.0, front
    )

    assert canonique.value == "rear"
    assert naif.value == "rear_left_corner"
    assert canonique != naif, (
        "sans cette différence, le test ne prouverait rien sur la formule"
    )
