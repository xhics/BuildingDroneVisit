"""Le Router : décider comment reconstruire, et le motiver (collecte V2).

Ce que ces tests protègent avant tout : **la décision ne dérive jamais du
nombre brut d'images**. Sur ce site, trois cent treize vues portent le run de
visibilité et un seul besoin est partiellement couvert — un compteur d'images
aurait conclu à une couverture largement suffisante.

Ils protègent ensuite la séparation des deux sources : les besoins disent ce
qui est photographiquement couvert, le manifeste de site dit quels objets
existent et sont ciblables. Les fondre ferait d'un objet non résolu une lacune
de couverture.
"""

from __future__ import annotations

from hotel_pipeline.router import (
    CRITICAL_OBJECTS,
    DemandStanding,
    ObjectStanding,
    ProxyZone,
    Route,
    decide,
    standing_of,
)

#: Le bâtiment est établi et géoréférencé : sans cela, tout est bloqué, et
#: aucun autre test ne dirait rien d'intéressant.
SITE_SAIN = {"BUILDING_MAIN": ObjectStanding.TARGETABLE}

#: Une toiture qualifiée : le proxy qui rend l'hybride possible.
TOITURE = ProxyZone(
    zone="ROOFLINE_MAIN", artifact="lidar_mnt", qualified=True,
    note="emprise et hauteur mesurées",
)


def _besoin(demand_id, vues=0, requis=1, ciblable=True, status="open"):
    return DemandStanding(
        demand_id=demand_id, status=status, viewpoints_found=vues,
        viewpoints_required=requis, targetable=ciblable,
    )


# --- jamais depuis le nombre brut d'images ------------------------------------


def test_a_flood_of_images_does_not_make_a_site_photo_ready() -> None:
    """Le cœur du contrat.

    Trois cents vues autour d'un bâtiment peuvent n'en documenter aucune façade
    utilement : sur ce site, six acquisitions ont été réfutées une à une. La
    route se lit sur les besoins, non sur un compteur.
    """
    besoins = [_besoin(f"obligation:FACADE_{n}") for n in "ABCD"]

    decision = decide(
        "essai", besoins, SITE_SAIN, [TOITURE],
        artifacts=["visibility_run_X"],
        inputs={"assets_in_manifest": 313},
    )

    assert decision.route is not Route.PHOTO_FIRST_READY, (
        "313 assets au manifeste, zéro besoin couvert : un compteur d'images "
        "aurait conclu à l'inverse"
    )
    assert decision.independent_viewpoints == 0
    assert len(decision.demands_open) == 4


def test_the_decision_cites_what_founds_it() -> None:
    """Une décision qu'on ne peut pas contester n'est pas une décision."""
    decision = decide(
        "essai",
        [_besoin("obligation:FACADE_PRIMARY", vues=1, status="partially_met"),
         _besoin("obligation:FACADE_REAR")],
        SITE_SAIN, [TOITURE], artifacts=["visibility_run_X", "orientation_Y"],
    )
    publié = decision.as_dict()

    assert publié["rationale"], "une route sans motif est un verdict"
    assert publié["photographic"]["open"] == ["obligation:FACADE_REAR"]
    assert publié["active_artifacts"] == ["orientation_Y", "visibility_run_X"]
    assert publié["contract_version"] >= 1
    assert publié["decided_at"], "une décision se date, sinon rien ne la périme"
    assert publié["limits"], "ce que la décision ne prétend pas établir"


# --- les deux sources ne se fondent pas ---------------------------------------


def test_an_unresolved_object_is_not_a_coverage_gap() -> None:
    """`PARKING_HOTEL` n'est pas une façade non photographiée.

    L'un demande une localisation ou une preuve, l'autre une prise de vue. Les
    confondre enverrait quelqu'un photographier un objet dont on ignore
    l'emplacement.
    """
    objets = dict(SITE_SAIN, PARKING_HOTEL=ObjectStanding.REFUTED)

    decision = decide(
        "essai", [_besoin("obligation:FACADE_PRIMARY", vues=1)],
        objets, [TOITURE], artifacts=[],
    )
    publié = decision.as_dict()

    assert "PARKING_HOTEL" not in publié["photographic"]["open"]
    assert publié["site"]["by_standing"]["refuted"] == ["PARKING_HOTEL"]


def test_a_known_object_without_geometry_is_reported_apart() -> None:
    """`PROPERTY_SIGN` est connu, sans géométrie : on sait qu'il existe, pas
    où le viser. Ce n'est ni satisfait, ni simplement ouvert."""
    decision = decide(
        "essai",
        [_besoin("obligation:PROPERTY_SIGN", ciblable=False),
         _besoin("obligation:FACADE_PRIMARY", vues=1)],
        dict(SITE_SAIN, PROPERTY_SIGN=ObjectStanding.KNOWN_NOT_TARGETABLE),
        [TOITURE], artifacts=[],
    )

    assert decision.demands_not_targetable == ["obligation:PROPERTY_SIGN"]
    assert "obligation:PROPERTY_SIGN" not in decision.demands_open, (
        "un besoin non ciblable n'est pas un besoin qu'une caméra comblerait"
    )
    assert any(
        "localiser" in action for action in decision.next_actions
    ), "l'action attendue est une localisation, non une prise de vue"


def test_standing_distinguishes_refuted_from_merely_unresolved() -> None:
    """Un objet démenti n'est pas un objet dont on manque de données."""
    assert standing_of("inferred", True) is ObjectStanding.TARGETABLE
    assert standing_of("inferred", False) is ObjectStanding.KNOWN_NOT_TARGETABLE
    assert standing_of("unresolved", False) is ObjectStanding.UNRESOLVED
    assert standing_of("unresolved", False, refuted=True) is ObjectStanding.REFUTED


# --- les quatre routes ---------------------------------------------------------


def test_a_missing_critical_object_blocks_everything() -> None:
    """L'ordre des tests n'est pas indifférent : le prérequis l'emporte sur la
    couverture, car photographier sans savoir quoi viser ne produit rien."""
    decision = decide(
        "essai",
        [_besoin(f"obligation:D{n}", vues=3) for n in range(4)],
        {"BUILDING_MAIN": ObjectStanding.UNRESOLVED}, [TOITURE], artifacts=[],
    )

    assert decision.route is Route.BLOCKED_PREREQUISITES, (
        "tous les besoins sont couverts, et pourtant on ne sait pas où viser"
    )
    assert decision.blocking and "BUILDING_MAIN" in decision.blocking[0]


def test_full_coverage_without_proxy_is_photo_first() -> None:
    decision = decide(
        "essai", [_besoin(f"obligation:D{n}", vues=2) for n in range(4)],
        SITE_SAIN, proxies=[], artifacts=[],
    )

    assert decision.route is Route.PHOTO_FIRST_READY
    assert not decision.demands_open and not decision.demands_partial


def test_partial_coverage_with_a_qualified_proxy_is_hybrid() -> None:
    decision = decide(
        "essai",
        [_besoin("obligation:FACADE_PRIMARY", vues=1, status="partially_met"),
         _besoin("obligation:FACADE_REAR")],
        SITE_SAIN, [TOITURE], artifacts=[],
    )

    assert decision.route is Route.HYBRID_READY
    assert decision.demands_open == ["obligation:FACADE_REAR"]


def test_an_unqualified_proxy_does_not_carry_a_hybrid_route() -> None:
    """Un proxy non qualifié ne comble rien : le retenir ferait passer une
    forme non mesurée pour une observation."""
    brouillon = ProxyZone(
        zone="TERRAIN_MAIN", artifact="estimation", qualified=False,
        note="hauteur non mesurée",
    )

    decision = decide(
        "essai", [_besoin("obligation:FACADE_REAR")],
        SITE_SAIN, [brouillon], artifacts=[],
    )

    assert decision.route is Route.CAPTURE_REQUIRED, (
        "sans proxy qualifié, ce qui manque se prend à la caméra"
    )


def test_no_coverage_and_no_proxy_requires_a_capture() -> None:
    decision = decide(
        "essai", [_besoin(f"obligation:D{n}") for n in range(3)],
        SITE_SAIN, proxies=[], artifacts=[],
    )

    assert decision.route is Route.CAPTURE_REQUIRED
    assert len(decision.next_actions) == 3
    assert not decision.blocking, "rien n'empêche d'y aller : ce n'est pas un blocage"


# --- ce qui compte comme couvert -----------------------------------------------


def test_a_demand_needing_two_viewpoints_is_not_met_by_one() -> None:
    """Un seul point de vue ne donne pas la parallaxe que le besoin réclame."""
    decision = decide(
        "essai", [_besoin("obligation:FACADE_PRIMARY", vues=1, requis=2)],
        SITE_SAIN, [TOITURE], artifacts=[],
    )

    assert decision.demands_partial == ["obligation:FACADE_PRIMARY"]
    assert not decision.demands_satisfied


def test_the_critical_set_stays_a_deliberate_choice() -> None:
    """Y verser tous les objets bloquerait sur un panneau non localisé ; n'y
    rien mettre laisserait partir sans savoir où viser."""
    assert "BUILDING_MAIN" in CRITICAL_OBJECTS
    assert "PROPERTY_SIGN" not in CRITICAL_OBJECTS
    assert "PARKING_HOTEL" not in CRITICAL_OBJECTS
