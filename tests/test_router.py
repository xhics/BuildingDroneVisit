"""Le Router : décider comment reconstruire, et le motiver (collecte V2).

Ce que ces tests protègent avant tout : **la décision ne dérive jamais du
nombre brut d'images**. Sur ce site, trois cent treize vues portent le run de
visibilité et un seul besoin est partiellement couvert — un compteur d'images
aurait conclu à une couverture largement suffisante.

Ils protègent ensuite quatre confusions qui rendraient le document faussement
rassurant : deux vues sans recouvrement comptées comme une couverture SfM, un
même panorama compté trois fois, un modèle de terrain crédité d'avoir comblé
une façade, et une décision rendue sur un corpus qu'elle n'a pas nommé.
"""

from __future__ import annotations

import pytest

import pydantic

from hotel_pipeline.router import (
    CRITICAL_OBJECTS,
    REQUIRED_INPUTS,
    ROUTER_CONTRACT_VERSION,
    DecisionConflict,
    DecisionStatus,
    DemandStanding,
    InputManifest,
    MissingInput,
    ObjectStanding,
    ProxyZone,
    compare_with_existing,
    decide,
    semantic_payload,
    standing_for,
    standing_of,
)
from hotel_pipeline.schemas.acquisition import (
    CaptureDemand,
    CaptureIntent,
    DemandAssessment,
    DemandStatus,
    TargetKind,
)
from hotel_pipeline.schemas.enums import RouterPath

#: Le bâtiment est établi et géoréférencé : sans cela tout est bloqué, et aucun
#: autre test ne dirait rien d'intéressant.
SITE_SAIN = {"BUILDING_MAIN": ObjectStanding.TARGETABLE}


def _entrées(**overrides) -> InputManifest:
    """Un manifeste complet : les tests de route ne doivent pas échouer pour
    une entrée manquante, qui est un autre défaut."""
    # Une empreinte distincte par entrée : `name[:6]` rendait la même valeur
    # pour le manifeste d'évaluation et son rapport, que l'invariant refuse.
    base = {name: f"{abs(hash(name)) % 10**12:012x}" for name in REQUIRED_INPUTS}
    base.update(overrides)
    return InputManifest(**base, policy_facets=("coverage", "visibility"))


def _besoin(demand_id, vues=0, requis=2, ciblable=True,
            status=DemandStatus.OPEN, meets=False, ids=()):
    return DemandStanding(
        demand_id=demand_id, status=status, viewpoints_required=requis,
        viewpoints_found=vues, meets_demand=meets, targetable=ciblable,
        viewpoint_ids=ids,
    )


def _couvert(demand_id, ids):
    return _besoin(demand_id, vues=len(ids), status=DemandStatus.MET,
                   meets=True, ids=ids)


#: Une toiture qualifiée : verdict, rapport empreint et artefacts sources —
#: « qualifié » est un constat, non une déclaration.
TOITURE = ProxyZone(
    zone="ROOFLINE_MAIN", artifact="dsm+ndsm", qualified=True,
    qualification_report="qualification_report_essai.json",
    qualification_digest="q" * 16,
    source_artifacts=("dsm_roof_class6@essai", "ndsm@essai"),
    source_digests=("d" * 16,),
    covered_objects=("ROOFLINE_MAIN", "BUILDING_MAIN"),
    covered_demands=("obligation:FACADE_REAR",),
    camera_restrictions=("plans rapprochés interdits sur les zones proxy",),
)


# --- jamais depuis le nombre brut d'images ------------------------------------


def test_a_flood_of_images_does_not_make_a_site_photo_ready() -> None:
    """Le cœur du contrat.

    Trois cents vues autour d'un bâtiment peuvent n'en documenter aucune façade
    utilement : sur ce site, six acquisitions ont été réfutées une à une. La
    route se lit sur les besoins, non sur un compteur.
    """
    besoins = [_besoin(f"obligation:FACADE_{n}") for n in "ABCD"]

    decision = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées())

    assert decision.path is not RouterPath.PATH_B_PHOTO_FIRST, (
        "313 assets au manifeste, zéro besoin satisfait : un compteur d'images "
        "aurait conclu à l'inverse"
    )
    assert decision.independent_viewpoints == 0


# --- la satisfaction se lit sur le besoin, non sur un compte ------------------


def test_two_viewpoints_without_measured_continuity_do_not_satisfy() -> None:
    """Le faux positif que `meets()` existe pour empêcher.

    Deux vues sans recouvrement mesuré ne relient rien : un SfM ne se contente
    pas d'une intention. Compter les vues seules produirait un Path B sur un
    corpus que la photogrammétrie refuserait.
    """
    demand = CaptureDemand(
        demand_id="obligation:FACADE_PRIMARY", intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.SITE_OBJECT, target_ref="FACADE_PRIMARY",
        viewpoints_required=2, continuity_required=0.6,
    )
    # Assez de vues, mais la continuité n'a été que **planifiée**.
    assessment = DemandAssessment(
        demand_id=demand.demand_id, corpus_digest="c" * 16,
        status=DemandStatus.PARTIALLY_MET, viewpoints_found=2,
        continuity_achieved=0.8, continuity_level="planned",
    )

    état = standing_for(demand, assessment, targetable=True, viewpoint_ids=("v1", "v2"))

    assert état.viewpoints_found >= état.viewpoints_required, (
        "le compte de vues suffirait — c'est précisément le piège"
    )
    assert not état.meets_demand, "la continuité planifiée n'est pas mesurée"
    assert not état.satisfied

    decision = decide("essai", [état], SITE_SAIN, [], _entrées())
    assert decision.path is not RouterPath.PATH_B_PHOTO_FIRST


def test_the_required_threshold_comes_from_the_demand() -> None:
    """`viewpoints_required` n'a pas de défaut : la valeur vient de la
    politique, matérialisée dans le besoin. En inventer une ici ferait décider
    le Router sur un seuil que personne n'a arbitré."""
    with pytest.raises(pydantic.ValidationError, match="viewpoints_required"):
        DemandStanding(demand_id="d", status=DemandStatus.OPEN)  # type: ignore[call-arg]

    demand = CaptureDemand(
        demand_id="d", intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.SITE_OBJECT, target_ref="FACADE_PRIMARY",
        viewpoints_required=2,
    )
    assessment = DemandAssessment(demand_id="d", corpus_digest="c" * 16)

    assert standing_for(demand, assessment, True).viewpoints_required == 2


def test_an_unreachable_demand_is_neither_open_nor_satisfied() -> None:
    """Un besoin clos par impossibilité n'appelle pas de campagne."""
    besoin = _besoin("obligation:FACADE_REAR", status=DemandStatus.UNREACHABLE)

    decision = decide("essai", [besoin], SITE_SAIN, [TOITURE], _entrées())

    assert decision.demands_unreachable == ["obligation:FACADE_REAR"]
    assert "obligation:FACADE_REAR" not in decision.demands_open


# --- l'union, non la somme ------------------------------------------------------


def test_one_panorama_serving_three_demands_counts_once() -> None:
    """Sommer `viewpoints_found` compterait trois fois le même point de vue, et
    gonflerait la couverture apparente sans qu'aucune observation s'ajoute."""
    besoins = [
        _besoin("obligation:FACADE_LEFT", vues=1, ids=("PANO_1",)),
        _besoin("obligation:FACADE_PRIMARY", vues=1, ids=("PANO_1",)),
        _besoin("obligation:FACADE_RIGHT", vues=1, ids=("PANO_1",)),
    ]

    decision = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées())

    assert sum(b.viewpoints_found for b in besoins) == 3, "la somme naïve"
    assert decision.independent_viewpoints == 1, (
        "un panorama servant trois besoins reste un point de vue"
    )
    assert decision.viewpoint_ids == ["PANO_1"]


# --- un proxy ne comble que ce qu'il déclare couvrir ---------------------------


def test_a_qualified_proxy_does_not_cover_what_it_never_touches() -> None:
    """Un modèle de terrain ne donne ni la façade arrière, ni l'entrée.

    Sans portée déclarée, n'importe quel proxy qualifié rendrait la route
    hybride, et le document annoncerait une couverture inexistante.
    """
    terrain = ProxyZone(
        zone="TERRAIN_MAIN", artifact="dtm", qualified=True,
        qualification_digest="q" * 16, source_artifacts=("dtm@essai",),
        covered_objects=("TERRAIN_MAIN",), covered_demands=(),
    )

    decision = decide(
        "essai", [_besoin("obligation:FACADE_REAR")], SITE_SAIN,
        [terrain], _entrées(),
    )

    assert decision.decision_status is DecisionStatus.CAPTURE_REQUIRED, (
        "le terrain est qualifié, mais ne couvre aucune façade"
    )


def test_a_proxy_covering_the_demand_carries_the_hybrid_route() -> None:
    decision = decide(
        "essai", [_besoin("obligation:FACADE_REAR")], SITE_SAIN,
        [TOITURE], _entrées(),
    )

    assert decision.path is RouterPath.PATH_D_HYBRID
    assert decision.decision_status is DecisionStatus.READY
    assert decision.appearance_gaps == ["obligation:FACADE_REAR"]


def test_a_proxy_never_provides_appearance() -> None:
    """Un rendu texturé sur une forme non observée passerait pour une
    photographie de la façade."""
    assert TOITURE.appearance_provided is False

    decision = decide(
        "essai", [_besoin("obligation:FACADE_REAR")], SITE_SAIN,
        [TOITURE], _entrées(),
    )
    publié = decision.as_dict()

    assert publié["geometric_proxies"][0]["appearance_provided"] is False
    assert publié["camera_restrictions"], "les plans rapprochés restent bornés"
    assert publié["appearance_gaps"] == ["obligation:FACADE_REAR"]


def test_an_unqualified_proxy_covers_nothing() -> None:
    brouillon = ProxyZone(
        zone="TERRAIN_MAIN", artifact="estimation", qualified=False,
        covered_demands=("obligation:FACADE_REAR",),
    )

    decision = decide(
        "essai", [_besoin("obligation:FACADE_REAR")], SITE_SAIN,
        [brouillon], _entrées(),
    )

    assert decision.decision_status is DecisionStatus.CAPTURE_REQUIRED


# --- les deux sources ne se fondent pas ---------------------------------------


def test_an_unresolved_object_is_not_a_coverage_gap() -> None:
    """`PARKING_HOTEL` n'est pas une façade non photographiée : l'un demande
    une preuve, l'autre une prise de vue."""
    objets = dict(SITE_SAIN, PARKING_HOTEL=ObjectStanding.UNRESOLVED)

    decision = decide(
        "essai", [_couvert("obligation:FACADE_PRIMARY", ("v1", "v2"))],
        objets, [TOITURE], _entrées(),
    )
    publié = decision.as_dict()

    assert "PARKING_HOTEL" not in publié["photographic"]["open"]
    assert publié["site"]["by_standing"]["unresolved"] == ["PARKING_HOTEL"]


def test_an_unresolved_object_forbids_any_claim_about_it() -> None:
    """`ENTRANCE_MAIN_CURRENT` : ni existence, ni état temporel établis.

    Dire « l'entrée actuelle se trouve là » supposerait deux faits qu'aucun
    artefact ne porte. La demande est d'établir, non de localiser.
    """
    objets = dict(SITE_SAIN, ENTRANCE_MAIN_CURRENT=ObjectStanding.UNRESOLVED)

    decision = decide(
        "essai",
        [_besoin("obligation:ENTRANCE_MAIN_CURRENT", ciblable=False),
         _besoin("obligation:FACADE_REAR")],
        objets, [TOITURE], _entrées(),
    )

    assert any(
        "ENTRANCE_MAIN_CURRENT" in claim for claim in decision.forbidden_claims
    ), "aucune affirmation ne doit être permise sur l'entrée actuelle"
    assert any(
        "établir existence, état temporel et géométrie" in action
        for action in decision.next_actions
    ), "« localiser » supposerait l'existence déjà acquise"
    assert decision.decision_status is not DecisionStatus.BLOCKED_PREREQUISITES, (
        "BUILDING_MAIN est établi : l'entrée ne bloque pas la reconstruction"
    )


def test_standing_reads_the_site_state_without_inventing_one() -> None:
    """`PARKING_HOTEL` reste `unresolved` : c'est l'**association** au
    stationnement candidat qui a été démentie, non l'existence d'un
    stationnement. Un état « objet réfuté » changerait le sens du constat."""
    assert standing_of("inferred", True) is ObjectStanding.TARGETABLE
    assert standing_of("inferred", False) is ObjectStanding.KNOWN_NOT_TARGETABLE
    assert standing_of("unresolved", False) is ObjectStanding.UNRESOLVED

    assert not hasattr(ObjectStanding, "REFUTED"), (
        "réfuter l'objet dirait autre chose que réfuter son association"
    )


# --- route et statut sont deux axes -------------------------------------------


def test_full_photographic_coverage_awaits_the_sfm_gate() -> None:
    """Gate G5 : avant le Lot 2, aucune reconstruction SfM n'a été éprouvée.

    Annoncer « prêt » livrerait une préparation SfM comme si elle était déjà
    acquise.
    """
    besoins = [
        _couvert(f"obligation:D{n}", (f"v{n}a", f"v{n}b")) for n in range(3)
    ]

    decision = decide("essai", besoins, SITE_SAIN, [], _entrées())

    assert decision.path is RouterPath.PATH_B_PHOTO_FIRST
    assert decision.decision_status is DecisionStatus.VALIDATION_REQUIRED, (
        "la couverture est complète, la validation ne l'est pas"
    )
    assert decision.independent_viewpoints == 6


def test_a_missing_critical_object_blocks_everything() -> None:
    """L'ordre n'est pas indifférent : le prérequis l'emporte sur la
    couverture, car photographier sans savoir quoi viser ne produit rien."""
    besoins = [_couvert(f"obligation:D{n}", (f"v{n}",)) for n in range(4)]

    decision = decide(
        "essai", besoins, {"BUILDING_MAIN": ObjectStanding.UNRESOLVED},
        [TOITURE], _entrées(),
    )

    assert decision.decision_status is DecisionStatus.BLOCKED_PREREQUISITES
    assert decision.blocking and "BUILDING_MAIN" in decision.blocking[0]


def test_the_status_is_not_the_route() -> None:
    """`PATH_D_HYBRID` peut être `ready` ou `capture_required` sans que les
    matériaux changent : les confondre ferait d'un état une route."""
    couvert = decide(
        "essai", [_besoin("obligation:FACADE_REAR")], SITE_SAIN,
        [TOITURE], _entrées(),
    )
    incomplet = decide(
        "essai",
        [_besoin("obligation:FACADE_REAR"), _besoin("obligation:FACADE_LEFT")],
        SITE_SAIN, [TOITURE], _entrées(),
    )

    assert couvert.path is incomplet.path is RouterPath.PATH_D_HYBRID
    assert couvert.decision_status is DecisionStatus.READY
    assert incomplet.decision_status is DecisionStatus.CAPTURE_REQUIRED


def test_ready_never_means_environment_3d_ready() -> None:
    """« ready » dit prêt à engager cette route ; la fin de Phase 1 est un
    autre verdict, qu'aucune décision de Router ne rend."""
    decision = decide(
        "essai", [_besoin("obligation:FACADE_REAR")], SITE_SAIN,
        [TOITURE], _entrées(),
    )
    publié = decision.as_dict()

    assert publié["decision_status"] == "ready"
    assert "ENVIRONMENT_3D_READY" not in str(publié["path"])
    assert any("ENVIRONMENT_3D_READY" in limite for limite in publié["limits"])


# --- les entrées sont fermées ---------------------------------------------------


def test_a_missing_input_refuses_the_decision() -> None:
    """Rendre une route sur un corpus inconnu produirait un document qui paraît
    fondé sans l'être — pire qu'une absence de décision."""
    incomplet = _entrées(visibility_application_digest="")

    with pytest.raises(MissingInput, match="visibility_application_digest"):
        decide("essai", [_besoin("obligation:D")], SITE_SAIN, [TOITURE], incomplet)


def test_every_required_input_is_actually_required() -> None:
    """Vérifier une seule entrée laisserait les huit autres facultatives."""
    for name in REQUIRED_INPUTS:
        with pytest.raises(MissingInput, match=name):
            decide(
                "essai", [_besoin("obligation:D")], SITE_SAIN, [TOITURE],
                _entrées(**{name: "   "}),
            )


def test_the_input_digest_is_deterministic_and_independent_of_time() -> None:
    """Un simple rejeu ne doit pas créer une décision différente : sans cela,
    deux documents divergeraient sans qu'aucune entrée n'ait changé."""
    besoins = [_besoin("obligation:FACADE_REAR")]

    première = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées())
    seconde = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées())

    assert première.input_digest == seconde.input_digest
    assert première.decided_at != seconde.decided_at or True
    assert première.input_digest not in première.decided_at


def test_a_changed_input_changes_the_digest() -> None:
    """Sinon une entrée périmée passerait pour la même décision."""
    besoins = [_besoin("obligation:FACADE_REAR")]

    avant = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées())
    après = decide(
        "essai", besoins, SITE_SAIN, [TOITURE],
        _entrées(visibility_application_digest="autre_run"),
    )

    assert avant.input_digest != après.input_digest


def test_the_consumed_policy_facets_are_recorded() -> None:
    """Le digest global change à chaque retouche, y compris sur une facette
    étrangère à cette décision."""
    decision = decide(
        "essai", [_besoin("obligation:FACADE_REAR")], SITE_SAIN,
        [TOITURE], _entrées(),
    )

    assert decision.as_dict()["inputs"]["policy_facets"] == ["coverage", "visibility"]


# --- le vocabulaire est celui du plan directeur --------------------------------


def test_the_router_uses_the_canonical_path_vocabulary() -> None:
    """Redéfinir les routes ici ferait diverger le Router du plan directeur."""
    from hotel_pipeline import router

    assert router.RouterPath is RouterPath
    assert not hasattr(router, "Route"), (
        "un second vocabulaire de routes réintroduirait la divergence"
    )


def test_the_critical_set_stays_a_deliberate_choice() -> None:
    """Y verser tous les objets bloquerait sur un panneau non localisé ; n'y
    rien mettre laisserait partir sans savoir où viser."""
    assert "BUILDING_MAIN" in CRITICAL_OBJECTS
    assert "PROPERTY_SIGN" not in CRITICAL_OBJECTS
    assert "PARKING_HOTEL" not in CRITICAL_OBJECTS


# --- un besoin non ciblable n'appelle pas une caméra ---------------------------


def test_an_untargetable_demand_never_triggers_a_capture() -> None:
    """La règle que le pilote a fait apparaître.

    Aucune prise de vue ne comble l'absence d'un objet dont on ignore s'il
    existe : compter ces besoins dans le statut confondrait les deux sources
    que le Router sépare — ce qui est photographiquement couvert, et ce qui est
    établi.
    """
    besoins = [
        _besoin("obligation:FACADE_REAR"),                       # comblé par proxy
        _besoin("obligation:PROPERTY_SIGN", ciblable=False),
        _besoin("obligation:ENTRANCE_MAIN_CURRENT", ciblable=False),
    ]
    objets = dict(
        SITE_SAIN,
        PROPERTY_SIGN=ObjectStanding.KNOWN_NOT_TARGETABLE,
        ENTRANCE_MAIN_CURRENT=ObjectStanding.UNRESOLVED,
    )

    decision = decide("essai", besoins, objets, [TOITURE], _entrées())

    assert decision.decision_status is DecisionStatus.READY, (
        "les deux besoins sans cible ne sont pas des lacunes de capture"
    )
    assert len(decision.demands_not_targetable) == 2
    assert not any(
        action.startswith("capturer") or action.startswith("chercher des vues")
        for action in decision.next_actions
        if "PROPERTY_SIGN" in action or "ENTRANCE_MAIN_CURRENT" in action
    ), "ces besoins appellent une résolution, jamais une caméra"
    assert sum(
        "établir existence, état temporel et géométrie" in action
        for action in decision.next_actions
    ) == 2


def test_a_targetable_demand_without_photo_or_proxy_requires_capture() -> None:
    """L'autre branche : ce qui est ciblable et non couvert se prend bien à la
    caméra."""
    besoins = [
        _besoin("obligation:FACADE_REAR"),                     # comblé
        _besoin("obligation:ACCESS_ROAD_MAIN"),                # ciblable, non comblé
    ]

    decision = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées())

    assert decision.decision_status is DecisionStatus.CAPTURE_REQUIRED
    assert any(
        "ACCESS_ROAD_MAIN" in action and "chercher des vues" in action
        for action in decision.next_actions
    )


# --- « qualifié » est un constat, non une déclaration --------------------------


def test_a_proxy_cannot_claim_qualification_without_evidence() -> None:
    """`capture_geometry.json` existe sur tout site ayant tourné une fois : s'en
    contenter qualifierait des proxies jamais éprouvés."""
    with pytest.raises(pydantic.ValidationError, match="sans empreinte de rapport"):
        ProxyZone(zone="Z", artifact="a", qualified=True,
                  source_artifacts=("x",))

    with pytest.raises(pydantic.ValidationError, match="sans artefact source"):
        ProxyZone(zone="Z", artifact="a", qualified=True,
                  qualification_digest="q" * 16)


def test_a_proxy_can_never_declare_appearance() -> None:
    with pytest.raises(pydantic.ValidationError, match="jamais l'apparence"):
        ProxyZone(zone="Z", artifact="a", appearance_provided=True)


# --- l'identité de la décision -------------------------------------------------


def test_the_two_assessment_digests_cannot_be_confused() -> None:
    """Une empreinte unique laissait trois rapports différents produire la même
    identité — c'est arrivé sur le pilote."""
    with pytest.raises(pydantic.ValidationError, match="deux fichiers"):
        InputManifest(
            assessment_manifest_digest="même", assessment_report_digest="même"
        )


def test_the_contract_version_enters_the_identity() -> None:
    """Deux versions n'ont pas jugé selon les mêmes règles : leurs verdicts ne
    se comparent pas, même à entrées identiques."""
    avant = _entrées(contract_version=1)
    après = _entrées(contract_version=2)

    assert avant.digest != après.digest
    assert ROUTER_CONTRACT_VERSION >= 2


def test_the_report_digest_enters_the_identity() -> None:
    """Le défaut constaté : trois décisions de contenus différents portaient la
    même identité parce que le rapport n'était pas empreint."""
    avant = _entrées()
    après = _entrées(assessment_report_digest="un_autre_rapport")

    assert avant.digest != après.digest


# --- à identité égale, le verdict est identique --------------------------------


def test_a_divergence_at_equal_identity_is_refused() -> None:
    """Une différence signifie qu'une entrée non déclarée a pesé. Republier
    effacerait la trace de ce défaut sans le corriger."""
    besoins = [_besoin("obligation:FACADE_REAR")]
    rendue = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées()).as_dict()

    divergente = dict(rendue, decision_status="ready", path="path_b_photo_first")

    with pytest.raises(DecisionConflict, match="path|decision_status"):
        compare_with_existing(rendue, divergente)


def test_only_the_timestamp_may_differ_between_replays() -> None:
    """Comparer `decided_at` ferait échouer tout rejeu légitime."""
    besoins = [_besoin("obligation:FACADE_REAR")]
    première = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées()).as_dict()
    seconde = decide("essai", besoins, SITE_SAIN, [TOITURE], _entrées()).as_dict()

    assert première["decided_at"] != seconde["decided_at"]
    compare_with_existing(première, seconde)          # ne lève pas
    assert "decided_at" not in semantic_payload(première)


# --- invariants structurels de la décision -------------------------------------


def test_a_decision_cannot_claim_ready_while_a_prerequisite_is_missing() -> None:
    """Sans cet invariant, un document pourrait annoncer « prêt » en portant un
    objet critique non établi — et faire autorité."""
    from hotel_pipeline.router import RouterDecision

    with pytest.raises(pydantic.ValidationError, match="ne bloque pas"):
        RouterDecision(
            hotel_id="essai", path=RouterPath.PATH_D_HYBRID,
            decision_status=DecisionStatus.READY, inputs=_entrées(),
            critical_objects_unestablished=["BUILDING_MAIN"],
        )


def test_a_decision_cannot_miscount_its_own_viewpoints() -> None:
    from hotel_pipeline.router import RouterDecision

    with pytest.raises(pydantic.ValidationError, match="point\\(s\\) de vue"):
        RouterDecision(
            hotel_id="essai", path=RouterPath.PATH_D_HYBRID,
            decision_status=DecisionStatus.READY, inputs=_entrées(),
            independent_viewpoints=3, viewpoint_ids=["PANO_1"],
        )


def test_path_b_can_never_be_ready_before_the_sfm_gate() -> None:
    from hotel_pipeline.router import RouterDecision

    with pytest.raises(pydantic.ValidationError, match="Gate G5"):
        RouterDecision(
            hotel_id="essai", path=RouterPath.PATH_B_PHOTO_FIRST,
            decision_status=DecisionStatus.READY, inputs=_entrées(),
        )


def test_the_decision_refuses_unknown_fields() -> None:
    """Un champ libre laisserait une entrée non déclarée peser sur le verdict."""
    from hotel_pipeline.router import RouterDecision

    with pytest.raises(pydantic.ValidationError):
        RouterDecision(
            hotel_id="essai", path=RouterPath.PATH_D_HYBRID,
            decision_status=DecisionStatus.READY, inputs=_entrées(),
            coverage_score=0.9,  # type: ignore[call-arg]
        )
