"""Bootstrap générique du profil et de la politique (portabilité, commit 1).

Ce qui est éprouvé ici : un projet neuf ne peut hériter ni du calibrage, ni de
l'identité, ni de la langue du pilote — et le pilote existant ne bouge pas d'un
octet pour autant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hotel_pipeline import capabilities
from hotel_pipeline.capabilities import (
    Capability,
    CapabilityUnavailable,
    Requirement,
    available,
    require,
)
from hotel_pipeline.cli import app
from hotel_pipeline.provenance import policy_digest
from hotel_pipeline.schemas import PipelinePolicy, PropertyProfile
from hotel_pipeline.schemas.policy import UNCALIBRATED

runner = CliRunner()


class Context:
    """Contexte minimal, pour éprouver la matrice sans monter un projet."""

    def __init__(self, profile=None, policy=None, defaults=(), spatial=None):
        self.profile = profile
        self.policy = policy
        self.policy_defaults_applied = tuple(defaults)
        self.spatial_reference = spatial


def profile(**overrides) -> PropertyProfile:
    fields = dict(
        property_id="p", address="1 rue Test", official_name="Hôtel Test",
        country_code="CA", timezone="America/Toronto", ocr_languages=["fr"],
    )
    fields.update(overrides)
    return PropertyProfile(**fields)


# --- la politique du pilote ne bouge pas ------------------------------------


#: Empreinte de la politique du pilote, par version. Elle bouge quand la
#: politique gagne un seuil décisionnel — c'est le rôle d'une empreinte — mais
#: jamais quand une refactorisation ne change aucune valeur. Les rapports
#: publiés conservent celle de leur époque : ils n'ont pas été produits avec la
#: politique d'aujourd'hui, et le prétendre serait faux.
PILOT_POLICY_DIGESTS = {
    "a4564b71ddeec56e": "avant les seuils de secteur et de séparation",
    "9275a7e32eeb0431": "avec sector_observer_half_width_deg et viewpoint_separation_m",
    "18422c85a67d09fe": "avec la section coverage — cibles de couverture",
    "27133809b5a4f3da": "avec les bornes d'enrichissement et d'expansion de séquence",
    "b2090cf303c3ee23": "avec adaptive_search — préférence de parallaxe non monotone",
    "fbfb02571f608b79": (
        "avec max_distance_to_target_m — le classement seul ne bornait pas la "
        "distance : le premier candidat était retenu à n'importe quelle portée"
    ),
    "e5d4fa407286fa72": (
        "avec heading_tolerance_deg — position de l'observateur et orientation "
        "de la caméra sont deux questions distinctes"
    ),
    "ea1d51576b06f645": (
        "max_distance_to_target_m renommé automatic_candidate_max_distance_m — "
        "hors de portée de recommandation n'est pas inutilisable : sans les "
        "intrinsèques de la caméra, la distance seule ne prouve rien"
    ),
    "4034de29c03892b8": (
        "avec framing_merge_bearing_deg et les deux résolutions — une preview "
        "se vérifie en miniature, jamais en pleine résolution"
    ),
}


def test_the_pilot_policy_digest_moves_only_on_a_real_change() -> None:
    """Une empreinte change quand une valeur change, jamais autrement.

    Le socle `Calibrated` n'est donc pas un modèle — en hériter aurait placé
    les deux champs de calibration en tête du dump et changé l'empreinte sans
    qu'aucune valeur ne bouge. Ajouter deux seuils décisionnels, au contraire,
    la déplace à juste titre.
    """
    path = Path("work/welcominns-boucherville/00_manifest/pipeline_policy.json")
    if not path.is_file():  # pragma: no cover — dépend du corpus local
        pytest.skip("corpus du pilote absent")

    loaded = PipelinePolicy.model_validate_json(path.read_text("utf-8"))
    digest = policy_digest(loaded)

    assert digest in PILOT_POLICY_DIGESTS, (
        f"empreinte {digest!r} non répertoriée : si la politique a gagné une "
        "valeur, inscrivez-la ; sinon, une refactorisation a déplacé un champ"
    )
    # Les valeurs du pilote, elles, ne bougent pas.
    assert loaded.model.calibration_id == "welcominns-2026-08-36-images"
    assert loaded.model.calibrated_on_sites == 1


def test_a_refactoring_never_moves_the_digest_on_its_own() -> None:
    """Le contrôle qui compte : deux politiques de mêmes valeurs, une empreinte.

    C'est ce qui distingue un vrai changement d'un déplacement de champ.
    """
    from hotel_pipeline.schemas import DEFAULT_POLICY

    twin = PipelinePolicy.model_validate_json(DEFAULT_POLICY.model_dump_json())

    assert policy_digest(twin) == policy_digest(DEFAULT_POLICY)


def test_the_calibration_fields_are_never_moved_to_the_front() -> None:
    """L'ordre porte l'empreinte : le vérifier évite de le casser par mégarde.

    Le mode de défaillance observé est précis — hériter d'un socle pydantic
    remonte les deux champs en tête. Ils n'ont pas à être derniers partout
    (`qualification` porte ensuite ses deux tables de seuils), mais jamais
    premiers.
    """
    from hotel_pipeline.schemas.policy import (
        ModelPolicy, QualificationPolicy, TerrainPolicy,
    )

    for section in (ModelPolicy, TerrainPolicy, QualificationPolicy):
        order = list(section.model_fields)
        assert order[0] not in ("calibration_id", "calibrated_on_sites"), section.__name__
        assert order.index("calibration_id") + 1 == order.index("calibrated_on_sites")


# --- aucun défaut ne nomme un établissement ---------------------------------


def test_a_new_policy_inherits_no_calibration() -> None:
    fresh = PipelinePolicy()

    for section in (fresh.model, fresh.terrain, fresh.qualification):
        assert section.calibration_id == UNCALIBRATED
        assert section.calibrated_on_sites == 0
        assert not section.is_calibrated


def test_a_named_campaign_without_sites_is_refused() -> None:
    with pytest.raises(ValueError, match="zéro site"):
        PipelinePolicy.model_validate(
            {"qualification": {"calibration_id": "pilote-v1", "calibrated_on_sites": 0}}
        )


def test_the_pilot_marker_is_recognised_as_uncalibrated() -> None:
    """Une politique déjà écrite porte une autre formulation. La convertir
    aurait modifié un fichier dont l'empreinte est publiée."""
    legacy = PipelinePolicy.model_validate(
        {"terrain": {"calibration_id": "non-calibré — valeurs initiales, un seul site"}}
    )

    assert not legacy.terrain.is_calibrated
    assert not legacy.terrain.names_a_campaign


# --- le profil déclare ce qui était supposé ---------------------------------


def test_a_profile_must_declare_country_timezone_and_languages() -> None:
    for missing in ("country_code", "timezone", "ocr_languages"):
        fields = dict(
            property_id="p", address="a", official_name="X",
            country_code="FR", timezone="Europe/Paris", ocr_languages=["fr"],
        )
        del fields[missing]
        with pytest.raises(ValueError, match=missing):
            PropertyProfile(**fields)


def test_no_language_is_assumed_from_the_country() -> None:
    """« fr, en » était le repli du pilote, pas une propriété du monde."""
    lyon = profile(country_code="FR", timezone="Europe/Paris", ocr_languages=["fr"])

    assert lyon.ocr_languages == ["fr"]
    assert "en" not in lyon.ocr_languages


def test_a_subdivision_stays_optional_rather_than_invented() -> None:
    assert profile().subdivision_code is None


# --- la matrice de capacités -------------------------------------------------


def test_every_capability_declares_its_requirements() -> None:
    """Une capacité non déclarée serait un passe-droit silencieux."""
    assert set(capabilities.REQUIREMENTS) == set(Capability)


def test_every_command_declares_the_capability_it_needs() -> None:
    """Un oubli ne doit pas devenir une permission tacite.

    Le contexte traitait une commande non déclarée comme une inspection —
    permissive. La matrice redevenait alors facultative pour qui l'oubliait.
    """
    import re
    from pathlib import Path

    source = Path("src/hotel_pipeline/cli.py").read_text("utf-8")
    # `= _context(...)` ne capture que les **appels** : la définition de
    # `_context`, qui mentionne aussi `hotel_id`, n'en est pas un.
    invocations = re.findall(r"=\s*_context\(([^)]*)\)", source)

    assert len(invocations) >= 18, f"seulement {len(invocations)} appels trouvés"
    undeclared = [call for call in invocations if "Capability." not in call]
    assert undeclared == []


def test_an_undeclared_capability_stops_the_command() -> None:
    from hotel_pipeline.cli import _context

    with pytest.raises(RuntimeError, match="sans capacité déclarée"):
        _context("hotel-test", None)


def test_identity_classification_refuses_to_run_without_a_profile() -> None:
    with pytest.raises(CapabilityUnavailable) as raised:
        require(Context(policy=PipelinePolicy()), Capability.IDENTITY_CLASSIFICATION)

    assert raised.value.missing == [Requirement.PROFILE]
    # L'erreur nomme ce qui manque et comment l'obtenir.
    assert "l'établissement visé n'est pas décrit" in str(raised.value)
    assert "hotel-pipeline init" in str(raised.value)


def test_targeted_collection_also_needs_a_position() -> None:
    check = available(
        Context(profile=profile(), policy=PipelinePolicy()),
        Capability.TARGETED_COLLECTION,
    )

    assert check.missing == [Requirement.POSITION]

    located = available(
        Context(profile=profile(lat=45.57, lon=-73.44), policy=PipelinePolicy()),
        Capability.TARGETED_COLLECTION,
    )
    assert located.satisfied


def test_geospatial_qualification_refuses_implicit_thresholds() -> None:
    """Un rapport citerait des seuils qu'aucun fichier ne porte."""
    implicit = Context(policy=PipelinePolicy(), defaults=("terrain.ring_m",))

    check = available(implicit, Capability.GEOSPATIAL_QUALIFICATION)

    assert Requirement.MATERIALISED_POLICY in check.missing


def test_geospatial_qualification_needs_no_identity_but_much_else() -> None:
    """Des seuils de terrain ne dépendent pas du nom de l'hôtel.

    C'est pourquoi la capacité s'appelle « géospatiale » : une qualification
    photographique viendra, et elle n'aura pas besoin du contexte spatial.
    """
    needs = capabilities.REQUIREMENTS[Capability.GEOSPATIAL_QUALIFICATION]

    assert Requirement.PROFILE not in needs
    assert set(needs) == {
        Requirement.MATERIALISED_POLICY,
        Requirement.SITE_MANIFEST,
        Requirement.EXPECTED_ARTIFACTS,
        Requirement.SPATIAL_CONTEXT,
        Requirement.VERTICAL_PROVENANCE,
    }


def test_bootstrap_requires_nothing_and_inspection_says_what_it_lacks() -> None:
    empty = Context()

    assert available(empty, Capability.BOOTSTRAP).satisfied
    inspection = available(empty, Capability.INSPECTION)

    # Permissive, mais pas silencieuse : c'est toute la différence avec
    # l'avertissement jaune qu'elle remplace.
    assert inspection.satisfied
    assert inspection.partial
    assert any("établissement visé" in item for item in inspection.partial)


def test_partial_results_are_allowed_only_where_declared() -> None:
    """« Partiel » n'est ni « sans prérequis » ni « sans profil ».

    La qualification géospatiale n'exige aucune identité et n'y figure
    pourtant pas : elle ne rend rien de partiel, ses prérequis sont
    satisfaits ou elle s'arrête.
    """
    assert capabilities.PARTIAL_CONTEXT_ALLOWED == {
        Capability.BOOTSTRAP, Capability.INSPECTION
    }
    assert Capability.GEOSPATIAL_QUALIFICATION not in capabilities.PARTIAL_CONTEXT_ALLOWED


def test_every_capability_needing_an_identity_refuses_to_run_without_one() -> None:
    """La qualification n'y figure pas, et c'est correct : des seuils
    géospatiaux ne dépendent d'aucune identité d'établissement."""
    without_profile = Context(policy=PipelinePolicy())

    blocked = {
        capability
        for capability in Capability
        if not available(without_profile, capability).satisfied
    }
    needing_identity = {
        capability
        for capability, needs in capabilities.REQUIREMENTS.items()
        if Requirement.PROFILE in needs
    }

    assert needing_identity <= blocked
    assert Capability.IDENTITY_CLASSIFICATION in blocked
    assert Capability.GEOSPATIAL_QUALIFICATION not in needing_identity


# --- bout en bout ------------------------------------------------------------


def workspace(tmp_path, monkeypatch, *options):
    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.chdir(tmp_path)
    return runner.invoke(app, ["init", "second-site", "--address", "1 rue Test", *options])


def test_init_writes_a_profile_when_told_who_where_and_in_what_language(
    tmp_path, monkeypatch
) -> None:
    result = workspace(
        tmp_path, monkeypatch,
        "--name", "Hôtel Second Site", "--country", "FR",
        "--timezone", "Europe/Paris", "--ocr-language", "fr",
    )

    assert result.exit_code == 0
    written = json.loads((tmp_path / "profiles/second-site.json").read_text("utf-8"))
    assert written["country_code"] == "FR"
    assert written["timezone"] == "Europe/Paris"
    assert written["ocr_languages"] == ["fr"]
    assert written["subdivision_code"] is None


def test_init_refuses_to_invent_a_country_or_a_language(tmp_path, monkeypatch) -> None:
    """Une adresse contenant « Québec » n'établit pas un territoire."""
    result = workspace(tmp_path, monkeypatch, "--name", "Hôtel Second Site")

    assert result.exit_code == 0
    assert not (tmp_path / "profiles/second-site.json").exists()
    assert "--country" in result.stdout
    assert "--timezone" in result.stdout


def test_a_project_without_a_profile_cannot_classify_identity(
    tmp_path, monkeypatch
) -> None:
    workspace(tmp_path, monkeypatch)

    result = runner.invoke(app, ["assets", "classify", "second-site"])

    assert result.exit_code == 1
    # Le refus part sur la sortie d'erreur : `result.output` la comprend.
    assert "identity_classification" in result.output
    assert "l'établissement visé n'est pas décrit" in result.output
    assert "hotel-pipeline init" in result.output
