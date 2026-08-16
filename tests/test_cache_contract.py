"""La clé de cache porte le contrat de réponse (collecte V2).

La clé ne dépendait que de la géographie. Une entrée obtenue sous l'ancien
contrat — sans `sequence`, `camera_parameters`, `width` ni `height` — se
relisait donc comme si elle les contenait : un lancement `online` aurait servi
194 lignes incomplètes sans appeler l'API, et reproduit exactement le défaut
qu'on venait de corriger.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.collectors import mapillary
from hotel_pipeline.providers.transport import (
    NetworkMode,
    NetworkRefused,
    reset_ledger,
    set_mode,
)


# --- la clé bouge avec le contrat ---------------------------------------------


def test_changing_the_requested_fields_moves_the_key(monkeypatch) -> None:
    """Demander autre chose, c'est demander autre chose.

    Sans cela, ajouter `sequence` aux champs laissait l'ancienne réponse
    répondre à la nouvelle question.
    """
    before = mapillary.contract_digest()

    monkeypatch.setattr(mapillary, "FIELDS", mapillary.FIELDS + ",computed_geometry")
    after = mapillary.contract_digest()

    assert after != before


def test_changing_the_parser_version_moves_the_key(monkeypatch) -> None:
    """Une réponse identique lue autrement n'est plus la même donnée.

    Dériver le champ de vision d'un rapport focal en est l'exemple : les champs
    n'avaient pas changé, leur interprétation si.
    """
    before = mapillary.contract_digest()

    monkeypatch.setattr(mapillary, "PARSER_VERSION", mapillary.PARSER_VERSION + 1)

    assert mapillary.contract_digest() != before


def test_the_digest_is_stable_for_an_unchanged_contract() -> None:
    """Sinon chaque exécution invaliderait le cache de la précédente."""
    assert mapillary.contract_digest() == mapillary.contract_digest()


def test_the_key_material_carries_no_secret(monkeypatch) -> None:
    """La clé vit sur le disque : une empreinte n'est pas un endroit où ranger
    un jeton."""
    digest = mapillary.contract_digest()

    assert len(digest) == 12
    assert digest.isalnum()
    for forbidden in ("http", "token", "OAuth", "Authorization"):
        assert forbidden not in digest


# --- ce que le mode fermé en fait ---------------------------------------------


def test_an_entry_from_the_previous_contract_is_a_miss_not_a_hit(monkeypatch) -> None:
    """En `cache_only`, elle doit refuser — jamais servir.

    Et le motif doit distinguer « absente » de « présente sous un contrat
    antérieur » : la première demande une collecte, la seconde dit qu'un rejeu
    figé ne peut plus servir.
    """
    from hotel_pipeline.providers.cache import cached_call, get_cache

    cache = get_cache()
    ancienne = "essai-contrat::ancien::zone"
    nouvelle = "essai-contrat::nouveau::zone"
    cache.delete(nouvelle)
    set_mode(NetworkMode.ONLINE)
    cached_call(ancienne, lambda: [{"id": "1"}])

    set_mode(NetworkMode.CACHE_ONLY)
    reset_ledger()
    try:
        with pytest.raises(NetworkRefused) as refusal:
            cached_call(nouvelle, lambda: [{"id": "1", "sequence": "s"}])
    finally:
        set_mode(None)
        cache.delete(ancienne)

    message = str(refusal.value)
    assert "cache_miss/incompatible" in message
    assert "contrat antérieur" in message


def test_a_source_with_no_entry_at_all_says_so(monkeypatch) -> None:
    """« Absente » et « périmée » ne se corrigent pas de la même façon."""
    from hotel_pipeline.providers.cache import cached_call, get_cache

    cache = get_cache()
    for key in list(cache.iterkeys()):
        if str(key).startswith("essai-vide::"):
            cache.delete(key)

    set_mode(NetworkMode.CACHE_ONLY)
    try:
        with pytest.raises(NetworkRefused) as refusal:
            cached_call("essai-vide::zone", lambda: [])
    finally:
        set_mode(None)

    assert "absente du cache" in str(refusal.value)
    assert "contrat antérieur" not in str(refusal.value)


# --- ce que le rapport doit montrer -------------------------------------------


def test_the_report_counts_what_the_candidates_really_carry() -> None:
    """Demander un champ ne garantit pas de le recevoir.

    Un corpus servi par un cache antérieur passerait pour enrichi ; ces
    chiffres le démentent immédiatement.
    """
    from hotel_pipeline.discover import _contract_coverage
    from hotel_pipeline.schemas.acquisition import CaptureCandidate

    def candidate(candidate_id: str, **overrides):
        fields = dict(
            candidate_id=candidate_id, source="mapillary", provider_id=candidate_id,
            camera_lat=45.5, camera_lon=-73.4,
        )
        fields.update(overrides)
        return CaptureCandidate(**fields)

    from hotel_pipeline.schemas.acquisition import ProjectionSupport

    coverage = _contract_coverage([
        candidate("complet", sequence_id="s1", requested_fov_deg=60.9,
                  observed_horizontal_fov_deg=60.9,
                  projection_support=ProjectionSupport.SUPPORTED,
                  advertised_width=4000, advertised_height=3000,
                  camera_type="perspective"),
        candidate("nu"),
        candidate("partiel", sequence_id="s2"),
    ])

    row = coverage["mapillary"]
    assert row["candidates"] == 3
    assert row["with_sequence"] == 2
    assert row["observed_fov"] == 1
    assert row["projection_supported"] == 1
    assert row["fov_unknown"] == 2
    assert row["with_dimensions"] == 1
    assert row["with_camera_type"] == 1


def test_a_partial_corpus_does_not_become_the_current_manifest(
    tmp_path, monkeypatch
) -> None:
    """Une source en échec ne doit pas se lire comme une source vide.

    Planifier sur un corpus dont Mapillary est absent ferait prendre son
    silence pour une absence de vues — et le plan couvrirait des besoins que
    personne n'a cherché à servir.
    """
    from hotel_pipeline.cli import _latest_candidates
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "travail"))
    workspace = Workspace("essai")
    sources = workspace.path("01_sources")
    (sources / "replays").mkdir(parents=True, exist_ok=True)
    assert workspace.root.resolve().is_relative_to(tmp_path.resolve())

    # Un corpus complet, puis un corpus partiel plus récent.
    (sources / "candidates_20260101T000000Z.json").write_text("{}", "utf-8")
    (sources / "replays" / "candidates_20260901T000000Z.json").write_text(
        "{}", "utf-8"
    )

    latest = _latest_candidates(workspace)

    assert latest is not None
    assert latest.name == "candidates_20260101T000000Z.json", (
        "le corpus partiel ne remplace pas le complet, même plus récent"
    )


# --- « inconnu » n'est pas « connu mais non supporté » -------------------------


def _image(**overrides):
    from hotel_pipeline.collectors.base import CollectedImage

    fields = dict(
        source="mapillary", source_id="1", url="http://exemple",
        lat=45.0, lon=-73.0,
    )
    fields.update(overrides)
    return CollectedImage(**fields)


def test_a_wide_angle_keeps_its_measured_field_of_view() -> None:
    """Les 18 vues du pilote font 134,2° : c'est une mesure, pas une absence.

    Les encoder `None` faisait passer un ultra-grand-angle pour une caméra sans
    métadonnées — deux situations qui n'appellent pas la même correction.
    """
    from hotel_pipeline.discover import candidates_from
    from hotel_pipeline.schemas.acquisition import ProjectionSupport

    built = candidates_from("mapillary", [_image(fov_deg=134.24)])[0]

    assert built.observed_horizontal_fov_deg == 134.24, "la mesure est conservée"
    assert built.requested_fov_deg is None, (
        "elle ne sert pas au cadrage : le modèle n'est pas validé là"
    )
    assert built.projection_support is ProjectionSupport.UNSUPPORTED_FOV
    assert "134.2" in built.projection_note
    assert "non validé au-delà de 120" in built.projection_note


def test_an_absent_field_of_view_stays_unknown() -> None:
    """La source ne publie rien : c'est une autre correction qu'appelle ce cas."""
    from hotel_pipeline.discover import candidates_from
    from hotel_pipeline.schemas.acquisition import ProjectionSupport

    built = candidates_from("mapillary", [_image()])[0]

    assert built.observed_horizontal_fov_deg is None
    assert built.projection_support is ProjectionSupport.UNKNOWN_INTRINSICS
    assert "ne publie pas" in built.projection_note


def test_a_panorama_declares_what_it_needs() -> None:
    """360° est vrai et inexploitable comme cadrage."""
    from hotel_pipeline.discover import candidates_from
    from hotel_pipeline.schemas.acquisition import ProjectionSupport

    built = candidates_from("mapillary", [_image(fov_deg=360.0)])[0]

    assert built.observed_horizontal_fov_deg == 360.0
    assert built.requested_fov_deg is None
    assert built.projection_support is ProjectionSupport.PANORAMIC_REQUIRES_EXTRACTION


def test_a_supported_optic_carries_both_values() -> None:
    from hotel_pipeline.discover import candidates_from
    from hotel_pipeline.schemas.acquisition import ProjectionSupport

    built = candidates_from("mapillary", [_image(fov_deg=60.93)])[0]

    assert built.requested_fov_deg == 60.93
    assert built.observed_horizontal_fov_deg == 60.93
    assert built.projection_support is ProjectionSupport.SUPPORTED


def test_the_coverage_separates_observed_from_supported() -> None:
    """Le format exigé : quatre comptes, non un seul.

    « FOV observé 195, supporté 177, non supporté 18, inconnu 0 » se lit d'un
    coup d'œil ; « with_fov 177 » laissait croire à 18 vues sans métadonnées.
    """
    from hotel_pipeline.discover import _contract_coverage, candidates_from

    corpus = candidates_from("mapillary", [
        *[_image(source_id=f"s{i}", fov_deg=60.9) for i in range(3)],
        *[_image(source_id=f"w{i}", fov_deg=134.24) for i in range(2)],
        _image(source_id="nu"),
    ])

    row = _contract_coverage(corpus)["mapillary"]

    assert row["observed_fov"] == 5, "les 134° comptent comme observés"
    assert row["projection_supported"] == 3
    assert row["projection_unsupported"] == 2
    assert row["fov_unknown"] == 1
