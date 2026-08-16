"""Street View en candidats V2 (collecte V2).

Le test d'acceptation central, en une phrase : plusieurs cadrages d'un panorama
produisent plusieurs acquisitions possibles et **un seul** point de vue ; des
panoramas réellement distincts produisent des points de vue distincts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from hotel_pipeline.collectors.streetview_v2 import (
    Framing,
    candidate_from,
    candidates_from,
    framings_towards,
    resolve_url,
)
from hotel_pipeline.plan import group_viewpoints


@dataclass(frozen=True)
class Panorama:
    pano_id: str
    lat: float = 45.5734
    lon: float = -73.4433
    date: str | None = "2024-06"
    copyright: str | None = "© Google"


# --- identité : le cadrage fait l'image ---------------------------------------


def test_two_framings_of_one_panorama_are_two_acquisitions() -> None:
    """Les nommer pareil en écraserait un, sans que rien ne le signale."""
    panorama = Panorama("pano-A")

    first = candidate_from(panorama, Framing(heading_deg=0.0))
    second = candidate_from(panorama, Framing(heading_deg=180.0))

    assert first.candidate_id != second.candidate_id
    assert first.provider_id == second.provider_id == "pano-A"


def test_the_same_framing_twice_is_one_acquisition() -> None:
    panorama = Panorama("pano-A")

    once = candidate_from(panorama, Framing(heading_deg=90.0))
    again = candidate_from(panorama, Framing(heading_deg=90.0))

    assert once.candidate_id == again.candidate_id


def test_every_component_of_the_framing_enters_the_identity() -> None:
    """Deux cadrages qui ne diffèrent que par l'ouverture sont deux images."""
    panorama = Panorama("pano-A")
    base = Framing(heading_deg=0.0)

    variants = [
        candidate_from(panorama, base),
        candidate_from(panorama, Framing(heading_deg=0.0, fov_deg=40.0)),
        candidate_from(panorama, Framing(heading_deg=0.0, pitch_deg=15.0)),
        candidate_from(panorama, Framing(heading_deg=0.0, size="1024x1024")),
    ]

    identifiers = [candidate.candidate_id for candidate in variants]
    assert len(set(identifiers)) == len(identifiers)


# --- points de vue : la position, pas le cadrage ------------------------------


def test_several_framings_of_one_panorama_are_one_viewpoint() -> None:
    """Le test d'acceptation central.

    Les compter séparément ferait croire un besoin servi par une parallaxe qui
    n'existe pas — huit caps sur un panorama, c'est huit fichiers et un point
    de vue.
    """
    panorama = Panorama("pano-A")
    framings = framings_towards(90.0, extra_offsets=(-30.0, 30.0, 180.0))
    candidates = candidates_from([panorama], framings)

    assert len(candidates) == 4

    grouped = group_viewpoints(candidates, separation_m=10.0)
    assert len(set(grouped.values())) == 1


def test_distinct_panoramas_are_distinct_viewpoints() -> None:
    first = Panorama("pano-A", lat=45.5734, lon=-73.4433)
    second = Panorama("pano-B", lat=45.5741, lon=-73.4440)

    candidates = candidates_from([first, second], [Framing(heading_deg=0.0)])
    grouped = group_viewpoints(candidates, separation_m=10.0)

    assert len(set(grouped.values())) == 2


def test_a_panorama_identity_beats_proximity() -> None:
    """Deux panoramas voisins restent deux points de vue.

    Le fournisseur les distingue : les fondre parce qu'ils sont proches
    reviendrait à corriger sa décision avec une tolérance arbitraire.
    """
    close_pair = [
        Panorama("pano-A", lat=45.5734, lon=-73.4433),
        Panorama("pano-B", lat=45.57341, lon=-73.44331),
    ]

    candidates = candidates_from(close_pair, [Framing(heading_deg=0.0)])
    grouped = group_viewpoints(candidates, separation_m=50.0)

    assert len(set(grouped.values())) == 2


def test_many_framings_never_inflate_the_viewpoint_count() -> None:
    """Le défaut chiffré à l'étape 2 : huit caps, huit fichiers, un point de vue."""
    panorama = Panorama("pano-A")
    framings = [Framing(heading_deg=angle) for angle in range(0, 360, 45)]

    candidates = candidates_from([panorama], framings)
    grouped = group_viewpoints(candidates, separation_m=10.0)

    assert len(candidates) == 8
    assert len(set(grouped.values())) == 1


# --- le cap est demandé, non observé ------------------------------------------


def test_the_heading_is_an_intention_never_a_measurement() -> None:
    """Le porter comme mesure ferait juger un secteur sur notre propre choix."""
    candidate = candidate_from(Panorama("pano-A"), Framing(heading_deg=42.0))

    assert candidate.requested_heading_deg == 42.0
    assert candidate.original_heading_deg is None
    assert candidate.computed_heading_deg is None
    assert candidate.heading_is_measured is False


def test_the_declared_framing_makes_the_geometry_computable() -> None:
    """C'est ce qui distingue Street View : le cadrage est connu d'avance."""
    candidate = candidate_from(Panorama("pano-A"), Framing(heading_deg=0.0))

    assert candidate.requested_fov_deg == 80.0
    assert candidate.advertised_width == 640
    assert candidate.advertised_height == 640


# --- aucune URL, aucun secret -------------------------------------------------


def test_no_url_is_persisted(tmp_path) -> None:
    candidates = candidates_from([Panorama("pano-A")], [Framing(heading_deg=0.0)])
    serialised = json.dumps([json.loads(c.model_dump_json()) for c in candidates])

    assert "://" not in serialised
    assert "key=" not in serialised
    assert "googleapis" not in serialised


def test_the_address_is_rebuilt_from_the_framing_alone() -> None:
    candidate = candidate_from(Panorama("pano-A"), Framing(heading_deg=42.0))

    url = resolve_url(candidate.request_spec, signed=False)

    assert "pano=pano-A" in url
    assert "heading=42.0" in url
    assert "fov=80.0" in url
    assert "size=640x640" in url
    assert "key=" not in url


def test_the_request_is_actually_signed(monkeypatch) -> None:
    """« L'appelant ajoutera la clé » était une intention que personne
    n'honorait : ni la mesure de volume, ni le téléchargement.

    Street View était donc validé par tests unitaires et inutilisable en réel.
    """
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "cle-d-essai")

    candidate = candidate_from(Panorama("pano-A"), Framing(heading_deg=0.0))
    signed = resolve_url(candidate.request_spec)

    assert "key=cle-d-essai" in signed


def test_the_unsigned_form_exists_for_reports(monkeypatch) -> None:
    """Une URL signée dans un fichier versionné y mettrait le secret."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "cle-d-essai")

    candidate = candidate_from(Panorama("pano-A"), Framing(heading_deg=0.0))

    assert "key=" not in resolve_url(candidate.request_spec, signed=False)


def test_the_acquisition_path_signs_its_request(monkeypatch) -> None:
    """Le raccord qui manquait, éprouvé là où il compte."""
    from hotel_pipeline.acquire import resolve_url as acquire_url

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "cle-d-essai")

    candidate = candidate_from(Panorama("pano-A"), Framing(heading_deg=0.0))
    url = acquire_url("street_view", candidate.request_spec)

    assert "key=cle-d-essai" in url


def test_an_incomplete_framing_cannot_be_resolved() -> None:
    """Une sphère ne devient une image qu'une fois entièrement cadrée."""
    with pytest.raises(ValueError, match="cadrage incomplet"):
        resolve_url({"pano_id": "pano-A", "heading_deg": "0.0"})


def test_the_acquisition_refuses_an_incomplete_framing() -> None:
    from hotel_pipeline.acquire import AcquisitionRefused, resolve_url as acquire_url

    with pytest.raises(AcquisitionRefused, match="cadrage incomplet"):
        acquire_url("street_view", {"pano_id": "pano-A"})


def test_street_view_now_has_a_resolver() -> None:
    from hotel_pipeline.acquire import RESOLVERS

    assert "street_view" in RESOLVERS


# --- ce que la source établit, et ce qu'elle n'établit pas --------------------


def test_outdoor_evidence_comes_from_the_source() -> None:
    """Street View publie de la voirie : ce n'est pas une supposition."""
    candidate = candidate_from(Panorama("pano-A"), Framing(heading_deg=0.0))

    assert candidate.outdoor_evidence
    assert "voirie" in candidate.outdoor_evidence


def test_a_capture_month_never_becomes_a_precise_instant() -> None:
    dated = candidate_from(Panorama("pano-A", date="2024-06"), Framing(heading_deg=0.0))
    undated = candidate_from(Panorama("pano-B", date=None), Framing(heading_deg=0.0))

    assert dated.captured_at == datetime(2024, 6, 1, tzinfo=timezone.utc)
    assert undated.captured_at is None


def test_the_announced_size_is_not_a_measurement() -> None:
    """Le service peut rendre moins : la mesure sur le fichier fera foi."""
    candidate = candidate_from(
        Panorama("pano-A"), Framing(heading_deg=0.0, size="2048x2048")
    )

    assert candidate.advertised_width == 2048
    # Rien n'est encore mesuré : aucun fichier n'existe. La taille du cadrage
    # figure parmi les résolutions disponibles, aux côtés du vocabulaire que le
    # plan manipule — Street View rend la taille demandée, dans la limite d'un
    # plafond, et non une liste fermée.
    assert "2048x2048" in candidate.available_resolutions
    # Les résolutions déclarées doivent être celles que la traduction produit :
    # déclarer « 256 » quand elle demande « 256x256 » faisait refuser le plan.
    from hotel_pipeline.acquisition_request import PROVIDER_RESOLUTIONS

    assert set(PROVIDER_RESOLUTIONS["street_view"].values()) <= set(
        candidate.available_resolutions
    )
