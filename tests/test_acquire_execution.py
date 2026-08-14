"""Exécution d'un plan d'acquisition (collecte V2, étape 3).

Le seul module qui télécharge, et il ne télécharge que ce qu'un plan consenti
porte. Ce qui est éprouvé : les trois refus qui le précèdent, l'absence d'URL
persistée, et le fait qu'un fichier soit **mesuré** plutôt que cru sur parole.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.acquire import AcquisitionRefused, resolve_url, run
from hotel_pipeline.schemas import Rights
from hotel_pipeline.schemas.acquisition import (
    REQUIRED_PLAN_DIGESTS,
    AcquisitionPlan,
    CaptureCandidate,
    CaptureIntent,
    PlannedAcquisition,
    PlanStatus,
)

DIGESTS = {name: f"{name[:4]}0" for name in REQUIRED_PLAN_DIGESTS}

def _jpeg(width: int = 64, height: int = 64) -> bytes:
    """Un vrai JPEG, produit plutôt que bricolé.

    `measure()` lit les dimensions du fichier, et **refuse** une image d'une
    seule couleur : c'est ainsi qu'une réponse de remplacement se reconnaît
    d'une photographie. La fixture porte donc un motif, pas un aplat.
    """
    import io

    from PIL import Image

    # Des blocs de 8 px : un damier plus fin est écrasé par la compression,
    # et l'image redeviendrait uniforme — donc refusée, à juste titre.
    image = Image.new("RGB", (width, height), (10, 120, 200))
    for x in range(width):
        for y in range(height):
            if (x // 8 + y // 8) % 2:
                image.putpixel((x, y), (240, 30, 60))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


JPEG = _jpeg()


def candidate(candidate_id: str = "c1", **overrides) -> CaptureCandidate:
    fields = dict(
        candidate_id=candidate_id, source="mapillary", provider_id=candidate_id,
        camera_lat=45.573, camera_lon=-73.443,
        request_spec={"provider_id": candidate_id, "resolution": "thumb_2048"},
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def plan(status: PlanStatus = PlanStatus.EXECUTABLE, **overrides) -> AcquisitionPlan:
    fields = dict(
        plan_id="p1", hotel_id="h", status=status,
        acquisitions=[
            PlannedAcquisition(
                candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
                serves_demands=["d1"], selection_rationale="essai",
                expected_bytes=len(JPEG),
            )
        ],
        **DIGESTS,
    )
    fields.update(overrides)
    return AcquisitionPlan(**fields)


def fake_fetcher(payload: bytes = JPEG):
    """Écrit un fichier sans réseau, et note ce qui lui a été demandé.

    La couture porte sur résolution **et** téléchargement : les séparer
    laissait la résolution appeler l'API derrière un téléchargeur injecté.
    """
    calls: list[str] = []

    def fetch(candidate, target):  # noqa: ANN001, ANN202
        calls.append(candidate.candidate_id)
        target.write_bytes(payload)
        return target

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


# --- les trois refus ----------------------------------------------------------


def test_a_draft_is_never_acquired(tmp_path) -> None:
    """Un brouillon n'a pas été montré au consentement."""
    fetcher = fake_fetcher()

    with pytest.raises(AcquisitionRefused, match="non consenti"):
        run(
            plan(PlanStatus.DRAFT), {"c1": candidate()}, tmp_path, DIGESTS,
            plan_digest="pd", fetcher=fetcher,
        )

    assert fetcher.calls == []


def test_a_stale_plan_is_refused_before_any_download(tmp_path) -> None:
    """Les images auraient été choisies pour un autre état."""
    fetcher = fake_fetcher()
    moved = dict(DIGESTS)
    moved["corpus_digest"] = "autre-chose"

    with pytest.raises(AcquisitionRefused, match="plan périmé"):
        run(
            plan(), {"c1": candidate()}, tmp_path, moved,
            plan_digest="pd", fetcher=fetcher,
        )

    assert fetcher.calls == []
    assert list(tmp_path.iterdir()) == []


def test_a_candidate_outside_the_manifest_is_not_downloaded(tmp_path) -> None:
    fetcher = fake_fetcher()

    acquired, report = run(
        plan(), {}, tmp_path, DIGESTS, plan_digest="pd", fetcher=fetcher
    )

    assert acquired == []
    assert fetcher.calls == []
    assert "n'existe plus" in report.failed["c1"]


def test_an_unknown_source_has_no_resolver(tmp_path) -> None:
    """Inventer une URL reviendrait à deviner le protocole d'un fournisseur."""
    with pytest.raises(AcquisitionRefused, match="sans résolveur"):
        resolve_url("source-inconnue", {"provider_id": "1"})


def test_a_candidate_without_a_provider_id_cannot_be_resolved() -> None:
    with pytest.raises(AcquisitionRefused, match="provider_id"):
        resolve_url("mapillary", {})


# --- ce qui est acquis est mesuré ---------------------------------------------


def test_an_acquired_file_is_measured_not_believed(tmp_path) -> None:
    """Les dimensions viennent du fichier, jamais du candidat."""
    fetcher = fake_fetcher()

    acquired, report = run(
        plan(), {"c1": candidate(advertised_width=9999, advertised_height=9999)},
        tmp_path, DIGESTS, plan_digest="pd", fetcher=fetcher,
        rights=Rights.OPEN_DATA,
    )

    asset = acquired[0]
    assert asset.width == 64 and asset.height == 64
    assert asset.file_size_bytes == len(JPEG)
    # L'annoncé survit **à côté** du mesuré, sans le remplacer.
    assert asset.acquisition.advertised_width == 9999
    assert report.bytes_downloaded == len(JPEG)


def test_the_asset_identity_comes_from_the_provider_never_the_url(tmp_path) -> None:
    acquired, _ = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(), rights=Rights.OPEN_DATA,
    )

    asset = acquired[0]
    assert asset.source_url_or_id == "c1"
    assert "://" not in asset.source_url_or_id


def test_no_url_is_written_into_the_asset(tmp_path) -> None:
    acquired, _ = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(), rights=Rights.OPEN_DATA,
    )

    assert "://" not in acquired[0].model_dump_json()


def test_the_provenance_binds_the_file_to_the_plan_that_chose_it(tmp_path) -> None:
    acquired, report = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd0", fetcher=fake_fetcher(), rights=Rights.OPEN_DATA,
    )

    provenance = acquired[0].acquisition
    assert provenance.plan_id == "p1"
    assert provenance.plan_digest == "pd0"
    assert provenance.candidate_id == "c1"
    assert provenance.run_id == report.run_id
    assert provenance.intents == [CaptureIntent.BUILDING_CAPTURE]


def test_exterior_is_never_presumed(tmp_path) -> None:
    """Une vue d'intérieur déclarée extérieure fausserait la couverture."""
    from hotel_pipeline.schemas import ExteriorInterior

    acquired, _ = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(), rights=Rights.OPEN_DATA,
    )

    assert acquired[0].exterior_or_interior is ExteriorInterior.UNKNOWN


# --- le volume téléchargé se confronte au volume consenti ---------------------


def test_the_report_compares_downloaded_against_consented(tmp_path) -> None:
    _, report = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(), rights=Rights.OPEN_DATA,
    )

    volume = report.as_dict()["volume"]
    assert volume["consented_bytes"] == len(JPEG)
    assert volume["downloaded_bytes"] == len(JPEG)
    assert volume["within_consent"] is True


def test_a_download_larger_than_announced_is_visible_in_the_report(tmp_path) -> None:
    """Un dépassement doit se lire, pas se découvrir sur le disque."""
    _, report = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS, plan_digest="pd",
        fetcher=fake_fetcher(JPEG + b"\x00" * 5000), rights=Rights.OPEN_DATA,
    )

    volume = report.as_dict()["volume"]
    assert volume["downloaded_bytes"] > volume["consented_bytes"]
    assert volume["within_consent"] is False


# --- le fichier acquis est confronté à son empreinte --------------------------


def test_a_tampered_file_is_caught_by_verification(tmp_path) -> None:
    from hotel_pipeline.acquisition import verify_acquired

    acquired, _ = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(), rights=Rights.OPEN_DATA,
    )
    assert verify_acquired(acquired) == []

    (tmp_path / "c1.jpg").write_bytes(JPEG + b"altered")

    problems = verify_acquired(acquired)
    assert any("empreinte du fichier" in problem for problem in problems)
