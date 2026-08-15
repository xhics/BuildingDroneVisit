"""Volume d'une acquisition : mesuré, jamais estimé (collecte V2).

Le plan séparait déjà connu et inconnu, et refusait un consentement partiel.
Restait à alimenter le connu — sans quoi Street View, qui n'annonce aucune
taille, était définitivement inacquérable.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.plan import build
from hotel_pipeline.schemas.acquisition import (
    REQUIRED_PLAN_DIGESTS,
    CaptureCandidate,
    CaptureDemand,
    CaptureIntent,
    TargetKind,
    VolumeStatus,
)
from hotel_pipeline.volumes import IMPLAUSIBLE_BYTES, VolumeReport, measure

DIGESTS = {name: f"{name[:4]}0" for name in REQUIRED_PLAN_DIGESTS}


def candidate(candidate_id: str = "c1", **overrides) -> CaptureCandidate:
    fields = dict(
        candidate_id=candidate_id, source="mapillary", provider_id=candidate_id,
        camera_lat=45.573, camera_lon=-73.443,
        request_spec={"provider_id": candidate_id, "resolution": "thumb_2048"},
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def demand(**overrides) -> CaptureDemand:
    fields = dict(
        demand_id="d1", intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="front",
    )
    fields.update(overrides)
    return CaptureDemand(**fields)


def prober(sizes: dict):
    """Sonde d'essai : rend ce qu'on lui dit, sans réseau."""
    calls: list[str] = []

    def probe(candidate):  # noqa: ANN001, ANN202
        calls.append(candidate.candidate_id)
        value = sizes.get(candidate.candidate_id, "absent")
        if isinstance(value, Exception):
            raise value
        return None if value == "absent" else value

    probe.calls = calls  # type: ignore[attr-defined]
    return probe


# --- mesurer n'est pas télécharger --------------------------------------------


def test_a_measured_size_makes_the_volume_exact() -> None:
    report = measure([candidate("c1")], prober({"c1": 12_345}))

    assert report.measured == {"c1": 12_345}
    assert report.known_bytes == 12_345

    plan, _, _ = build(
        "h", [candidate("c1")], [demand()], DIGESTS, sizes=report.measured
    )
    assert plan.volume_status is VolumeStatus.EXACT
    assert plan.known_bytes == 12_345


def test_an_undeclared_length_stays_unknown() -> None:
    """La deviner serait pire que l'ignorer."""
    report = measure([candidate("c1")], prober({}))

    assert report.measured == {}
    assert "ne déclare pas de longueur" in report.unmeasured["c1"]


def test_nothing_is_estimated_from_the_dimensions() -> None:
    """640×640 ne dit rien du taux de compression."""
    large = candidate("c1", advertised_width=4096, advertised_height=4096)

    report = measure([large], prober({}))

    assert report.measured == {}
    assert report.known_bytes == 0


def test_an_unmeasured_candidate_is_never_counted_as_zero() -> None:
    """Un total « exact » reposant sur des zéros serait exact et faux."""
    report = measure([candidate("c1"), candidate("c2")], prober({"c1": 1000}))

    plan, _, _ = build(
        "h",
        [candidate("c1"), candidate("c2", camera_lat=45.5745, camera_lon=-73.4445)],
        [demand(viewpoints_required=2)], DIGESTS, sizes=report.measured,
    )

    assert plan.known_bytes == 1000
    assert plan.unknown_size_items == ["c2"]
    assert plan.volume_status is VolumeStatus.PARTIAL


def test_a_partial_measurement_still_refuses_consent() -> None:
    """C'est la bonne réponse, non un échec."""
    from hotel_pipeline.plan import consent

    report = measure([candidate("c1"), candidate("c2")], prober({"c1": 1000}))
    plan, _, _ = build(
        "h",
        [candidate("c1"), candidate("c2", camera_lat=45.5745, camera_lon=-73.4445)],
        [demand(viewpoints_required=2)], DIGESTS, sizes=report.measured,
    )

    assert plan.volume_status is not VolumeStatus.EXACT
    # Le plan reste consentable en tant qu'objet ; c'est la CLI qui refuse un
    # consentement sur un total partiel, et le statut le lui dit.
    assert consent(plan, DIGESTS).volume_status is VolumeStatus.PARTIAL


# --- ce qui est invraisemblable n'est pas mesuré ------------------------------


def test_an_implausible_length_is_refused_rather_than_consented() -> None:
    """Deux gigaoctets signalent un résolveur fautif, pas une photographie."""
    report = measure([candidate("c1")], prober({"c1": IMPLAUSIBLE_BYTES + 1}))

    assert report.measured == {}
    assert "invraisemblable" in report.unmeasured["c1"]


def test_a_zero_length_is_not_a_free_download() -> None:
    report = measure([candidate("c1")], prober({"c1": 0}))

    assert report.measured == {}
    assert "invraisemblable" in report.unmeasured["c1"]


def test_a_failing_probe_is_recorded_not_silently_skipped() -> None:
    report = measure(
        [candidate("c1")], prober({"c1": RuntimeError("503 du service")})
    )

    assert report.measured == {}
    assert "503" in report.unmeasured["c1"]


# --- Street View devient mesurable --------------------------------------------


def test_street_view_is_unacquirable_until_its_size_is_measured() -> None:
    """L'endpoint image n'annonce rien : sans mesure, rien n'est acquérable."""
    from hotel_pipeline.collectors.streetview_v2 import Framing, candidates_from

    class Panorama:
        pano_id, lat, lon, date, copyright = "pano-A", 45.5734, -73.4433, "2024-06", None

    candidates = candidates_from([Panorama()], [Framing(heading_deg=0.0)])

    without, _, _ = build("h", candidates, [demand()], DIGESTS)
    assert without.volume_status is VolumeStatus.UNKNOWN

    measured = measure(candidates, prober({candidates[0].candidate_id: 48_000}))
    with_sizes, _, _ = build(
        "h", candidates, [demand()], DIGESTS, sizes=measured.measured
    )

    assert with_sizes.volume_status is VolumeStatus.EXACT
    assert with_sizes.known_bytes == 48_000


# --- le rapport dit comment il a obtenu ce qu'il annonce ----------------------


def test_the_report_states_that_no_body_was_downloaded() -> None:
    report = measure([candidate("c1")], prober({"c1": 1000}))
    published = report.as_dict()

    assert "sans télécharger le corps" in published["note"]
    assert "ni estimée" in published["note"]
    assert published["known_bytes"] == 1000


def test_the_report_names_every_unmeasured_candidate() -> None:
    report = measure(
        [candidate("c1"), candidate("c2"), candidate("c3")],
        prober({"c2": 500}),
    )
    published = report.as_dict()

    assert published["measured"] == 1
    assert published["unmeasured"] == 2
    assert set(published["reasons"]) == {"c1", "c3"}


def test_measuring_is_opt_in_and_says_so(tmp_path, monkeypatch) -> None:
    """Interroger un service facturé à l'appel doit rester un geste explicite."""
    import json

    from typer.testing import CliRunner

    from hotel_pipeline.cli import app
    from hotel_pipeline.schemas.acquisition import CandidateManifest, CaptureDemandManifest
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(app, [
        "init", "hotel-test", "--address", "1 rue Test", "--name", "Hôtel Test",
        "--country", "CA", "--timezone", "America/Toronto", "--ocr-language", "fr",
        "--lat", "45.573", "--lon", "-73.443",
    ])
    workspace = Workspace("hotel-test")
    workspace.write_json(
        "01_sources/capture_demands.json",
        json.loads(
            CaptureDemandManifest(hotel_id="hotel-test", demands=[demand()])
            .model_dump_json()
        ),
    )
    workspace.write_json(
        "01_sources/candidates_20260814T000000000000Z.json",
        json.loads(
            CandidateManifest(hotel_id="hotel-test", candidates=[candidate()])
            .model_dump_json()
        ),
    )

    result = runner.invoke(app, ["assets", "plan", "hotel-test"])

    assert result.exit_code == 0, result.output
    assert "volumes non mesurés" in result.output
    assert "--measure-volumes" in result.output
