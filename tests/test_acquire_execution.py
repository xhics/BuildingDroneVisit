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
    requested: list[str] = []

    def fetch(candidate, target, request=None, ceiling=None):  # noqa: ANN001, ANN202
        calls.append(candidate.candidate_id)
        # Ce que le téléchargeur reçoit **réellement** : sans cette trace, le
        # plan pouvait annoncer 256 et l'image arriver en 2048.
        requested.append(
            request.provider_resolution if request is not None else "sans-requête"
        )
        # Le plafond est respecté ici comme il le serait sur le réseau : un
        # faux téléchargeur qui l'ignorerait ne prouverait rien.
        if ceiling is not None and len(payload) > ceiling:
            from hotel_pipeline.download import DownloadRefused

            raise DownloadRefused(
                f"dépassement : {len(payload)} octets pour un plafond de {ceiling}"
            )
        target.write_bytes(payload)
        return target

    fetch.calls = calls  # type: ignore[attr-defined]
    fetch.requested = requested  # type: ignore[attr-defined]
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
        plan_digest="pd", fetcher=fake_fetcher(),
    )

    asset = acquired[0]
    assert asset.source_url_or_id == "c1"
    assert "://" not in asset.source_url_or_id


def test_no_url_is_written_into_the_asset(tmp_path) -> None:
    acquired, _ = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(),
    )

    assert "://" not in acquired[0].model_dump_json()


def test_the_provenance_binds_the_file_to_the_plan_that_chose_it(tmp_path) -> None:
    acquired, report = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd0", fetcher=fake_fetcher(),
    )

    provenance = acquired[0].acquisition
    assert provenance.plan_id == "p1"
    assert provenance.plan_digest == "pd0"
    assert provenance.candidate_id == "c1"
    assert provenance.run_id == report.run_id
    assert provenance.intents == [CaptureIntent.BUILDING_CAPTURE]


def test_acquisition_establishes_no_right(tmp_path) -> None:
    """`--rights owned` permettait d'écrire un statut juridique sans preuve.

    L'acquisition constate d'où vient un fichier ; elle ne tranche rien.
    """
    acquired, _ = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(),
    )

    asset = acquired[0]
    assert asset.rights is Rights.PUBLIC_UNCLEARED
    assert asset.rights_encumbered is False
    assert asset.rights_history == []
    # Et donc : rien n'est éligible production à la sortie de l'acquisition.
    assert asset.usable_in_production is False


def test_a_source_licence_is_kept_as_a_claim_not_an_authorisation(tmp_path) -> None:
    """Afficher « CC BY » ne prouve pas qu'on détenait les droits de l'accorder."""
    claiming = candidate()
    claiming = claiming.model_copy(
        update={"request_spec": {**claiming.request_spec, "licence_claim": "CC BY-SA 4.0"}}
    )

    acquired, _ = run(
        plan(), {"c1": claiming}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(),
    )

    asset = acquired[0]
    assert asset.rights is Rights.PUBLIC_UNCLEARED
    assert "revendiquée par la source" in asset.rights_note
    assert "CC BY-SA 4.0" in asset.rights_note


def test_exterior_is_never_presumed(tmp_path) -> None:
    """Une vue d'intérieur déclarée extérieure fausserait la couverture."""
    from hotel_pipeline.schemas import ExteriorInterior

    acquired, _ = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(),
    )

    assert acquired[0].exterior_or_interior is ExteriorInterior.UNKNOWN


# --- le volume téléchargé se confronte au volume consenti ---------------------


def test_the_report_compares_downloaded_against_consented(tmp_path) -> None:
    _, report = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(),
    )

    volume = report.as_dict()["volume"]
    assert volume["consented_bytes"] == len(JPEG)
    assert volume["downloaded_bytes"] == len(JPEG)
    assert volume["within_consent"] is True


def test_a_download_larger_than_announced_is_refused_not_just_reported(
    tmp_path,
) -> None:
    """Le dépassement était constaté après coup, donc déjà sur le disque.

    Il est désormais refusé pendant le flux : le fichier n'atteint jamais sa
    place, et aucun asset n'est publié.
    """
    acquired, report = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS, plan_digest="pd",
        fetcher=fake_fetcher(JPEG + b"\x00" * 5000),
    )

    assert acquired == [], "aucun asset publié"
    assert "dépassement" in report.failed["c1"]
    assert report.published == 0
    assert list(tmp_path.glob("*.jpg")) == [], "aucun fichier final"
    assert report.outcomes["c1"]["bytes_published"] == 0


# --- le fichier acquis est confronté à son empreinte --------------------------


def test_a_tampered_file_is_caught_by_verification(tmp_path) -> None:
    from hotel_pipeline.acquisition import verify_acquired

    acquired, _ = run(
        plan(), {"c1": candidate()}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(),
    )
    assert verify_acquired(acquired) == []

    (tmp_path / "c1.jpg").write_bytes(JPEG + b"altered")

    problems = verify_acquired(acquired)
    assert any("empreinte du fichier" in problem for problem in problems)


# --- ce qui est demandé, mesuré et publié doit être la même chose ------------


def test_a_preview_downloads_a_thumbnail_and_says_so(tmp_path) -> None:
    """Le défaut d'origine : le plan annonçait 256, l'image arrivait en 2048.

    `fetch()` ne recevait que le candidat, dont `request_spec` portait encore
    `thumb_2048` ; la provenance inscrivait pourtant `256`. Elle décrivait donc
    un fichier qui n'était pas celui du disque.
    """
    fetcher = fake_fetcher()
    planned = plan(acquisitions=[
        PlannedAcquisition(
            candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="aperçu",
            resolution="256", expected_bytes=len(JPEG),
        )
    ])

    acquired, report = run(
        planned, {"c1": candidate("c1")}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fetcher,
    )

    assert fetcher.requested == ["thumb_256"], (
        "le téléchargeur doit recevoir la résolution fournisseur, non celle "
        "du vocabulaire du plan"
    )
    assert acquired
    trace = report.requested["c1"]
    assert trace["provider_resolution"] == "thumb_256", (
        "la provenance décrit le fichier obtenu"
    )
    assert trace["semantic_resolution"] == "256", (
        "et conserve ce que le plan demandait"
    )
    assert trace["request_digest"]


def test_a_full_acquisition_downloads_the_full_image(tmp_path) -> None:
    fetcher = fake_fetcher()
    planned = plan(acquisitions=[
        PlannedAcquisition(
            candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="complet",
            resolution="2048", expected_bytes=len(JPEG),
        )
    ])

    acquired, report = run(
        planned, {"c1": candidate("c1")}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fetcher,
    )

    assert fetcher.requested == ["thumb_2048"]
    assert report.requested["c1"]["provider_resolution"] == "thumb_2048"


def test_an_untranslatable_resolution_downloads_nothing(tmp_path) -> None:
    """Un plan qu'on ne sait pas exécuter ne doit pas s'exécuter à moitié."""
    fetcher = fake_fetcher()
    planned = plan(acquisitions=[
        PlannedAcquisition(
            candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="essai",
            resolution="une-résolution-inconnue", expected_bytes=len(JPEG),
        )
    ])
    limited = candidate("c1", available_resolutions=["thumb_2048"])

    with pytest.raises(AcquisitionRefused, match="aucune acquisition lancée"):
        run(planned, {"c1": limited}, tmp_path, DIGESTS,
            plan_digest="pd", fetcher=fetcher)

    assert fetcher.calls == [], "rien n'a été téléchargé"


def test_the_provenance_names_the_file_that_is_on_disk(tmp_path) -> None:
    """La provenance décrit ce qui a été obtenu, non ce qui était demandé.

    Elle inscrivait `acquisition.resolution` — le vocabulaire du plan — alors
    que `fetch` téléchargeait selon `request_spec`. Un lecteur y aurait lu
    « 256 » sur une image de 2048.
    """
    planned = plan(acquisitions=[
        PlannedAcquisition(
            candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="aperçu",
            resolution="256", expected_bytes=len(JPEG),
        )
    ])

    acquired, _ = run(
        planned, {"c1": candidate("c1")}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(),
    )

    provenance = acquired[0].acquisition
    assert provenance.resolution == "thumb_256", (
        "la résolution inscrite est celle du fournisseur"
    )
    assert provenance.requested_resolution == "256", (
        "ce que le plan demandait reste consultable, à côté"
    )
    assert provenance.request_digest, "l'empreinte de la requête est publiée"


def test_a_request_that_differs_from_the_consented_one_is_refused(tmp_path) -> None:
    """Le consentement porte sur des requêtes précises.

    Si celles qu'on émet diffèrent, ce qui a été accepté n'est pas ce qui
    serait téléchargé — et le volume consenti ne décrit plus rien.
    """
    fetcher = fake_fetcher()
    planned = plan(acquisitions=[
        PlannedAcquisition(
            candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="essai",
            resolution="256", expected_bytes=len(JPEG),
            # Une empreinte qui ne correspond à aucune requête réelle : le plan
            # a été consenti pour autre chose.
            request_digest="0" * 16,
        )
    ])

    with pytest.raises(AcquisitionRefused, match="différentes de celles consenties"):
        run(planned, {"c1": candidate("c1")}, tmp_path, DIGESTS,
            plan_digest="pd", fetcher=fetcher)

    assert fetcher.calls == [], "rien n'a été téléchargé"


def test_a_matching_digest_lets_the_acquisition_proceed(tmp_path) -> None:
    """Sans quoi le verrou bloquerait tout, y compris le cas légitime."""
    from hotel_pipeline.acquisition_request import resolve

    subject = candidate("c1")
    acquisition = PlannedAcquisition(
        candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
        serves_demands=["d1"], selection_rationale="essai",
        resolution="256", expected_bytes=len(JPEG),
    )
    request = resolve(subject, acquisition)
    planned = plan(acquisitions=[
        acquisition.model_copy(update={"request_digest": request.digest})
    ])

    acquired, report = run(
        planned, {"c1": subject}, tmp_path, DIGESTS,
        plan_digest="pd", fetcher=fake_fetcher(),
    )

    assert acquired
    assert report.requested["c1"]["request_digest"] == request.digest, (
        "la même empreinte, du plan jusqu'au fichier produit"
    )


# --- publication : les six ou aucun -------------------------------------------


def _six_plan(sizes):
    """Un plan de six acquisitions, chacune avec sa taille annoncée."""
    return plan(acquisitions=[
        PlannedAcquisition(
            candidate_id=f"c{i}", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="essai",
            resolution="256", expected_bytes=size,
        )
        for i, size in enumerate(sizes, start=1)
    ])


def test_the_sixth_failure_publishes_nothing_of_the_first_five(tmp_path) -> None:
    """Publier ce qui a réussi ferait croire à une acquisition partielle
    consentie, et le manifeste décrirait un lot qui n'a jamais existé.
    """
    candidates = {f"c{i}": candidate(f"c{i}") for i in range(1, 7)}
    planned = _six_plan([len(JPEG)] * 6)

    appels: list[str] = []

    def fetch(subject, target, request=None, ceiling=None):  # noqa: ANN001
        appels.append(subject.candidate_id)
        if subject.candidate_id == "c6":
            # Le sixième dépasse : les cinq premiers sont déjà en staging.
            from hotel_pipeline.download import DownloadRefused

            raise DownloadRefused("dépassement sur le sixième")
        target.write_bytes(JPEG)
        return target

    acquired, report = run(
        planned, candidates, tmp_path, DIGESTS, plan_digest="pd", fetcher=fetch,
    )

    assert len(appels) == 6, "les six ont bien été tentés"
    assert acquired == [], "aucun asset publié"
    assert report.published == 0
    assert sorted(p.name for p in tmp_path.glob("*.jpg")) == [], (
        "aucun fichier final des cinq premiers"
    )
    assert not (tmp_path / ".staging").exists() or not list(
        (tmp_path / ".staging").iterdir()
    ), "le staging est vidé"


def test_six_successes_publish_atomically(tmp_path) -> None:
    """Sans quoi le refus bloquerait aussi le cas nominal."""
    candidates = {f"c{i}": candidate(f"c{i}") for i in range(1, 7)}
    planned = _six_plan([len(JPEG)] * 6)

    acquired, report = run(
        planned, candidates, tmp_path, DIGESTS, plan_digest="pd",
        fetcher=fake_fetcher(),
    )

    assert len(acquired) == 6
    assert report.published == 6
    assert len(list(tmp_path.glob("*.jpg"))) == 6
    for asset in acquired:
        assert ".staging" not in asset.local_path, (
            "l'asset porte sa place définitive, non celle du staging"
        )


def test_the_receipt_separates_the_four_byte_counts(tmp_path) -> None:
    """Déclaré, reçu, en staging, publié : les fondre masquerait le cas qu'on
    veut voir."""
    candidates = {"c1": candidate("c1")}
    planned = _six_plan([len(JPEG)])[:1] if False else plan(acquisitions=[
        PlannedAcquisition(
            candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="essai",
            resolution="256", expected_bytes=len(JPEG),
        )
    ])

    _, report = run(
        planned, candidates, tmp_path, DIGESTS, plan_digest="pd",
        fetcher=fake_fetcher(),
    )

    outcome = report.outcomes["c1"]
    assert outcome["declared_bytes"] == len(JPEG)
    assert outcome["bytes_received"] == len(JPEG)
    assert outcome["bytes_staged"] == len(JPEG)
    assert outcome["bytes_published"] == len(JPEG)


def test_a_refusal_reports_zero_published_despite_bytes_received(tmp_path) -> None:
    """Le cas qui distingue les quatre comptes."""
    candidates = {"c1": candidate("c1"), "c2": candidate("c2")}
    planned = _six_plan([len(JPEG), len(JPEG)])

    def fetch(subject, target, request=None, ceiling=None):  # noqa: ANN001
        if subject.candidate_id == "c2":
            from hotel_pipeline.download import DownloadRefused

            raise DownloadRefused("corps trop court")
        target.write_bytes(JPEG)
        return target

    _, report = run(
        planned, candidates, tmp_path, DIGESTS, plan_digest="pd", fetcher=fetch,
    )

    assert report.outcomes["c1"]["bytes_staged"] == len(JPEG)
    assert report.outcomes["c1"]["bytes_published"] == 0, (
        "reçu et mis en staging, jamais publié"
    )
    assert report.outcomes["c2"]["bytes_received"] == 0
    assert "trop court" in report.outcomes["c2"]["refused"]


# --- le consentement s'attache aux requêtes et au plafond ---------------------


def test_acquiring_a_request_that_was_not_consented_is_refused(tmp_path) -> None:
    """Réécrire une résolution après l'accord téléchargerait autre chose sous
    le même consentement."""
    from hotel_pipeline.plan import consent

    subject = candidate("c1")
    acquisition = PlannedAcquisition(
        candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
        serves_demands=["d1"], selection_rationale="essai",
        resolution="256", expected_bytes=len(JPEG),
        request_digest="empreinte-consentie",
    )
    accepted = consent(
        plan(acquisitions=[acquisition]), DIGESTS, download_contract_version=1
    )

    assert accepted.consented_max_bytes == len(JPEG)
    assert accepted.consented_request_digests == ["empreinte-consentie"]
    assert accepted.consented_download_contract_version == 1

    # La requête réelle diffère de celle consentie : `run` doit refuser.
    fetcher = fake_fetcher()
    with pytest.raises(AcquisitionRefused, match="différentes de celles consenties"):
        run(accepted, {"c1": subject}, tmp_path, DIGESTS,
            plan_digest="pd", fetcher=fetcher)

    assert fetcher.calls == [], "rien n'a été téléchargé"


def test_consent_refuses_a_partial_volume(tmp_path) -> None:
    """Consentir à un total dont une part n'est pas mesurée serait consentir à
    ce qui n'a pas été montré."""
    from hotel_pipeline.plan import PlanRefused, consent

    incomplet = plan(acquisitions=[
        PlannedAcquisition(
            candidate_id="c1", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="essai",
            expected_bytes=len(JPEG), request_digest="d1",
        ),
        PlannedAcquisition(
            candidate_id="c2", intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["d1"], selection_rationale="essai",
            request_digest="d2",
        ),
    ])

    with pytest.raises(PlanRefused, match="inconnue"):
        consent(incomplet, DIGESTS)
