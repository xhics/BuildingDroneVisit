"""Exécution d'un plan d'acquisition (collecte V2, étape 3).

Le seul module de la chaîne qui télécharge, et il ne télécharge que ce qu'un
plan **exécutable** porte. Trois refus le précèdent :

```text
brouillon        un plan non consenti ne s'acquiert pas
plan périmé      une empreinte qui a bougé, c'est un choix fait pour un autre état
hors du plan     un candidat non planifié n'a jamais été montré au consentement
```

L'adresse se reconstruit ici, à partir de `request_spec`. C'est le seul endroit
où elle existe : le manifeste n'en garde aucune, puisqu'une URL de CDN expire
et qu'une URL signée porterait la clé d'API jusque dans un fichier versionné.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .logging import get_logger
from .schemas.acquisition import AcquisitionPlan, AcquisitionProvenance, PlanStatus

log = get_logger("acquire")


class AcquisitionRefused(RuntimeError):
    """Rien n'a été téléchargé, et rien n'a été écrit."""


@dataclass
class AcquireReport:
    """Ce qui a été téléchargé, ce qui a échoué, et combien d'octets."""

    run_id: str = ""
    plan_id: str = ""
    planned: int = 0
    acquired: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    bytes_downloaded: int = 0
    bytes_consented: int = 0

    #: Registre des appels de cette acquisition. Publié **même sur échec** :
    #: une exécution interrompue a coûté des appels, et les taire donnerait à
    #: croire qu'elle n'a rien consommé.
    transport: dict = field(default_factory=dict)

    #: Ce qui a été **réellement** demandé, par candidat : résolution
    #: fournisseur et empreinte de requête. Sans cette trace, rien ne permet de
    #: vérifier que le fichier obtenu est celui que le plan décrivait.
    requested: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "planned": self.planned,
            "requested": self.requested,
            "transport": self.transport,
            "acquired": self.acquired,
            "failed": self.failed,
            "volume": {
                "consented_bytes": self.bytes_consented,
                "downloaded_bytes": self.bytes_downloaded,
                "within_consent": self.bytes_downloaded <= self.bytes_consented,
            },
        }


#: Résolveurs d'adresse, par source. Une source sans résolveur ne se télécharge
#: pas : inventer une URL reviendrait à deviner le protocole d'un fournisseur.
RESOLVERS: dict[str, str] = {
    "mapillary": "thumb_2048_url rendu par l'API Graph, à la demande",
    "street_view": "endpoint image, reconstruit depuis le cadrage demandé",
}


def resolve_url(source: str, request_spec: dict[str, str]) -> str:
    """Reconstruit l'adresse d'un candidat au moment du téléchargement.

    Mapillary ne publie pas d'URL durable : la vignette se redemande à l'API
    Graph, qui en rend une valable quelques minutes. C'est précisément pourquoi
    le manifeste n'en conserve aucune.
    """
    if source not in RESOLVERS:
        raise AcquisitionRefused(
            f"source {source!r} sans résolveur d'adresse : le protocole de "
            f"téléchargement n'est pas déclaré. Connus : {sorted(RESOLVERS)}"
        )

    if source == "street_view":
        from .collectors.streetview_v2 import resolve_url as street_view_url

        try:
            return street_view_url(request_spec)
        except ValueError as exc:
            raise AcquisitionRefused(str(exc)) from exc

    provider_id = request_spec.get("provider_id")
    if not provider_id:
        raise AcquisitionRefused(
            "candidat sans `provider_id` : l'adresse ne peut pas être reconstruite"
        )

    from .collectors.mapillary import thumbnail_url

    return thumbnail_url(provider_id, request_spec.get("resolution", "thumb_2048"))


def check_executable(
    plan: AcquisitionPlan, digests: dict[str, str | None], policy=None  # noqa: ANN001
) -> list[str]:
    """Le plan peut-il être exécuté maintenant, et tel quel ?"""
    from .acquisition import plan_is_current

    return plan_is_current(plan, digests, policy)


def run(
    plan: AcquisitionPlan,
    candidates: dict,
    destination: Path,
    digests: dict[str, str | None],
    plan_digest: str,
    fetcher=None,  # noqa: ANN001 — injecté pour éprouver sans réseau
    run_id: str | None = None,
    policy=None,  # noqa: ANN001
) -> tuple[list, AcquireReport]:
    """Télécharge ce que le plan porte, et rien d'autre.

    `fetcher` reçoit (candidat, chemin) et rend le chemin écrit. La couture
    porte sur les deux gestes à la fois : résoudre l'adresse et la télécharger
    sont **une seule** interaction avec le fournisseur, et les séparer laissait
    la résolution appeler le réseau derrière un téléchargeur pourtant injecté.
    """
    from .acquisition import as_asset, new_run_id
    from .rights import acquisition_rights
    from .schemas import Rights

    if plan.status is not PlanStatus.EXECUTABLE:
        raise AcquisitionRefused(
            f"plan {plan.plan_id!r} à l'état « {plan.status.value} » : un plan "
            "non consenti ne s'acquiert pas"
        )

    stale = check_executable(plan, digests, policy)
    if stale:
        raise AcquisitionRefused(
            "plan périmé — les images auraient été choisies pour un autre "
            "état : " + " ; ".join(stale)
        )

    from .providers.transport import ledger as transport_ledger, reset_ledger

    reset_ledger()
    report = AcquireReport(
        run_id=run_id or new_run_id(),
        plan_id=plan.plan_id,
        planned=len(plan.acquisitions),
        bytes_consented=plan.known_bytes,
    )
    # Résolues **avant** toute écriture : un plan dont une acquisition ne se
    # traduit pas annoncerait un volume qu'il ne saurait pas obtenir, et le
    # découvrir au milieu du lot laisserait des fichiers orphelins.
    from .acquisition_request import RequestUnresolvable, resolve_all

    try:
        requests_by_candidate = resolve_all(candidates, plan.acquisitions)
    except RequestUnresolvable as exc:
        raise AcquisitionRefused(f"aucune acquisition lancée — {exc}") from exc

    # Le consentement portait sur des requêtes précises. Si celles qu'on
    # s'apprête à émettre diffèrent, ce qui a été accepté n'est pas ce qui
    # serait téléchargé — et le volume consenti ne décrit plus rien.
    diverged = [
        acquisition.candidate_id
        for acquisition in plan.acquisitions
        if acquisition.request_digest
        and acquisition.candidate_id in requests_by_candidate
        and requests_by_candidate[acquisition.candidate_id].digest
        != acquisition.request_digest
    ]
    if diverged:
        raise AcquisitionRefused(
            "requête(s) différentes de celles consenties : "
            f"{sorted(diverged)} — le volume accepté ne les décrit pas"
        )

    destination.mkdir(parents=True, exist_ok=True)
    acquired = []

    for acquisition in plan.acquisitions:
        candidate = candidates.get(acquisition.candidate_id)
        if candidate is None:
            report.failed[acquisition.candidate_id] = (
                "candidat absent du manifeste : le plan cite une vue qui n'existe plus"
            )
            continue

        request = requests_by_candidate.get(acquisition.candidate_id)
        if request is None:
            report.failed[acquisition.candidate_id] = (
                "requête non résolue : ce que le plan demande ne se traduit "
                "pas dans les termes de la source"
            )
            continue

        try:
            target = destination / f"{acquisition.candidate_id}.jpg"
            written = (fetcher or fetch)(candidate, target, request)
        except (AcquisitionRefused, OSError, RuntimeError, ValueError) as exc:
            report.failed[acquisition.candidate_id] = str(exc)
            continue

        provenance = AcquisitionProvenance(
            provider_id=candidate.provider_id,
            plan_id=plan.plan_id,
            plan_digest=plan_digest,
            candidate_id=candidate.candidate_id,
            intents=list(acquisition.intents),
            primary_intent=acquisition.primary_intent,
            queried_lat=candidate.queried_lat, queried_lon=candidate.queried_lon,
            returned_lat=candidate.camera_lat, returned_lon=candidate.camera_lon,
            original_heading_deg=candidate.original_heading_deg,
            computed_heading_deg=candidate.computed_heading_deg,
            requested_heading_deg=candidate.requested_heading_deg,
            requested_fov_deg=candidate.requested_fov_deg,
            # Ce que l'optique **vaut**, à côté de ce qui a servi au cadrage :
            # un fichier acquis d'un ultra-grand-angle doit le dire.
            observed_horizontal_fov_deg=candidate.observed_horizontal_fov_deg,
            projection_support=candidate.projection_support,
            # Ce que ce fichier venait vérifier, besoin par besoin : la preview
            # arrivait sinon sans rattachement, et son constat n'aurait su à
            # quelle exigence répondre.
            serves_demands=list(acquisition.serves_demands),
            demand_levels=dict(acquisition.demand_levels or {}),
            requested_pitch_deg=candidate.requested_pitch_deg,
            sequence_id=candidate.sequence_id,
            panorama_id=candidate.panorama_id,
            camera_type=candidate.camera_type,
            # La résolution **fournisseur**, non celle du vocabulaire du plan :
            # inscrire « 256 » sur une image obtenue en `thumb_2048` décrirait
            # un fichier qui n'est pas celui du disque.
            resolution=request.provider_resolution,
            requested_resolution=request.semantic_resolution,
            request_digest=request.digest,
            acquired_at=datetime.now(timezone.utc),
            # Annoncées, conservées **à côté** des mesures : les faire
            # coïncider d'office masquerait un rendu tronqué ou redimensionné.
            advertised_width=candidate.advertised_width,
            advertised_height=candidate.advertised_height,
            run_id=report.run_id,
        )

        # L'acquisition constate un fait ; elle ne tranche aucun droit. Une
        # source tierce téléchargée est `public_uncleared`, et la licence
        # revendiquée par le fournisseur reste une **revendication**.
        report.requested[acquisition.candidate_id] = {
            "semantic_resolution": request.semantic_resolution,
            "provider_resolution": request.provider_resolution,
            "request_digest": request.digest,
        }
        asset = as_asset(candidate, provenance, written, Rights.PUBLIC_UNCLEARED)
        asset = asset.model_copy(
            update=acquisition_rights(candidate.request_spec.get("licence_claim"))
        )
        acquired.append(asset)
        report.acquired += 1
        report.bytes_downloaded += asset.file_size_bytes or 0

    report.transport = transport_ledger().as_dict()
    log.info(
        "acquisition %s : %d/%d fichier(s), %d octet(s) sur %d consenti(s)",
        report.run_id, report.acquired, report.planned,
        report.bytes_downloaded, report.bytes_consented,
    )
    return acquired, report


def fetch(candidate, target: Path, request=None) -> Path:  # noqa: ANN001
    """Résout l'adresse puis la télécharge : un seul geste, un seul refus.

    C'est ici, et nulle part ailleurs, qu'une URL existe.

    `request` porte la résolution **fournisseur** décidée au plan. Sans elle,
    le téléchargement retombait sur celle qu'y avait laissée la découverte :
    le plan annonçait 256, l'image arrivait en 2048, et la provenance
    inscrivait 256.
    """
    import requests

    from .providers.cache import ensure_online

    from .providers import transport

    spec = request.request_spec if request is not None else candidate.request_spec
    url = resolve_url(candidate.source, spec)
    response = transport.get(
        candidate.source, transport.Stage.DOWNLOAD, url,
        timeout=60, stream=True,
        request_digest=getattr(request, "digest", None),
        what="acquisition d'image",
    )
    response.raise_for_status()

    written = 0
    with target.open("wb") as handle:
        for chunk in response.iter_content(1 << 16):
            written += handle.write(chunk)

    # Ce qui a **réellement** été écrit, à côté de ce qui était annoncé : leur
    # écart est précisément ce qu'un rapport doit rendre visible.
    attempt = transport.last_attempt()
    if attempt is not None:
        transport.record_written(attempt, written)
    return target
