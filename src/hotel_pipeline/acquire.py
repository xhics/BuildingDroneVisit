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

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "planned": self.planned,
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


def check_executable(plan: AcquisitionPlan, digests: dict[str, str | None]) -> list[str]:
    """Le plan peut-il être exécuté maintenant, et tel quel ?"""
    from .acquisition import plan_is_current

    return plan_is_current(plan, digests)


def run(
    plan: AcquisitionPlan,
    candidates: dict,
    destination: Path,
    digests: dict[str, str | None],
    plan_digest: str,
    fetcher=None,  # noqa: ANN001 — injecté pour éprouver sans réseau
    run_id: str | None = None,
    rights=None,  # noqa: ANN001
) -> tuple[list, AcquireReport]:
    """Télécharge ce que le plan porte, et rien d'autre.

    `fetcher` reçoit (candidat, chemin) et rend le chemin écrit. La couture
    porte sur les deux gestes à la fois : résoudre l'adresse et la télécharger
    sont **une seule** interaction avec le fournisseur, et les séparer laissait
    la résolution appeler le réseau derrière un téléchargeur pourtant injecté.
    """
    from .acquisition import as_asset, new_run_id
    from .schemas import Rights

    if plan.status is not PlanStatus.EXECUTABLE:
        raise AcquisitionRefused(
            f"plan {plan.plan_id!r} à l'état « {plan.status.value} » : un plan "
            "non consenti ne s'acquiert pas"
        )

    stale = check_executable(plan, digests)
    if stale:
        raise AcquisitionRefused(
            "plan périmé — les images auraient été choisies pour un autre "
            "état : " + " ; ".join(stale)
        )

    report = AcquireReport(
        run_id=run_id or new_run_id(),
        plan_id=plan.plan_id,
        planned=len(plan.acquisitions),
        bytes_consented=plan.known_bytes,
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

        try:
            target = destination / f"{acquisition.candidate_id}.jpg"
            written = (fetcher or fetch)(candidate, target)
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
            requested_pitch_deg=candidate.requested_pitch_deg,
            sequence_id=candidate.sequence_id,
            panorama_id=candidate.panorama_id,
            camera_type=candidate.camera_type,
            resolution=acquisition.resolution,
            acquired_at=datetime.now(timezone.utc),
            # Annoncées, conservées **à côté** des mesures : les faire
            # coïncider d'office masquerait un rendu tronqué ou redimensionné.
            advertised_width=candidate.advertised_width,
            advertised_height=candidate.advertised_height,
            run_id=report.run_id,
        )

        asset = as_asset(candidate, provenance, written, rights or Rights.UNKNOWN)
        acquired.append(asset)
        report.acquired += 1
        report.bytes_downloaded += asset.file_size_bytes or 0

    log.info(
        "acquisition %s : %d/%d fichier(s), %d octet(s) sur %d consenti(s)",
        report.run_id, report.acquired, report.planned,
        report.bytes_downloaded, report.bytes_consented,
    )
    return acquired, report


def fetch(candidate, target: Path) -> Path:  # noqa: ANN001
    """Résout l'adresse puis la télécharge : un seul geste, un seul refus.

    C'est ici, et nulle part ailleurs, qu'une URL existe.
    """
    import requests

    from .providers.cache import ensure_online

    url = resolve_url(candidate.source, candidate.request_spec)
    ensure_online("acquisition d'image")
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()
    with target.open("wb") as handle:
        for chunk in response.iter_content(1 << 16):
            handle.write(chunk)
    return target
