"""Boucle de vérification des recadrages (Lot 2).

La géométrie propose, les pixels corrigent — mais la correction ne servait à
rien tant qu'elle n'était pas **écrite**. `verified_prominence` existait sur
`RecropOpportunity` et rien ne le remplissait : la sélection retombait donc sur
la distance, et proposait la rue résidentielle arrière où des pavillons absents
du modèle d'obstacles bouchent la vue. Six candidats, six maisons.

Ce module ferme la boucle :

1. **proposer** les recadrages (géométrie) ;
2. **acquérir** ceux qu'on n'a pas encore vus ;
3. **lire** la prominence sur les pixels ;
4. **persister** le verdict, par `(panorama, cap, champ)` ;
5. **re-sélectionner** — la prominence vérifiée départage désormais.

Le registre est append-only et relu à chaque exécution : une vue jugée une fois
n'est pas re-téléchargée, et une campagne interrompue reprend où elle en était.
Les scores y sont conservés même quand ils sont mauvais — savoir qu'un cadrage
ne montre rien est une information, non un échec à oublier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger

log = get_logger("recrop-verify")

#: Registre des vérifications, sous le workspace.
REGISTER = "01_sources/recrop_verifications.json"

#: Regroupement des caps : deux recadrages à moins de cet écart rendent la même
#: image, et partagent donc leur verdict.
HEADING_BUCKET_DEG = 5.0


@dataclass
class VerificationRegister:
    """Ce que les pixels ont dit de chaque recadrage, une fois pour toutes."""

    entries: dict[str, dict] = field(default_factory=dict)

    @staticmethod
    def key(panorama_id: str, heading_deg: float, fov_deg: float) -> str:
        """Identité d'un recadrage : panorama, cap groupé, champ groupé.

        Le champ fait partie de la clé : le même cap à 70° et à 25° ne montre
        pas la même chose — mesuré sur le pilote, 0,396 contre 0,997.
        """
        return (
            f"{panorama_id}"
            f"::{int(heading_deg // HEADING_BUCKET_DEG)}"
            f"::{int(fov_deg // HEADING_BUCKET_DEG)}"
        )

    def get(self, panorama_id: str, heading_deg: float, fov_deg: float) -> dict | None:
        return self.entries.get(self.key(panorama_id, heading_deg, fov_deg))

    def record(
        self,
        panorama_id: str,
        heading_deg: float,
        fov_deg: float,
        *,
        score: float | None,
        verdict: str,
        facade_id: str | None = None,
        path: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.entries[self.key(panorama_id, heading_deg, fov_deg)] = {
            "panorama_id": panorama_id,
            "heading_deg": round(heading_deg, 1),
            "fov_deg": round(fov_deg, 1),
            "facade_id": facade_id,
            "score": round(score, 4) if score is not None else None,
            "verdict": verdict,
            "reason": reason,
            "path": path,
        }

    def as_payload(self) -> dict:
        return {
            "contract_version": 1,
            "verifications": sorted(
                self.entries.values(),
                key=lambda row: (row["panorama_id"], row["heading_deg"]),
            ),
        }


def load_register(workspace) -> VerificationRegister:  # noqa: ANN001
    """Relit le registre, ou en ouvre un vide."""
    payload = workspace.read_json(REGISTER) or {}
    register = VerificationRegister()
    for row in payload.get("verifications") or []:
        panorama = row.get("panorama_id")
        if not panorama:
            continue
        register.entries[
            VerificationRegister.key(
                panorama, row.get("heading_deg", 0.0), row.get("fov_deg", 70.0)
            )
        ] = row
    return register


def save_register(workspace, register: VerificationRegister) -> Path:  # noqa: ANN001
    return workspace.write_json(REGISTER, register.as_payload())


def apply_known(opportunities: list, register: VerificationRegister) -> tuple[int, int]:
    """Reporte sur les propositions ce que le registre sait déjà.

    Returns:
        `(connus, inconnus)`. Un recadrage inconnu garde
        `verified_prominence=None` — jamais vérifié n'est pas jamais bon.
    """
    known = unknown = 0
    for opportunity in opportunities:
        row = register.get(
            opportunity.panorama_id, opportunity.heading_deg, opportunity.fov_deg
        )
        if row is None or row.get("score") is None:
            unknown += 1
            continue
        opportunity.verified_prominence = float(row["score"])
        known += 1
    return known, unknown


def fetch_and_read(
    opportunities: list,
    register: VerificationRegister,
    *,
    cache_dir: Path,
    reader=None,  # noqa: ANN001
    fetcher=None,  # noqa: ANN001
    limit: int | None = None,
) -> tuple[int, int]:
    """Acquiert les recadrages inconnus, les lit, et inscrit le verdict.

    Chaque acquisition est une requête facturée : on ne redemande jamais un
    recadrage déjà jugé, et `limit` borne la campagne.

    Returns:
        `(vérifiés, ignorés)`.
    """
    from .subject_prominence import ProminenceReader

    pending = [
        o for o in opportunities
        if register.get(o.panorama_id, o.heading_deg, o.fov_deg) is None
    ]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return 0, 0

    if fetcher is None:
        fetcher = _street_view_fetcher
    if reader is None:
        reader = ProminenceReader()

    cache_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[tuple[object, Path]] = []
    skipped = 0
    for opportunity in pending:
        name = (
            f"{opportunity.facade_id}_{opportunity.panorama_id[:14]}"
            f"_{opportunity.heading_deg:.0f}h_{opportunity.fov_deg:.0f}f.jpg"
        )
        path = cache_dir / name
        if not path.is_file():
            try:
                payload = fetcher(opportunity)
            except Exception as exc:  # réseau, quota, panorama retiré
                register.record(
                    opportunity.panorama_id, opportunity.heading_deg,
                    opportunity.fov_deg, score=None, verdict="unfetched",
                    facade_id=opportunity.facade_id, reason=str(exc)[:120],
                )
                skipped += 1
                continue
            if not payload:
                register.record(
                    opportunity.panorama_id, opportunity.heading_deg,
                    opportunity.fov_deg, score=None, verdict="unfetched",
                    facade_id=opportunity.facade_id,
                    reason="réponse vide ou trop courte",
                )
                skipped += 1
                continue
            path.write_bytes(payload)
        fetched.append((opportunity, path))

    if not fetched:
        return 0, skipped

    readings = reader.read_many([(str(p), p) for _, p in fetched])
    for (opportunity, path), reading in zip(fetched, readings):
        register.record(
            opportunity.panorama_id,
            opportunity.heading_deg,
            opportunity.fov_deg,
            score=reading.score if reading.measured else None,
            verdict=reading.verdict,
            facade_id=opportunity.facade_id,
            path=str(path),
            reason=None if reading.measured else reading.reason,
        )
        if reading.measured:
            opportunity.verified_prominence = reading.score

    log.info(
        "vérification : %d recadrage(s) lu(s), %d ignoré(s)",
        len(fetched), skipped,
    )
    return len(fetched), skipped


def _street_view_fetcher(opportunity) -> bytes | None:  # noqa: ANN001
    """Acquiert un recadrage Street View. Seul point où une clé est jointe."""
    import requests

    from .config import secret

    response = requests.get(
        "https://maps.googleapis.com/maps/api/streetview",
        params={
            "size": "640x640",
            "pano": opportunity.panorama_id,
            "heading": f"{opportunity.heading_deg:.1f}",
            "fov": f"{opportunity.fov_deg:.1f}",
            "pitch": f"{opportunity.pitch_deg:.1f}",
            "key": secret("GOOGLE_MAPS_API_KEY"),
        },
        timeout=30,
    )
    response.raise_for_status()
    # Street View rend une vignette « pas d'image » de quelques kilo-octets
    # plutôt qu'une erreur : la taille est le seul signal disponible.
    if len(response.content) < 5000:
        return None
    return response.content


__all__ = [
    "HEADING_BUCKET_DEG",
    "REGISTER",
    "VerificationRegister",
    "apply_known",
    "fetch_and_read",
    "load_register",
    "save_register",
]
