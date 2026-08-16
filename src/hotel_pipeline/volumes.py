"""Volume d'une acquisition : mesuré, jamais estimé (collecte V2).

Le plan sépare depuis le début volume connu et volume inconnu, et refuse un
consentement qui porterait sur un total partiel. Restait à alimenter le connu :
aucune source ne l'annonçait, donc tout tombait en `unknown` et rien n'était
acquérable — un refus correct, mais définitif.

La taille se **mesure** par une requête d'en-tête. Ce n'est pas un
téléchargement : on demande au service ce qu'il rendrait, sans recevoir le
corps. La distinction compte, puisque le consentement porte précisément sur ce
qui va être téléchargé.

Ce qu'on refuse de faire, et qui serait plus simple :

```text
estimer depuis les dimensions   640×640 ne dit rien du taux de compression
prendre une taille moyenne      un plan « exact » reposerait sur une moyenne
compter zéro l'inconnu          un total exact et faux
```

Une taille non obtenue reste **inconnue**. Le plan le dira, et le consentement
restera refusé — ce qui est la bonne réponse, non un échec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("volumes")

#: Longueur au-delà de laquelle un en-tête est jugé invraisemblable pour une
#: image de rue. Une réponse de 2 Go signale une erreur de résolveur, non une
#: photographie, et l'accepter ferait consentir à n'importe quoi.
IMPLAUSIBLE_BYTES = 200 * 1024 * 1024


@dataclass
class VolumeReport:
    """Ce qui a été mesuré, ce qui reste inconnu, et pourquoi."""

    measured: dict[str, int] = field(default_factory=dict)
    unmeasured: dict[str, str] = field(default_factory=dict)

    @property
    def known_bytes(self) -> int:
        return sum(self.measured.values())

    def as_dict(self) -> dict:
        return {
            "measured": len(self.measured),
            "unmeasured": len(self.unmeasured),
            "known_bytes": self.known_bytes,
            "reasons": self.unmeasured,
            "note": (
                "tailles obtenues par requête d'en-tête, sans télécharger le "
                "corps. Une taille non obtenue reste inconnue : elle n'est ni "
                "estimée depuis les dimensions, ni comptée pour zéro"
            ),
        }


def measure(requests: list, prober=None) -> VolumeReport:  # noqa: ANN001
    """Mesure la taille de chaque **requête résolue**, sans télécharger.

    Les `ResolvedAcquisitionRequest` d'un plan, non les candidats d'un
    manifeste : mesurer `thumb_2048` quand le plan demande `thumb_256`
    annonçait un volume qui n'était pas celui du téléchargement, et le
    consentement portait sur un chiffre faux.

    `prober` reçoit une requête et rend une taille en octets, ou `None` si le
    service ne la déclare pas. L'injecter rend la mesure éprouvable sans clé ni
    réseau — et c'est la même couture que l'acquisition, pour la même raison :
    une résolution d'adresse cachée derrière un faux téléchargeur ne prouverait
    rien.
    """
    report = VolumeReport()
    probe = prober or content_length

    for candidate in requests:
        try:
            size = probe(candidate)
        except (OSError, RuntimeError, ValueError) as exc:
            report.unmeasured[candidate.candidate_id] = str(exc)[:120]
            continue

        if size is None:
            report.unmeasured[candidate.candidate_id] = (
                "le service ne déclare pas de longueur"
            )
            continue
        if size <= 0 or size > IMPLAUSIBLE_BYTES:
            report.unmeasured[candidate.candidate_id] = (
                f"longueur invraisemblable pour une image : {size} octets"
            )
            continue

        report.measured[candidate.candidate_id] = size

    log.info(
        "volumes : %d mesuré(s) pour %d octets, %d inconnu(s)",
        len(report.measured), report.known_bytes, len(report.unmeasured),
    )
    return report


def content_length(request) -> int | None:  # noqa: ANN001
    """Longueur annoncée par le service, sans recevoir le corps.

    Une requête `HEAD` demande les en-têtes seuls. Certains services n'y
    répondent pas, ou omettent `Content-Length` : dans les deux cas la taille
    reste inconnue, et la deviner serait pire que l'ignorer.

    Le `HEAD` porte sur **ce qui sera téléchargé** : le `request_spec` de la
    requête résolue, résolution fournisseur comprise. Mesurer celui du candidat
    interrogeait une autre image que celle du plan.
    """
    import requests as http

    from .acquire import resolve_url
    from .providers.cache import ensure_online

    ensure_online("mesure de volume")
    url = resolve_url(request.source, request.request_spec)

    response = http.head(url, timeout=30, allow_redirects=True)
    response.raise_for_status()

    declared = response.headers.get("Content-Length")
    if declared is None:
        return None
    try:
        return int(declared)
    except ValueError:
        return None
