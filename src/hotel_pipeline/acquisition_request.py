"""Ce qui sera **réellement** demandé au fournisseur (collecte V2).

Le plan parle un vocabulaire — `256`, `2048` — et chaque fournisseur le sien :
`thumb_256`, `640x640`. Entre les deux, personne ne traduisait. Le plan
annonçait donc `256` tandis que `request_spec` conservait `thumb_2048`, et la
provenance inscrivait la résolution **planifiée** sans que rien ne garantisse
qu'elle fût celle du fichier obtenu.

Trois conséquences, toutes silencieuses :

```text
mesurer un fichier      HEAD sur thumb_2048
en télécharger un autre GET sur thumb_2048 alors que 256 était voulu
publier une provenance  « resolution: 256 » sur une image de 2048
```

Cet objet ferme la brèche : il est construit une fois, à partir du couple
`(candidat, acquisition planifiée)`, et c'est **lui** que consomment la
liaison, le `HEAD`, le téléchargement et la provenance. Aucun d'eux ne
reconstruit sa propre version de la question.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("acquisition-request")


class RequestUnresolvable(RuntimeError):
    """Le fournisseur ne sait pas servir ce que le plan demande."""


#: Traduction du vocabulaire du plan vers celui de chaque source. Une table
#: déclarée, non une heuristique de nom : deviner marcherait pour `thumb_256`
#: et échouerait au premier fournisseur nommant autrement.
PROVIDER_RESOLUTIONS: dict[str, dict[str, str]] = {
    "mapillary": {
        "256": "thumb_256",
        "2048": "thumb_2048",
        # « Le mieux que la source sache faire », sans exiger un nombre : ce
        # que 2048 vaut chez l'un ne vaut pas chez l'autre.
        "full_available": "thumb_2048",
    },
    "street_view": {
        "256": "256x256",
        # Pas de « 2048 » : l'API Static plafonne à 640 px, et le premier
        # passage réel l'a montré — 640x640 servi pour 2048x2048 demandés. La
        # traduire en 640 aurait tronqué la demande en silence ; l'omettre fait
        # refuser **avant** tout appel.
        "full_available": "640x640",
    },
}

#: Capacité maximale **vérifiée** de chaque source, en pixels du plus grand
#: côté. Constatée sur les réponses réelles, non lue dans une documentation :
#: c'est ce que le fournisseur a effectivement servi.
PROVIDER_MAX_PIXELS: dict[str, int] = {
    "street_view": 640,
    "mapillary": 2048,
}


@dataclass(frozen=True)
class ResolvedAcquisitionRequest:
    """Ce qui sera demandé, dans les termes du fournisseur.

    Immuable : une requête qu'on peut modifier après consentement n'est plus
    celle qui a été consentie.
    """

    candidate_id: str
    source: str

    #: Ce que le **plan** demande — `256`, `2048`.
    semantic_resolution: str

    #: Ce que le **fournisseur** comprend — `thumb_256`, `640x640`.
    provider_resolution: str

    #: Paramètres assainis : de quoi reconstruire l'adresse, et rien qui
    #: ressemble à un secret ou à une URL.
    request_spec: dict = field(default_factory=dict)

    width_px: int | None = None
    height_px: int | None = None

    @property
    def digest(self) -> str:
        """Empreinte de **cette** requête, résolution fournisseur comprise.

        Changer la résolution change l'empreinte : c'est ce qui permet au
        consentement de porter sur une demande précise, et non sur un candidat
        dont on redéfinirait le contenu ensuite.
        """
        payload = json.dumps(
            {
                "candidate_id": self.candidate_id,
                "source": self.source,

                "request_spec": self.request_spec,
            },
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "semantic_resolution": self.semantic_resolution,
            "provider_resolution": self.provider_resolution,
            "request_spec": self.request_spec,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "digest": self.digest,
        }


def _dimensions(provider_resolution: str) -> tuple[int | None, int | None]:
    """Dimensions déductibles du nom de résolution, quand il en porte."""
    if "x" in provider_resolution:
        parts = provider_resolution.split("x", 1)
        if all(part.isdigit() for part in parts):
            return int(parts[0]), int(parts[1])
    tail = provider_resolution.rsplit("_", 1)[-1]
    if tail.isdigit():
        side = int(tail)
        return side, side
    return None, None


def resolve(candidate, acquisition) -> ResolvedAcquisitionRequest:  # noqa: ANN001
    """Traduit une acquisition planifiée dans les termes de sa source.

    Refuse plutôt que de deviner : servir une autre résolution que celle
    planifiée téléchargerait un fichier dont personne n'a mesuré la taille, et
    la provenance décrirait une image qui n'est pas celle du disque.
    """
    source = candidate.source
    table = PROVIDER_RESOLUTIONS.get(source)
    if table is None:
        raise RequestUnresolvable(
            f"source {source!r} sans table de résolutions : ce que le plan "
            f"demande ne se traduit pas. Connues : {sorted(PROVIDER_RESOLUTIONS)}"
        )

    wanted = acquisition.resolution

    # Une exigence chiffrée au-delà de ce que la source sait servir rend cette
    # source **inéligible** pour ce besoin : la rabattre silencieusement
    # livrerait autre chose que ce qui a été planifié, et le premier passage
    # réel l'a fait — 640x640 pour 2048x2048.
    ceiling = PROVIDER_MAX_PIXELS.get(source)
    if ceiling is not None and wanted.isdigit() and int(wanted) > ceiling:
        raise RequestUnresolvable(
            f"{candidate.candidate_id} : {source} ne dépasse pas {ceiling} px "
            f"(capacité vérifiée) ; {wanted} demandés. Employez "
            "« full_available » si le besoin n'exige pas ce nombre, ou "
            "cherchez cette vue ailleurs."
        )

    provider_resolution = table.get(wanted)
    if provider_resolution is None:
        # Le vocabulaire du fournisseur peut être employé directement — un
        # cadrage Street View nomme sa taille — mais seulement s'il figure
        # parmi ce que le candidat déclare savoir servir.
        if wanted in (candidate.available_resolutions or []):
            provider_resolution = wanted
        else:
            raise RequestUnresolvable(
                f"{candidate.candidate_id} : {source} ne sait pas servir "
                f"{wanted!r} ; traductions connues : {sorted(table)}"
            )

    if (
        candidate.available_resolutions
        and provider_resolution not in candidate.available_resolutions
    ):
        raise RequestUnresolvable(
            f"{candidate.candidate_id} : {provider_resolution!r} absent des "
            f"résolutions déclarées {candidate.available_resolutions}"
        )

    # La résolution **fournisseur** entre dans les paramètres : sans elle, le
    # téléchargement retombait sur celle qu'y avait laissée la découverte.
    spec = dict(candidate.request_spec or {})
    spec["resolution"] = provider_resolution
    if source == "street_view":
        spec["size"] = provider_resolution

    width, height = _dimensions(provider_resolution)
    return ResolvedAcquisitionRequest(
        candidate_id=candidate.candidate_id,
        source=source,
        semantic_resolution=wanted,
        provider_resolution=provider_resolution,
        request_spec=spec,
        width_px=width,
        height_px=height,
    )


def resolve_all(candidates: dict, acquisitions: list) -> dict:  # noqa: ANN001
    """Résout toutes les acquisitions d'un plan, ou refuse le lot entier.

    Rien de partiel : un plan dont une acquisition ne se traduit pas annoncerait
    un volume qu'il ne saurait pas obtenir.
    """
    resolved: dict[str, ResolvedAcquisitionRequest] = {}
    problems: list[str] = []

    for acquisition in acquisitions:
        candidate = candidates.get(acquisition.candidate_id)
        if candidate is None:
            # Un candidat absent se **rapporte** acquisition par acquisition :
            # il n'y a rien à traduire, mais rien non plus qui empêche les
            # autres. Refuser le lot entier ferait d'une vue disparue un
            # blocage général.
            continue
        try:
            resolved[acquisition.candidate_id] = resolve(candidate, acquisition)
        except RequestUnresolvable as exc:
            problems.append(str(exc))

    if problems:
        raise RequestUnresolvable(" ; ".join(problems))
    return resolved
