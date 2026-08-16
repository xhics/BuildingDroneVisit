"""Télécharger sous plafond, vérifier, puis seulement publier (collecte V2).

Le consentement porte sur 133 030 octets exacts. Sans borne, un serveur servant
davantage remplirait le disque avant qu'on s'en aperçoive : `bytes_written`
était inscrit **après** l'écriture, donc trop tard pour refuser.

Trois moments, dans cet ordre :

```text
plafonné    le flux s'arrête avant d'écrire le chunk qui dépasserait
vérifié     format décodé, dimensions, empreinte — sur le fichier obtenu
publié      transactionnellement, les six ou aucun
```

Le format se **décode**, il ne se déduit pas d'une extension : un serveur qui
sert une page d'erreur en `.jpg` passerait sinon pour une image. Les dimensions
se comparent au contrat du fournisseur, qui n'est pas le même partout — Street
View rend exactement la taille demandée, Mapillary une miniature dont seul le
plus grand côté est garanti.

Rien n'est publié tant que les six ne sont pas prêts. L'échec du sixième doit
laisser le manifeste inchangé et aucun fichier final des cinq premiers :
publier ce qui a réussi ferait croire à une acquisition partielle consentie.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger

log = get_logger("download")

#: Version du contrat de téléchargement. Elle entre dans la facette de
#: politique : changer ce que « télécharger » garantit doit périmer un plan
#: mesuré sous l'ancien contrat, non passer inaperçu.
DOWNLOAD_CONTRACT_VERSION = 1

#: Formats acceptés, par leur **contenu**. Une extension ne prouve rien.
ACCEPTED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG"})


class DownloadRefused(RuntimeError):
    """Rien n'a été publié, et le temporaire est supprimé."""


@dataclass
class DownloadOutcome:
    """Ce qu'un téléchargement a coûté et produit, étape par étape.

    Quatre comptes, non un seul : ce que le service annonce, ce qui est arrivé,
    ce qui attend en staging, ce qui est publié. Les fondre masquerait
    précisément le cas qu'on veut voir — un corps plus gros que déclaré, ou une
    publication interrompue.
    """

    candidate_id: str = ""
    request_digest: str = ""

    declared_bytes: int | None = None
    bytes_received: int = 0
    bytes_staged: int = 0
    bytes_published: int = 0

    checksum: str = ""
    image_format: str | None = None
    width: int | None = None
    height: int | None = None

    refused: str | None = None

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "request_digest": self.request_digest,
            "declared_bytes": self.declared_bytes,
            "bytes_received": self.bytes_received,
            "bytes_staged": self.bytes_staged,
            "bytes_published": self.bytes_published,
            "checksum": self.checksum,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "refused": self.refused,
        }


@dataclass
class Budget:
    """Le plafond consenti, et ce qu'il en reste.

    Deux plafonds simultanés : celui de chaque acquisition — sa taille déclarée
    — et celui du lot. Le second seul laisserait un fichier consommer la part
    des autres.
    """

    total_consented: int
    spent: int = 0

    #: Plafond individuel par acquisition, depuis le `HEAD`.
    per_request: dict = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return max(self.total_consented - self.spent, 0)

    def ceiling_for(self, candidate_id: str) -> int:
        """Le plus contraignant des deux plafonds."""
        individual = self.per_request.get(candidate_id)
        if individual is None:
            return self.remaining
        return min(individual, self.remaining)


def stream_to(
    response, target: Path, ceiling: int, declared: int | None,
    chunk_size: int = 1 << 16,
) -> int:  # noqa: ANN001
    """Écrit le corps, en s'arrêtant **avant** de dépasser le plafond.

    Le contrôle précède l'écriture : vérifier après aurait déjà mis les octets
    sur le disque, et « refuser » ne serait plus qu'un constat.
    """
    written = 0
    with target.open("wb") as handle:
        for chunk in response.iter_content(chunk_size):
            if not chunk:
                continue
            if written + len(chunk) > ceiling:
                raise DownloadRefused(
                    f"dépassement : {written + len(chunk)} octets dépasseraient "
                    f"le plafond de {ceiling}. Le flux est interrompu avant "
                    "écriture, et le fichier partiel supprimé."
                )
            handle.write(chunk)
            written += len(chunk)

    if declared is not None and written < declared:
        raise DownloadRefused(
            f"corps de {written} octets pour {declared} annoncés : le HEAD et "
            "le GET ne décrivent pas la même réponse"
        )
    return written


def inspect(path: Path) -> tuple[str, int, int, str]:
    """Format **décodé**, dimensions et empreinte du fichier obtenu.

    Une extension ne prouve rien : un serveur qui sert une page d'erreur en
    `.jpg` passerait pour une image, et l'asset publierait un fichier illisible.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            fmt, (width, height) = image.format, image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise DownloadRefused(
            f"contenu non décodable comme image : {str(exc)[:80]}"
        ) from exc

    if fmt not in ACCEPTED_FORMATS:
        raise DownloadRefused(
            f"format {fmt!r} hors des formats acceptés {sorted(ACCEPTED_FORMATS)}"
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return fmt, width, height, digest


def check_dimensions(source: str, request, width: int, height: int) -> None:  # noqa: ANN001
    """Confronte les dimensions au contrat **du fournisseur**.

    Ils ne promettent pas la même chose. Street View rend exactement la taille
    demandée ; Mapillary rend une miniature dont seul le plus grand côté est
    garanti, l'autre suivant le rapport de l'original. Exiger l'égalité stricte
    partout rejetterait des images conformes.
    """
    expected = getattr(request, "width_px", None), getattr(request, "height_px", None)

    if source == "street_view":
        if expected[0] and (width, height) != expected:
            raise DownloadRefused(
                f"dimensions {width}x{height} au lieu de "
                f"{expected[0]}x{expected[1]} demandées"
            )
        return

    if source == "mapillary":
        longest = expected[0]
        if longest and max(width, height) > longest:
            raise DownloadRefused(
                f"miniature {width}x{height} : le plus grand côté dépasse "
                f"{longest} demandés"
            )
        return

    # Source sans contrat déclaré : on ne rejette pas sur une règle inventée,
    # mais les dimensions restent publiées et confrontables.
    log.info(
        "%s : aucun contrat de dimensions déclaré, %dx%d reçus",
        source, width, height,
    )


def verify(path: Path, source: str, request) -> DownloadOutcome:  # noqa: ANN001
    """Tout ce qui doit être vrai avant qu'un fichier devienne un asset."""
    fmt, width, height, digest = inspect(path)
    check_dimensions(source, request, width, height)

    return DownloadOutcome(
        candidate_id=getattr(request, "candidate_id", ""),
        request_digest=getattr(request, "digest", ""),
        checksum=digest,
        image_format=fmt,
        width=width,
        height=height,
        bytes_staged=path.stat().st_size,
    )
