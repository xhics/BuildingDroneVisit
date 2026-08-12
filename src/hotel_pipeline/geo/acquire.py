"""Acquisition d'une tuile, sous protocole strict (Lot 1B §9).

Un téléchargement partiel qui porte le nom du fichier final est pire qu'un
échec : tout ce qui suit le croira valide. Le fichier n'est donc nommé qu'après
avoir satisfait **toutes** les vérifications.

Neuf règles, dans cet ordre :

1. écrire dans un `.part`, au même endroit que la cible — un renommage entre
   systèmes de fichiers n'est pas atomique ;
2. conserver la validation TLS, suivre les redirections, capturer
   `Content-Length`, `ETag` et `Last-Modified` ;
3. écrire en flux, sans charger le fichier en mémoire ;
4. exiger exactement la taille annoncée ;
5. vérifier la signature `LASF` ;
6. calculer l'empreinte SHA-256 ;
7. renommer seulement après tout cela ;
8. enregistrer URL, taille, empreinte, date et en-têtes ;
9. en cas d'échec, produire un rapport — et **aucune** source citable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from ..logging import get_logger
from ..providers.cache import ensure_online
from ..schemas.site import GeoSourceProvenance

log = get_logger("acquire")

CHUNK_SIZE = 1 << 20
TIMEOUT = 300

#: Signature d'un fichier LAS ou LAZ, quatre premiers octets de l'en-tête.
LAS_SIGNATURE = b"LASF"


class AcquisitionError(RuntimeError):
    """Échec d'acquisition. Aucune source citable n'en découle."""


@dataclass
class AcquisitionResult:
    url: str
    path: Path | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    retrieved_at: datetime | None = None
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not None and self.error is None

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "path": str(self.path) if self.path else None,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "http_headers": self.headers,
            "error": self.error,
            "succeeded": self.succeeded,
        }


def download_tile(
    url: str,
    destination: Path,
    expected_bytes: int,
    expect_signature: bytes | None = LAS_SIGNATURE,
) -> AcquisitionResult:
    """Télécharge une tuile et ne la nomme qu'une fois toutes ses garanties tenues."""
    ensure_online(f"téléchargement {url}")
    result = AcquisitionResult(url=url)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Le `.part` vit à côté de la cible : `replace()` n'est atomique qu'au sein
    # d'un même système de fichiers.
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    written = 0

    try:
        with requests.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True) as response:
            response.raise_for_status()
            result.headers = {
                name: response.headers[name]
                for name in ("Content-Length", "ETag", "Last-Modified", "Content-Type")
                if name in response.headers
            }

            announced = response.headers.get("Content-Length")
            if announced and announced.isdigit() and int(announced) != expected_bytes:
                raise AcquisitionError(
                    f"taille annoncée {announced} ≠ taille autorisée {expected_bytes}"
                )

            with partial.open("wb") as handle:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)

        # 4. taille exacte
        if written != expected_bytes:
            raise AcquisitionError(
                f"{written} octets reçus, {expected_bytes} attendus — "
                "téléchargement incomplet ou fichier modifié"
            )

        # 5. signature
        if expect_signature:
            with partial.open("rb") as handle:
                signature = handle.read(len(expect_signature))
            if signature != expect_signature:
                raise AcquisitionError(
                    f"signature {signature!r} ≠ {expect_signature!r} — "
                    "le contenu n'est pas un fichier LAS/LAZ"
                )

        result.size_bytes = written
        result.sha256 = digest.hexdigest()
        result.retrieved_at = datetime.now(timezone.utc)

        # 7. nommage final, une fois seulement
        partial.replace(destination)
        result.path = destination

        log.info(
            "tuile acquise : %s, %d octets, sha256 %s",
            destination.name,
            written,
            result.sha256[:16],
        )
        return result

    except (AcquisitionError, requests.RequestException, OSError) as exc:
        partial.unlink(missing_ok=True)
        result.error = str(exc)
        log.error("acquisition échouée (%s) : %s", url.rsplit("/", 1)[-1], exc)
        return result


def provenance_from(
    result: AcquisitionResult, tile, dataset: str = "Données LiDAR du Québec"
) -> GeoSourceProvenance:  # noqa: ANN001
    """Provenance citable d'une acquisition réussie.

    Refuse de décrire un échec : une source citable dont le fichier n'existe
    pas rendrait tout objet dérivé invérifiable.
    """
    if not result.succeeded:
        raise AcquisitionError(
            "aucune provenance citable pour une acquisition échouée : "
            f"{result.error}"
        )

    return GeoSourceProvenance(
        source_id=f"lidar-quebec-{tile.tile_id}",
        dataset=dataset,
        vintage=str(tile.acquired_on.year) if tile.acquired_on else None,
        tile_id=tile.tile_id,
        crs_horizontal=tile.crs_horizontal,
        crs_vertical=tile.crs_vertical,
        point_density_per_m2=tile.point_density_per_m2,
        carries_elevation=True,
        file_digest=result.sha256,
        licence=tile.licence,
        retrieved_at=result.retrieved_at,
        notes=f"classification {tile.classification}, {tile.file_format}",
    )
