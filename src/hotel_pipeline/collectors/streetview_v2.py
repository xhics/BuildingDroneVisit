"""Street View en candidats V2 : cadrages multiples, points de vue réels.

Une image Mapillary existe telle quelle. Un panorama Street View, non : c'est
une sphère, et l'image n'existe qu'au moment où on la cadre. Deux conséquences
que tout le reste du lot a préparées :

```text
identité      deux cadrages d'un panorama = deux acquisitions
point de vue  deux cadrages d'un panorama = un seul point de vue
```

Les confondre coûte cher dans les deux sens. Nommer les deux cadrages pareil en
écraserait un — le corpus perdrait une vue sans que rien ne le signale. Les
compter comme deux observations indépendantes ferait croire un besoin servi par
une parallaxe qui n'existe pas, et aucun SfM n'en tirerait de profondeur.

L'étape 2 avait chiffré le défaut d'origine : huit caps sur un panorama
donnaient huit fichiers, huit photographies, **un** point de vue.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logging import get_logger
from ..schemas.acquisition import (
    CaptureCandidate,
    IdentityStrategy,
    capture_identity,
)

log = get_logger("streetview-v2")

#: Cadrages demandés par défaut autour d'un panorama. Chacun produit une
#: acquisition distincte ; tous partagent le même point de vue.
DEFAULT_FOV_DEG = 80.0
DEFAULT_PITCH_DEG = 0.0
DEFAULT_SIZE = "640x640"


@dataclass(frozen=True)
class Framing:
    """Un cadrage demandé à la sphère : c'est lui qui fait l'image."""

    heading_deg: float
    fov_deg: float = DEFAULT_FOV_DEG
    pitch_deg: float = DEFAULT_PITCH_DEG
    size: str = DEFAULT_SIZE

    def as_request_spec(self, pano_id: str) -> dict[str, str]:
        """De quoi reconstruire l'adresse, et rien de plus.

        Aucune URL n'est conservée : celle de Street View porte la clé d'API
        dès qu'elle est signée, et le manifeste est versionné.
        """
        return {
            "pano_id": pano_id,
            "heading_deg": f"{self.heading_deg:.1f}",
            "fov_deg": f"{self.fov_deg:.1f}",
            "pitch_deg": f"{self.pitch_deg:.1f}",
            "size": self.size,
        }


def candidate_from(panorama, framing: Framing, distance_m: float | None = None) -> CaptureCandidate:  # noqa: ANN001
    """Un cadrage d'un panorama, en candidat.

    Le cap est **demandé**, non observé : nous le dirigeons vers l'empreinte.
    Le porter comme `original_heading_deg` en ferait une mesure du fournisseur,
    et la géométrie de candidat s'y fierait pour juger un secteur.
    """
    pano_id = panorama.pano_id

    return CaptureCandidate(
        candidate_id=capture_identity(
            "street_view", pano_id,
            heading_deg=framing.heading_deg, fov_deg=framing.fov_deg,
            pitch_deg=framing.pitch_deg, size=framing.size,
        ),
        source="street_view",
        provider_id=pano_id,
        panorama_id=pano_id,
        camera_lat=panorama.lat,
        camera_lon=panorama.lon,
        # Le cadrage est une intention : il vit dans les champs `requested_*`.
        requested_heading_deg=framing.heading_deg % 360.0,
        requested_fov_deg=framing.fov_deg,
        requested_pitch_deg=framing.pitch_deg,
        heading_is_measured=False,
        captured_at=_captured_at(panorama.date),
        advertised_width=_dimension(framing.size, 0),
        advertised_height=_dimension(framing.size, 1),
        # Street View rend la taille demandée : ce n'est pas une liste fermée
        # mais un plafond. Déclarer le seul cadrage nominal faisait refuser à
        # l'exécution tout plan demandant un aperçu.
        available_resolutions=sorted({framing.size, *PLAN_RESOLUTIONS}),
        request_spec=framing.as_request_spec(pano_id),
        # Street View publie des vues de voirie : la preuve d'extériorité vient
        # de la source elle-même, non d'une supposition sur le contenu.
        outdoor_evidence="Google Street View — imagerie de voirie extérieure",
    )


def candidates_from(panoramas: list, framings: list[Framing]) -> list[CaptureCandidate]:
    """Tous les cadrages de tous les panoramas.

    Les panoramas sont supposés **déjà dédupliqués** par identifiant : le
    collecteur le fait, et le refaire ici masquerait un doublon d'index.
    """
    produced = [
        candidate_from(panorama, framing)
        for panorama in panoramas
        for framing in framings
    ]
    log.info(
        "Street View : %d candidat(s) pour %d panorama(s) et %d cadrage(s)",
        len(produced), len(panoramas), len(framings),
    )
    return produced


def discover_panoramas(
    corridors: list[dict],
    spacing_m: float,
    snap_radius_m: int,
    probe=None,  # noqa: ANN001 — injecté pour éprouver sans clé ni réseau
) -> tuple[list, dict[str, str]]:
    """Cherche les panoramas le long des corridors, sans télécharger d'image.

    L'endpoint de métadonnées est gratuit et dit s'il existe un panorama, où et
    à quelle date : toute la sélection s'y fait. L'endpoint image, facturé,
    n'intervient qu'à l'acquisition — et seulement pour ce qu'un plan consenti
    porte.

    Les panoramas sont dédupliqués par identifiant : deux points d'échantillon
    voisins rendent le même, et l'empiler gonflerait le volume annoncé au plan.
    """
    from .streetview import panorama_at, sample_road_network

    seen: dict[str, object] = {}
    skipped: dict[str, str] = {}

    points = sample_road_network(corridors, spacing_m=spacing_m)
    if not points:
        skipped["corridors"] = (
            "aucun point d'échantillon : les corridors n'ont pas de géométrie"
        )
        return [], skipped

    lookup = probe or _default_probe(snap_radius_m)

    for lat, lon in points:
        try:
            panorama = lookup(lat, lon)
        except (OSError, RuntimeError, ValueError) as exc:
            skipped[f"{lat:.5f},{lon:.5f}"] = str(exc)[:100]
            continue
        if panorama is None or not panorama.pano_id:
            continue
        seen.setdefault(panorama.pano_id, panorama)

    log.info(
        "Street View : %d panorama(s) distinct(s) sur %d point(s) de corridor",
        len(seen), len(points),
    )
    return list(seen.values()), skipped


def _default_probe(snap_radius_m: int):  # noqa: ANN202
    from ..config import secret
    from .streetview import panorama_at

    def probe(lat: float, lon: float):  # noqa: ANN202
        return panorama_at(
            lat, lon, secret("GOOGLE_MAPS_API_KEY"), radius_m=snap_radius_m
        )

    return probe


def framings_for_targets(panorama, targets: list) -> list[Framing]:  # noqa: ANN001
    """Un cadrage par cible visée depuis ce panorama.

    Le cap n'est pas choisi au hasard ni tourné en huit : il est **dirigé** vers
    ce qu'un besoin demande. Balayer l'horizon produirait des acquisitions que
    rien ne réclame, et le consentement porterait sur elles.
    """
    from ..visibility import bearing_deg

    framings: dict[float, Framing] = {}
    for target in targets:
        centroid = target.shape.centroid
        heading = bearing_deg(panorama.lat, panorama.lon, centroid.y, centroid.x)
        framings.setdefault(round(heading, 1), Framing(heading_deg=heading))
    return list(framings.values())


def framings_towards(bearing_deg: float, extra_offsets: tuple[float, ...] = ()) -> list[Framing]:
    """Un cadrage vers la cible, et ceux qu'on veut en plus.

    Les décalages produisent des **acquisitions** supplémentaires, jamais des
    points de vue : ils élargissent la couverture angulaire depuis une même
    position, ce qui sert le contexte mais n'apporte aucune parallaxe.
    """
    return [
        Framing(heading_deg=(bearing_deg + offset) % 360.0)
        for offset in (0.0, *extra_offsets)
    ]


def resolve_url(request_spec: dict[str, str], *, signed: bool = True) -> str:
    """Reconstruit l'adresse d'un cadrage, au moment de la requête.

    C'est le seul endroit où elle existe, et la clé y est jointe **ici** :
    « l'appelant l'ajoutera » était une intention que ni la mesure de volume ni
    le téléchargement n'honoraient, et Street View restait inutilisable en réel.

    `signed=False` rend l'adresse sans clé, pour les journaux et les rapports :
    une URL signée dans un fichier versionné y mettrait le secret.
    """
    from .streetview import IMAGE_URL

    missing = sorted(
        key for key in ("pano_id", "heading_deg", "fov_deg", "pitch_deg", "size")
        if not request_spec.get(key)
    )
    if missing:
        raise ValueError(
            f"cadrage incomplet : {missing} — une sphère ne devient une image "
            "qu'une fois entièrement cadrée"
        )

    url = (
        f"{IMAGE_URL}?size={request_spec['size']}"
        f"&pano={request_spec['pano_id']}"
        f"&heading={request_spec['heading_deg']}"
        f"&fov={request_spec['fov_deg']}"
        f"&pitch={request_spec['pitch_deg']}"
    )
    if not signed:
        return url

    from ..config import secret

    return f"{url}&key={secret('GOOGLE_MAPS_API_KEY')}"


#: Vocabulaire des résolutions que le **plan** manipule. Chaque collecteur les
#: traduit dans le sien au moment d'acquérir : sans cette déclaration, un plan
#: parlant « 256 » et un candidat parlant « 640x640 » ne se reconnaissaient
#: pas, et les neuf acquisitions du brouillon auraient été refusées.
PLAN_RESOLUTIONS: frozenset[str] = frozenset({"256", "2048"})


def _captured_at(date: str | None):  # noqa: ANN201
    """Street View publie « AAAA-MM » : on n'en fait pas un instant précis."""
    from datetime import datetime, timezone

    if not date:
        return None
    parts = date.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError):
        return None
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _dimension(size: str, index: int) -> int | None:
    """Dimension **demandée**, à ne pas confondre avec celle du fichier.

    Elle est annoncée : le service peut rendre moins, et c'est la mesure sur
    le fichier acquis qui fera foi.
    """
    try:
        return int(size.split("x")[index])
    except (ValueError, IndexError):
        return None
