"""Provenance d'un panorama, et ce qu'elle dit de sa **pose**.

Toute l'approche de projection repose sur une hypothèse : la pose vient des
métadonnées — position, cap, champ — et non d'une reconstruction. L'hypothèse
tient, mais pas uniformément, et la nuance a un coût mesuré.

Street View mélange deux familles sous une même API :

- les captures **Google Car**, géoréférencées par la trajectoire du véhicule ;
- les **photosphères utilisateur**, dont la position est posée à la main sur
  une carte par leur auteur.

Rien ne les distingue dans l'asset : `attribution` porte partout « © Google
Street View ». Seul le champ `copyright` de l'endpoint metadata tranche, et il
n'était pas conservé jusqu'ici.

Mesuré sur le pilote, en comparant le cap déclaré au cap géométrique vers le
centroïde du bâtiment :

```text
famille                n   écart médian   écart max
Google Car             8          3,3°         7,6°
photosphère            2         16,5°        16,5°
```

Les deux seules vues du lot de validation dont l'incrustation du modèle tombait
à côté du bâtiment étaient les deux photosphères — identifiées visuellement
*avant* que leur provenance soit connue. Sur celle qui portait deux recadrages
distincts, réconcilier les deux vues demande de déplacer la caméra de ~17 m :
l'erreur est bien dans la **position**, non dans le cap, puisque deux caps
différents dévient du même côté.

Conséquence pratique : une pose Google Car est utilisable telle quelle ; une
pose photosphère doit être raffinée avant de servir d'ancrage. Ce module
n'affirme pas qu'elle est fausse — il dit qu'elle n'est **pas attestée**, ce
qui est une troisième valeur, distincte de « bonne » et de « mauvaise ».
"""

from __future__ import annotations

from dataclasses import dataclass

from .logging import get_logger

log = get_logger("pano-provenance")

#: Mention de `copyright` des captures véhiculées, seules géoréférencées par
#: la trajectoire. Toute autre mention désigne un contributeur.
SURVEY_COPYRIGHT = "© Google"

#: Écart cap-métadonnée / cap-géométrique au-delà duquel une pose est tenue
#: pour suspecte, quelle que soit sa famille. Calibré au-dessus du maximum
#: observé sur les captures véhiculées du pilote (7,6°), pour ne pas
#: disqualifier une pose saine.
BEARING_TOLERANCE_DEG = 10.0


@dataclass(frozen=True)
class PoseProvenance:
    """Ce qui atteste — ou non — la pose d'un panorama."""

    panorama_id: str
    copyright: str | None
    #: Écart mesuré au cap géométrique, quand la géométrie permet de le poser.
    bearing_error_deg: float | None = None
    #: Collecteur d'origine. Le `copyright` Google ne dit rien d'une image
    #: Mapillary : sans cette distinction, tout un corpus tiers se rangeait
    #: sous « provenance inconnue », ce qui est vrai mais illisible — on ne
    #: voyait plus les panoramas Street View réellement non résolus.
    source: str = "street_view"

    @property
    def surveyed(self) -> bool:
        """Position issue d'une trajectoire véhicule, non d'une saisie main."""
        return (self.copyright or "").strip() == SURVEY_COPYRIGHT

    @property
    def pose_status(self) -> str:
        """`attested` / `needs_refinement` / `unknown_provenance`.

        Jamais `bad` : une pose non attestée n'est pas démontrée fausse, et
        l'affirmer ferait jeter des vues que le raffinement PnP récupère.
        """
        if self.source != "street_view":
            # Hors Street View, `copyright` n'est pas le bon discriminant :
            # l'absence de mention n'est pas un défaut de résolution.
            return "foreign_source"
        if self.copyright is None:
            return "unknown_provenance"
        if not self.surveyed:
            return "needs_refinement"
        if (
            self.bearing_error_deg is not None
            and self.bearing_error_deg > BEARING_TOLERANCE_DEG
        ):
            # Une capture véhiculée peut dériver aussi — tunnels, canyons
            # urbains. La famille oriente, la mesure tranche.
            return "needs_refinement"
        return "attested"

    @property
    def usable_as_anchor(self) -> bool:
        """La pose peut-elle ancrer une projection sans raffinement ?"""
        return self.pose_status == "attested"

    def as_dict(self) -> dict:
        return {
            "panorama_id": self.panorama_id,
            "source": self.source,
            "copyright": self.copyright,
            "surveyed": self.surveyed,
            "pose_status": self.pose_status,
            "usable_as_anchor": self.usable_as_anchor,
            "bearing_error_deg": (
                round(self.bearing_error_deg, 1)
                if self.bearing_error_deg is not None else None
            ),
        }


def copyrights_from_cache() -> dict[str, str]:
    """Relit les `copyright` des métadonnées déjà payées.

    Les appels metadata sont gratuits mais mis en cache ; les relire ne coûte
    rien et évite de redemander ce qu'on a déjà. Un cache absent rend un
    dictionnaire vide — l'inconnu reste l'inconnu, non un défaut.
    """
    from .providers.cache import get_cache

    try:
        cache = get_cache()
        keys = [k for k in cache.iterkeys() if str(k).startswith("streetview-meta")]
    except Exception as exc:  # cache absent, verrouillé, corrompu
        log.warning("cache de métadonnées illisible (%s) : provenance inconnue", exc)
        return {}

    found: dict[str, str] = {}
    for key in keys:
        payload = cache.get(key)
        if not isinstance(payload, dict) or payload.get("status") != "OK":
            continue
        panorama_id = payload.get("pano_id")
        if panorama_id:
            found[panorama_id] = (payload.get("copyright") or "").strip()

    surveyed = sum(1 for c in found.values() if c == SURVEY_COPYRIGHT)
    log.info(
        "provenance : %d panorama(s), %d véhiculé(s), %d contributeur(s)",
        len(found), surveyed, len(found) - surveyed,
    )
    return found


def classify(
    panorama_ids,  # noqa: ANN001
    copyrights: dict[str, str] | None = None,
    bearing_errors: dict[str, float] | None = None,
    sources: dict[str, str] | None = None,
) -> dict[str, PoseProvenance]:
    """Attribue une provenance à chaque panorama.

    Un panorama absent du cache reçoit `copyright=None` — donc
    `unknown_provenance`, jamais `attested` par défaut. Supposer la meilleure
    famille en l'absence de preuve reproduirait l'erreur que ce module corrige.
    """
    copyrights = copyrights if copyrights is not None else copyrights_from_cache()
    bearing_errors = bearing_errors or {}
    sources = sources or {}
    return {
        panorama_id: PoseProvenance(
            panorama_id=panorama_id,
            copyright=copyrights.get(panorama_id),
            bearing_error_deg=bearing_errors.get(panorama_id),
            source=sources.get(panorama_id, "street_view"),
        )
        for panorama_id in panorama_ids
    }


def bearing_error(
    origin: tuple[float, float],
    centroid,  # noqa: ANN001
    heading_deg: float,
) -> float:
    """Écart entre un cap déclaré et le cap qui viserait le bâtiment.

    C'est un contrôle **faible** : il ne détecte qu'une erreur de cap, et une
    erreur de position ne s'y voit que par la rotation qu'elle induit. Sur le
    pilote, un décalage de 17 m s'y est traduit par 16,5°.
    """
    import math

    geometric = math.degrees(
        math.atan2(centroid.x - origin[0], centroid.y - origin[1])
    ) % 360.0
    return abs((heading_deg - geometric + 180.0) % 360.0 - 180.0)


def summarise(provenances: dict[str, PoseProvenance]) -> dict:
    """Bilan par statut, pour dire ce qui est ancrable sans raffinement."""
    counts: dict[str, int] = {}
    for provenance in provenances.values():
        counts[provenance.pose_status] = counts.get(provenance.pose_status, 0) + 1
    total = len(provenances) or 1
    return {
        "total": len(provenances),
        "by_status": counts,
        "attested_fraction": round(counts.get("attested", 0) / total, 3),
    }


__all__ = [
    "BEARING_TOLERANCE_DEG",
    "SURVEY_COPYRIGHT",
    "PoseProvenance",
    "bearing_error",
    "classify",
    "copyrights_from_cache",
    "summarise",
]
