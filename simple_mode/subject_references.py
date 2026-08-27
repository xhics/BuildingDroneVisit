"""Jeu de références multi-angles d'un bâtiment, pour la génération Ref2VA.

Un essai manuel a montré ce qui fonctionne, et corrigé deux de mes erreurs :
quatre photographies **réelles du même bâtiment sous des angles différents**,
déclarées ``fully_preserved``, suffisent à ce qu'un modèle tienne l'identité
d'un édifice pendant tout un mouvement de caméra — là où six images tirées
d'un même rendu de synthèse échouaient.

La différence tient à la **diversité angulaire** : des vues franchement
distinctes laissent deviner le volume, tandis que des images voisines d'une
même source n'apprennent rien de plus qu'une seule.

Ce module compose ce jeu automatiquement, pour n'importe quelle adresse, en
mêlant deux registres :

- **au sol**, des panoramas Street View répartis autour du bâtiment, qui en
  donnent les façades et les matériaux de près ;
- **en l'air**, une vue verticale ou oblique, sans laquelle le modèle ignore
  tout de l'implantation, des abords et du plan du site.

Les deux sont nécessaires : les vues au sol ne disent rien de ce qui entoure
le bâtiment, la vue aérienne ne dit rien de ses façades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .geo_utils import bearing_deg
from .street_view import Panorama, download_image, find_panoramas

#: Le checkpoint Ref2VA accepte 9 images ; on garde une place pour l'aérien.
MAX_REFERENCES = 9

#: Nombre de secteurs angulaires à couvrir autour du bâtiment. Six donne un
#: angle tous les 60°, assez pour cerner un volume sans saturer le jeu.
DEFAULT_SECTORS = 6


@dataclass
class SubjectReferences:
    """Les vues d'un même sujet, au sol et en l'air."""

    ground: list[Path] = field(default_factory=list)
    aerial: list[Path] = field(default_factory=list)
    #: Description architecturale, rédigée d'après les photos.
    description_fr: str = ""

    def all_images(self, limit: int = MAX_REFERENCES) -> list[Path]:
        """Références dans l'ordre où le prompt les désignera.

        L'aérienne vient en dernier : les vues de façade portent l'identité
        du bâtiment, qui prime, et le prompt les cite en premier.
        """
        images = [*self.ground, *self.aerial]
        return images[:limit]

    def describe_fr(self) -> str:
        return (
            f"{len(self.ground)} vue(s) au sol + {len(self.aerial)} vue(s) aérienne(s)"
        )


#: Distance de prise de vue recherchée. Une référence utile montre le
#: bâtiment **en entier** : collé à la façade, on ne photographie qu'un mur
#: ou une arcade, et le modèle reproduit alors ce point de vue piéton au lieu
#: du volume. Constaté en pratique — des panoramas à 14-30 m ont produit une
#: vue sous les arcades plutôt qu'une orbite.
IDEAL_DISTANCE_M = 160.0


def _spread_by_bearing(
    panoramas: list[Panorama], lat: float, lon: float, sectors: int
) -> list[Panorama]:
    """Un panorama par secteur angulaire autour du bâtiment.

    Deux critères, l'un après l'autre : couvrir le tour du bâtiment, et dans
    chaque secteur retenir la vue **la plus proche du recul idéal** — ni
    collée à la façade, ni si lointaine que le sujet s'y perde.
    """
    buckets: dict[int, Panorama] = {}
    # Seules les prises de vue officielles conviennent : les contributions
    # privées autour d'un hôtel sont presque toutes des intérieurs — salons,
    # couloirs, salles de réunion — et donnaient un jeu de références décrivant
    # une salle lambrissée au lieu de la façade.
    outdoor = [p for p in panoramas if p.is_official]
    for panorama in outdoor or panoramas:
        # Azimut du bâtiment vers le panorama : c'est lui qui dit de quel côté
        # se trouve la vue, alors que `heading_to_center_deg` dit où regarder.
        angle = bearing_deg(lat, lon, panorama.lat, panorama.lon)
        sector = int(angle // (360 / sectors)) % sectors
        current = buckets.get(sector)
        if current is None or abs(panorama.distance_m - IDEAL_DISTANCE_M) < abs(
            current.distance_m - IDEAL_DISTANCE_M
        ):
            buckets[sector] = panorama
    return [buckets[k] for k in sorted(buckets)]


def collect(
    lat: float,
    lon: float,
    api_key: str,
    out_dir: str | Path,
    *,
    sectors: int = DEFAULT_SECTORS,
    aerial_images: list[str | Path] | None = None,
    # Champ large : il faut que l'édifice tienne entier dans le cadre, sinon
    # la référence ne montre qu'un fragment de façade.
    fov: int = 100,
) -> SubjectReferences:
    """Compose le jeu de références multi-angles d'un bâtiment.

    ``aerial_images`` accepte les vues déjà produites — image satellite ou
    image extraite du rendu 3D. Une oblique issue du rendu vaut mieux qu'une
    verticale satellite : elle montre le bâtiment sous un angle proche de
    celui qu'aura la vidéo.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Le rayon par défaut de `find_panoramas` (90 m) ne laisse pas assez de
    # recul pour cadrer un grand bâtiment : on cherche plus loin.
    panoramas = find_panoramas(
        lat, lon, api_key,
        ring_radii_m=(80.0, 140.0, 200.0, 260.0),
        samples_per_ring=14,
        max_distance_m=300.0,
    )
    chosen = _spread_by_bearing(panoramas, lat, lon, sectors)

    ground: list[Path] = []
    for index, panorama in enumerate(chosen):
        destination = out_dir / f"angle_{index:02d}_{panorama.pano_id[:8]}.jpg"
        try:
            download_image(panorama, api_key, destination, fov=fov, size="1024x768")
        except Exception:  # noqa: BLE001 — un angle manquant n'invalide pas le jeu
            continue
        ground.append(destination)

    aerial = [Path(a) for a in (aerial_images or []) if Path(a).exists()]
    return SubjectReferences(ground=ground, aerial=aerial)


_DESCRIBE_PROMPT = (
    "Décris l'architecture de ce bâtiment pour qu'un modèle de génération "
    "d'images puisse le reproduire fidèlement : matériaux et leurs couleurs, "
    "forme des toitures, tourelles ou clochetons, rythme et forme des "
    "fenêtres, ornements caractéristiques, nombre approximatif d'étages. "
    "Deux à trois phrases, en français, sans préambule."
)


def describe_subject(
    references: SubjectReferences, openai_key: str, *, model: str = "gpt-4o-mini"
) -> str:
    """Rédige la description architecturale à partir des vues réunies.

    Le prompt qui a réussi décrivait précisément la brique rouge, les
    toitures de cuivre vert et les tourelles. Sans cette description, le
    modèle comble par un bâtiment vraisemblable mais différent : la nommer
    pèse autant que les images elles-mêmes.
    """
    import base64

    from openai import OpenAI

    images = references.all_images()[:4]
    if not images:
        return ""

    content: list[dict] = [{"type": "text", "text": _DESCRIBE_PROMPT}]
    for path in images:
        encoded = base64.b64encode(Path(path).read_bytes()).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )

    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}]
    )
    references.description_fr = (response.choices[0].message.content or "").strip()
    return references.description_fr


def build_ref2va_prompt(
    references: SubjectReferences,
    *,
    establishment_fr: str,
    move_fr: str,
    duration_s: int,
    time_of_day_fr: str = "",
    additions_fr: list[str] | None = None,
) -> str:
    """Prompt Ref2VA à six champs, en ``fully_preserved``.

    ``fully_preserved`` et non ``weak_reference`` : l'essai manuel montre que
    la fidélité déclarée est ce qui tient l'identité du bâtiment. J'avais
    choisi l'inverse pour éviter une copie trop littérale — mais ce défaut-là
    venait du mode ``flf2v``, qui verrouille deux images, pas d'un excès de
    fidélité aux références.
    """
    images = references.all_images()
    ground_count = min(len(references.ground), len(images))

    labels = []
    for index in range(len(images)):
        kind = (
            "vue au sol de la façade" if index < ground_count else "vue aérienne du site"
        )
        labels.append(f"<Picture {index + 1}> : {kind}")

    retention = "\n".join(f"- <Picture {i + 1}> : fully_preserved" for i in range(len(images)))
    additions = additions_fr or []
    additions_clause = (
        f"\nÀ faire vivre, absent des références : {'; '.join(additions)}."
        if additions
        else ""
    )
    light = time_of_day_fr or "lumière naturelle directionnelle"

    return (
        "subject_definitions:\n"
        f"<Subject 1> {establishment_fr}. {references.description_fr}\n"
        + "\n".join(labels)
        + "\n\nsummary:\n"
        "reference generation. Plan aérien continu autour du bâtiment, filmé au drone.\n\n"
        "retention_analysis:\n"
        f"{retention}\n"
        "<Subject 1> est fully_preserved sur tout le plan : matériaux, couleurs, "
        "toitures, tourelles et rythme des ouvertures restent ceux des références. "
        "La composition est partially_preserved — le cadrage évolue avec le "
        "mouvement, le bâtiment reste le sujet. Les vues au sol donnent les "
        "façades, la vue aérienne donne l'implantation et les abords.\n\n"
        "detailed_description:\n"
        f"[Shot 1] {move_fr} Le bâtiment reste le sujet principal pendant tout le "
        f"mouvement, qui dure {duration_s} secondes sans coupure. La caméra garde "
        f"une vitesse régulière.{additions_clause}\n"
        f"{light}. Rendu photographique : flou de mouvement naturel, profondeur de "
        "champ, matières et reflets réalistes.\n\n"
        "overall_soundscape:\nN/A\n\n"
        "non_diegetic_music:\nN/A\n"
    )


__all__ = [
    "DEFAULT_SECTORS",
    "MAX_REFERENCES",
    "SubjectReferences",
    "build_ref2va_prompt",
    "collect",
    "describe_subject",
]
