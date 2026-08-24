"""Silhouettes lues dans les images au sol, là où le LiDAR ne voit rien.

Un relevé aérien décrit une enveloppe supérieure : il donne la hauteur d'une
couronne, jamais son profil vu de la rue. Or c'est ce profil que montre un plan
d'établissement — un conifère effilé et un érable étalé occupent le même disque
vu du ciel et n'ont pas la même silhouette depuis l'entrée.

Les photographies portent cette information. Le module la lit avec un modèle à
vocabulaire ouvert : les catégories sont décrites en langage naturel, ce qui
évite d'entraîner un classifieur et permet d'ajouter une nature en écrivant une
phrase.

Un modèle fermé a été essayé et écarté : DeepLabv3 pré-entraîné sur VOC ne
connaît ni arbre, ni bâtiment, ni ciel — ses vingt et une classes rendaient
cent pour cent de « background » sur nos images de rue.
"""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-silhouette")

#: Natures reconnues, décrites en langage naturel. Chaque intitulé décrit une
#: scène concrète : un modèle contrastif n'encode pas la négation, et une
#: catégorie définie par ce qu'elle n'est pas l'emporterait à tort.
CLASS_PROMPTS: dict[str, str] = {
    "vegetation": "green tree foliage, leaves and branches",
    "conifere": "a tall narrow evergreen conifer tree",
    "batiment": "a brick or concrete building facade with windows",
    "ciel": "plain empty sky",
    "sol": "asphalt road, parking lot or paved ground",
    "herbe": "green lawn or grass",
    "neige": "snow covered ground",
    "mobilier": "a street lamp post, sign pole or traffic sign",
    "vehicule": "a parked car, van or bus",
}

#: Côté d'une tuile analysée, en pixels. Trente-deux pixels donnent une carte
#: assez fine pour suivre le contour d'une couronne sans exploser le nombre
#: d'inférences.
TILE_PX = 32

#: En deçà, l'attribution est trop incertaine pour être publiée : la tuile
#: sort en `indetermine` plutôt que d'être forcée dans la classe la mieux
#: notée. Réglé par mesure sur le corpus : la marge médiane vaut 0,024 à cette
#: résolution, et un seuil de 0,02 laissait quarante-quatre pour cent des
#: tuiles indécises — soit davantage que ce qu'il tranchait.
MIN_MARGIN = 0.008

#: Natures qui comptent comme végétation dans un profil vertical.
VEGETATION_CLASSES = frozenset({"vegetation", "conifere", "herbe"})


@dataclass
class SilhouetteMap:
    """Carte de natures d'une image, tuile par tuile."""

    asset_id: str
    labels: np.ndarray
    classes: list[str]
    tile_px: int
    bearing_deg: float | None = None

    def fraction(self, name: str) -> float:
        if name not in self.classes:
            return 0.0
        index = self.classes.index(name)
        return float((self.labels == index).sum() / self.labels.size)

    def vegetation_profile(self) -> np.ndarray:
        """Part de végétation par bande horizontale, du haut vers le bas.

        C'est ce profil que le LiDAR ne peut pas donner : il dit *comment* la
        masse végétale se répartit en hauteur, donc si l'objet est effilé ou
        étalé, dégagé au pied ou touffu jusqu'au sol.
        """
        indices = [
            self.classes.index(name)
            for name in VEGETATION_CLASSES
            if name in self.classes
        ]
        if not indices:
            return np.zeros(self.labels.shape[0])
        mask = np.isin(self.labels, indices)
        return mask.mean(axis=1)

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "bearing_deg": None if self.bearing_deg is None else round(self.bearing_deg, 1),
            "tile_px": self.tile_px,
            "shape": list(self.labels.shape),
            "fractions": {
                name: round(self.fraction(name), 4)
                for name in self.classes
                if self.fraction(name) > 0.005
            },
            "vegetation_profile": [
                round(float(v), 3) for v in self.vegetation_profile()
            ],
        }


@dataclass
class SilhouetteReading:
    """Ce que l'ensemble des vues établit sur un site."""

    hotel_id: str
    maps: list[SilhouetteMap] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def mean_fraction(self, name: str) -> float:
        if not self.maps:
            return 0.0
        return float(np.mean([m.fraction(name) for m in self.maps]))

    def as_dict(self) -> dict:
        names = sorted({c for m in self.maps for c in m.classes})
        return {
            "hotel_id": self.hotel_id,
            "views": len(self.maps),
            "mean_fractions": {
                name: round(self.mean_fraction(name), 4)
                for name in names
                if self.mean_fraction(name) > 0.005
            },
            "provenance": self.provenance,
            "maps": [m.as_dict() for m in self.maps],
            "caveats": [
                "les natures sont attribuées par ressemblance à des "
                "descriptions en langage naturel, non par un modèle entraîné "
                "sur ce site : une tuile ambiguë sort `indetermine`",
                "une silhouette lue depuis la rue ne dit rien de la face "
                "opposée de l'objet : elle complète le relevé aérien, elle ne "
                "le remplace pas",
                "les vues sont datées : un profil relevé en hiver ne décrit "
                "pas le même feuillage qu'en été",
            ],
        }


#: Racine du cache de lecture. Les silhouettes ne dépendent que de l'image et
#: des réglages de lecture : rien dans la scène ne peut les changer, et les
#: recalculer à chaque rendu coûtait plusieurs minutes pour un résultat
#: identique au précédent.
CACHE_DIRNAME = "silhouette_cache"


def _cache_key(image_path: Path, tile_px: int, model: str) -> str:
    """Empreinte de ce qui détermine la lecture, et de rien d'autre.

    Le contenu de l'image, non son chemin : une photo réacquise sous un autre
    nom ne mérite pas d'être relue. Le modèle et la taille de tuile en font
    partie — les changer change le résultat, et un cache qui l'ignorerait
    servirait une lecture périmée.
    """
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    # Le lissage fait partie de la lecture : le changer périme le cache.
    digest.update(
        f"|{tile_px}|{model}|{sorted(CLASS_PROMPTS)}"
        f"|{NEIGHBOUR_WEIGHT}|{SMOOTHING_PASSES}".encode()
    )
    return digest.hexdigest()[:32]


def _cache_load(path: Path, asset_id: str, bearing_deg: float | None):
    """Relit une lecture mise en cache, ou rend `None` si rien n'est réutilisable."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SilhouetteMap(
            asset_id=asset_id,
            labels=np.asarray(payload["labels"], dtype=int),
            classes=list(payload["classes"]),
            tile_px=int(payload["tile_px"]),
            bearing_deg=bearing_deg,
        )
    except (OSError, ValueError, KeyError) as exc:
        # Un cache illisible n'est pas une panne : on relit l'image.
        log.info("cache de silhouette ignoré (%s) : %s", exc, path.name)
        return None


def _cache_store(path: Path, found: "SilhouetteMap") -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "labels": found.labels.tolist(),
                    "classes": found.classes,
                    "tile_px": found.tile_px,
                }
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        log.info("cache de silhouette non écrit (%s)", exc)


#: Poids accordé au voisinage d'une tuile dans le lissage. Le score propre
#: reste dominant : le voisinage tranche les cas douteux, il ne réécrit pas une
#: lecture nette.
NEIGHBOUR_WEIGHT = 0.45

#: Nombre de passes de lissage. Deux suffisent à propager une évidence d'une
#: tuile à ses voisines sans étaler une classe sur toute l'image.
SMOOTHING_PASSES = 2


def smooth_scores(scores: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Fait voter le voisinage de chaque tuile, sans lui donner le dernier mot.

    Une tuile de trente-deux pixels est classée seule, alors qu'une scène
    réelle est continue : un pan de toiture ne devient pas véhicule sur une
    tuile pour redevenir toiture sur la suivante. Mesuré sur ce pilote, dix
    pour cent des tuiles contredisaient leurs quatre voisines, et trente-huit
    pour cent restaient indécises faute d'un écart suffisant entre les deux
    meilleures natures.

    Le lissage porte sur les **scores**, non sur les étiquettes : additionner
    des évidences a un sens, faire voter des décisions déjà prises en perd. Une
    tuile dont la lecture propre est franche garde donc la sienne — le
    voisinage ne fait pencher que ce qui hésitait.
    """
    grid = scores.reshape(rows, cols, -1)
    for _pass in range(SMOOTHING_PASSES):
        padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge")
        neighbours = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) * 0.25
        grid = grid + NEIGHBOUR_WEIGHT * (neighbours - grid)
    return grid.reshape(rows * cols, -1)


def read_image(
    embedder,
    image_path: Path,
    asset_id: str,
    bearing_deg: float | None = None,
    tile_px: int = TILE_PX,
    cache_dir: Path | None = None,
) -> SilhouetteMap | None:
    """Attribue une nature à chaque tuile d'une image.

    L'encodeur est celui du dépistage d'identité : un seul modèle chargé sert
    les deux usages, et les vecteurs de texte se calculent une fois pour tout
    le corpus.
    """
    import torch
    from PIL import Image

    image_path = Path(image_path)
    cached_path = None
    if cache_dir is not None and image_path.is_file():
        model = f"{embedder.model_name}/{embedder.pretrained}"
        cached_path = Path(cache_dir) / f"{_cache_key(image_path, tile_px, model)}.json"
        cached = _cache_load(cached_path, asset_id, bearing_deg)
        if cached is not None:
            log.info("%s : silhouette relue du cache", image_path.name)
            return cached

    embedder.load()
    names = list(CLASS_PROMPTS)
    text_vectors = embedder.encode_text([CLASS_PROMPTS[n] for n in names])

    try:
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            rows, cols = height // tile_px, width // tile_px
            if rows < 2 or cols < 2:
                return None

            tiles = []
            for row in range(rows):
                for col in range(cols):
                    box = (
                        col * tile_px,
                        row * tile_px,
                        (col + 1) * tile_px,
                        (row + 1) * tile_px,
                    )
                    tiles.append(embedder._preprocess(image.crop(box)))
    except (OSError, ValueError) as exc:
        log.info("image illisible, écartée : %s (%s)", image_path, exc)
        return None

    with torch.no_grad():
        batch = torch.stack(tiles)
        vectors = embedder._model.encode_image(batch)
    vectors = (vectors / vectors.norm(dim=-1, keepdim=True)).cpu().numpy()

    scores = vectors @ text_vectors.T

    # Une tuile dont les deux meilleures natures se valent n'est pas tranchée :
    # une façade de brique derrière un branchage nu ressemble aux deux.
    # Le voisinage tranche avant cette indécision : une tuile isolée au milieu
    # d'une façade est plus probablement une façade qu'une nature à part.
    raw_margin = np.sort(scores, axis=1)
    raw_margin = raw_margin[:, -1] - raw_margin[:, -2]

    scores = smooth_scores(scores, rows, cols)
    best = scores.argmax(axis=1)
    ordered = np.sort(scores, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]

    # Le lissage rapproche les scores : mesuré sur ce pilote, l'écart médian
    # se contracte d'un facteur 1,27. Appliquer le seuil brut à des scores
    # lissés le rendrait mécaniquement plus sévère et déclarerait indécises
    # des tuiles que le voisinage vient justement de trancher. Le seuil suit
    # donc la même contraction, mesurée sur cette image et non supposée.
    reference = float(np.median(raw_margin))
    contracted = float(np.median(margin))
    scale = contracted / reference if reference > 1e-9 else 1.0
    undecided = margin < MIN_MARGIN * scale
    labels = np.where(undecided, len(names), best).reshape(rows, cols)

    log.info(
        "%s : %d tuile(s), %d indéterminée(s)",
        image_path.name,
        labels.size,
        int(undecided.sum()),
    )
    found = SilhouetteMap(
        asset_id=asset_id,
        labels=labels,
        classes=names + ["indetermine"],
        tile_px=tile_px,
        bearing_deg=bearing_deg,
    )
    if cached_path is not None:
        _cache_store(cached_path, found)
    return found


def read_views(
    hotel_id: str,
    views: list[tuple[str, Path, float | None]],
    embedder=None,
    tile_px: int = TILE_PX,
    cache_dir: Path | None = None,
) -> SilhouetteReading:
    """Lit les silhouettes de plusieurs vues d'un même site."""
    from .. identity.embedding import ImageEmbedder

    embedder = embedder or ImageEmbedder()
    reading = SilhouetteReading(hotel_id=hotel_id)

    for asset_id, path, bearing in views:
        found = read_image(
            embedder, Path(path), asset_id, bearing, tile_px, cache_dir=cache_dir
        )
        if found is not None:
            reading.maps.append(found)

    reading.provenance = {
        "model": f"{embedder.model_name}/{embedder.pretrained}",
        "tile_px": tile_px,
        "classes": list(CLASS_PROMPTS),
        "views_requested": len(views),
        "views_read": len(reading.maps),
    }
    log.info("silhouettes : %d vue(s) lue(s)", len(reading.maps))
    return reading


#: Profils de référence, du sommet vers le pied. Ils décrivent la part de la
#: largeur occupée à chaque niveau — la forme vue de la rue, que le relevé
#: aérien ne donne pas.
PROFILE_SHAPES: dict[str, tuple[float, ...]] = {
    # Effilé : large au pied, pointu au sommet.
    "conique": (0.15, 0.35, 0.55, 0.75, 0.90, 1.00, 1.00, 0.95),
    # Étalé : couronne haute et large, tronc dégagé.
    "etale": (0.55, 0.90, 1.00, 1.00, 0.85, 0.45, 0.20, 0.15),
    # Colonne : largeur constante.
    "colonnaire": (0.70, 0.90, 1.00, 1.00, 1.00, 0.95, 0.85, 0.75),
    # Buisson : masse basse, rien en hauteur.
    "arbustif": (0.10, 0.25, 0.55, 0.85, 1.00, 1.00, 1.00, 1.00),
}


def infer_shape(profile: np.ndarray) -> tuple[str, float]:
    """Choisit le profil de référence le plus proche d'un relevé.

    C'est ici que la lecture au sol complète le relevé aérien : le LiDAR donne
    la position et la hauteur, l'image dit si la masse est effilée, étalée ou
    colonnaire. Le rendu peut alors poser un volume qui ressemble à l'objet
    plutôt qu'un cylindre uniforme.

    La forme retenue est une **hypothèse plausible**, non une mesure : elle est
    rendue avec son score de ressemblance pour que rien ne la prenne pour un
    relevé.
    """
    values = np.asarray(profile, dtype=np.float64)
    band = values[values > 0.01]
    if band.size < 3:
        return "indetermine", 0.0

    # Le profil est ramené à huit niveaux et normalisé : seule la forme compte,
    # pas la hauteur absolue ni la distance de prise de vue.
    resampled = np.interp(
        np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, band.size), band
    )
    peak = resampled.max()
    if peak <= 0:
        return "indetermine", 0.0
    resampled = resampled / peak

    best_name, best_score = "indetermine", -1.0
    for name, reference in PROFILE_SHAPES.items():
        distance = float(np.mean(np.abs(resampled - np.asarray(reference))))
        score = 1.0 - distance
        if score > best_score:
            best_name, best_score = name, score
    return best_name, float(best_score)


def shape_hints(reading: SilhouetteReading) -> dict:
    """Ce que l'ensemble des vues suggère sur la forme de la végétation."""
    shapes: dict[str, int] = {}
    scores: list[float] = []
    for silhouette_map in reading.maps:
        name, score = infer_shape(silhouette_map.vegetation_profile())
        if name == "indetermine":
            continue
        shapes[name] = shapes.get(name, 0) + 1
        scores.append(score)

    dominant = max(shapes, key=shapes.get) if shapes else "indetermine"
    return {
        "dominant_shape": dominant,
        "by_shape": shapes,
        "mean_score": round(float(np.mean(scores)), 3) if scores else 0.0,
        "views_used": len(scores),
        "caveat": (
            "la forme est déduite d'un profil vu depuis la rue : c'est une "
            "hypothèse plausible sur l'allure de la masse végétale, jamais une "
            "mesure de sa géométrie"
        ),
    }
