"""Le bâtiment est-il dans l'image ? Jugé par la géométrie, confirmé par l'œil.

La mesure de visibilité répond à une question voisine mais différente : « un
obstacle s'interpose-t-il entre cette caméra et le bâtiment ? » Une ligne de vue
dégagée ne prouve pas que le bâtiment soit **dans le cadre** — la caméra peut
viser ailleurs, ou le sujet sortir de l'image. C'est pourquoi la projection de
visibilité refuse de promouvoir quoi que ce soit.

Ce module tranche l'autre question, et seulement elle. Deux évidences
indépendantes doivent concorder :

- **le cadrage**, calculé du cap, du champ et du secteur angulaire occupé par
  l'emprise — il dit qu'à cette position, avec cette orientation, le bâtiment
  tombe dans l'image ;
- **le contenu**, lu par un modèle entraîné — il dit qu'on y voit bel et bien
  une façade, et non un ciel vide ou une haie.

L'une sans l'autre ne suffit pas. Un cap erroné place le bâtiment au bon
endroit d'une image qui montre autre chose ; un modèle voyant « un bâtiment »
ne dit pas *lequel*. Exiger les deux évite ces deux erreurs symétriques.

**Ce que ce jugement n'est pas.** Ce n'est pas une revue humaine, et il ne s'en
substitue pas : le verdict porte sa méthode, et une décision de personne prime
toujours. Ce n'est pas non plus un jugement d'identité — que le bâtiment vu
soit bien le bon relève de `identity/`, qui a ses propres preuves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..logging import get_logger

log = get_logger("geo-in-frame")

#: Part de l'emprise devant tomber dans le cadre pour que la géométrie
#: conclue. En deçà, le bâtiment n'est qu'effleuré par le bord de l'image et
#: n'apporte pas de façade exploitable.
IN_FRAME_MIN = 0.55

#: Largeur apparente minimale, en fraction de la largeur d'image. Mesuré sur
#: ce pilote, la médiane est de 0,17 : un bâtiment occupant moins d'un
#: vingtième du cadre est trop lointain pour porter de la structure.
WIDTH_MIN = 0.05

#: Part de l'horizon devant porter du bâti pour confirmer. Un seul secteur sur
#: huit suffit : mesuré sur ce pilote, une vue montrant l'hôtel en compte un,
#: une vue d'autoroute aucun. Exiger davantage écarterait les vues de trois
#: quarts, où le bâtiment n'occupe qu'un bord du cadre.
CONTENT_MIN = 0.12

#: Bande d'image balayée, en fractions de la hauteur. L'horizon d'une vue de
#: rue : sous le ciel, au-dessus de la chaussée.
HORIZON_BAND = (0.48, 0.68)

#: Secteurs verticaux du balayage. Huit donnent des vignettes assez larges
#: pour qu'une façade y soit reconnaissable, assez étroites pour qu'elle ne
#: soit pas noyée.
HORIZON_SECTORS = 8

#: Descriptions soumises au modèle. La classe utile d'abord, puis ce qu'on
#: pourrait confondre avec elle : un modèle sans alternative dit toujours oui.
CONTENT_PROMPTS: dict[str, str] = {
    "batiment": "a building facade with windows, doors and walls",
    "route": "an empty road or parking lot with no building",
    "vegetation": "trees, hedges and foliage filling the view",
    "ciel": "plain sky or a blank featureless surface",
    "interieur": "an indoor room seen from inside",
}

#: Classes valant confirmation. Les autres sont des alternatives de contrôle.
POSITIVE_CLASSES = frozenset({"batiment"})


@dataclass
class InFrameVerdict:
    """Ce qu'on conclut d'une vue, et sur quelles évidences."""

    asset_id: str
    #: `True`, `False`, ou `None` quand une évidence manque.
    in_frame: bool | None
    reason: str
    in_frame_fraction: float | None = None
    width_fraction: float | None = None
    content_score: float | None = None
    content_class: str | None = None
    method: str = "geometry+content"

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "in_frame": self.in_frame,
            "reason": self.reason,
            "in_frame_fraction": (
                round(self.in_frame_fraction, 3)
                if self.in_frame_fraction is not None
                else None
            ),
            "width_fraction": (
                round(self.width_fraction, 3) if self.width_fraction is not None else None
            ),
            "content_score": (
                round(self.content_score, 4) if self.content_score is not None else None
            ),
            "content_class": self.content_class,
            "method": self.method,
        }


@dataclass
class InFrameReport:
    verdicts: list[InFrameVerdict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def visible(self) -> list[str]:
        return [v.asset_id for v in self.verdicts if v.in_frame is True]

    @property
    def absent(self) -> list[str]:
        return [v.asset_id for v in self.verdicts if v.in_frame is False]

    @property
    def undecided(self) -> list[str]:
        return [v.asset_id for v in self.verdicts if v.in_frame is None]

    def as_dict(self) -> dict:
        return {
            "visible_count": len(self.visible),
            "absent_count": len(self.absent),
            "undecided_count": len(self.undecided),
            "total": len(self.verdicts),
            "visible": self.visible,
            "verdicts": [v.as_dict() for v in self.verdicts],
            "provenance": self.provenance,
            "caveats": [
                "ce verdict dit que le bâtiment est dans le cadre, non que "
                "c'est le bon bâtiment — l'identité se juge ailleurs",
                "il ne remplace pas une revue humaine : une décision de "
                "personne prime toujours sur cette mesure",
                "un cadrage tiré du champ déclaré par la politique décrit la "
                "consigne du lot, non l'optique de cette prise de vue",
            ],
        }


def _geometry_verdict(framing) -> tuple[bool | None, str, float | None, float | None]:  # noqa: ANN001
    """Ce que le cadrage seul permet de dire."""
    if framing is None:
        return None, "aucun cadrage calculé pour cette vue", None, None
    if not framing.get("horizontal_computable"):
        return (
            None,
            framing.get("horizontal_reason") or "cadrage non calculable",
            None,
            None,
        )

    fraction = framing.get("target_in_frame_fraction")
    width = framing.get("unclipped_width_fraction")
    if fraction is None or width is None:
        return None, "cadrage incomplet", fraction, width

    if fraction < IN_FRAME_MIN:
        return (
            False,
            f"emprise hors cadre à {1 - fraction:.0%} (seuil {IN_FRAME_MIN:.0%})",
            fraction,
            width,
        )
    if width < WIDTH_MIN:
        return (
            False,
            f"bâtiment trop petit dans l'image ({width:.1%} de la largeur, "
            f"seuil {WIDTH_MIN:.0%})",
            fraction,
            width,
        )
    return True, "emprise cadrée et de taille exploitable", fraction, width


def _content_verdict(embedder, image_path) -> tuple[str | None, float | None]:  # noqa: ANN001
    """Un bâtiment occupe-t-il un secteur de l'horizon ?

    Une lecture globale échoue ici, et il a fallu regarder les images pour le
    comprendre : sur une vue de rue, le bâtiment tient dans une bande mince
    au-dessus de la chaussée, cerné de neige et de ciel. Interrogé sur
    l'ensemble du cadre, le modèle répond « route » — ce qui est vrai de
    l'image, et faux de la question posée. Une vue montrant l'hôtel, enseigne
    lisible, était ainsi écartée à 1,00 de confiance.

    Le balayage suit donc la géométrie du problème : la bande d'horizon,
    découpée en secteurs verticaux. Un seul secteur portant une façade suffit —
    c'est ce que le cadrage prétend, et il n'est pas demandé au bâtiment de
    remplir l'image.
    """
    import numpy as np
    import torch
    from PIL import Image

    names = list(CONTENT_PROMPTS)
    try:
        embedder.load()
        text = np.asarray(embedder.encode_text([CONTENT_PROMPTS[n] for n in names]))
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            if width < 64 or height < 64:
                return None, None
            top, bottom = int(height * HORIZON_BAND[0]), int(height * HORIZON_BAND[1])
            crops = [
                embedder._preprocess(
                    image.crop(
                        (
                            int(width * k / HORIZON_SECTORS),
                            top,
                            int(width * (k + 1) / HORIZON_SECTORS),
                            bottom,
                        )
                    )
                )
                for k in range(HORIZON_SECTORS)
            ]
    except (OSError, ValueError) as exc:
        log.info("contenu illisible (%s) : %s", exc, image_path)
        return None, None

    with torch.no_grad():
        vectors = embedder._model.encode_image(torch.stack(crops))
    vectors = (vectors / vectors.norm(dim=-1, keepdim=True)).cpu().numpy()

    labels = (vectors @ text.T).argmax(axis=1)
    building = int((labels == names.index("batiment")).sum())
    # La part de l'horizon occupée par du bâti : elle dit à la fois s'il y en
    # a et combien, là où un simple booléen perdrait la nuance.
    share = building / HORIZON_SECTORS
    if building:
        return "batiment", share
    # Sans bâti, on rend la nature qui domine l'horizon : c'est elle qui
    # explique le refus dans le rapport.
    dominant = names[int(np.bincount(labels, minlength=len(names)).argmax())]
    return dominant, share


def judge(
    assets,  # noqa: ANN001
    framings: dict[str, dict],
    embedder=None,  # noqa: ANN001
    workspace=None,  # noqa: ANN001
) -> InFrameReport:
    """Juge, vue par vue, si le bâtiment cible tombe dans l'image.

    Le contenu n'est lu que pour les vues dont la géométrie conclut : demander
    au modèle de trancher une vue déjà écartée coûterait une inférence pour
    rien, et lui donner le dernier mot contre la géométrie reviendrait à lui
    faire dire où pointe la caméra — ce qu'il ne sait pas.
    """
    report = InFrameReport()
    checked = 0

    for asset in assets:
        framing = framings.get(asset.id)
        decided, reason, fraction, width = _geometry_verdict(framing)

        if decided is not True:
            report.verdicts.append(
                InFrameVerdict(
                    asset_id=asset.id,
                    in_frame=decided,
                    reason=reason,
                    in_frame_fraction=fraction,
                    width_fraction=width,
                    method="geometry",
                )
            )
            continue

        if embedder is None or workspace is None or not asset.local_path:
            report.verdicts.append(
                InFrameVerdict(
                    asset_id=asset.id,
                    in_frame=None,
                    reason=(
                        "cadrage favorable, contenu non vérifié : une "
                        "géométrie seule ne prouve pas ce que montre l'image"
                    ),
                    in_frame_fraction=fraction,
                    width_fraction=width,
                    method="geometry",
                )
            )
            continue

        path = workspace.path(asset.local_path)
        if not path.is_file():
            report.verdicts.append(
                InFrameVerdict(
                    asset_id=asset.id,
                    in_frame=None,
                    reason="fichier absent du poste : contenu invérifiable",
                    in_frame_fraction=fraction,
                    width_fraction=width,
                    method="geometry",
                )
            )
            continue

        label, score = _content_verdict(embedder, path)
        checked += 1
        if label is None:
            report.verdicts.append(
                InFrameVerdict(
                    asset_id=asset.id,
                    in_frame=None,
                    reason="image illisible : contenu invérifiable",
                    in_frame_fraction=fraction,
                    width_fraction=width,
                    method="geometry",
                )
            )
            continue

        agrees = label in POSITIVE_CLASSES and score >= CONTENT_MIN
        report.verdicts.append(
            InFrameVerdict(
                asset_id=asset.id,
                in_frame=agrees,
                reason=(
                    f"cadrage favorable ({fraction:.0%} dans l'image) et "
                    f"façade lue sur {score:.0%} de l'horizon"
                    if agrees
                    else (
                        f"cadrage favorable mais l'horizon ne montre que "
                        f"{label!r} : la caméra ne vise pas le bâtiment"
                    )
                ),
                in_frame_fraction=fraction,
                width_fraction=width,
                content_score=score,
                content_class=label,
            )
        )

    report.provenance = {
        "in_frame_min": IN_FRAME_MIN,
        "width_min": WIDTH_MIN,
        "content_min": CONTENT_MIN,
        "classes": list(CONTENT_PROMPTS),
        "horizon_band": list(HORIZON_BAND),
        "horizon_sectors": HORIZON_SECTORS,
        "content_checked": checked,
        "model": getattr(embedder, "model_name", None) if embedder else None,
    }
    log.info(
        "cadre : %d vue(s) montrent la cible, %d non, %d indécidable(s)",
        len(report.visible),
        len(report.absent),
        len(report.undecided),
    )
    return report


__all__ = [
    "CONTENT_MIN",
    "HORIZON_BAND",
    "HORIZON_SECTORS",
    "IN_FRAME_MIN",
    "WIDTH_MIN",
    "InFrameReport",
    "InFrameVerdict",
    "judge",
]
