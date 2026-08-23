"""Du score au verdict, avec un seuil qui se calibre au lieu d'être décrété.

Un seuil écrit en dur ne survit pas au changement de site : 0,65 sépare l'hôtel
des pavillons à Boucherville, et ne veut rien dire pour une tour urbaine
entourée d'autres tours. Le seuil est donc **dérivé de la distribution des
scores du corpus**, et le nombre en dur ne sert que de garde-fou.

Le verdict n'est jamais binaire. `uncertain` est un résultat de plein droit :
c'est lui qui envoie une image en revue humaine plutôt qu'en production.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ..logging import get_logger

log = get_logger("identity-verdict")

#: Écart des moyennes en deçà duquel une coupure au plancher trahit une
#: population unique plutôt qu'une frontière située plus bas.
FALLBACK_GAP = 0.20

#: Part des scores retenue quand Otsu ne trouve aucune frontière interne.
#: Réglé haut : sur un corpus large, les vues réellement utiles sont rares.
FALLBACK_PERCENTILE = 92.0

#: Bornes du seuil calibré. En deçà de `FLOOR`, tout devient « l'hôtel » ;
#: au-delà de `CEILING`, même les vraies vues de l'arrière sont rejetées.
THRESHOLD_FLOOR = 0.55
THRESHOLD_CEILING = 0.80

#: Largeur de la zone d'incertitude autour du seuil, en points de cosinus.
#: Sert de repli quand la dispersion du corpus n'est pas connue.
UNCERTAIN_BAND = 0.06

#: Largeur de la bande d'indécision, en écarts-types des scores du corpus.
#: Une largeur fixe produit un volume d'indécision arbitraire : mesuré sur ce
#: pilote, ±0,06 mettait en revue 32 % du corpus — non parce que ces images
#: étaient ambiguës, mais parce que la distribution était resserrée autour du
#: seuil. Exprimée en écarts-types, la bande suit la dispersion réelle.
UNCERTAIN_BAND_SIGMA = 0.35

#: Bornes de la bande dérivée, pour qu'un corpus très dispersé ne rende pas
#: tout indécis, ni un corpus très concentré ne supprime toute prudence.
MIN_UNCERTAIN_BAND = 0.02
MAX_UNCERTAIN_BAND = 0.08

#: En dessous, les ancres se contredisent : aucun verdict n'est publiable.
MIN_ANCHOR_COHERENCE = 0.55

#: Écart minimal entre les moyennes des deux groupes, en points de cosinus,
#: pour croire qu'ils sont bien deux. Contrairement à la variance d'Otsu, cet
#: écart ne dépend pas des effectifs : dix vraies images noyées dans cent
#: cinquante donnent une variance minuscule mais un écart de moyennes net.
MIN_MEAN_GAP = 0.12


class IdentityStatus(StrEnum):
    """Ce que le modèle établit sur l'appartenance d'une image."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNCERTAIN = "uncertain"
    #: Aucune ancre exploitable : le pipeline ne sait pas à quoi comparer.
    UNDECIDABLE = "undecidable"


@dataclass
class IdentityVerdict:
    """Le jugement porté sur une image, et ce qui l'explique."""

    asset_id: str
    status: IdentityStatus
    score: float
    threshold: float
    nearest_anchor: str | None
    reason: str

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "status": str(self.status),
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "nearest_anchor": self.nearest_anchor,
            "reason": self.reason,
        }


def calibrate_threshold(scores: list[float]) -> tuple[float, str]:
    """Cherche la coupure qui sépare le mieux deux populations de scores.

    Le critère est celui d'Otsu : la coupure retenue **maximise la variance
    inter-classes**, c'est-à-dire sépare au mieux « le bâtiment » du « reste ».

    Une méthode plus intuitive a été essayée puis écartée : couper au plus
    grand écart entre deux scores consécutifs. Elle suppose deux groupes bien
    détachés. Un corpus de rue réel n'a pas cette forme — sa distribution est
    continue, du pavillon voisin jusqu'à la façade de face en passant par
    toutes les vues partielles — et le plus grand écart tombait alors dans le
    bruit de la queue haute, plaçant le seuil au-dessus de presque toutes les
    vraies images du site.
    """
    if len(scores) < 8:
        return THRESHOLD_FLOOR, (
            f"corpus trop petit ({len(scores)} images) pour calibrer : "
            "plancher appliqué"
        )

    values = np.asarray(scores, dtype=np.float64)
    candidates = np.linspace(THRESHOLD_FLOOR, THRESHOLD_CEILING, 96)

    best_cut, best_variance, best_gap = THRESHOLD_FLOOR, -1.0, 0.0
    for cut in candidates:
        low, high = values[values < cut], values[values >= cut]
        # Une coupure qui vide l'un des deux côtés ne sépare rien.
        if low.size < 2 or high.size < 2:
            continue
        weight_low = low.size / values.size
        between = weight_low * (1.0 - weight_low) * (high.mean() - low.mean()) ** 2
        if between > best_variance:
            best_cut = float(cut)
            best_variance = float(between)
            best_gap = float(high.mean() - low.mean())

    if best_variance < 0:
        return THRESHOLD_CEILING, (
            "aucune coupure ne sépare deux groupes : plafond appliqué"
        )

    # Otsu coupe *toujours* en deux, y compris une population homogène qui n'en
    # forme qu'une. Mesuré sur ce pilote : un lot ne contenant que des maisons
    # du voisinage produisait une coupure au plancher, et promouvait en
    # « références » les pavillons les moins dissemblables.
    #
    # Le critère de séparation porte sur l'**écart des moyennes**, non sur la
    # variance inter-classes : celle-ci s'effondre dès que le groupe recherché
    # est petit — dix vraies vues parmi cent cinquante donnent 0,004, contre
    # 0,021 pour deux groupes de tailles comparables — et un seuil posé dessus
    # rejetterait justement les corpus où les bonnes images sont rares.
    if best_gap < MIN_MEAN_GAP:
        return THRESHOLD_CEILING, (
            f"séparation trop faible (écart des moyennes {best_gap:.3f} < "
            f"{MIN_MEAN_GAP}) : le corpus ne contient probablement qu'une "
            "population, plafond appliqué pour ne rien promouvoir à tort"
        )

    # Une coupure calée sur une borne n'a rien séparé : elle dit seulement que
    # la vraie frontière est hors de la plage explorée. Mesuré sur ce pilote,
    # élargir le corpus faisait tomber le seuil au plancher, et vingt pour cent
    # des images ressortaient `match` — dont des bâtiments de brique sans
    # rapport avec le site. Le quantile haut prend alors le relais : il retient
    # une part fixe des meilleurs scores plutôt qu'une frontière introuvable.
    # Deux situations mènent au plancher, et une seule est un échec. Quand les
    # deux groupes sont franchement séparés, l'écart des moyennes reste large
    # et la coupure au plancher est correcte : la frontière est simplement plus
    # bas que la plage explorée. Quand la population est unique, cet écart
    # s'affaisse — c'est là que le quantile prend le relais.
    if best_cut <= THRESHOLD_FLOOR + 1e-6 and best_gap < FALLBACK_GAP:
        fallback = float(np.percentile(values, FALLBACK_PERCENTILE))
        cut = float(np.clip(fallback, THRESHOLD_FLOOR, THRESHOLD_CEILING))
        retained = int((values >= cut).sum())
        return cut, (
            f"coupure d'Otsu calée sur le plancher : aucune frontière dans la "
            f"plage explorée, quantile {FALLBACK_PERCENTILE:.0f} % appliqué "
            f"({cut:.3f}), {retained}/{values.size} images retenues"
        )

    retained = int((values >= best_cut).sum())
    return best_cut, (
        f"coupure d'Otsu à {best_cut:.3f} (écart des moyennes {best_gap:.3f}), "
        f"{retained}/{values.size} images retenues"
    )


def uncertain_band(scores: list[float]) -> float:
    """Largeur de la bande d'indécision, dérivée de la dispersion du corpus."""
    if len(scores) < 8:
        return UNCERTAIN_BAND
    spread = float(np.std(np.asarray(scores, dtype=np.float64)))
    return float(
        np.clip(spread * UNCERTAIN_BAND_SIGMA, MIN_UNCERTAIN_BAND, MAX_UNCERTAIN_BAND)
    )


def judge(
    asset_id: str,
    score: float,
    threshold: float,
    nearest_anchor: str | None,
    anchor_count: int,
    anchor_coherence: float,
    band: float | None = None,
) -> IdentityVerdict:
    """Tranche, ou déclare qu'il n'y a pas de quoi trancher."""
    if anchor_count == 0:
        return IdentityVerdict(
            asset_id, IdentityStatus.UNDECIDABLE, score, threshold, None,
            "aucune ancre confirmée : rien n'atteste à quoi ressemble ce site",
        )
    if anchor_coherence < MIN_ANCHOR_COHERENCE:
        return IdentityVerdict(
            asset_id, IdentityStatus.UNDECIDABLE, score, threshold, nearest_anchor,
            f"ancres incohérentes entre elles ({anchor_coherence:.2f}) : elles "
            "ne montrent pas toutes le même bâtiment",
        )

    band = UNCERTAIN_BAND if band is None else band
    if score >= threshold + band:
        return IdentityVerdict(
            asset_id, IdentityStatus.MATCH, score, threshold, nearest_anchor,
            f"ressemblance {score:.3f} nettement au-dessus du seuil {threshold:.3f}",
        )
    if score <= threshold - band:
        return IdentityVerdict(
            asset_id, IdentityStatus.MISMATCH, score, threshold, nearest_anchor,
            f"ressemblance {score:.3f} nettement en dessous du seuil "
            f"{threshold:.3f} : bâtiment différent",
        )
    return IdentityVerdict(
        asset_id, IdentityStatus.UNCERTAIN, score, threshold, nearest_anchor,
        f"ressemblance {score:.3f} dans la zone d'indécision autour de "
        f"{threshold:.3f} : revue humaine requise",
    )
