"""Décider si un sparse mérite qu'on lance le dense.

La reconstruction dense est l'étape la plus coûteuse du pipeline, et la seule
qui produise un livrable. Elle est aussi celle qui ne dit rien quand elle
échoue : un maillage troué ne distingue pas un bâtiment mal observé d'un solve
fragmenté, et l'on cherche ensuite dans le dense une cause qui vient du sparse.

Ce module tranche avant. Il ne mesure pas la qualité du sparse en général — il
répond à une question précise : **ce sparse contient-il de quoi reconstruire le
bâtiment demandé ?** Trois conditions, chacune éliminatoire pour une raison
différente :

- **la connexité** — un solve fragmenté n'a pas de repère commun, et deux
  composantes ne se fusionnent pas faute de correspondances ; le dense y
  produirait plusieurs morceaux sans échelle partagée ;
- **la couverture** — un bâtiment observé sur une seule face donne un maillage
  dont trois côtés sont inventés par interpolation ;
- **la triangulation** — des vues confondues ne portent aucune profondeur,
  quel que soit leur nombre.

Chaque refus porte sa raison et ce qu'il faudrait pour lever l'obstacle : un
gate qui dit seulement « non » oblige à deviner la suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("dense-admission")

#: Part des images enregistrées qui doit tenir dans une seule composante. En
#: deçà, le solve décrit plusieurs scènes disjointes plutôt qu'une.
MIN_CONNECTED_FRACTION = 0.6

#: Nombre minimal d'images dans la composante principale. Trois est le seuil
#: que COLMAP retient pour tenir un point pour observé ; en deçà, la géométrie
#: n'est pas contrainte.
MIN_COMPONENT_IMAGES = 3

#: Part des cellules de façade devant être triangulables. Un bâtiment dont
#: moins du tiers des murs est observé deux fois donnera un maillage
#: majoritairement interpolé.
MIN_TRIANGULABLE_FRACTION = 0.35

#: Erreur de reprojection, en pixels, au-delà de laquelle les poses ne sont pas
#: assez justes pour porter un dense.
MAX_REPROJECTION_PX = 3.0


@dataclass
class AdmissionCheck:
    """Une condition, son verdict et ce qu'il faudrait pour la lever."""

    name: str
    passed: bool
    measured: float | None
    required: float | None
    reason: str
    remedy: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "measured": (
                round(self.measured, 3) if self.measured is not None else None
            ),
            "required": self.required,
            "reason": self.reason,
            "remedy": self.remedy,
        }


@dataclass
class AdmissionVerdict:
    """Ce que le sparse autorise, et ce qu'il interdit encore."""

    checks: list[AdmissionCheck] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def blocking(self) -> list[AdmissionCheck]:
        return [check for check in self.checks if not check.passed]

    def as_dict(self) -> dict:
        return {
            "admitted": self.admitted,
            "checks": [check.as_dict() for check in self.checks],
            "blocking": [check.name for check in self.blocking],
            "remedies": [
                check.remedy for check in self.blocking if check.remedy
            ],
            "provenance": self.provenance,
            "caveats": [
                "ce contrôle porte sur la géométrie du sparse, non sur "
                "l'apparence : un solve admis peut encore manquer de texture",
                "admettre n'est pas garantir — c'est écarter les cas dont on "
                "sait déjà que le dense ne peut rien tirer",
            ],
        }


def evaluate(
    registered_images: int,
    largest_component: int,
    triangulable_fraction: float | None = None,
    reprojection_px: float | None = None,
) -> AdmissionVerdict:
    """Juge si un sparse mérite le dense, condition par condition.

    Les mesures absentes ne bloquent pas : elles produisent un contrôle non
    concluant, dit comme tel. Refuser faute de mesure reviendrait à traiter
    l'ignorance comme une preuve.
    """
    verdict = AdmissionVerdict()

    # 1. Connexité — la condition qui ne se rattrape pas en aval.
    fraction = largest_component / max(registered_images, 1)
    connected = (
        largest_component >= MIN_COMPONENT_IMAGES
        and fraction >= MIN_CONNECTED_FRACTION
    )
    verdict.checks.append(
        AdmissionCheck(
            name="connexite",
            passed=connected,
            measured=fraction,
            required=MIN_CONNECTED_FRACTION,
            reason=(
                f"{largest_component} image(s) sur {registered_images} dans la "
                f"composante principale ({fraction:.0%})"
            ),
            remedy=(
                ""
                if connected
                else (
                    "acquérir des vues de liaison entre les groupes : deux "
                    "composantes sans correspondance commune ne fusionnent pas"
                )
            ),
        )
    )

    # 2. Couverture — un bâtiment vu d'un seul côté n'est pas reconstructible.
    if triangulable_fraction is None:
        verdict.checks.append(
            AdmissionCheck(
                name="couverture",
                passed=True,
                measured=None,
                required=MIN_TRIANGULABLE_FRACTION,
                reason="aucune carte d'observation fournie — contrôle non concluant",
                remedy="lancer `geo observation-map` pour mesurer la couverture",
            )
        )
    else:
        covered = triangulable_fraction >= MIN_TRIANGULABLE_FRACTION
        verdict.checks.append(
            AdmissionCheck(
                name="couverture",
                passed=covered,
                measured=triangulable_fraction,
                required=MIN_TRIANGULABLE_FRACTION,
                reason=(
                    f"{triangulable_fraction:.0%} des cellules de façade sont "
                    "triangulables"
                ),
                remedy=(
                    ""
                    if covered
                    else (
                        "acquérir depuis les positions que la carte "
                        "d'observation recommande"
                    )
                ),
            )
        )

    # 3. Justesse des poses — mesurée quand le solve la rapporte.
    if reprojection_px is None:
        verdict.checks.append(
            AdmissionCheck(
                name="reprojection",
                passed=True,
                measured=None,
                required=MAX_REPROJECTION_PX,
                reason="erreur de reprojection non rapportée par le solve",
                remedy="",
            )
        )
    else:
        precise = reprojection_px <= MAX_REPROJECTION_PX
        verdict.checks.append(
            AdmissionCheck(
                name="reprojection",
                passed=precise,
                measured=reprojection_px,
                required=MAX_REPROJECTION_PX,
                reason=f"erreur moyenne de {reprojection_px:.2f} px",
                remedy=(
                    ""
                    if precise
                    else (
                        "revoir les intrinsèques : une erreur de reprojection "
                        "élevée vient souvent d'une focale mal estimée"
                    )
                ),
            )
        )

    verdict.provenance = {
        "min_connected_fraction": MIN_CONNECTED_FRACTION,
        "min_component_images": MIN_COMPONENT_IMAGES,
        "min_triangulable_fraction": MIN_TRIANGULABLE_FRACTION,
        "max_reprojection_px": MAX_REPROJECTION_PX,
        "registered_images": registered_images,
        "largest_component": largest_component,
    }
    log.info(
        "admission au dense : %s (%d/%d contrôle(s) franchi(s))",
        "oui" if verdict.admitted else "non",
        sum(1 for c in verdict.checks if c.passed),
        len(verdict.checks),
    )
    return verdict


__all__ = [
    "MAX_REPROJECTION_PX",
    "MIN_COMPONENT_IMAGES",
    "MIN_CONNECTED_FRACTION",
    "MIN_TRIANGULABLE_FRACTION",
    "AdmissionCheck",
    "AdmissionVerdict",
    "evaluate",
]
