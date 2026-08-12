"""Étapes du pipeline (plan directeur §18).

`collect` est implémentée au Lot 1. Les étapes suivantes lèvent
`StepNotImplemented` : le squelette s'exécute de bout en bout et s'arrête
proprement sur la première étape non construite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .logging import get_logger
from .schemas import EntranceVersion, ExteriorInterior
from .schemas.spatial import SpatialManifest

log = get_logger("steps")

#: Ordre des étapes de la Phase 1 (plan directeur §18).
STEP_ORDER: tuple[str, ...] = (
    "collect",
    "preflight",
    "reconstruct",
    "align",
    "validate",
)

ELEMENTS_FILE = "01_sources/overpass_elements.json"


class StepNotImplemented(RuntimeError):
    """Étape prévue par le plan directeur, pas encore construite."""

    def __init__(self, step: str, lot: str) -> None:
        super().__init__(f"l'étape {step!r} n'est pas implémentée — prévue au {lot}.")
        self.step = step
        self.lot = lot


class StepBlocked(RuntimeError):
    """Étape suspendue en attente d'une décision humaine.

    Aucune attente interactive sur une VM facturée : l'étape écrit son état,
    libère la machine, et se reprend à la session suivante.
    """

    def __init__(self, step: str, awaiting: str, expected_form: str) -> None:
        super().__init__(f"étape {step!r} bloquée : {awaiting}")
        self.step = step
        self.awaiting = awaiting
        self.expected_form = expected_form


@dataclass(frozen=True)
class Step:
    name: str
    lot: str
    summary: str


STEPS: dict[str, Step] = {
    "collect": Step(
        "collect",
        "Lot 1 puis Lot 4",
        "Résolution de propriété, collecte, droits et manifeste d'assets.",
    ),
    "preflight": Step(
        "preflight", "Lot 2 puis Lot 4", "Cascade G0 à G5, du comptage au SfM sparse réel."
    ),
    "reconstruct": Step(
        "reconstruct", "Lot 2 puis Lot 5", "Route de reconstruction et production du modèle 3D."
    ),
    "align": Step("align", "Lot 6", "Géoréférencement, alignement et environnement composite."),
    "validate": Step(
        "validate", "Lot 7", "Carte de confiance, comparaison aux références, rapport."
    ),
}


def _collect(workspace) -> None:  # noqa: ANN001 — Workspace, import circulaire
    """Résolution de propriété et qualification des médias (Lot 1).

    S'arrête sur chacun des deux verrous humains identifiés au complément §4 :
    la confirmation du bâtiment, puis la version de l'entrée.
    """
    from .intake import ASSET_MANIFEST_NAME
    from .resolve import check_separations, resolve
    from .schemas import AssetManifest

    project = workspace.read_manifest()

    # --- 1. vérité spatiale ---------------------------------------------
    spatial = workspace.read_spatial()
    if spatial is None:
        spatial = resolve(project.hotel_id, project.address)
        workspace.write_spatial(spatial)

        elements = _fetch_elements(spatial)
        workspace.write_json(ELEMENTS_FILE, elements)

    if not spatial.confirmed_building_id:
        ranked = spatial.ranked()[:5]
        listing = "\n".join(
            f"      {c.feature_id}  score={c.score:.2f}  "
            f"{c.distance_to_geocode_m:.0f} m  {c.area_m2:.0f} m²  "
            f"{c.tags.get('name', '(sans nom)')}"
            for c in ranked
        )
        raise StepBlocked(
            "collect",
            f"confirmation de BUILDING_MAIN parmi {len(spatial.candidates)} candidat(s)",
            "hotel-pipeline confirm-building <hotel> <feature_id> "
            "--by <auteur> --rationale <justification>\n"
            f"    candidats les mieux classés :\n{listing}",
        )

    # --- 2. séparations géométriques ------------------------------------
    elements = workspace.read_json(ELEMENTS_FILE) or []
    spatial.assertions = check_separations(spatial, elements)
    workspace.write_spatial(spatial)

    failed = spatial.failed_assertions()
    if failed:
        detail = "; ".join(f"{a.name} — {a.detail}" for a in failed)
        raise StepBlocked(
            "collect",
            f"séparation géométrique non satisfaite : {detail}",
            "corriger l'identification (confirm-building) ou enregistrer une "
            "correction humaine justifiée",
        )

    # --- 3. médias et droits --------------------------------------------
    assets_path = workspace.path("00_manifest", ASSET_MANIFEST_NAME)
    if not assets_path.is_file():
        raise StepBlocked(
            "collect",
            "inventaire des médias et de leurs droits",
            "hotel-pipeline assets import <hotel> <inventaire.csv> "
            "[--images-root <dir>]",
        )

    assets = AssetManifest.model_validate_json(assets_path.read_text("utf-8"))
    eligible = assets.production_eligible()
    if not eligible:
        raise StepBlocked(
            "collect",
            f"aucun asset éligible production sur {len(assets.assets)} inventorié(s)",
            "hotel-pipeline assets promote <hotel> <id> [<id>...] — "
            "après revue des droits",
        )

    exteriors = [a for a in eligible if a.exterior_or_interior is ExteriorInterior.EXTERIOR]
    undated = [a for a in exteriors if a.entrance_version is EntranceVersion.UNKNOWN]
    if undated:
        raise StepBlocked(
            "collect",
            f"version d'entrée indéterminée pour {len(undated)} extérieur(s) éligible(s) — "
            "non déductible visuellement sans référence datée",
            "hotel-pipeline assets set-entrance-version <hotel> <id> "
            "<pre_2024|post_2024> : " + ", ".join(a.id for a in undated[:10]),
        )

    log.info(
        "collect terminé : bâtiment %s confirmé, %d asset(s) éligible(s), %d extérieur(s)",
        spatial.confirmed_building_id,
        len(eligible),
        len(exteriors),
    )


def _fetch_elements(spatial: SpatialManifest) -> list[dict]:
    """Réinterroge Overpass pour conserver les éléments bruts.

    Le cache disque évite un second appel réseau réel ; les conserver permet
    de rejouer les séparations sans dépendre du réseau.
    """
    from .providers import features_around

    assert spatial.geocode is not None
    return features_around(
        spatial.geocode.lat, spatial.geocode.lon, spatial.search_radius_m
    )


def run_step(name: str, workspace) -> None:  # noqa: ANN001
    """Exécute une étape."""
    if name == "collect":
        _collect(workspace)
        return

    step = STEPS[name]
    raise StepNotImplemented(step.name, step.lot)


__all__ = [
    "ELEMENTS_FILE",
    "STEPS",
    "STEP_ORDER",
    "Step",
    "StepBlocked",
    "StepNotImplemented",
    "run_step",
]
