"""Migration d'un profil vers les champs de portabilité.

Trois champs deviennent obligatoires — pays, fuseau, langues d'OCR — parce
qu'ils étaient jusqu'ici supposés : le territoire valait `QC`, le fuseau valait
UTC, et les langues retombaient sur « fr, en ». Un profil qui ne les déclare
pas ne peut pas être servi ailleurs qu'au Québec.

Leur ajout **déplace l'empreinte du profil**, citée par une vingtaine de
rapports déjà publiés. C'est voulu : l'empreinte existe pour détecter qu'un
profil a changé, et il a changé. Ce que la migration doit prouver, c'est que
rien de **décisionnel** n'a bougé — ni identité, ni concurrents, ni position,
ni travaux. Le reçu porte les deux empreintes et cette preuve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .logging import get_logger
from .provenance import profile_digest
from .schemas import PropertyProfile

log = get_logger("migrate-profile")

#: Champs ajoutés par cette migration. Tout le reste doit rester identique.
ADDED_FIELDS = ("country_code", "subdivision_code", "timezone")

#: Champs dont dépend une décision déjà prise. Le reçu prouve leur stabilité.
DECISION_BEARING = (
    "property_id", "address", "official_name", "aliases", "competitor_names",
    "renovation_events", "room_count", "expected_levels", "footprint_min_m2",
    "footprint_max_m2", "lat", "lon", "website_url", "place_query",
)


class ProfileMigrationRefused(RuntimeError):
    """Rien n'a été écrit."""


@dataclass
class ProfileMigrationReport:
    property_id: str = ""
    digest_before: str = ""
    digest_after: str = ""
    added: dict = field(default_factory=dict)
    decisions_unchanged: bool = True
    migrated_at: str = ""

    def as_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "digest_before": self.digest_before,
            "digest_after": self.digest_after,
            "added": self.added,
            "decisions_unchanged": self.decisions_unchanged,
            "migrated_at": self.migrated_at,
            "note": (
                "l'empreinte change parce que le profil déclare désormais son "
                "pays, son fuseau et ses langues. Aucun champ portant une "
                "décision déjà prise n'a été modifié : identité, concurrents, "
                "position et travaux sont identiques avant et après."
            ),
        }


def migrate_payload(
    payload: dict, country_code: str, timezone_name: str,
    ocr_languages: list[str] | None = None, subdivision_code: str | None = None,
) -> tuple[dict, ProfileMigrationReport]:
    """Ajoute les champs de portabilité. Aucune valeur n'est devinée.

    Le pays et le fuseau sont demandés à l'appelant : les déduire de l'adresse
    reproduirait le défaut qu'on corrige — une chaîne contenant « Québec »
    n'établit pas un territoire, et deviner ferait de la migration une source
    d'autorité qu'elle n'a pas.
    """
    report = ProfileMigrationReport(
        property_id=payload.get("property_id", "?"),
        migrated_at=datetime.now(timezone.utc).isoformat(),
    )
    before = {key: payload.get(key) for key in DECISION_BEARING}

    migrated = dict(payload)
    migrated["country_code"] = country_code
    migrated["timezone"] = timezone_name
    if subdivision_code is not None:
        migrated["subdivision_code"] = subdivision_code
    if ocr_languages is not None:
        migrated["ocr_languages"] = ocr_languages
    elif not migrated.get("ocr_languages"):
        raise ProfileMigrationRefused(
            f"profil {report.property_id!r} sans langue d'OCR déclarée : "
            "précisez-les, elles ne se devinent pas du pays"
        )

    report.added = {key: migrated.get(key) for key in ADDED_FIELDS}
    report.decisions_unchanged = before == {
        key: migrated.get(key) for key in DECISION_BEARING
    }
    if not report.decisions_unchanged:
        raise ProfileMigrationRefused(
            "migration refusée : un champ portant une décision a changé"
        )
    return migrated, report


#: Ce qu'on inscrit quand l'empreinte antérieure n'est pas connue de l'appelant.
DIGEST_NOT_RECOMPUTABLE = "non recalculable — schéma antérieur"


def migrate_file(
    path: Path, country_code: str, timezone_name: str,
    ocr_languages: list[str] | None = None, subdivision_code: str | None = None,
    digest_before: str | None = None,
) -> tuple[PropertyProfile, ProfileMigrationReport]:
    """Migre le profil, valide le résultat, et rend les deux empreintes.

    `digest_before` est **fourni**, non recalculé. L'empreinte antérieure était
    celle du dump complet du modèle d'alors ; le modèle ayant changé, la
    reproduire demanderait de réimplémenter l'ancien schéma, et une
    reconstruction approchée serait pire qu'une absence — elle porterait
    l'autorité d'une empreinte sans en avoir la valeur. On la constate donc
    dans les rapports publiés, ou on déclare qu'on ne la connaît pas.
    """
    payload = json.loads(path.read_text("utf-8"))

    migrated, report = migrate_payload(
        payload, country_code, timezone_name, ocr_languages, subdivision_code
    )
    profile = PropertyProfile.model_validate(migrated)

    report.digest_before = digest_before or DIGEST_NOT_RECOMPUTABLE
    report.digest_after = profile_digest(profile)
    log.info(
        "profil %s migré : empreinte %s → %s",
        report.property_id, report.digest_before, report.digest_after,
    )
    return profile, report
