"""Registre canonique et factuel des familles photographiques du Lot 1B.

Le registre ne lance aucune collecte. Il confronte les familles exigées par le
plan aux preuves déjà publiées et distingue une interrogation courante, un
stock historique, une indisponibilité documentée et une absence de preuve.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


SOURCE_REGISTRY_CONTRACT_VERSION = 1


class SourceFamilyState(StrEnum):
    QUERIED_CURRENT = "queried_current"
    EVIDENCE_PRESENT = "evidence_present"
    UNAVAILABLE_DOCUMENTED = "unavailable_documented"
    PENDING_MANUAL = "pending_manual"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_EVIDENCED = "not_evidenced"


class SourceFamilyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str
    priority: str
    required_for_campaign: bool = True
    collector_id: str | None = None
    state: SourceFamilyState
    asset_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    evidence: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)
    campaign_closed: bool

    @model_validator(mode="after")
    def _closure_requires_a_terminal_state(self) -> "SourceFamilyRecord":
        terminal = {
            SourceFamilyState.QUERIED_CURRENT,
            SourceFamilyState.UNAVAILABLE_DOCUMENTED,
        }
        if self.campaign_closed != (self.state in terminal):
            raise ValueError("campaign_closed contredit l'état de la famille")
        return self


class SourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: int = SOURCE_REGISTRY_CONTRACT_VERSION
    hotel_id: str
    generated_at: str
    execution_mode: str = "local_only"
    network_requests: int = Field(default=0, ge=0)
    input_digests: dict[str, str]
    families: list[SourceFamilyRecord] = Field(min_length=1)
    required_families: int = Field(ge=1)
    closed_families: int = Field(ge=0)
    closure_complete: bool

    @model_validator(mode="after")
    def _counts_and_ids_are_closed(self) -> "SourceRegistry":
        ids = [row.family_id for row in self.families]
        if len(ids) != len(set(ids)):
            raise ValueError("famille source dupliquée")
        required = [row for row in self.families if row.required_for_campaign]
        closed = [row for row in required if row.campaign_closed]
        if self.required_families != len(required):
            raise ValueError("required_families contredit les familles")
        if self.closed_families != len(closed):
            raise ValueError("closed_families contredit les familles")
        if self.closure_complete != (len(required) == len(closed)):
            raise ValueError("closure_complete contredit les familles")
        required_inputs = {"asset_manifest", "candidate_manifest", "lot_1b_plan"}
        if set(self.input_digests) != required_inputs:
            raise ValueError("empreintes du registre de sources incomplètes")
        return self


# Ce catalogue est le reflet exécutable du §8 du plan Lot 1B. L'absence d'un
# collecteur reste visible comme NOT_IMPLEMENTED ; elle n'efface pas la famille.
_FAMILIES = (
    ("tripadvisor_traveler", "A", "tripadvisor", True),
    ("social_official", "A", None, True),
    ("hotel_project_team", "A", None, True),
    ("iceportal", "B", None, True),
    ("booking", "B", None, True),
    ("expedia_media", "B", None, True),
    ("foursquare", "B", None, True),
    ("flickr", "C", "flickr", True),
    ("google_places", "C", "places", True),
    ("hotel_website", "C", "website", True),
    ("discovery_directories", "C", None, True),
    ("yelp_apple_bing", "C", None, True),
    ("mapillary", "STREET_OPEN", "mapillary", True),
    ("kartaview", "STREET_OPEN", "kartaview", True),
    ("panoramax", "MONITOR", None, False),
    ("wikimedia_commons", "MONITOR", "commons", False),
    ("street_view", "STREET", "street_view", True),
)


#: Reçus d'indisponibilité, append-only. Un reçu retiré laisse sa trace : le
#: registre doit pouvoir expliquer pourquoi une famille a cessé d'être close.
UNAVAILABILITY_RECEIPTS = "00_manifest/source_unavailability.json"

_FAMILY_IDS = frozenset(row[0] for row in _FAMILIES)


class UnavailabilityReceipt(BaseModel):
    """Constat qu'une famille ne peut pas être interrogée, et pourquoi.

    Un reçu documente une **indisponibilité observée**, jamais une famille que
    l'on a simplement choisi de ne pas interroger : sans cette distinction, la
    campagne se clôturerait en déclarant absente toute source coûteuse.
    """

    model_config = ConfigDict(extra="forbid")

    family_id: str
    reason: str = Field(min_length=1)
    recorded_by: str = Field(min_length=1)
    recorded_at: str
    withdrawn: bool = False

    @model_validator(mode="after")
    def _family_is_known(self) -> "UnavailabilityReceipt":
        if self.family_id not in _FAMILY_IDS:
            raise ValueError(f"famille inconnue : {self.family_id}")
        return self


#: Reçus de campagne : une famille réellement interrogée hors découverte
#: ciblée. Le registre ne lisait que les manifestes de candidats, or seule la
#: découverte ciblée en produit — un collecteur exécuté directement ne laissait
#: donc aucune trace, et sa famille restait indéfiniment ouverte.
CAMPAIGN_RECEIPTS = "00_manifest/source_campaigns.json"


class CampaignReceipt(BaseModel):
    """Constat qu'une famille a été interrogée, et ce qu'elle a rendu.

    `returned` peut valoir zéro : une source interrogée qui ne rend rien est
    close tout de même. C'est l'interrogation qui ferme la campagne, pas la
    moisson — confondre les deux rouvrirait toute source pauvre.
    """

    model_config = ConfigDict(extra="forbid")

    family_id: str
    query: str = Field(min_length=1)
    returned: int = Field(ge=0)
    evidence: str = Field(min_length=1)
    recorded_by: str = Field(min_length=1)
    recorded_at: str

    @model_validator(mode="after")
    def _family_is_known(self) -> "CampaignReceipt":
        if self.family_id not in _FAMILY_IDS:
            raise ValueError(f"famille inconnue : {self.family_id}")
        return self


def _receipts(workspace) -> list[UnavailabilityReceipt]:  # noqa: ANN001
    raw = workspace.read_json(UNAVAILABILITY_RECEIPTS)
    if raw is None:
        return []
    return [UnavailabilityReceipt.model_validate(row) for row in raw]


def _campaigns(workspace) -> list[CampaignReceipt]:  # noqa: ANN001
    raw = workspace.read_json(CAMPAIGN_RECEIPTS)
    if raw is None:
        return []
    return [CampaignReceipt.model_validate(row) for row in raw]


def record_campaign(  # noqa: ANN001
    workspace, family_id: str, query: str, returned: int, evidence: str, by: str
) -> Path:
    """Consigne qu'une famille a été interrogée, sans réécrire les précédents."""
    receipt = CampaignReceipt(
        family_id=family_id,
        query=query,
        returned=returned,
        evidence=evidence,
        recorded_by=by,
        recorded_at=datetime.now(timezone.utc).isoformat(),
    )
    history = _campaigns(workspace)
    history.append(receipt)
    return workspace.write_json(
        CAMPAIGN_RECEIPTS, [row.model_dump(mode="json") for row in history]
    )


def record_unavailable(workspace, family_id: str, reason: str, by: str) -> Path:  # noqa: ANN001
    """Consigne un reçu d'indisponibilité, sans réécrire les précédents."""
    receipt = UnavailabilityReceipt(
        family_id=family_id,
        reason=reason,
        recorded_by=by,
        recorded_at=datetime.now(timezone.utc).isoformat(),
    )
    history = _receipts(workspace)
    history.append(receipt)
    return workspace.write_json(
        UNAVAILABILITY_RECEIPTS, [row.model_dump(mode="json") for row in history]
    )


def withdraw_unavailable(workspace, family_id: str, by: str, reason: str) -> Path:  # noqa: ANN001
    """Retire un reçu devenu faux, en conservant l'historique.

    Le retrait est lui-même un enregistrement : une famille redevenue
    interrogeable doit rouvrir la campagne, pas disparaître du registre.
    """
    history = _receipts(workspace)
    active = [row for row in history if row.family_id == family_id and not row.withdrawn]
    if not active:
        raise ValueError(f"aucun reçu actif pour {family_id}")
    for row in active:
        row.withdrawn = True
    history.append(UnavailabilityReceipt(
        family_id=family_id,
        reason=f"retrait : {reason}",
        recorded_by=by,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        withdrawn=True,
    ))
    return workspace.write_json(
        UNAVAILABILITY_RECEIPTS, [row.model_dump(mode="json") for row in history]
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_candidates(workspace) -> Path:  # noqa: ANN001
    candidates = sorted(
        workspace.path("01_sources").glob("candidates_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("aucun manifeste canonique de candidats")
    return candidates[-1]


def build(workspace) -> Path:  # noqa: ANN001
    assets = workspace.read_assets()
    if assets is None:
        raise FileNotFoundError("manifeste d'assets absent")
    candidates_path = _current_candidates(workspace)
    plan_path = Path(__file__).resolve().parents[2] / "PLAN_LOT_1B_WELCOMINNS.md"
    if not plan_path.is_file():
        raise FileNotFoundError("plan Lot 1B absent")

    candidates = json.loads(candidates_path.read_text("utf-8"))
    asset_counts = Counter(str(asset.source) for asset in assets.assets)
    candidate_counts = Counter(
        str(row.get("source")) for row in candidates.get("candidates", [])
    )
    current_candidate_sources = {
        source for source, count in candidate_counts.items() if count > 0
    }

    campaigns = {row.family_id: row for row in _campaigns(workspace)}

    documented = {
        row.family_id for row in _receipts(workspace) if not row.withdrawn
    }
    receipt_reasons = {
        row.family_id: row.reason
        for row in _receipts(workspace)
        if not row.withdrawn
    }

    records: list[SourceFamilyRecord] = []
    for family_id, priority, collector_id, required in _FAMILIES:
        asset_count = asset_counts.get(collector_id or "", 0)
        candidate_count = candidate_counts.get(collector_id or "", 0)
        if collector_id in current_candidate_sources:
            state = SourceFamilyState.QUERIED_CURRENT
            evidence = [f"01_sources/{candidates_path.name}"]
            reason = "candidats présents dans le manifeste canonique courant"
        elif family_id in campaigns:
            # Une interrogation réelle vaut celle de la découverte ciblée : la
            # trace diffère, le fait est le même.
            receipt = campaigns[family_id]
            state = SourceFamilyState.QUERIED_CURRENT
            evidence = [CAMPAIGN_RECEIPTS, receipt.evidence]
            reason = (
                f"famille interrogée : {receipt.returned} résultat(s) pour "
                f"{receipt.query!r}"
            )
        elif family_id in documented:
            # Après l'interrogation courante : une famille réellement interrogée
            # ne doit jamais être déclarée indisponible par un reçu périmé.
            state = SourceFamilyState.UNAVAILABLE_DOCUMENTED
            evidence = [UNAVAILABILITY_RECEIPTS]
            reason = receipt_reasons[family_id]
        elif asset_count:
            state = SourceFamilyState.EVIDENCE_PRESENT
            evidence = ["00_manifest/asset_manifest.json"]
            reason = "assets présents, sans preuve d'interrogation de campagne courante"
        elif family_id == "hotel_project_team":
            state = SourceFamilyState.PENDING_MANUAL
            evidence = ["PLAN_LOT_1B_WELCOMINNS.md §8 priorité A"]
            reason = "demande directe à l'hôtel, au dossier municipal ou aux intervenants non consignée"
        elif collector_id is None:
            state = SourceFamilyState.NOT_IMPLEMENTED
            evidence = ["PLAN_LOT_1B_WELCOMINNS.md §8"]
            reason = "aucun collecteur ni reçu d'indisponibilité canonique"
        else:
            state = SourceFamilyState.NOT_EVIDENCED
            evidence = ["01_sources/gather_report.json"]
            reason = "collecteur présent, mais aucune interrogation ou indisponibilité exploitable n'est consignée"
        records.append(SourceFamilyRecord(
            family_id=family_id,
            priority=priority,
            required_for_campaign=required,
            collector_id=collector_id,
            state=state,
            asset_count=asset_count,
            candidate_count=candidate_count,
            evidence=evidence,
            reason=reason,
            campaign_closed=state in {
                SourceFamilyState.QUERIED_CURRENT,
                SourceFamilyState.UNAVAILABLE_DOCUMENTED,
            },
        ))

    required_rows = [row for row in records if row.required_for_campaign]
    closed_rows = [row for row in required_rows if row.campaign_closed]
    registry = SourceRegistry(
        hotel_id=workspace.hotel_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        input_digests={
            "asset_manifest": _sha256(workspace.assets_path),
            "candidate_manifest": _sha256(candidates_path),
            "lot_1b_plan": _sha256(plan_path),
        },
        families=records,
        required_families=len(required_rows),
        closed_families=len(closed_rows),
        closure_complete=len(required_rows) == len(closed_rows),
    )
    return workspace.write_json(
        "00_manifest/source_registry.json", registry.model_dump(mode="json")
    )

