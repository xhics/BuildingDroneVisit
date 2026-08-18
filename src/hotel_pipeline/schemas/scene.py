"""Contrats du paquet 3D destiné aux consommateurs vidéo.

Le paquet décrit ce qui est réellement mesuré et ce qui reste un proxy. Il ne
transforme jamais un export techniquement lisible en verdict
``ENVIRONMENT_3D_READY`` : ce verdict appartient au gate Phase 1.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Phase1Status(StrEnum):
    ENVIRONMENT_3D_READY = "ENVIRONMENT_3D_READY"
    NEEDS_AUTHORIZED_CAPTURE = "NEEDS_AUTHORIZED_CAPTURE"
    NEEDS_MANUAL_CORRECTION = "NEEDS_MANUAL_CORRECTION"
    GEO_FIRST_PROXY_ONLY = "GEO_FIRST_PROXY_ONLY"
    REJECTED_DATA_INSUFFICIENT = "REJECTED_DATA_INSUFFICIENT"


class GateState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class EvidenceClass(StrEnum):
    MEASURED = "measured"
    INFERRED = "inferred"
    PROXY = "proxy"


class GateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate_id: str
    requirement: str
    state: GateState
    evidence: list[str] = Field(min_length=1)


class Phase1Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: int = 1
    hotel_id: str
    generated_at: str
    status: Phase1Status
    router_decision_digest: str
    input_digests: dict[str, str]
    checks: list[GateCheck] = Field(min_length=1)
    blocking_reasons: list[str]
    human_review_approved: bool = False

    @model_validator(mode="after")
    def _ready_requires_every_gate_and_human_review(self) -> "Phase1Verdict":
        if self.status is Phase1Status.ENVIRONMENT_3D_READY:
            incomplete = [c.gate_id for c in self.checks if c.state is not GateState.PASSED]
            if incomplete:
                raise ValueError(
                    "ENVIRONMENT_3D_READY avec gates non franchis : "
                    + ", ".join(incomplete)
                )
            if not self.human_review_approved:
                raise ValueError("ENVIRONMENT_3D_READY exige une revue humaine explicite")
            if self.blocking_reasons:
                raise ValueError("ENVIRONMENT_3D_READY ne peut porter de blocage")
        return self


class PackageFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: str
    evidence_class: EvidenceClass
    source_refs: list[str] = Field(min_length=1)


class CameraPose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: int = Field(ge=0)
    position_local_m: tuple[float, float, float]
    look_at_local_m: tuple[float, float, float]
    azimuth_deg: float = Field(ge=0, lt=360)
    elevation_deg: float = Field(gt=-90, lt=90)
    distance_m: float = Field(gt=0)
    fov_horizontal_deg: float = Field(gt=0, le=120)

    #: Ce que la pose regarde, quand c'est connu — `FACADE_PRIMARY`…
    faces: str | None = None

    #: Vrai quand la pose cadre une surface dont **aucune apparence n'a été
    #: observée**. La pose reste dans le chemin : la retirer masquerait la
    #: lacune au lieu de la déclarer, et un consommateur ne saurait plus
    #: qu'un secteur entier n'a jamais été photographié.
    blind_field: bool = False


class VirtualCameraPath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str
    simulation_only: bool
    derivation: str
    poses: list[CameraPose] = Field(min_length=8)

    @model_validator(mode="after")
    def _never_masquer_un_chemin_virtuel(self) -> "VirtualCameraPath":
        if not self.simulation_only:
            raise ValueError(
                "une orbite synthétique ne vaut pas autorisation de capture physique"
            )
        if len({pose.frame for pose in self.poses}) != len(self.poses):
            raise ValueError("frames caméra dupliquées")
        return self


class ScenePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: int = 1
    hotel_id: str
    package_id: str
    generated_at: str
    status: str
    horizontal_crs: str
    vertical_datum: str
    local_origin_projected: tuple[float, float, float]
    units: str = "metres"
    input_digests: dict[str, str]
    phase1_verdict: str
    files: list[PackageFile] = Field(min_length=1)
    camera_paths: list[VirtualCameraPath] = Field(min_length=1)
    rights_summary: dict[str, int]
    forbidden_claims: list[str]
    blind_visual_fields: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(min_length=1)
    video_generation: dict

    @model_validator(mode="after")
    def _package_type_is_valid(self) -> "ScenePackage":
        allowed = {
            "hybrid_proxy_package",
            "reconstructed_photo_first",
            "reconstructed_hybrid",
        }
        if self.status not in allowed:
            raise ValueError(f"status de paquet invalide : {self.status}")
        paths = [row.path for row in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("fichier dupliqué dans le paquet")
        if self.video_generation.get("real_provider_call_performed") is not False:
            raise ValueError("le paquet ne doit pas prétendre avoir appelé un fournisseur")
        return self
