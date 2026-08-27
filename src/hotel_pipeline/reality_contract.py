"""Strict One Reality Model contract shared by every geometry consumer."""

from __future__ import annotations

from dataclasses import dataclass


class RealityContractError(RuntimeError):
    """Raised when production code attempts to bypass CanonicalSceneMesh."""


@dataclass(frozen=True)
class MeshConsumerReceipt:
    consumer: str
    input_mesh_digest: str
    legacy_geometry_paths_used: int = 0

    def as_dict(self) -> dict:
        return {
            "consumer": self.consumer,
            "input_mesh_digest": self.input_mesh_digest,
            "legacy_geometry_paths_used": self.legacy_geometry_paths_used,
        }


def require_canonical_mesh(mesh, consumer: str):  # noqa: ANN001
    from .conditioning.canonical_mesh import CanonicalSceneMesh

    if not isinstance(mesh, CanonicalSceneMesh):
        raise RealityContractError(f"{consumer}: CanonicalSceneMesh required")
    mesh.validate_triangle_metadata()
    return MeshConsumerReceipt(consumer, mesh.mesh_digest())


def audit_consumer_receipts(canonical_digest: str, receipts: list[MeshConsumerReceipt]) -> dict:
    rows = [receipt.as_dict() for receipt in receipts]
    passed = all(
        row["input_mesh_digest"] == canonical_digest
        and row["legacy_geometry_paths_used"] == 0
        for row in rows
    )
    return {
        "canonical_mesh_digest": canonical_digest,
        "consumers": rows,
        "legacy_geometry_paths_used": sum(row["legacy_geometry_paths_used"] for row in rows),
        "passed": passed,
    }


__all__ = [
    "MeshConsumerReceipt", "RealityContractError", "audit_consumer_receipts",
    "require_canonical_mesh",
]
