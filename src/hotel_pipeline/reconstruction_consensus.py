"""Consensus et sélection de reconstruction (Lot 2 — P3).

Ce module compare plusieurs `ReconstructionRun`, aligne leurs poses en
Sim(3) via l'algorithme d'Umeyama, et sélectionne la meilleure reconstruction
selon des critères quantitatifs. Il produit également un `ReconstructionConsensusReport`
et des entrées `CameraConsensusEntry` par image.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict, Field

from .geometry_align import (
    alignment_rmse,
    apply_sim3,
    umeyama_sim3,
)
from .schemas.reconstruction import (
    CameraConsensusEntry,
    ReconstructionConsensusReport,
    ReconstructionRun,
    SparseConsensusGate,
    ViewGraphManifest,
)
from .workspace import Workspace


#: Familie de solveur par backend (pour la porte A).
_SOLVER_FAMILY = {
    "colmap_incremental": "classical",
    "colmap_global": "classical",
    "gluemap": "hybrid",
    "mp_sfm": "classical",
    "mapanything": "feedforward",
    "vggt": "feedforward",
    "brush": "classical",
    "gsplat": "classical",
    "synthetic": "classical",
}


# ---------------------------------------------------------------------------
# Helpers géométriques
# ---------------------------------------------------------------------------


def resolve_model_dir(output_path: str | Path) -> Path:
    """Répertoire contenant `normalized/` pour un `output_path` de run.

    `output_path` désigne souvent le modèle lui-même (`.../sparse`), alors que
    `normalized/` est publié à côté, dans le répertoire du run. Tester
    seulement `is_dir()` ne corrige rien dans ce cas : `sparse/` **est** un
    répertoire, si bien qu'on cherchait `sparse/normalized/images` et qu'on
    retournait zéro caméra en silence — portes de stabilité, novel-view et
    consensus comprises.

    On remonte donc jusqu'au premier ancêtre portant `normalized/`.
    """
    path = Path(output_path)
    if not path.exists():
        # Un chemin inexistant ne doit pas être « rattrapé » par son parent :
        # on retournerait les caméras d'un autre modèle.
        return path
    if not path.is_dir():
        path = path.parent
    candidate = path
    for _ in range(3):
        if (candidate / "normalized" / "images").is_file():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return path


def _is_pose_line(line: str) -> bool:
    """Une ligne de pose finit par `CAMERA_ID NAME` ; une ligne d'observations
    n'est faite que de nombres."""
    parts = line.split()
    if len(parts) < 10:
        return False
    try:
        float(parts[-1])
    except ValueError:
        return True
    return False


def _load_colmap_camera_centers(run_dir: Path) -> dict[str, np.ndarray]:
    """Charge les centres de caméra depuis un modèle COLMAP normalisé.

    Retourne {asset_id: center_3d} où asset_id est dérivé du nom de fichier
    sans extension.
    """
    run_dir = resolve_model_dir(run_dir)
    images_file = run_dir / "normalized" / "images"
    if not images_file.is_file():
        return {}

    # COLMAP écrit **deux** lignes par image : la pose, puis les
    # observations « X Y POINT3D_ID … ». Lire toutes les lignes prenait ces
    # observations pour des poses — autant de caméras fantômes.
    _lines = [
        line for line in images_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if any(not _is_pose_line(line) for line in _lines):
        _lines = _lines[::2]

    centers: dict[str, np.ndarray] = {}
    for line in _lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9] if len(parts) > 9 else parts[8]

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])
        t = np.array([tx, ty, tz])
        center = -R.T @ t
        asset_id = Path(name).stem
        centers[asset_id] = center
    return centers


def _umeyama_sim3(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sim(3) amenant `source` sur `target`. Délègue à `geometry_align`.

    Conservé comme alias pour ne pas casser les appels existants ; la seule
    implémentation vit désormais dans `geometry_align.umeyama_sim3`.
    """
    return umeyama_sim3(source, target)


def _apply_sim3(points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    return apply_sim3(points, R, t, s)


def _alignment_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return alignment_rmse(a, b)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ConsensusBuilder:
    """Construit un `ReconstructionConsensusReport` depuis plusieurs runs."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def build(self, run_ids: list[str]) -> ReconstructionConsensusReport:
        runs = [self._load_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None and r.status == "completed"]
        if len(runs) < 2:
            raise ValueError("au moins deux runs complétés sont nécessaires pour un consensus")

        # Charger les centres de caméra pour chaque run
        run_centers = []
        for r in runs:
            centers = self._load_run_centers(r)
            run_centers.append(centers)

        pairwise = self._pairwise_alignment_errors(runs, run_centers)
        camera_consensus = self._camera_consensus(runs, run_centers)

        consensus_id = (
            f"consensus-{runs[0].reconstruction_input_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        selected = self._select_best_run(runs, pairwise, camera_consensus)

        return ReconstructionConsensusReport(
            consensus_id=consensus_id,
            reconstruction_input_id=runs[0].reconstruction_input_id,
            run_ids=[r.run_id for r in runs],
            pairwise_alignment_errors=pairwise,
            camera_consensus=camera_consensus,
            selected_run_id=selected.run_id if selected else None,
            selection_rationale=self._selection_rationale(selected, runs, pairwise) if selected else None,
        )

    def _load_run(self, run_id: str) -> ReconstructionRun | None:
        path = self.workspace.path("07_reconstruction", "runs", f"{run_id}.json")
        if not path.is_file():
            return None
        try:
            return ReconstructionRun.model_validate_json(path.read_text("utf-8"))
        except Exception:
            return None

    def _load_run_centers(self, run: ReconstructionRun) -> dict[str, np.ndarray]:
        if not run.output_path:
            return {}
        run_dir = resolve_model_dir(run.output_path)

        normalized_dir = run_dir / "normalized"
        if not normalized_dir.is_dir():
            normalized_dir = run_dir.parent / "normalized"
        if not normalized_dir.is_dir():
            normalized_dir = run_dir.parent.parent / "normalized"
        if not normalized_dir.is_dir():
            return {}
        return _load_colmap_camera_centers(normalized_dir.parent)

    def _pairwise_alignment_errors(
        self,
        runs: list[ReconstructionRun],
        run_centers: list[dict[str, np.ndarray]],
    ) -> dict[str, float]:
        """Estime l'erreur d'alignement Sim(3) entre chaque paire de runs."""
        errors: dict[str, float] = {}
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                key = f"{runs[i].run_id}__{runs[j].run_id}"
                centers_i = run_centers[i]
                centers_j = run_centers[j]
                common = sorted(set(centers_i) & set(centers_j))
                if len(common) < 3:
                    mi = runs[i].metrics.get("alignment_rmse_m", 1.0)
                    mj = runs[j].metrics.get("alignment_rmse_m", 1.0)
                    errors[key] = round(abs(mi - mj), 4) if isinstance(mi, (int, float)) and isinstance(mj, (int, float)) else 1.0
                    continue

                X = np.array([centers_i[k] for k in common])
                Y = np.array([centers_j[k] for k in common])
                # On amène Y sur X : la source est le premier argument.
                R, t, s = umeyama_sim3(Y, X)
                Y_aligned = apply_sim3(Y, R, t, s)
                errors[key] = round(alignment_rmse(Y_aligned, X), 4)
        return errors

    def _camera_consensus(
        self,
        runs: list[ReconstructionRun],
        run_centers: list[dict[str, np.ndarray]],
    ) -> list[CameraConsensusEntry]:
        """Construit le consensus par image avec alignement Sim(3) réel."""
        entries: list[CameraConsensusEntry] = []
        all_asset_ids: set[str] = set()
        for centers in run_centers:
            all_asset_ids.update(centers.keys())

        if not all_asset_ids:
            return entries

        # Référence = premier run complété avec centres
        ref_idx = 0
        ref_centers = run_centers[ref_idx]
        ref_run = runs[ref_idx]

        for asset_id in sorted(all_asset_ids):
            backends = []
            spreads_t = []
            spreads_r = []
            spreads_f = []
            aberrants = []

            aligned_centers = []
            for r, centers in zip(runs, run_centers):
                if asset_id not in centers:
                    continue
                backends.append(r.backend)
                c = centers[asset_id].reshape(1, 3)

                if r.run_id == ref_run.run_id:
                    R, t, s = np.eye(3), np.zeros(3), 1.0
                else:
                    common = sorted(set(ref_centers) & set(centers))
                    if len(common) < 3:
                        R, t, s = np.eye(3), np.zeros(3), 1.0
                    else:
                        X = np.array([ref_centers[k] for k in common])
                        Y = np.array([centers[k] for k in common])
                        # Source = ce run, target = la référence.
                        R, t, s = umeyama_sim3(Y, X)

                aligned = apply_sim3(c, R, t, s).flatten()
                aligned_centers.append(aligned)

            if len(aligned_centers) >= 2:
                arr = np.array(aligned_centers)
                median_c = np.median(arr, axis=0)
                spreads_t = [float(np.linalg.norm(c - median_c)) for c in aligned_centers]
                # Approximation rotation spread via axes variance (MVP)
                spreads_r = [0.0] * len(aligned_centers)

            confidence = "none"
            if len(backends) >= 3:
                confidence = "high"
            elif len(backends) == 2:
                confidence = "medium"

            entries.append(CameraConsensusEntry(
                asset_id=asset_id,
                backends=backends,
                translation_spread_m=round(float(np.mean(spreads_t)), 3) if spreads_t else 0.0,
                rotation_spread_deg=round(float(np.mean(spreads_r)), 3) if spreads_r else 0.0,
                focal_spread_px=0.0,
                confidence=confidence,
                aberrants=aberrants,
            ))

        return entries

    @staticmethod
    def _select_best_run(
        runs: list[ReconstructionRun],
        pairwise: dict[str, float],
        consensus: list[CameraConsensusEntry],
    ) -> ReconstructionRun | None:
        """Sélectionne le meilleur run selon les métriques."""
        scored = []
        for r in runs:
            m = r.metrics if isinstance(r.metrics, dict) else {}
            registered = m.get("registered_ratio", 0.0)
            error = m.get("alignment_rmse_m", 1.0)
            score = registered - error
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    @staticmethod
    def _selection_rationale(
        selected: ReconstructionRun,
        runs: list[ReconstructionRun],
        pairwise: dict[str, float],
    ) -> str:
        m = selected.metrics if isinstance(selected.metrics, dict) else {}
        return (
            f"run {selected.run_id} sélectionné : "
            f"registered_ratio={m.get('registered_ratio', 0):.2f}, "
            f"alignment_rmse={m.get('alignment_rmse_m', 0):.3f}m"
        )


def publish_consensus(
    report: ReconstructionConsensusReport,
    workspace: Workspace,
) -> Path:
    """Publie le rapport de consensus sous `07_reconstruction/consensus/`."""
    path = workspace.path("07_reconstruction", "consensus", f"{report.consensus_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path


def _load_colmap_points3d(run_dir: Path) -> list[tuple[float, float]]:
    """Charge (track_length, error_px) depuis points3D.txt normalisé."""
    p = run_dir / "normalized" / "points3D"
    if not p.is_file():
        return []
    out: list[tuple[float, float]] = []
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            out.append((float(parts[6]), float(parts[7])))
        except ValueError:
            continue
    return out


def _load_view_graph(workspace: Workspace) -> ViewGraphManifest | None:
    vg_dir = workspace.path("07_reconstruction", "view_graphs")
    if not vg_dir.is_dir():
        return None
    files = sorted(vg_dir.glob("vg-*.json"))
    if not files:
        return None
    try:
        return ViewGraphManifest.model_validate_json(files[-1].read_text("utf-8"))
    except Exception:
        return None


def _estimate_image_diagonal(workspace: Workspace) -> float | None:
    """Médiane des diagonales d'image depuis le manifeste d'assets."""
    try:
        from .schemas import AssetManifest

        assets = AssetManifest.model_validate_json(
            workspace.assets_path.read_text("utf-8")
        )
        diags = []
        for a in assets.assets:
            if a.width and a.height:
                diags.append(float((a.width**2 + a.height**2) ** 0.5))
        if diags:
            return float(np.median(diags))
    except Exception:
        pass
    return None


def build_sparse_consensus_gate(workspace: Workspace, run: ReconstructionRun) -> SparseConsensusGate:
    """Construit la Porte A (consensus caméra/pose) depuis un run de reconstruction.

    Combine l'enregistrement COLMAP (centres de caméra), le ViewGraph (composante
    maximale, qualité des intrinsèques, ratios d'inliers) et les statistiques
    points3D (erreur de reprojection, longueur de piste). Avec un seul run, une
    seule famille de solveur est représentée : `independent_families_agreeing=1`.
    """
    run_dir = Path(run.output_path) if run.output_path else None
    if run_dir and run_dir.is_dir():
        norm = run_dir / "normalized"
        if not norm.is_dir():
            norm = run_dir.parent
    else:
        norm = None

    centers = _load_colmap_camera_centers(norm) if norm else {}
    registered = len(centers)

    vg = _load_view_graph(workspace)
    if vg is not None:
        largest_component = vg.report.largest_component
        intrinsics_quality = vg.report.intrinsics_quality
        valid_inliers = [p.inlier_ratio for p in vg.pairs if p.status == "valid"]
        inlier_ratio_median = float(np.median(valid_inliers)) if valid_inliers else 0.0
    else:
        largest_component = max(registered, 1)
        intrinsics_quality = "poor"
        inlier_ratio_median = 0.0

    pts = _load_colmap_points3d(norm) if norm else []
    if pts:
        track_lengths = [t for t, _ in pts]
        errors = [e for _, e in pts]
        median_reprojection_px = float(np.median(errors))
        track_length_median = float(np.median(track_lengths))
        image_diag = _estimate_image_diagonal(workspace) or 1000.0
        median_reprojection_normalized = round(median_reprojection_px / image_diag, 4)
    else:
        median_reprojection_px = 0.0
        track_length_median = 0.0
        median_reprojection_normalized = 0.0

    selected = registered
    try:
        from .reconstruction_input import prepare_input

        im, _ = prepare_input(workspace.hotel_id)
        selected = len(im.selected_asset_ids)
    except Exception:
        pass
    raw_registration_rate = round(registered / selected, 3) if selected else 0.0

    localization = None
    localization_dir = workspace.path("07_reconstruction", "localization")
    if localization_dir.is_dir():
        from .schemas import LocalizationManifest

        for path in sorted(localization_dir.glob("*.json"), reverse=True):
            try:
                candidate = LocalizationManifest.model_validate_json(path.read_text("utf-8"))
            except Exception:
                continue
            if candidate.reconstruction_input_id == run.reconstruction_input_id:
                localization = candidate
                break

    if localization is not None:
        registration_rate = localization.validated_registration_rate
        validated_component = localization.validated_main_component_ratio
        anchor_images = localization.measured_anchor_images
        localized_images = localization.measured_localized_images
        inferred_images = localization.inferred_images
        rejected_images = localization.rejected_images
        localization_id = localization.localization_run_id
        anchor_model_ready = False
        anchor_manifest = workspace.path(
            "07_reconstruction", "anchors", f"{localization.anchor_model_id}.json"
        )
        if anchor_manifest.is_file():
            try:
                from .schemas import AnchorModelManifest

                anchor_model_ready = (
                    AnchorModelManifest.model_validate_json(
                        anchor_manifest.read_text("utf-8")
                    ).status
                    == "ready"
                )
            except Exception:
                anchor_model_ready = False
        external_consistency = (
            anchor_model_ready
            and localization.measured_anchor_images >= localization.policy.min_anchor_images
        )
        largest_component = round(
            validated_component * (anchor_images + localized_images)
        )
    else:
        # Une inscription dans le modèle brut n'est plus comptée comme pose
        # validée. L'absence de preuve reste un zéro explicite.
        registration_rate = 0.0
        validated_component = 0.0
        anchor_images = localized_images = inferred_images = rejected_images = 0
        localization_id = None
        external_consistency = False
        largest_component = 0

    solver_families = [_SOLVER_FAMILY.get(getattr(run, "backend", ""), "classical")]

    return SparseConsensusGate(
        raw_registration_rate=raw_registration_rate,
        raw_registered_images=registered,
        registration_rate=registration_rate,
        validated_registration_rate=registration_rate,
        measured_anchor_images=anchor_images,
        measured_localized_images=localized_images,
        inferred_images=inferred_images,
        rejected_images=rejected_images,
        validated_main_component_ratio=validated_component,
        external_pose_consistency=external_consistency,
        localization_manifest_id=localization_id,
        largest_component_size=largest_component,
        median_reprojection_px=round(median_reprojection_px, 3),
        median_reprojection_normalized=median_reprojection_normalized,
        track_length_median=round(track_length_median, 3),
        inlier_ratio_median=round(inlier_ratio_median, 3),
        intrinsics_quality=intrinsics_quality,
        camera_consensus={},
        solver_families=solver_families,
        independent_families_agreeing=1,
    )


__all__ = [
    "ConsensusBuilder",
    "publish_consensus",
    "build_sparse_consensus_gate",
]
