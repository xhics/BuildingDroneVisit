"""Exécution des reconstructions SfM (Lot 2 — P2).

Ce module fournit :
- `ReconstructionRunner` : exécute un backend sur un `ReconstructionInputManifest`
- adapters COLMAP incremental et global
- schéma commun `ReconstructionRun`

Les backends exportent vers une représentation normalisée (caméras, points,
images) pour que Brush, gsplat et Blender ne dépendent pas du solveur.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import ReconstructionBackend, ReconstructionRun
from .workspace import Workspace


class ReconstructionRunner:
    """Exécute un backend de reconstruction sur un snapshot gelé."""

    def __init__(self, workspace: Workspace, *, colmap_binary: str = "colmap"):
        self.workspace = workspace
        self.colmap_binary = colmap_binary

    def run(
        self,
        input_manifest: ReconstructionInputManifest,
        backend: ReconstructionBackend = ReconstructionBackend.COLMAP_INCREMENTAL,
        *,
        selected_asset_ids: list[str] | None = None,
    ) -> ReconstructionRun:
        """Exécute la reconstruction et retourne un `ReconstructionRun`.

        Pour le MVP, seul COLMAP incremental est implémenté. Les autres
        backends sont des placeholders qui journalisent l'intention.

        Args:
            input_manifest: manifeste d'entrée gelé
            backend: backend de reconstruction
            selected_asset_ids: override optionnel des assets sélectionnés
                (utilisé pour les tests de cohorte temporelle)
        """
        run_id = _new_run_id(backend.value, input_manifest.reconstruction_input_id)
        started = datetime.now(timezone.utc).isoformat()

        if backend is ReconstructionBackend.COLMAP_INCREMENTAL:
            result = _run_colmap_incremental(self, input_manifest, run_id, selected_asset_ids=selected_asset_ids)
        elif backend is ReconstructionBackend.COLMAP_GLOBAL:
            result = _run_colmap_global(self, input_manifest, run_id, selected_asset_ids=selected_asset_ids)
        elif backend is ReconstructionBackend.MAP_ANYTHING:
            result = _run_map_anything(self, input_manifest, run_id, selected_asset_ids=selected_asset_ids)
        elif backend is ReconstructionBackend.VGGT:
            result = _run_vggt(self, input_manifest, run_id, selected_asset_ids=selected_asset_ids)
        elif backend is ReconstructionBackend.GLUEMAP:
            result = _run_gluemap(self, input_manifest, run_id, selected_asset_ids=selected_asset_ids)
        elif backend is ReconstructionBackend.MP_SFM:
            result = _run_mpsfm(self, input_manifest, run_id, selected_asset_ids=selected_asset_ids)
        else:
            result = ReconstructionRun(
                run_id=run_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id,
                backend=backend.value,
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"backend {backend.value} non implémenté dans cette phase",
            )

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_run_id(backend: str, input_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{backend}-{input_id}-{stamp}"


def _workspace_root(workspace: Workspace) -> Path:
    return workspace.path("07_reconstruction")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _setup_run_images(
    runner: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_dir: Path,
    *,
    selected_asset_ids: list[str] | None = None,
) -> tuple[list[str], dict, list[str]]:
    """Prépare le répertoire d'images et retourne la liste des chemins copiés."""
    from .schemas import AssetManifest

    assets = AssetManifest.model_validate_json(
        runner.workspace.assets_path.read_text("utf-8")
    )
    by_id = {a.id: a for a in assets.assets}

    if selected_asset_ids is not None:
        selected = [aid for aid in selected_asset_ids if aid in by_id]
    else:
        selected = [aid for aid in input_manifest.selected_asset_ids if aid in by_id]
    if not selected:
        raise ValueError("aucun asset sélectionné pour la reconstruction")

    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for aid in selected:
        asset = by_id[aid]
        if asset.local_path:
            src = runner.workspace.path(asset.local_path)
            if src.is_file():
                dst = image_dir / src.name
                if not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
                image_paths.append(str(dst))

    if not image_paths:
        raise ValueError("aucune image accessible pour la reconstruction")
    return selected, by_id, image_paths


def _run_colmap_feature_extractor(
    runner: ReconstructionRunner,
    image_paths: list[str],
    db_path: Path,
) -> None:
    image_dir = db_path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for src in image_paths:
        dst = image_dir / Path(src).name
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)

    cmd = [
        runner.colmap_binary,
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=False)


def _run_colmap_exhaustive_matcher(runner: ReconstructionRunner, db_path: Path) -> None:
    cmd = [
        runner.colmap_binary,
        "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "0",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=False)


def _export_colmap_normalized(sparse_model: Path, run_dir: Path, selected: list[str], by_id: dict) -> None:
    """Exporte cameras, images, points3D vers un format normalisé."""
    import shutil
    out = run_dir / "normalized"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("cameras", "images", "points3D"):
        src = sparse_model / name
        if src.exists():
            shutil.copy2(src, out / name)


def _parse_colmap_metrics(sparse_model: Path | None, selected: list[str], by_id: dict) -> dict:
    if sparse_model is None or not sparse_model.exists():
        return {"registered_ratio": 0.0}

    images_file = sparse_model / "images"
    points3d_file = sparse_model / "points3D"
    if not images_file.exists():
        return {"registered_ratio": 0.0}

    lines = images_file.read_text().splitlines()
    registered = 0
    track_lengths: list[int] = []
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            registered += 1

    # Parser points3D pour erreurs de reprojection et longueurs de tracks
    reproj_errors: list[float] = []
    if points3d_file.exists():
        for line in points3d_file.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 8:
                reproj_errors.append(float(parts[7]))
                # Track length = nombre de paires IMAGE_ID POINT2D_IDX
                track_len = max(0, (len(parts) - 8)) // 2
                if track_len > 0:
                    track_lengths.append(track_len)

    metrics = {
        "registered_ratio": round(registered / len(selected), 3) if selected else 0.0,
        "registered_images": registered,
        "selected_images": len(selected),
    }
    if reproj_errors:
        metrics["median_reprojection_error"] = round(float(np.median(reproj_errors)), 4)
        metrics["mean_reprojection_error"] = round(float(np.mean(reproj_errors)), 4)
        metrics["max_reprojection_error"] = round(float(np.max(reproj_errors)), 4)
    if track_lengths:
        metrics["median_track_length"] = round(float(np.median(track_lengths)), 2)
        metrics["mean_track_length"] = round(float(np.mean(track_lengths)), 2)
        metrics["max_track_length"] = int(np.max(track_lengths))
    return metrics


# ---------------------------------------------------------------------------
# COLMAP incremental
# ---------------------------------------------------------------------------


def _run_colmap_incremental(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    workspace = self.workspace
    root = _workspace_root(workspace)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "database.db"

    started = datetime.now(timezone.utc).isoformat()

    try:
        selected, by_id, image_paths = _setup_run_images(self, input_manifest, run_dir, selected_asset_ids=selected_asset_ids)
        _run_colmap_feature_extractor(self, image_paths, db_path)
        _run_colmap_exhaustive_matcher(self, db_path)
        sparse_dir = run_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.colmap_binary,
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(run_dir / "images"),
            "--output_path", str(sparse_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        sparse_model = None
        if (sparse_dir / "0").exists():
            sparse_model = sparse_dir / "0"
            _export_colmap_normalized(sparse_model, run_dir, selected, by_id)

        finished = datetime.now(timezone.utc).isoformat()
        metrics = _parse_colmap_metrics(sparse_model, selected, by_id)
        if proc.returncode != 0 and not sparse_model:
            error = proc.stderr.strip() or proc.stdout.strip() or "mapper a échoué"
        else:
            error = None

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.COLMAP_INCREMENTAL.value,
            status="completed" if sparse_model else "failed",
            started_at=started,
            finished_at=finished,
            metrics=metrics,
            output_path=str(sparse_model) if sparse_model else None,
            error=error,
        )

    except FileNotFoundError as exc:
        if "No such file or directory" in str(exc) and self.colmap_binary in str(exc):
            return ReconstructionRun(
                run_id=run_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id,
                backend=ReconstructionBackend.COLMAP_INCREMENTAL.value,
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"binaire COLMAP introuvable : {self.colmap_binary}",
            )
        raise
    except Exception as exc:
        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.COLMAP_INCREMENTAL.value,
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


def _run_colmap_global(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    workspace = self.workspace
    root = _workspace_root(workspace)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "database.db"

    started = datetime.now(timezone.utc).isoformat()

    try:
        selected, by_id, image_paths = _setup_run_images(self, input_manifest, run_dir, selected_asset_ids=selected_asset_ids)
        _run_colmap_feature_extractor(self, image_paths, db_path)
        _run_colmap_exhaustive_matcher(self, db_path)
        sparse_dir = run_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.colmap_binary,
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(run_dir / "images"),
            "--output_path", str(sparse_dir),
            "--Mapper.init_mode", "global",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        sparse_model = None
        if (sparse_dir / "0").exists():
            sparse_model = sparse_dir / "0"
            _export_colmap_normalized(sparse_model, run_dir, selected, by_id)

        finished = datetime.now(timezone.utc).isoformat()
        metrics = _parse_colmap_metrics(sparse_model, selected, by_id)
        error = None if sparse_model else (proc.stderr.strip() or proc.stdout.strip() or "mapper global a échoué")

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.COLMAP_GLOBAL.value,
            status="completed" if sparse_model else "failed",
            started_at=started,
            finished_at=finished,
            metrics=metrics,
            output_path=str(sparse_model) if sparse_model else None,
            error=error,
        )

    except FileNotFoundError as exc:
        if "No such file or directory" in str(exc) and self.colmap_binary in str(exc):
            return ReconstructionRun(
                run_id=run_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id,
                backend=ReconstructionBackend.COLMAP_GLOBAL.value,
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"binaire COLMAP introuvable : {self.colmap_binary}",
            )
        raise
    except Exception as exc:
        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.COLMAP_GLOBAL.value,
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Feed-forward backends (MapAnything, VGGT)
# ---------------------------------------------------------------------------


def _run_gluemap(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    """Exécute GLUEMAP (placeholder P2)."""
    workspace = self.workspace
    root = _workspace_root(workspace)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    try:
        selected, by_id, image_paths = _setup_run_images(self, input_manifest, run_dir, selected_asset_ids=selected_asset_ids)
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "gluemap",
            "--image_path", str(image_dir),
            "--output_path", str(run_dir / "gluemap_out"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        output_dir = run_dir / "gluemap_out"
        output_path = str(output_dir) if output_dir.exists() and any(output_dir.iterdir()) else None
        error = None if proc.returncode == 0 and output_path else (proc.stderr.strip() or proc.stdout.strip() or "gluemap a échoué")

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.GLUEMAP.value,
            status="completed" if output_path else "failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            metrics={"registered_ratio": 0.0},
            output_path=output_path,
            error=error,
        )
    except FileNotFoundError as exc:
        if "gluemap" in str(exc):
            return ReconstructionRun(
                run_id=run_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id,
                backend=ReconstructionBackend.GLUEMAP.value,
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="binaire GLUEMAP introuvable",
            )
        raise
    except Exception as exc:
        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.GLUEMAP.value,
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


def _run_mpsfm(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    """Exécute MP-SfM (placeholder P2)."""
    workspace = self.workspace
    root = _workspace_root(workspace)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    try:
        selected, by_id, image_paths = _setup_run_images(self, input_manifest, run_dir, selected_asset_ids=selected_asset_ids)
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mpsfm",
            "--image_path", str(image_dir),
            "--output_path", str(run_dir / "mpsfm_out"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        output_dir = run_dir / "mpsfm_out"
        output_path = str(output_dir) if output_dir.exists() and any(output_dir.iterdir()) else None
        error = None if proc.returncode == 0 and output_path else (proc.stderr.strip() or proc.stdout.strip() or "mpsfm a échoué")

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.MP_SFM.value,
            status="completed" if output_path else "failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            metrics={"registered_ratio": 0.0},
            output_path=output_path,
            error=error,
        )
    except FileNotFoundError as exc:
        if "mpsfm" in str(exc):
            return ReconstructionRun(
                run_id=run_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id,
                backend=ReconstructionBackend.MP_SFM.value,
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="binaire MP-SfM introuvable",
            )
        raise
    except Exception as exc:
        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.MP_SFM.value,
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Feed-forward backends (MapAnything, VGGT)
# ---------------------------------------------------------------------------


def _run_feed_forward(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    backend: ReconstructionBackend,
    binary: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    """Exécute un vérificateur feed-forward (MapAnything, VGGT).

    Ces backends ne produisent pas de modèle COLMAP mais des poses et
    de la géométrie. Pour le MVP, on journalise l'intention et on
    échoue proprement si le binaire n'est pas disponible.
    """
    workspace = self.workspace
    root = _workspace_root(workspace)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    try:
        selected, by_id, image_paths = _setup_run_images(self, input_manifest, run_dir, selected_asset_ids=selected_asset_ids)
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        cmd = [binary, "--image_path", str(image_dir), "--output_path", str(run_dir / "feed_forward")]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        output_dir = run_dir / "feed_forward"
        output_path = str(output_dir) if output_dir.exists() and any(output_dir.iterdir()) else None
        error = None if proc.returncode == 0 and output_path else (proc.stderr.strip() or proc.stdout.strip() or f"{backend.value} a échoué")

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=backend.value,
            status="completed" if output_path else "failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            metrics={"registered_ratio": 0.0},
            output_path=output_path,
            error=error,
        )
    except FileNotFoundError as exc:
        if binary in str(exc):
            return ReconstructionRun(
                run_id=run_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id,
                backend=backend.value,
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"binaire {backend.value} introuvable : {binary}",
            )
        raise
    except Exception as exc:
        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=backend.value,
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


def _run_map_anything(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    return _run_feed_forward(self, input_manifest, run_id, ReconstructionBackend.MAP_ANYTHING, "mapanything", selected_asset_ids=selected_asset_ids)


def _run_vggt(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    return _run_feed_forward(self, input_manifest, run_id, ReconstructionBackend.VGGT, "vggt", selected_asset_ids=selected_asset_ids)


# ---------------------------------------------------------------------------
# Helpers publics
# ---------------------------------------------------------------------------


def publish_run(run: ReconstructionRun, workspace: Workspace) -> Path:
    """Publie un `ReconstructionRun` sous `07_reconstruction/runs/`."""
    root = _workspace_root(workspace)
    path = root / "runs" / f"{run.run_id}.json"
    _write_json(path, run.model_dump(mode="json"))
    return path


def load_run(run_id: str, workspace: Workspace) -> ReconstructionRun | None:
    """Charge un `ReconstructionRun` publié."""
    path = _workspace_root(workspace) / "runs" / f"{run_id}.json"
    if not path.is_file():
        return None
    return ReconstructionRun.model_validate_json(path.read_text("utf-8"))


__all__ = [
    "ReconstructionRunner",
    "publish_run",
    "load_run",
]
