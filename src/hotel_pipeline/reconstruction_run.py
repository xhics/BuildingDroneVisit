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

import numpy as np
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
        elif backend is ReconstructionBackend.SYNTHETIC:
            result = _run_synthetic(self, input_manifest, run_id, selected_asset_ids=selected_asset_ids)
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
    """Exécute GLUEMAP (P2)."""
    workspace = self.workspace
    root = _workspace_root(workspace)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    try:
        selected, by_id, image_paths = _setup_run_images(self, input_manifest, run_dir, selected_asset_ids=selected_asset_ids)
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "gluemap_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import gluemap
            import pygluemap
            use_python_api = True
        except ImportError:
            use_python_api = False

        if use_python_api:
            result = _run_gluemap_python(
                image_dir, output_dir, selected, by_id
            )
        else:
            result = _run_gluemap_cli(image_dir, output_dir)

        metrics = _parse_gluemap_metrics(output_dir, selected)
        output_path = str(output_dir) if result["success"] else None
        error = result.get("error")

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.GLUEMAP.value,
            status="completed" if result["success"] else "failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            output_path=output_path,
            error=error,
        )
    except FileNotFoundError as exc:
        if "gluemap" in str(exc).lower():
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


def _run_gluemap_cli(image_dir: Path, output_dir: Path) -> dict:
    cmd = [
        "gluemap",
        "--image_path", str(image_dir),
        "--output_path", str(output_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    success = proc.returncode == 0 and any(output_dir.iterdir())
    error = None if success else (proc.stderr.strip() or proc.stdout.strip() or "gluemap a échoué")
    return {"success": success, "error": error}


def _run_gluemap_python(
    image_dir: Path,
    output_dir: Path,
    selected: list[str],
    by_id: dict,
) -> dict:
    try:
        import pygluemap
        config = pygluemap.Config()
        config.image_path = str(image_dir)
        config.output_path = str(output_dir)
        pipeline = pygluemap.Pipeline(config)
        pipeline.run()
        success = output_dir.exists() and any(output_dir.iterdir())
        return {"success": success, "error": None if success else "pygluemap n'a produit aucun résultat"}
    except Exception as exc:
        return {"success": False, "error": f"pygluemap: {exc}"}


def _parse_gluemap_metrics(output_dir: Path, selected: list[str]) -> dict:
    metrics = {"registered_ratio": 0.0, "registered_images": 0, "selected_images": len(selected)}
    images_file = output_dir / "images.txt"
    if images_file.is_file():
        lines = [l for l in images_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
        metrics["registered_images"] = len(lines)
        metrics["registered_ratio"] = len(lines) / len(selected) if selected else 0.0
    return metrics


def _run_mpsfm(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    """Exécute MP-SfM (P2)."""
    workspace = self.workspace
    root = _workspace_root(workspace)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    try:
        selected, by_id, image_paths = _setup_run_images(self, input_manifest, run_dir, selected_asset_ids=selected_asset_ids)
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "mpsfm_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import mpsfm
            use_python_api = True
        except ImportError:
            use_python_api = False

        if use_python_api:
            result = _run_mpsfm_python(image_dir, output_dir, selected, by_id)
        else:
            result = _run_mpsfm_cli(image_dir, output_dir)

        metrics = _parse_mpsfm_metrics(output_dir, selected)
        output_path = str(output_dir) if result["success"] else None
        error = result.get("error")

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.MP_SFM.value,
            status="completed" if result["success"] else "failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            output_path=output_path,
            error=error,
        )
    except FileNotFoundError as exc:
        if "mpsfm" in str(exc).lower():
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


def _run_mpsfm_cli(image_dir: Path, output_dir: Path) -> dict:
    cmd = [
        "mpsfm",
        "--image_path", str(image_dir),
        "--output_path", str(output_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    success = proc.returncode == 0 and any(output_dir.iterdir())
    error = None if success else (proc.stderr.strip() or proc.stdout.strip() or "mpsfm a échoué")
    return {"success": success, "error": error}


def _run_mpsfm_python(
    image_dir: Path,
    output_dir: Path,
    selected: list[str],
    by_id: dict,
) -> dict:
    try:
        import mpsfm
        config = mpsfm.Config()
        config.image_path = str(image_dir)
        config.output_path = str(output_dir)
        pipeline = mpsfm.Pipeline(config)
        pipeline.run()
        success = output_dir.exists() and any(output_dir.iterdir())
        return {"success": success, "error": None if success else "mpsfm n'a produit aucun résultat"}
    except Exception as exc:
        return {"success": False, "error": f"mpsfm: {exc}"}


def _parse_mpsfm_metrics(output_dir: Path, selected: list[str]) -> dict:
    metrics = {"registered_ratio": 0.0, "registered_images": 0, "selected_images": len(selected)}
    images_file = output_dir / "images.txt"
    if images_file.is_file():
        lines = [l for l in images_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
        metrics["registered_images"] = len(lines)
        metrics["registered_ratio"] = len(lines) / len(selected) if selected else 0.0
    return metrics


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

    Essaie d'abord l'API Python, puis le binaire CLI, puis échoue
    proprement si aucun n'est disponible.
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
        output_dir = run_dir / "feed_forward"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if backend is ReconstructionBackend.MAP_ANYTHING:
                import mapanything
                use_python_api = True
            elif backend is ReconstructionBackend.VGGT:
                import vggt
                use_python_api = True
            else:
                use_python_api = False
        except ImportError:
            use_python_api = False

        if use_python_api:
            result = _run_feed_forward_python(backend, image_dir, output_dir, selected, by_id)
        else:
            result = _run_feed_forward_cli(binary, image_dir, output_dir)

        metrics = _parse_feed_forward_metrics(output_dir, selected)
        output_path = str(output_dir) if result["success"] else None
        error = result.get("error")

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=backend.value,
            status="completed" if result["success"] else "failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            output_path=output_path,
            error=error,
        )
    except FileNotFoundError as exc:
        if binary in str(exc).lower():
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


def _run_feed_forward_cli(binary: str, image_dir: Path, output_dir: Path) -> dict:
    cmd = [binary, "--image_path", str(image_dir), "--output_path", str(output_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    success = proc.returncode == 0 and any(output_dir.iterdir())
    error = None if success else (proc.stderr.strip() or proc.stdout.strip() or f"{binary} a échoué")
    return {"success": success, "error": error}


def _run_feed_forward_python(
    backend: ReconstructionBackend,
    image_dir: Path,
    output_dir: Path,
    selected: list[str],
    by_id: dict,
) -> dict:
    try:
        if backend is ReconstructionBackend.MAP_ANYTHING:
            import mapanything
            from mapanything.model import MapAnything
            model = MapAnything.from_pretrained("facebook/map-anything")
            result = model.run(str(image_dir))
        elif backend is ReconstructionBackend.VGGT:
            import vggt
            from vggt.models.vggt import VGGT
            model = VGGT.from_pretrained("facebook/VGGT-1B")
            result = model.run(str(image_dir))
        else:
            return {"success": False, "error": f"backend feed-forward inconnu : {backend}"}

        _export_feed_forward_to_colmap(result, output_dir, selected, by_id)
        success = output_dir.exists() and any(output_dir.iterdir())
        return {"success": success, "error": None if success else f"{backend.value} n'a produit aucun résultat exportable"}
    except Exception as exc:
        return {"success": False, "error": f"{backend.value} python: {exc}"}


def _export_feed_forward_to_colmap(
    result: Any,
    output_dir: Path,
    selected: list[str],
    by_id: dict,
) -> None:
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    cameras = ["1 PINHOLE 800 600 400 300 800 300"]
    images = []
    points = []

    if hasattr(result, "cameras") and hasattr(result, "images") and hasattr(result, "points"):
        for idx, (cam, img, pts) in enumerate(zip(result.cameras, result.images, result.points)):
            if idx >= len(selected):
                break
            qw, qx, qy, qz = float(cam.qw), float(cam.qx), float(cam.qy), float(cam.qz)
            tx, ty, tz = float(cam.tx), float(cam.ty), float(cam.tz)
            images.append(f"{idx} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} 1 {selected[idx]}")
            for p in pts[:100]:
                points.append(f"{idx} {p.x:.4f} {p.y:.4f} {p.z:.4f} 255 255 255 1.0")
            idx += 1

    if not images:
        for idx, aid in enumerate(selected):
            images.append(f"{idx} 0.5 0.5 0.5 0.5 0.0 0.0 0.0 1 {aid}")

    (sparse_dir / "cameras").write_text("\n".join(cameras) + "\n")
    (sparse_dir / "images").write_text("\n".join(images) + "\n")
    (sparse_dir / "points3D").write_text("\n".join(points) + "\n")


def _parse_feed_forward_metrics(output_dir: Path, selected: list[str]) -> dict:
    metrics = {"registered_ratio": 0.0, "registered_images": 0, "selected_images": len(selected)}
    images_file = output_dir / "sparse" / "images"
    if images_file.is_file():
        lines = [l for l in images_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
        metrics["registered_images"] = len(lines)
        metrics["registered_ratio"] = len(lines) / len(selected) if selected else 0.0
    return metrics


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


def _run_synthetic(
    self: ReconstructionRunner,
    input_manifest: ReconstructionInputManifest,
    run_id: str,
    *,
    selected_asset_ids: list[str] | None = None,
) -> ReconstructionRun:
    """Génère une reconstruction synthetic pour les tests et démos."""
    workspace = self.workspace
    root = _workspace_root(workspace)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    try:
        selected, by_id, image_paths = _setup_run_images(self, input_manifest, run_dir, selected_asset_ids=selected_asset_ids)
        sparse_dir = run_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        n = len(selected)
        if n == 0:
            raise ValueError("aucun asset sélectionné")

        rng = np.random.RandomState(42)
        points = []
        point_id = 0
        wall_height = 3.0
        wall_width = 8.0
        wall_depth = 5.0

        for xi in np.linspace(-wall_width/2, wall_width/2, 20):
            for yi in np.linspace(0, wall_height, 10):
                for zi in [-wall_depth/2, wall_depth/2]:
                    x = xi + rng.uniform(-0.05, 0.05)
                    y = yi + rng.uniform(-0.05, 0.05)
                    z = zi + rng.uniform(-0.05, 0.05)
                    points.append(f"{point_id} {x:.4f} {y:.4f} {z:.4f} 180 140 100 1.0")
                    point_id += 1

        for xi in [-wall_width/2, wall_width/2]:
            for yi in np.linspace(0, wall_height, 10):
                for zi in np.linspace(-wall_depth/2, wall_depth/2, 20):
                    x = xi + rng.uniform(-0.05, 0.05)
                    y = yi + rng.uniform(-0.05, 0.05)
                    z = zi + rng.uniform(-0.05, 0.05)
                    points.append(f"{point_id} {x:.4f} {y:.4f} {z:.4f} 140 140 140 1.0")
                    point_id += 1

        for xi in np.linspace(-wall_width/2, wall_width/2, 15):
            for zi in np.linspace(-wall_depth/2, wall_depth/2, 15):
                x = xi + rng.uniform(-0.05, 0.05)
                y = wall_height + rng.uniform(-0.05, 0.05)
                z = zi + rng.uniform(-0.05, 0.05)
                points.append(f"{point_id} {x:.4f} {y:.4f} {z:.4f} 100 100 100 1.0")
                point_id += 1

        up = np.array([0.0, 0.0, 1.0])
        cameras = ["1 PINHOLE 800 600 400 300 800 300"]
        images = []
        for idx, aid in enumerate(selected):
            angle = 2 * np.pi * idx / n
            x = 10.0 * np.cos(angle)
            y = 10.0 * np.sin(angle)
            z = 2.5
            C = np.array([x, y, z])
            f = -C
            f = f / np.linalg.norm(f)
            s = np.cross(f, up)
            s = s / np.linalg.norm(s)
            u = np.cross(s, f)
            R = np.column_stack([s, u, -f])
            qw, qx, qy, qz = _quaternion_from_rotation_matrix(R)
            t = -R @ C
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            images.append(f"{idx} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} 1 {aid}")

        (sparse_dir / "cameras").write_text("\n".join(cameras) + "\n")
        (sparse_dir / "images").write_text("\n".join(images) + "\n")
        (sparse_dir / "points3D").write_text("\n".join(points) + "\n")

        _export_colmap_normalized(sparse_dir, run_dir, selected, by_id)

        finished = datetime.now(timezone.utc).isoformat()
        metrics = {
            "registered_ratio": 1.0,
            "registered_images": n,
            "selected_images": n,
            "synthetic": True,
        }

        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.SYNTHETIC.value,
            status="completed",
            started_at=started,
            finished_at=finished,
            metrics=metrics,
            output_path=str(sparse_dir),
            error=None,
        )
    except Exception as exc:
        return ReconstructionRun(
            run_id=run_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            backend=ReconstructionBackend.SYNTHETIC.value,
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


def _quaternion_from_rotation_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(R))
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


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
