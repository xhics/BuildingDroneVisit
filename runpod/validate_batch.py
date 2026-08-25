"""Valide le lot RunPod avant facturation et ses sorties avant rapatriement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(checks: list[dict], check_id: str, passed: bool, evidence: str) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), "evidence": evidence})


def _safe_child(root: Path, relative: str) -> Path | None:
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def validate_input(bundle: Path) -> dict:
    checks: list[dict] = []
    manifest_path = bundle / "shape_input.json"
    if not manifest_path.is_file():
        _check(checks, "manifest", False, str(manifest_path))
        return _report("input", checks)
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        _check(checks, "manifest", False, str(exc))
        return _report("input", checks)
    images = manifest.get("images") or []
    _check(checks, "manifest", True, str(manifest_path))
    _check(checks, "image_count", len(images) >= 8, f"{len(images)} image(s), minimum 8")
    placed = int(manifest.get("placed") or 0)
    span = float(manifest.get("angular_span_deg") or 0.0)
    _check(checks, "placed_views", placed >= 6, f"{placed} vue(s) placee(s), minimum 6")
    _check(checks, "angular_span", span >= 120.0, f"{span:.1f} deg, minimum 120 deg")
    policy = manifest.get("usage_policy") or {}
    _check(
        checks,
        "usage_scope",
        policy.get("scope") == "experimental_demo_only"
        and policy.get("production_eligible") is False,
        json.dumps(policy, sort_keys=True),
    )

    seen: set[str] = set()
    readable = 0
    integrity = 0
    unique = True
    for item in images:
        candidate = _safe_child(bundle, str(item.get("path") or ""))
        if candidate is None or not candidate.is_file():
            continue
        try:
            from PIL import Image

            with Image.open(candidate) as image:
                image.verify()
            readable += 1
        except Exception:
            continue
        digest = _digest(candidate)
        if digest == item.get("sha256") and candidate.stat().st_size == item.get("bytes"):
            integrity += 1
        if digest in seen:
            unique = False
        seen.add(digest)
    _check(checks, "readable_images", readable == len(images), f"{readable}/{len(images)}")
    _check(checks, "image_integrity", integrity == len(images), f"{integrity}/{len(images)}")
    _check(checks, "unique_images", unique and len(seen) == len(images), f"{len(seen)}/{len(images)}")
    return _report("input", checks, {"hotel_id": manifest.get("hotel_id")})


def _ply_summary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    vertices = None
    samples: list[list[float]] = []
    actual_rows = 0
    malformed = False
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline().strip()
        if first != "ply":
            raise ValueError("en-tete PLY absent")
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("fin d'en-tete PLY absente")
            fields = line.strip().split()
            if fields[:2] == ["element", "vertex"]:
                vertices = int(fields[2])
            if fields == ["end_header"]:
                break
        for line in stream:
            if not line.strip():
                continue
            fields = line.split()
            actual_rows += 1
            if len(fields) < 3:
                malformed = True
                continue
            if len(samples) < 1000:
                try:
                    samples.append([float(fields[0]), float(fields[1]), float(fields[2])])
                except ValueError:
                    malformed = True
    if vertices is None:
        raise ValueError("compte de sommets PLY absent")
    finite = all(math.isfinite(value) for point in samples for value in point)
    spans = [
        max(point[axis] for point in samples) - min(point[axis] for point in samples)
        for axis in range(3)
    ] if samples else [0.0, 0.0, 0.0]
    return {
        "vertices": vertices,
        "actual_rows": actual_rows,
        "rows_match": actual_rows == vertices and not malformed,
        "samples": len(samples),
        "finite": finite,
        "spans": spans,
    }


def validate_output(out: Path, backends: list[str], expected_images: int) -> dict:
    checks: list[dict] = []
    summaries: dict[str, dict] = {}
    for backend in backends:
        root = out / backend
        run_path = root / "shape_run.json"
        try:
            run = json.loads(run_path.read_text("utf-8"))
            ply = _ply_summary(root / "shape.ply")
        except (OSError, ValueError) as exc:
            _check(checks, f"{backend}_artifacts", False, str(exc))
            continue
        summaries[backend] = {"run": run, "ply": ply}
        _check(checks, f"{backend}_artifacts", True, str(root))
        _check(
            checks,
            f"{backend}_cuda",
            run.get("device") == "cuda",
            f"device={run.get('device')}",
        )
        _check(
            checks,
            f"{backend}_images",
            int(run.get("images") or 0) == expected_images,
            f"{run.get('images')}/{expected_images}",
        )
        _check(
            checks,
            f"{backend}_points",
            ply["vertices"] >= 1000
            and ply["rows_match"]
            and int(run.get("points") or 0) == ply["vertices"],
            f"{ply['actual_rows']}/{ply['vertices']} sommet(s)",
        )
        _check(
            checks,
            f"{backend}_finite_noncollapsed",
            ply["finite"] and max(ply["spans"]) > 1e-5,
            f"spans={ply['spans']}",
        )
        if backend == "vggt":
            cameras_path = root / "cameras.json"
            try:
                cameras = json.loads(cameras_path.read_text("utf-8"))
                count = len(cameras.get("cameras") or [])
            except (OSError, ValueError):
                count = 0
            _check(checks, "vggt_cameras", count == expected_images, f"{count}/{expected_images}")
    return _report("output", checks, {"backends": summaries})


def _report(kind: str, checks: list[dict], details: dict | None = None) -> dict:
    passed = bool(checks) and all(item["passed"] for item in checks)
    return {
        "contract_version": 1,
        "kind": kind,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "checks": checks,
        "details": details or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    before = sub.add_parser("input")
    before.add_argument("--bundle", type=Path, required=True)
    before.add_argument("--report", type=Path)
    after = sub.add_parser("output")
    after.add_argument("--out", type=Path, required=True)
    after.add_argument("--backends", default="vggt")
    after.add_argument("--expected-images", type=int, required=True)
    after.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = (
        validate_input(args.bundle)
        if args.mode == "input"
        else validate_output(
            args.out,
            [item.strip() for item in args.backends.split(",") if item.strip()],
            args.expected_images,
        )
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
