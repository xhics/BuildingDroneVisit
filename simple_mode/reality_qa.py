"""Reality-preserving QA for generative video passes.

The renderer is the geometric authority.  A V2V model may improve appearance,
but it is not allowed to move roof lines, windows or the silhouette.  This
module therefore compares generated frames against their deterministic source
using tolerant edge correspondence.  It also selects a real appearance photo
by matching local features against the source video instead of blindly using
the first hotel photo.

The algorithms deliberately fail soft when OpenCV is unavailable: generation
can still run, but the audit reports ``available=False`` and no structural
claim is made.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoStructureAssessment:
    available: bool
    accepted: bool
    sampled_frames: int
    median_score: float | None
    p10_score: float | None
    minimum_required_median: float
    minimum_required_p10: float
    reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _cv2():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _gray(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    if cv2 is None:
        raise RuntimeError("opencv unavailable")
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def edge_structure_score(
    source_frame: np.ndarray,
    generated_frame: np.ndarray,
    *,
    tolerance_px: int = 3,
) -> float | None:
    """Tolerant bidirectional structural-edge agreement for one frame.

    A few pixels of antialiasing or V2V texture change are allowed.  Large
    changes to roof lines, window rhythms or silhouette are not.
    """
    cv2 = _cv2()
    if cv2 is None:
        return None

    source = _gray(source_frame)
    generated = _gray(generated_frame)
    if generated.shape != source.shape:
        generated = cv2.resize(
            generated,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    # Light blur suppresses brick/noise micro-edges so the score focuses on
    # architecture rather than whether the AI added film grain.
    source = cv2.GaussianBlur(source, (3, 3), 0)
    generated = cv2.GaussianBlur(generated, (3, 3), 0)
    src_edges = cv2.Canny(source, 70, 160) > 0
    gen_edges = cv2.Canny(generated, 70, 160) > 0

    src_count = int(src_edges.sum())
    gen_count = int(gen_edges.sum())
    if src_count < 100 or gen_count < 100:
        return None

    radius = max(1, int(tolerance_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    src_near = cv2.dilate(src_edges.astype(np.uint8), kernel) > 0
    gen_near = cv2.dilate(gen_edges.astype(np.uint8), kernel) > 0

    # Recall protects the geometry authority: every strong source edge should
    # still exist.  Precision receives lower weight because a photorealistic
    # pass may legitimately add material micro-detail.
    recall = float((src_edges & gen_near).sum()) / src_count
    precision = float((gen_edges & src_near).sum()) / gen_count
    return float(0.8 * recall + 0.2 * precision)


def _video_frame_at(capture, index: int):  # noqa: ANN001
    cv2 = _cv2()
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, frame = capture.read()
    return frame if ok else None


def assess_video_structure(
    source_video: str | Path,
    generated_video: str | Path,
    *,
    samples: int = 12,
    minimum_median: float = 0.58,
    minimum_p10: float = 0.40,
    tolerance_px: int = 3,
) -> VideoStructureAssessment:
    """Audit V2V output against the deterministic geometric source."""
    cv2 = _cv2()
    if cv2 is None:
        return VideoStructureAssessment(
            False,
            True,
            0,
            None,
            None,
            minimum_median,
            minimum_p10,
            "opencv unavailable",
        )

    source_video, generated_video = Path(source_video), Path(generated_video)
    source = cv2.VideoCapture(str(source_video))
    generated = cv2.VideoCapture(str(generated_video))
    if not source.isOpened() or not generated.isOpened():
        source.release()
        generated.release()
        return VideoStructureAssessment(
            False,
            False,
            0,
            None,
            None,
            minimum_median,
            minimum_p10,
            "video cannot be opened",
        )

    source_count = max(1, int(source.get(cv2.CAP_PROP_FRAME_COUNT)))
    generated_count = max(1, int(generated.get(cv2.CAP_PROP_FRAME_COUNT)))
    count = max(2, min(int(samples), source_count, generated_count))
    positions = np.linspace(0.0, 1.0, count)

    scores: list[float] = []
    for position in positions:
        source_index = round(position * (source_count - 1))
        generated_index = round(position * (generated_count - 1))
        source_frame = _video_frame_at(source, source_index)
        generated_frame = _video_frame_at(generated, generated_index)
        if source_frame is None or generated_frame is None:
            continue
        score = edge_structure_score(
            source_frame, generated_frame, tolerance_px=tolerance_px
        )
        if score is not None:
            scores.append(score)

    source.release()
    generated.release()
    if len(scores) < 2:
        return VideoStructureAssessment(
            False,
            True,
            len(scores),
            None,
            None,
            minimum_median,
            minimum_p10,
            "insufficient structural frames",
        )

    values = np.asarray(scores, dtype=np.float64)
    median = float(np.median(values))
    p10 = float(np.quantile(values, 0.10))
    accepted = median >= minimum_median and p10 >= minimum_p10
    return VideoStructureAssessment(
        True,
        accepted,
        len(scores),
        round(median, 4),
        round(p10, 4),
        minimum_median,
        minimum_p10,
        None if accepted else "generated structure drifted from source render",
    )


def _representative_video_frame(path: Path) -> np.ndarray | None:
    cv2 = _cv2()
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame = _video_frame_at(capture, round((frame_count - 1) * 0.35))
    capture.release()
    return frame


def _orb_match_score(reference_frame: np.ndarray, candidate_path: Path) -> int:
    cv2 = _cv2()
    if cv2 is None:
        return 0
    image = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
    if image is None:
        return 0

    reference_gray = _gray(reference_frame)
    candidate_gray = _gray(image)
    # Cap resolution for predictable ORB cost while preserving layout.
    max_side = 1200
    if max(candidate_gray.shape) > max_side:
        scale = max_side / max(candidate_gray.shape)
        candidate_gray = cv2.resize(
            candidate_gray,
            (
                max(1, round(candidate_gray.shape[1] * scale)),
                max(1, round(candidate_gray.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    if max(reference_gray.shape) > max_side:
        scale = max_side / max(reference_gray.shape)
        reference_gray = cv2.resize(
            reference_gray,
            (
                max(1, round(reference_gray.shape[1] * scale)),
                max(1, round(reference_gray.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    orb = cv2.ORB_create(nfeatures=2500)
    key_a, desc_a = orb.detectAndCompute(reference_gray, None)
    key_b, desc_b = orb.detectAndCompute(candidate_gray, None)
    if desc_a is None or desc_b is None or len(key_a) < 8 or len(key_b) < 8:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    return len(good)


def _source_directories(source_video: Path) -> list[Path]:
    directories: list[Path] = []
    for parent in list(source_video.parents)[:5]:
        candidate = parent / "sources"
        if candidate.is_dir() and candidate not in directories:
            directories.append(candidate)
    return directories


def select_appearance_reference(
    source_video: str | Path,
    *,
    minimum_matches: int = 12,
    max_candidates: int = 16,
) -> Path | None:
    """Choose the real hotel photo that best matches the current V2V shot.

    This avoids feeding an interior or unrelated pool photo as subject
    appearance merely because it happens to be ``source_00.jpg``.
    """
    source_video = Path(source_video)
    frame = _representative_video_frame(source_video)
    if frame is None:
        return None

    candidates: list[Path] = []
    for directory in _source_directories(source_video):
        candidates.extend(sorted(directory.glob("*.jpg")))
        candidates.extend(sorted(directory.glob("*.jpeg")))
        candidates.extend(sorted(directory.glob("*.png")))
    candidates = candidates[:max_candidates]
    if not candidates:
        return None

    scored = [(_orb_match_score(frame, candidate), candidate) for candidate in candidates]
    score, path = max(scored, key=lambda row: row[0])
    return path if score >= minimum_matches else None


__all__ = [
    "VideoStructureAssessment",
    "assess_video_structure",
    "edge_structure_score",
    "select_appearance_reference",
]
