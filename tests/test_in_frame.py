"""Le bâtiment est-il dans le cadre : géométrie d'abord, contenu ensuite."""

from __future__ import annotations

import pytest

from hotel_pipeline.geo.in_frame import (
    CONTENT_MIN,
    HORIZON_BAND,
    HORIZON_SECTORS,
    IN_FRAME_MIN,
    WIDTH_MIN,
    _geometry_verdict,
    judge,
)


def _framing(fraction: float = 0.9, width: float = 0.2, computable: bool = True) -> dict:
    return {
        "horizontal_computable": computable,
        "horizontal_reason": None if computable else "cadrage demandé non conservé",
        "target_in_frame_fraction": fraction,
        "unclipped_width_fraction": width,
    }


class _Asset:
    def __init__(self, identifier: str, local_path: str | None = "img.jpg"):
        self.id = identifier
        self.local_path = local_path


class TestGeometryVerdict:
    def test_well_framed_target_passes(self):
        decided, _reason, _f, _w = _geometry_verdict(_framing())
        assert decided is True

    def test_target_outside_the_frame_is_refused(self):
        decided, reason, _f, _w = _geometry_verdict(_framing(fraction=0.1))
        assert decided is False
        assert "hors cadre" in reason

    def test_target_too_small_is_refused(self):
        decided, reason, _f, _w = _geometry_verdict(_framing(width=0.01))
        assert decided is False
        assert "trop petit" in reason

    def test_absent_framing_is_undecided_not_refused(self):
        """Ne rien savoir n'est pas savoir que non."""
        decided, reason, _f, _w = _geometry_verdict(None)
        assert decided is None
        assert "aucun cadrage" in reason

    def test_incomputable_framing_carries_its_reason(self):
        decided, reason, _f, _w = _geometry_verdict(_framing(computable=False))
        assert decided is None
        assert "cadrage demandé" in reason

    def test_incomplete_framing_is_undecided(self):
        decided, _reason, _f, _w = _geometry_verdict(
            {"horizontal_computable": True, "target_in_frame_fraction": None,
             "unclipped_width_fraction": None}
        )
        assert decided is None


class TestJudgeWithoutContent:
    def test_geometry_alone_never_confirms(self):
        """Un cadrage favorable ne prouve pas ce que montre l'image."""
        report = judge([_Asset("A")], {"A": _framing()})
        verdict = report.verdicts[0]
        assert verdict.in_frame is None
        assert "contenu non vérifié" in verdict.reason

    def test_geometry_alone_can_refuse(self):
        report = judge([_Asset("A")], {"A": _framing(fraction=0.0)})
        assert report.verdicts[0].in_frame is False
        assert report.absent == ["A"]

    def test_verdicts_partition_the_corpus(self):
        assets = [_Asset("A"), _Asset("B"), _Asset("C")]
        framings = {"A": _framing(), "B": _framing(fraction=0.0), "C": _framing(width=0.0)}
        report = judge(assets, framings)
        total = len(report.visible) + len(report.absent) + len(report.undecided)
        assert total == 3


class _Embedder:
    """Modèle simulé : rend la classe convenue pour chaque chemin."""

    model_name = "test"

    def __init__(self, answers: dict[str, tuple[str, float]]):
        self.answers = answers


class _Workspace:
    def __init__(self, tmp_path, exists: bool = True):
        self._tmp = tmp_path
        self._exists = exists

    def path(self, *parts):
        target = self._tmp.joinpath(*parts)
        if self._exists:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return target


class TestJudgeWithContent:
    def test_building_on_the_horizon_confirms(self, tmp_path, monkeypatch):
        import hotel_pipeline.geo.in_frame as module

        monkeypatch.setattr(module, "_content_verdict", lambda e, p: ("batiment", 0.25))
        report = judge(
            [_Asset("A")], {"A": _framing()}, _Embedder({}), _Workspace(tmp_path)
        )
        assert report.verdicts[0].in_frame is True
        assert report.visible == ["A"]

    def test_road_only_refuses_despite_good_framing(self, tmp_path, monkeypatch):
        """Le cas mesuré : cadrage parfait, mais la caméra vise la chaussée."""
        import hotel_pipeline.geo.in_frame as module

        monkeypatch.setattr(module, "_content_verdict", lambda e, p: ("route", 0.0))
        report = judge(
            [_Asset("A")], {"A": _framing()}, _Embedder({}), _Workspace(tmp_path)
        )
        assert report.verdicts[0].in_frame is False
        assert "ne vise pas" in report.verdicts[0].reason

    def test_a_single_sector_is_enough(self, tmp_path, monkeypatch):
        """Un bâtiment au bord du cadre reste un bâtiment dans le cadre."""
        import hotel_pipeline.geo.in_frame as module

        monkeypatch.setattr(
            module, "_content_verdict", lambda e, p: ("batiment", 1 / HORIZON_SECTORS)
        )
        report = judge(
            [_Asset("A")], {"A": _framing()}, _Embedder({}), _Workspace(tmp_path)
        )
        assert report.verdicts[0].in_frame is True

    def test_unreadable_image_is_undecided(self, tmp_path, monkeypatch):
        import hotel_pipeline.geo.in_frame as module

        monkeypatch.setattr(module, "_content_verdict", lambda e, p: (None, None))
        report = judge(
            [_Asset("A")], {"A": _framing()}, _Embedder({}), _Workspace(tmp_path)
        )
        assert report.verdicts[0].in_frame is None
        assert "illisible" in report.verdicts[0].reason

    def test_missing_file_is_undecided(self, tmp_path):
        report = judge(
            [_Asset("A")], {"A": _framing()}, _Embedder({}), _Workspace(tmp_path, exists=False)
        )
        assert report.verdicts[0].in_frame is None
        assert "absent" in report.verdicts[0].reason

    def test_content_is_not_read_for_refused_geometry(self, tmp_path, monkeypatch):
        """Interroger le modèle sur une vue déjà écartée coûterait pour rien."""
        import hotel_pipeline.geo.in_frame as module

        called = []
        monkeypatch.setattr(
            module, "_content_verdict", lambda e, p: called.append(p) or ("batiment", 1.0)
        )
        judge(
            [_Asset("A")], {"A": _framing(fraction=0.0)}, _Embedder({}), _Workspace(tmp_path)
        )
        assert called == []


class TestReport:
    def test_report_serialises_with_its_caveats(self, tmp_path):
        payload = judge([_Asset("A")], {"A": _framing()}).as_dict()
        assert payload["total"] == 1
        assert payload["caveats"]
        # Le verdict ne prétend pas juger l'identité du bâtiment.
        assert any("identité" in c for c in payload["caveats"])

    def test_thresholds_are_coherent(self):
        assert 0.0 < WIDTH_MIN < IN_FRAME_MIN <= 1.0
        assert 0.0 < CONTENT_MIN <= 1.0
        assert 0.0 < HORIZON_BAND[0] < HORIZON_BAND[1] < 1.0
        assert HORIZON_SECTORS >= 4
