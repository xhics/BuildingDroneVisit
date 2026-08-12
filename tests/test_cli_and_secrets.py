"""CLI, fournisseurs et absence de secrets dans les journaux (§6, §18)."""

from __future__ import annotations

import logging

from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.config import check_providers
from hotel_pipeline.logging import SecretRedactingFilter, redact

runner = CliRunner()


class TestSecretRedaction:
    def test_known_secret_is_redacted(self, monkeypatch):
        monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|super-secret-value")
        assert "super-secret" not in redact("appel avec MLY|super-secret-value")

    def test_non_secret_env_var_untouched(self, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_WORK", "/workspace/work")
        assert redact("chemin /workspace/work") == "chemin /workspace/work"

    def test_short_values_not_redacted(self, monkeypatch):
        """Une valeur trop courte mutilerait du texte ordinaire."""
        monkeypatch.setenv("SOME_KEY", "on")
        assert redact("mode on") == "mode on"

    def test_log_record_is_scrubbed(self, monkeypatch):
        monkeypatch.setenv("VISION_API_KEY", "sk-abcdef0123456789")
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="clé=sk-abcdef0123456789", args=(), exc_info=None,
        )
        SecretRedactingFilter().filter(record)
        assert "sk-abcdef" not in record.msg


class TestProviderCheck:
    def test_keyless_providers_always_configured(self, monkeypatch):
        monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
        by_name = {s.provider.name: s for s in check_providers()}
        assert by_name["OSM"].configured
        assert not by_name["OSM"].blocking

    def test_missing_optional_provider_is_not_blocking(self, monkeypatch):
        """L'absence d'une source optionnelle ne provoque pas d'échec global (§6)."""
        monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
        by_name = {s.provider.name: s for s in check_providers()}
        assert not by_name["Mapillary"].configured
        assert not by_name["Mapillary"].blocking

    def test_command_reports_missing_token(self, monkeypatch):
        monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
        result = runner.invoke(app, ["provider-check"])
        assert result.exit_code == 0
        assert "missing MAPILLARY_TOKEN" in result.stdout

    def test_command_does_not_print_secret_values(self, monkeypatch):
        monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|leaky-token-value")
        result = runner.invoke(app, ["provider-check"])
        assert "leaky-token-value" not in result.stdout


class TestCommands:
    def test_smoke_passes(self):
        result = runner.invoke(app, ["smoke"])
        assert result.exit_code == 0, result.stdout
        assert "smoke test réussi" in result.stdout

    def test_init_then_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path))
        assert runner.invoke(app, ["init", "h", "--address", "1195 rue Ampère"]).exit_code == 0

        result = runner.invoke(app, ["status", "h"])
        assert result.exit_code == 0
        assert "1195 rue Ampère" in result.stdout

    def test_init_is_idempotent_without_force(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path))
        runner.invoke(app, ["init", "h", "--address", "a"])
        result = runner.invoke(app, ["init", "h", "--address", "autre"])
        assert result.exit_code == 0
        assert "existe déjà" in result.stdout
        assert runner.invoke(app, ["status", "h"]).stdout.count("autre") == 0

    def test_unimplemented_step_stops_cleanly(self, tmp_path, monkeypatch):
        """Acceptation Lot 0 : arrêt propre sur la première étape non construite."""
        monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path))
        runner.invoke(app, ["init", "h", "--address", "a"])

        result = runner.invoke(app, ["run-phase1", "h"])
        assert result.exit_code == 2
        assert "collect" in result.stdout
        assert "Lot 1" in result.stdout

    def test_status_on_unknown_hotel_is_actionable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path))
        result = runner.invoke(app, ["status", "absent"])
        assert result.exit_code != 0
