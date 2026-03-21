"""Tests for the Felix CLI — init, run, yaml_loader, and version."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from felix_agent_sdk.cli.init_command import run_init
from felix_agent_sdk.cli.main import main
from felix_agent_sdk.cli.yaml_loader import load_workflow_yaml
from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.workflows.config import SynthesisStrategy


# ---------------------------------------------------------------------------
# felix version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_command(self, capsys):
        code = main(["version"])
        assert code == 0
        output = capsys.readouterr().out
        assert "felix-agent-sdk" in output

    def test_no_command_shows_help(self, capsys):
        code = main([])
        assert code == 0


# ---------------------------------------------------------------------------
# felix init
# ---------------------------------------------------------------------------


class TestInit:
    def test_scaffolds_project(self, tmp_path):
        target = tmp_path / "myproject"
        code = run_init(str(target), "research")
        assert code == 0
        assert (target / "felix.yaml").exists()
        assert (target / "main.py").exists()
        assert (target / "requirements.txt").exists()
        assert (target / ".env.example").exists()

    def test_yaml_is_valid(self, tmp_path):
        target = tmp_path / "test_yaml"
        run_init(str(target), "research")
        with open(target / "felix.yaml") as f:
            data = yaml.safe_load(f)
        assert data["helix"] == "research_heavy"
        assert data["task"] is not None
        assert len(data["team"]) == 4

    def test_analysis_template(self, tmp_path):
        target = tmp_path / "analysis_proj"
        code = run_init(str(target), "analysis")
        assert code == 0
        with open(target / "felix.yaml") as f:
            data = yaml.safe_load(f)
        assert data["helix"] == "fast_convergence"

    def test_review_template(self, tmp_path):
        target = tmp_path / "review_proj"
        code = run_init(str(target), "review")
        assert code == 0
        with open(target / "felix.yaml") as f:
            data = yaml.safe_load(f)
        assert data["helix"] == "default"

    def test_existing_directory_fails(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        code = run_init(str(target), "research")
        assert code == 1

    def test_unknown_template_fails(self, tmp_path):
        code = run_init(str(tmp_path / "bad"), "nonexistent")
        assert code == 1

    def test_via_main(self, tmp_path, capsys):
        target = tmp_path / "cli_init"
        code = main(["init", str(target)])
        assert code == 0
        assert (target / "felix.yaml").exists()


# ---------------------------------------------------------------------------
# yaml_loader
# ---------------------------------------------------------------------------


class TestYamlLoader:
    def _write_yaml(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "felix.yaml"
        with open(p, "w") as f:
            yaml.dump(data, f)
        return p

    def test_minimal_config(self, tmp_path):
        path = self._write_yaml(tmp_path, {"task": "Test question"})
        config, task, provider_info = load_workflow_yaml(path)
        assert task == "Test question"
        assert len(config.team_composition) == 3  # default team
        assert config.max_rounds == 3

    def test_helix_preset(self, tmp_path):
        path = self._write_yaml(tmp_path, {
            "task": "Test",
            "helix": "research_heavy",
        })
        config, _, _ = load_workflow_yaml(path)
        expected = HelixConfig.research_heavy()
        assert config.helix_config.top_radius == expected.top_radius

    def test_helix_dict(self, tmp_path):
        path = self._write_yaml(tmp_path, {
            "task": "Test",
            "helix": {"top_radius": 5.0, "bottom_radius": 0.1},
        })
        config, _, _ = load_workflow_yaml(path)
        assert config.helix_config.top_radius == 5.0
        assert config.helix_config.bottom_radius == 0.1

    def test_team_parsing(self, tmp_path):
        path = self._write_yaml(tmp_path, {
            "task": "Test",
            "team": [
                {"type": "research"},
                {"type": "analysis"},
                {"type": "critic"},
                {"type": "critic"},
            ],
        })
        config, _, _ = load_workflow_yaml(path)
        assert len(config.team_composition) == 4
        types = [t for t, _ in config.team_composition]
        assert types == ["research", "analysis", "critic", "critic"]

    def test_scalars(self, tmp_path):
        path = self._write_yaml(tmp_path, {
            "task": "Test",
            "max_rounds": 5,
            "confidence_threshold": 0.9,
            "max_agents": 15,
            "synthesis_strategy": "best_result",
            "enable_dynamic_spawning": True,
            "max_dynamic_agents": 5,
        })
        config, _, _ = load_workflow_yaml(path)
        assert config.max_rounds == 5
        assert config.confidence_threshold == 0.9
        assert config.max_agents == 15
        assert config.synthesis_strategy == SynthesisStrategy.BEST_RESULT
        assert config.enable_dynamic_spawning is True
        assert config.max_dynamic_agents == 5

    def test_provider_info(self, tmp_path):
        path = self._write_yaml(tmp_path, {
            "task": "Test",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
        })
        _, _, info = load_workflow_yaml(path)
        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-sonnet-4-5"

    def test_missing_task_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, {"helix": "default"})
        with pytest.raises(ValueError, match="Missing required 'task'"):
            load_workflow_yaml(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_workflow_yaml("/nonexistent/felix.yaml")

    def test_invalid_helix_preset_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, {"task": "T", "helix": "bogus"})
        with pytest.raises(ValueError, match="Unknown helix preset"):
            load_workflow_yaml(path)

    def test_string_team_entries(self, tmp_path):
        path = self._write_yaml(tmp_path, {
            "task": "Test",
            "team": ["research", "analysis"],
        })
        config, _, _ = load_workflow_yaml(path)
        assert config.team_composition == [("research", {}), ("analysis", {})]


# ---------------------------------------------------------------------------
# felix init → felix run round-trip (scaffolded yaml is loadable)
# ---------------------------------------------------------------------------


class TestInitRunRoundTrip:
    def test_scaffolded_yaml_is_loadable(self, tmp_path):
        target = tmp_path / "roundtrip"
        run_init(str(target), "research")
        config, task, info = load_workflow_yaml(target / "felix.yaml")
        assert task is not None
        assert len(config.team_composition) > 0
        assert info["provider"] == "openai"
