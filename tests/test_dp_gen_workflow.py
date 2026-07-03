"""Integration tests for the DP-GEN active learning workflow.

Verifies data format compatibility between exploration, labeling,
and training stages, and validates the param.json configuration schema.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from dp_gen.exploration.explore import parse_model_deviation, select_candidates
from dp_gen.labeling.submit_abacus import prepare_abacus_inputs, collect_results


class TestModelDeviation:
    """Test exploration model deviation computation and candidate selection."""

    def test_parse_model_deviation_output_keys(self) -> None:
        """Output should contain expected keys with correct shapes."""
        result = parse_model_deviation(Path("mock_traj"))
        assert "max_dev_f" in result
        assert "avg_dev_f" in result
        assert "frames" in result
        assert len(result["max_dev_f"]) == 5000

    def test_select_candidates_budget(self) -> None:
        """Candidate selection should respect max_candidates budget."""
        rng = np.random.default_rng(42)
        deviations = rng.uniform(0.0, 0.5, size=1000)
        mock_result = {
            "max_dev_f": deviations,
            "avg_dev_f": deviations * 0.6,
            "frames": np.arange(1000),
        }
        candidates = select_candidates(mock_result, trust_lo=0.05, trust_hi=0.25,
                                       max_candidates=100)
        assert len(candidates) <= 100

    def test_select_candidates_all_in_range(self) -> None:
        """Selected candidates should have deviation >= trust_lo."""
        rng = np.random.default_rng(42)
        deviations = rng.uniform(0.0, 0.5, size=1000)
        mock_result = {
            "max_dev_f": deviations,
            "avg_dev_f": deviations * 0.6,
            "frames": np.arange(1000),
        }
        candidates = select_candidates(mock_result, trust_lo=0.05, trust_hi=0.25,
                                       max_candidates=200)
        selected_devs = deviations[candidates]
        assert np.all(selected_devs >= 0.05), (
            f"Found {np.sum(selected_devs < 0.05)} candidates below trust_lo"
        )


class TestABACUSLabeling:
    """Test ABACUS labeling pipeline."""

    def test_prepare_abacus_inputs_creates_dirs(
        self, tmp_path: Path
    ) -> None:
        """Should create task directories with STRU, INPUT, KPT files."""
        template_dir = Path("abacus/templates/BF3")
        # Skip if templates not available
        if not template_dir.exists():
            pytest.skip("ABACUS templates not available")

        output_dir = tmp_path / "labeling"
        task_dirs = prepare_abacus_inputs(
            [0, 1, 2],
            template_dir=str(template_dir),
            output_dir=str(output_dir),
        )
        assert len(task_dirs) == 3
        for td in task_dirs:
            assert (Path(td) / "STRU").exists()
            assert (Path(td) / "INPUT").exists()
            assert (Path(td) / "KPT").exists()

    def test_collect_results_summary(self, tmp_path: Path) -> None:
        """collect_results should return correct summary counts."""
        output_file = str(tmp_path / "training_data")
        # Should handle empty task list gracefully
        result = collect_results([], output_file=output_file)
        assert result["total"] == 0
        assert result["converged"] == 0


class TestParamJSON:
    """Test dp_gen/param.json configuration file."""

    @pytest.fixture
    def param_json(self) -> dict:
        """Load param.json."""
        param_path = Path("dp_gen/param.json")
        if not param_path.exists():
            pytest.skip("dp_gen/param.json not found")
        with open(param_path) as f:
            return json.load(f)

    def test_required_sections(self, param_json: dict) -> None:
        """param.json must contain all required top-level sections."""
        required = ["type_map", "model_devi", "train", "dpgen_data", "scf_config"]
        for section in required:
            assert section in param_json, f"Missing section: {section}"

    def test_type_map_matches_bf3(self, param_json: dict) -> None:
        """type_map should contain B and F for BF3 system."""
        assert "B" in param_json["type_map"]
        assert "F" in param_json["type_map"]

    def test_model_devi_bounds_positive(self, param_json: dict) -> None:
        """Trust bounds should be positive and lo < hi."""
        md = param_json["model_devi"]
        assert md["trust_lo"] > 0
        assert md["trust_hi"] > md["trust_lo"]

    def test_scf_threshold_reasonable(self, param_json: dict) -> None:
        """SCF threshold should be physically reasonable (1e-5 to 1e-8 eV)."""
        scf_thr = param_json["scf_config"]["scf_thr"]
        assert 1e-8 <= scf_thr <= 1e-5

    def test_numb_models_at_least_4(self, param_json: dict) -> None:
        """Ensemble should have at least 4 models for reliable deviation."""
        assert param_json["model_devi"]["numb_models"] >= 4
