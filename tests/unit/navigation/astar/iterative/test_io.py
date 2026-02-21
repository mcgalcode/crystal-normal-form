"""Tests for iterative A* I/O utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from cnf.navigation.astar.iterative._io import (
    write_round_json,
    write_energy_cache,
    write_manifest,
    serialize_result,
    ceiling_params_dict,
)


class TestWriteRoundJson:
    def test_writes_round_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "rounds").mkdir()

            round_data = {"round": 0, "paths": [], "ceiling": -10.5}
            write_round_json(output_dir, 0, round_data)

            round_file = output_dir / "rounds" / "round_000.json"
            assert round_file.exists()

            with open(round_file) as f:
                loaded = json.load(f)
            assert loaded == round_data

    def test_round_number_formatting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "rounds").mkdir()

            write_round_json(output_dir, 5, {"round": 5})
            write_round_json(output_dir, 42, {"round": 42})
            write_round_json(output_dir, 123, {"round": 123})

            assert (output_dir / "rounds" / "round_005.json").exists()
            assert (output_dir / "rounds" / "round_042.json").exists()
            assert (output_dir / "rounds" / "round_123.json").exists()


class TestWriteEnergyCache:
    def test_writes_cache_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            cache = {
                (1, 2, 3): -10.5,
                (4, 5, 6): -11.2,
            }
            write_energy_cache(output_dir, cache)

            cache_file = output_dir / "energy_cache.json"
            assert cache_file.exists()

            with open(cache_file) as f:
                loaded = json.load(f)

            # Keys are stringified tuples
            assert "(1, 2, 3)" in loaded
            assert loaded["(1, 2, 3)"] == -10.5

    def test_empty_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_energy_cache(output_dir, {})

            cache_file = output_dir / "energy_cache.json"
            assert cache_file.exists()

            with open(cache_file) as f:
                loaded = json.load(f)
            assert loaded == {}


class TestWriteManifest:
    def test_writes_manifest_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            params = {"xi": 1.5, "delta": 10}
            result = {"barrier": -10.5, "path_length": 42}
            timing = {"total_seconds": 123.4}

            write_manifest(output_dir, params, result, timing)

            manifest_file = output_dir / "manifest.json"
            assert manifest_file.exists()

            with open(manifest_file) as f:
                loaded = json.load(f)

            assert loaded["parameters"] == params
            assert loaded["result"] == result
            assert loaded["timing"] == timing


class TestSerializeResult:
    def test_serialize_found_result(self):
        result = {
            "ceiling": -10.0,
            "found": True,
            "iterations": 500,
            "barrier": -10.5,
            "path_length": 42,
            "path": [(1, 2, 3), (4, 5, 6)],
            "energies": [-10.2, -10.5],
            "extra_field": "should_be_ignored",
        }

        serialized = serialize_result(result)

        assert serialized["ceiling"] == -10.0
        assert serialized["found"] is True
        assert serialized["iterations"] == 500
        assert serialized["barrier"] == -10.5
        assert serialized["path_length"] == 42
        assert serialized["path"] == [[1, 2, 3], [4, 5, 6]]
        assert serialized["energies"] == [-10.2, -10.5]
        assert "extra_field" not in serialized

    def test_serialize_not_found_result(self):
        result = {
            "ceiling": -10.0,
            "found": False,
            "iterations": 1000,
        }

        serialized = serialize_result(result)

        assert serialized["ceiling"] == -10.0
        assert serialized["found"] is False
        assert serialized["iterations"] == 1000
        assert "barrier" not in serialized
        assert "path" not in serialized


class TestCeilingParamsDict:
    def test_builds_params_dict(self):
        # Create mock CNF-like objects with coords attribute
        class MockCNF:
            def __init__(self, coords, elements):
                self.coords = coords
                self.elements = elements

        start_cnfs = [MockCNF((1, 2, 3), ["Zr"])]
        goal_cnfs = [MockCNF((4, 5, 6), ["Zr"])]

        params = ceiling_params_dict(
            xi=1.5, delta=10, step_per_atom=0.5, num_ceilings=5,
            attempts_per_ceiling=3, max_passes=5, max_sweep_rounds=20,
            xi_factor=0.8, delta_factor=1.2, dropout=0.1, min_dropout=0.0,
            beam_width=1000, heuristic_mode="manhattan", heuristic_weight=0.5,
            n_workers=4, start_cnfs=start_cnfs, goal_cnfs=goal_cnfs,
            relax_endpoints=True,
        )

        assert params["algorithm"] == "ceiling_barrier_search"
        assert params["xi_initial"] == 1.5
        assert params["delta_initial"] == 10
        assert params["elements"] == ["Zr"]
        assert params["start_cnf_coords"] == [[1, 2, 3]]
        assert params["goal_cnf_coords"] == [[4, 5, 6]]
        assert params["relax_endpoints"] is True

    def test_empty_cnfs(self):
        params = ceiling_params_dict(
            xi=1.5, delta=10, step_per_atom=0.5, num_ceilings=5,
            attempts_per_ceiling=1, max_passes=5, max_sweep_rounds=20,
            xi_factor=0.8, delta_factor=1.2, dropout=0.1, min_dropout=0.0,
            beam_width=1000, heuristic_mode="manhattan", heuristic_weight=0.5,
            n_workers=1, start_cnfs=[], goal_cnfs=[],
        )

        assert params["elements"] == []
        assert params["start_cnf_coords"] == []
        assert params["goal_cnf_coords"] == []
