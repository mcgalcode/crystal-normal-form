"""Tests for iterative A* energy utilities."""

import pytest

from cnf.navigation.astar.iterative.core.energy import (
    evaluate_path_energies,
    path_barrier,
)


class MockCalculator:
    """Mock energy calculator for testing."""

    def __init__(self, energy_map=None):
        """
        Args:
            energy_map: Dict mapping coords tuples to energies.
                If None, returns hash-based pseudo-random energies.
        """
        self.energy_map = energy_map or {}
        self.call_count = 0

    def calculate_energy(self, cnf):
        self.call_count += 1
        coords = cnf.coords
        if coords in self.energy_map:
            return self.energy_map[coords]
        # Default: use sum of coords as energy
        return sum(coords) * 0.1


class MockCNF:
    """Mock CNF for testing."""

    def __init__(self, coords):
        self.coords = coords

    @classmethod
    def from_tuple(cls, coords, elements, xi, delta):
        return cls(coords)


class TestPathBarrier:
    def test_returns_max_energy(self):
        energies = [-10.0, -8.5, -12.0, -9.0]
        assert path_barrier(energies) == -8.5

    def test_single_energy(self):
        energies = [-10.0]
        assert path_barrier(energies) == -10.0

    def test_all_same(self):
        energies = [-10.0, -10.0, -10.0]
        assert path_barrier(energies) == -10.0

    def test_positive_energies(self):
        energies = [1.0, 5.0, 3.0, 2.0]
        assert path_barrier(energies) == 5.0


class TestEvaluatePathEnergies:
    def test_evaluates_all_points(self, monkeypatch):
        # Patch CrystalNormalForm.from_tuple
        import cnf.navigation.astar.iterative.core.energy as energy_module
        monkeypatch.setattr(energy_module, "CrystalNormalForm", MockCNF)

        path_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        energy_map = {
            (1, 2, 3): -10.0,
            (4, 5, 6): -8.0,
            (7, 8, 9): -12.0,
        }
        calc = MockCalculator(energy_map)
        cache = {}

        energies = evaluate_path_energies(
            path_tuples, ["Zr"], 1.5, 10, calc, cache
        )

        assert energies == [-10.0, -8.0, -12.0]
        assert calc.call_count == 3

    def test_uses_cache(self, monkeypatch):
        import cnf.navigation.astar.iterative.core.energy as energy_module
        monkeypatch.setattr(energy_module, "CrystalNormalForm", MockCNF)

        path_tuples = [(1, 2, 3), (4, 5, 6), (1, 2, 3)]  # First point repeated
        energy_map = {
            (1, 2, 3): -10.0,
            (4, 5, 6): -8.0,
        }
        calc = MockCalculator(energy_map)
        cache = {}

        energies = evaluate_path_energies(
            path_tuples, ["Zr"], 1.5, 10, calc, cache
        )

        assert energies == [-10.0, -8.0, -10.0]
        # Only 2 calls because (1,2,3) is cached after first evaluation
        assert calc.call_count == 2

    def test_populates_cache(self, monkeypatch):
        import cnf.navigation.astar.iterative.core.energy as energy_module
        monkeypatch.setattr(energy_module, "CrystalNormalForm", MockCNF)

        path_tuples = [(1, 2, 3), (4, 5, 6)]
        energy_map = {
            (1, 2, 3): -10.0,
            (4, 5, 6): -8.0,
        }
        calc = MockCalculator(energy_map)
        cache = {}

        evaluate_path_energies(path_tuples, ["Zr"], 1.5, 10, calc, cache)

        assert cache[(1, 2, 3)] == -10.0
        assert cache[(4, 5, 6)] == -8.0

    def test_uses_pre_populated_cache(self, monkeypatch):
        import cnf.navigation.astar.iterative.core.energy as energy_module
        monkeypatch.setattr(energy_module, "CrystalNormalForm", MockCNF)

        path_tuples = [(1, 2, 3), (4, 5, 6)]
        calc = MockCalculator()  # Empty map - would use default
        cache = {
            (1, 2, 3): -99.0,  # Pre-populated
            (4, 5, 6): -88.0,  # Pre-populated
        }

        energies = evaluate_path_energies(
            path_tuples, ["Zr"], 1.5, 10, calc, cache
        )

        assert energies == [-99.0, -88.0]
        assert calc.call_count == 0  # No calls needed

    def test_empty_path(self, monkeypatch):
        import cnf.navigation.astar.iterative.core.energy as energy_module
        monkeypatch.setattr(energy_module, "CrystalNormalForm", MockCNF)

        calc = MockCalculator()
        cache = {}

        energies = evaluate_path_energies([], ["Zr"], 1.5, 10, calc, cache)

        assert energies == []
        assert calc.call_count == 0
