"""Unit tests for cnf.navigation.utils."""

import math
import pytest
import numpy as np

from cnf.navigation.utils import compute_delta_for_step_size, min_bond_length
from cnf import UnitCell


class TestComputeDeltaForStepSize:
    """Tests for compute_delta_for_step_size function."""

    def test_anatase_various_step_sizes(self, ti_o2_anatase):
        """Test delta computation for TiO2 anatase with various step sizes."""
        lattice = ti_o2_anatase.lattice
        max_param = max(lattice.a, lattice.b, lattice.c)

        for target_step in [0.5, 0.3, 0.2, 0.1]:
            delta = compute_delta_for_step_size(ti_o2_anatase, target_step)
            actual_step = max_param / delta

            # Delta should be positive integer
            assert isinstance(delta, int)
            assert delta > 0

            # Actual step should be <= target (the guarantee)
            assert actual_step <= target_step

            # Should be close to but not exceed target
            assert actual_step >= target_step - (max_param / delta)

    def test_rutile_various_step_sizes(self, ti_o2_rutile):
        """Test delta computation for TiO2 rutile with various step sizes."""
        lattice = ti_o2_rutile.lattice
        max_param = max(lattice.a, lattice.b, lattice.c)

        for target_step in [0.5, 0.3, 0.2, 0.1]:
            delta = compute_delta_for_step_size(ti_o2_rutile, target_step)
            actual_step = max_param / delta

            assert isinstance(delta, int)
            assert delta > 0
            assert actual_step <= target_step

    def test_unit_cell_input(self, ti_o2_anatase):
        """Test that UnitCell input works correctly."""
        uc = UnitCell.from_pymatgen_structure(ti_o2_anatase)
        delta_from_struct = compute_delta_for_step_size(ti_o2_anatase, 0.3)
        delta_from_uc = compute_delta_for_step_size(uc, 0.3)
        assert delta_from_struct == delta_from_uc

    def test_mathematical_correctness(self, ti_o2_anatase):
        """Verify the formula: delta = ceil(max_param / step_size)."""
        lattice = ti_o2_anatase.lattice
        max_param = max(lattice.a, lattice.b, lattice.c)

        target_step = 0.3
        delta = compute_delta_for_step_size(ti_o2_anatase, target_step)
        expected = int(math.ceil(max_param / target_step))
        assert delta == expected

    def test_larger_step_gives_smaller_delta(self, ti_o2_anatase):
        """Larger step sizes should give smaller delta values."""
        delta_05 = compute_delta_for_step_size(ti_o2_anatase, 0.5)
        delta_03 = compute_delta_for_step_size(ti_o2_anatase, 0.3)
        delta_01 = compute_delta_for_step_size(ti_o2_anatase, 0.1)

        assert delta_05 < delta_03 < delta_01


class TestMinBondLength:
    """Tests for min_bond_length function."""

    def test_single_structure(self, ti_o2_anatase):
        """Test min bond length for a single structure."""
        min_dist = min_bond_length(ti_o2_anatase)

        # TiO2 should have Ti-O bonds around 1.9-2.0 Angstroms
        assert 1.5 < min_dist < 2.5
        assert isinstance(min_dist, float)

    def test_multiple_structures(self, ti_o2_anatase, ti_o2_rutile):
        """Test min bond length across multiple structures."""
        min_dist = min_bond_length([ti_o2_anatase, ti_o2_rutile])

        # Should be the minimum across both
        min_anatase = min_bond_length(ti_o2_anatase)
        min_rutile = min_bond_length(ti_o2_rutile)
        assert min_dist == min(min_anatase, min_rutile)

    def test_unit_cell_input(self, ti_o2_anatase):
        """Test that UnitCell input works."""
        uc = UnitCell.from_pymatgen_structure(ti_o2_anatase)
        min_from_struct = min_bond_length(ti_o2_anatase)
        min_from_uc = min_bond_length(uc)
        assert abs(min_from_struct - min_from_uc) < 1e-6
