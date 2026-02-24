import pytest
import numpy as np
from unittest.mock import Mock, patch

import helpers

from cnf import CrystalNormalForm
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation.search_filters import (
    FilterSet,
    VolumeLimitFilter,
    MinDistanceFilter,
    EnergyFilter,
)


@pytest.fixture
def zr_bcc_cnf(zr_bcc_mp):
    cnfs, _ = get_endpoint_cnfs(zr_bcc_mp, zr_bcc_mp, xi=1.5, delta=10)
    return cnfs[0]


@pytest.fixture
def zr_hcp_cnf(zr_hcp_mp):
    cnfs, _ = get_endpoint_cnfs(zr_hcp_mp, zr_hcp_mp, xi=1.5, delta=10)
    return cnfs[0]


class TestVolumeLimitFilter:

    def test_accepts_within_bounds(self, zr_bcc_cnf):
        struct = zr_bcc_cnf.reconstruct()
        vol = struct.volume
        filt = VolumeLimitFilter(vol * 0.5, vol * 1.5)
        assert filt.should_add_pt(zr_bcc_cnf, struct)

    def test_rejects_below_lower_bound(self, zr_bcc_cnf):
        struct = zr_bcc_cnf.reconstruct()
        vol = struct.volume
        filt = VolumeLimitFilter(vol * 1.5, vol * 2.0)
        assert not filt.should_add_pt(zr_bcc_cnf, struct)

    def test_rejects_above_upper_bound(self, zr_bcc_cnf):
        struct = zr_bcc_cnf.reconstruct()
        vol = struct.volume
        filt = VolumeLimitFilter(vol * 0.1, vol * 0.5)
        assert not filt.should_add_pt(zr_bcc_cnf, struct)

    def test_boundary_is_exclusive(self, zr_bcc_cnf):
        struct = zr_bcc_cnf.reconstruct()
        vol = struct.volume
        filt = VolumeLimitFilter(vol, vol * 1.5)
        assert not filt.should_add_pt(zr_bcc_cnf, struct)

    def test_from_struct(self, zr_bcc_mp):
        filt = VolumeLimitFilter.from_struct(zr_bcc_mp, low_ratio=0.8, high_ratio=1.2)
        vol = zr_bcc_mp.volume
        assert filt.vll == pytest.approx(vol * 0.8)
        assert filt.vul == pytest.approx(vol * 1.2)

    def test_from_endpoint_structs(self, zr_bcc_mp, zr_hcp_mp):
        structs = [zr_bcc_mp, zr_hcp_mp]
        filt = VolumeLimitFilter.from_endpoint_structs(structs)
        volumes = [s.volume for s in structs]
        assert filt.vll == pytest.approx(min(volumes) * 0.8)
        assert filt.vul == pytest.approx(max(volumes) * 1.2)

    def test_from_cnf(self, zr_bcc_cnf):
        filt = VolumeLimitFilter.from_cnf(zr_bcc_cnf)
        vol = zr_bcc_cnf.reconstruct().volume
        assert filt.vll == pytest.approx(vol * 0.8)
        assert filt.vul == pytest.approx(vol * 1.2)


class TestMinDistanceFilter:

    def test_accepts_valid_distances(self, zr_bcc_cnf):
        filt = MinDistanceFilter(0.5)
        struct = zr_bcc_cnf.reconstruct()
        assert filt.should_add_pt(zr_bcc_cnf, struct)

    def test_rejects_large_threshold(self, zr_bcc_cnf):
        filt = MinDistanceFilter(100.0)
        struct = zr_bcc_cnf.reconstruct()
        assert not filt.should_add_pt(zr_bcc_cnf, struct)

    def test_from_structures(self, zr_bcc_mp, zr_hcp_mp):
        from cnf.navigation.utils import min_bond_length
        structs = [zr_bcc_mp, zr_hcp_mp]
        filt = MinDistanceFilter.from_structures(structs, ratio=0.75)
        expected = min_bond_length(structs) * 0.75
        assert filt.dist == pytest.approx(expected)

    def test_from_structures_default_ratio(self, zr_bcc_mp):
        from cnf.navigation.utils import min_bond_length
        filt = MinDistanceFilter.from_structures([zr_bcc_mp])
        expected = min_bond_length([zr_bcc_mp]) * MinDistanceFilter.DEFAULT_BOND_RATIO
        assert filt.dist == pytest.approx(expected)

    def test_empty_list_rust_path(self):
        with patch.dict('os.environ', {'USE_RUST': '1'}):
            filt = MinDistanceFilter(1.0)
            valid_cnfs, valid_structs = filt.filter_nbs([], [])
            assert valid_cnfs == []

    def test_python_path(self, zr_bcc_cnf):
        with patch.dict('os.environ', {'USE_RUST': '0'}):
            filt = MinDistanceFilter(0.5)
            cnfs = [zr_bcc_cnf]
            structs = [zr_bcc_cnf.reconstruct()]
            valid_cnfs, _ = filt.filter_nbs(cnfs, structs)
            assert len(valid_cnfs) == 1

    def test_requires_structs_depends_on_rust(self):
        with patch.dict('os.environ', {'USE_RUST': '0'}):
            filt = MinDistanceFilter(1.5)
            assert filt.requires_structs is True

        with patch.dict('os.environ', {'USE_RUST': '1'}):
            filt = MinDistanceFilter(1.5)
            assert filt.requires_structs is False


class TestEnergyFilter:

    def test_uses_cache(self):
        mock_calc = Mock()
        mock_calc.calculate_energy.return_value = -10.0
        mock_cnf = Mock()
        mock_cnf.coords = (1, 2, 3, 4, 5, 6, 7)

        filt = EnergyFilter(max_energy=-5.0, calc=mock_calc)
        filt.should_add_pt(mock_cnf, None)
        filt.should_add_pt(mock_cnf, None)

        assert mock_calc.calculate_energy.call_count == 1

    def test_accepts_below_max(self):
        mock_calc = Mock()
        mock_calc.calculate_energy.return_value = -10.0
        mock_cnf = Mock()
        mock_cnf.coords = (1, 2, 3)

        filt = EnergyFilter(max_energy=-5.0, calc=mock_calc)
        assert filt.should_add_pt(mock_cnf, None)

    def test_rejects_above_max(self):
        mock_calc = Mock()
        mock_calc.calculate_energy.return_value = -2.0
        mock_cnf = Mock()
        mock_cnf.coords = (1, 2, 3)

        filt = EnergyFilter(max_energy=-5.0, calc=mock_calc)
        assert not filt.should_add_pt(mock_cnf, None)

    def test_rejects_at_max(self):
        mock_calc = Mock()
        mock_calc.calculate_energy.return_value = -5.0
        mock_cnf = Mock()
        mock_cnf.coords = (1, 2, 3)

        filt = EnergyFilter(max_energy=-5.0, calc=mock_calc)
        assert not filt.should_add_pt(mock_cnf, None)

    def test_requires_structs_false(self):
        filt = EnergyFilter(max_energy=-5.0)
        assert not filt.requires_structs

    def test_custom_cache(self):
        cache = {(1, 2, 3): -10.0}
        filt = EnergyFilter(max_energy=-5.0, cache=cache)
        assert filt._cache is cache


class TestFilterSet:

    def test_empty_filters(self, zr_bcc_cnf):
        fs = FilterSet([])
        result_cnfs, _ = fs.filter_cnfs([zr_bcc_cnf])
        assert len(result_cnfs) == 1

    def test_validates_requires_structs(self):
        mock_filter = Mock()
        mock_filter.requires_structs = True
        with pytest.raises(RuntimeError):
            FilterSet([mock_filter], use_structs=False)

    def test_allows_non_struct_filters(self):
        mock_filter = Mock()
        mock_filter.requires_structs = False
        fs = FilterSet([mock_filter], use_structs=False)
        assert len(fs.filters) == 1

    def test_chains_filters(self, zr_bcc_cnf):
        struct = zr_bcc_cnf.reconstruct()
        vol = struct.volume

        vol_filter = VolumeLimitFilter(vol * 0.5, vol * 1.5)
        with patch.dict('os.environ', {'USE_RUST': '0'}):
            dist_filter = MinDistanceFilter(0.5)

        fs = FilterSet([vol_filter, dist_filter])
        result_cnfs, _ = fs.filter_cnfs([zr_bcc_cnf])
        assert len(result_cnfs) == 1

    def test_add_filter(self):
        fs = FilterSet([])
        mock_filter = Mock()
        mock_filter.requires_structs = True
        fs.add_filter(mock_filter)
        assert len(fs.filters) == 1

    def test_filter_nbs_cnf_only(self, zr_bcc_cnf):
        struct = zr_bcc_cnf.reconstruct()
        vol = struct.volume
        filt = VolumeLimitFilter(vol * 0.5, vol * 1.5)
        result = filt.filter_nbs_cnf_only([zr_bcc_cnf])
        assert len(result) == 1

    def test_filters_in_order(self, zr_bcc_cnf, zr_hcp_cnf):
        # First filter passes both
        mock_filter1 = Mock()
        mock_filter1.requires_structs = True
        mock_filter1.filter_nbs.return_value = (
            [zr_bcc_cnf, zr_hcp_cnf],
            [zr_bcc_cnf.reconstruct(), zr_hcp_cnf.reconstruct()]
        )

        # Second filter removes one
        mock_filter2 = Mock()
        mock_filter2.requires_structs = True
        mock_filter2.filter_nbs.return_value = (
            [zr_bcc_cnf],
            [zr_bcc_cnf.reconstruct()]
        )

        fs = FilterSet([mock_filter1, mock_filter2])
        result_cnfs, _ = fs.filter_cnfs([zr_bcc_cnf, zr_hcp_cnf])
        assert len(result_cnfs) == 1
