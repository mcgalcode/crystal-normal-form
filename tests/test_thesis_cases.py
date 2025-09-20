import pytest
import numpy as np

from cnf.lattice import Superbasis, VonormList
from cnf.lattice.lnf_constructor import LatticeNormalFormConstructor
from cnf.lattice.selling import VonormListSellingReducer
from cnf.lattice.rounding import DiscretizedVonormComputer
from cnf.sublattice.gamma_matrices import GammaMatrixTuple
from cnf.sublattice.sublattice_generator import SublatticeGenerator, transform_lattice_vecs

@pytest.fixture
def zr_fcc_lnfs_and_diff(zr_fcc_primitive_lattice_vecs):
    sl_generator = SublatticeGenerator.for_index(2)
    xi = 1.5

    expected_lnf_1 = (6, 7, 13, 25, 13, 18, 20)
    expected_lnf_2 = (7, 7, 20, 20, 7, 20, 27)

    sublattice_vec_sets = sl_generator.generate_sublattice_vector_sets(zr_fcc_primitive_lattice_vecs)
    constructor = LatticeNormalFormConstructor(xi)

    all_lnfs = []
    for v in sublattice_vec_sets:
        print("New lattice!")
        all_lnfs.append(constructor.build_lnf_from_generating_vecs(v).lnf)
        print(all_lnfs[-1])

    sublattice_organizer = {
        1: [],
        2: []
    }

    for lnf, gm, slvs in zip(all_lnfs,
                             sl_generator.generating_matrix_group.ordered_matrices,
                             sublattice_vec_sets):
        
        diff1 = np.array(lnf.coords) - np.array(expected_lnf_1)
        diff2 = np.array(lnf.coords) - np.array(expected_lnf_2)
        minimum_diff_idx = np.argmin([np.abs(diff1).sum(), np.abs(diff2).sum()])
        diff = [diff1, diff2][minimum_diff_idx]
        matching_lnf = [expected_lnf_1, expected_lnf_2][minimum_diff_idx]
        sublattice_organizer[minimum_diff_idx + 1].append({
            "computed_lnf": lnf,
            "expected_lnf": matching_lnf,
            "diff": diff,
            "gamma_mat": gm,
            "generating_vecs": slvs
        })
        
    return sublattice_organizer

# @pytest.mark.skip
def test_produces_zr_bcc_vonorm_set_sublattice_1(zr_bcc_primitive_lattice_vecs):
    transform = GammaMatrixTuple(np.array([
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]))
    vecs = transform_lattice_vecs(zr_bcc_primitive_lattice_vecs, transform)
    xi = 1.5

    sb = Superbasis.from_generating_vecs(vecs)
    vnorms = DiscretizedVonormComputer.discretize_vonorm_list(sb.compute_vonorms(), xi)
    
    expected_thesis_vonorms = VonormList([6, 8, 17, 23, 6, 23, 25])
    assert vnorms.is_obtuse()
    assert sb.is_obtuse(tol=1e-5)
    # Note: because of our enhanced permutation sets, we might not
    # get the same maximally ascending string. If they are different
    # then our should be "greater" (more maximally ascending)
    assert vnorms.has_same_members(expected_thesis_vonorms)
    assert vnorms.tuple >= expected_thesis_vonorms.tuple

# @pytest.mark.xfail
def test_produces_zr_bcc_vonorm_set_sublattice_2(zr_bcc_primitive_lattice_vecs):
    transform = GammaMatrixTuple(np.array([
        [2, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ]))
    vecs = transform_lattice_vecs(zr_bcc_primitive_lattice_vecs, transform)
    xi = 1.5

    sb = Superbasis.from_generating_vecs(vecs)
    vnorms = sb.compute_vonorms()
    lnf_constructor = LatticeNormalFormConstructor(xi)
    lnf = lnf_constructor.build_lnf_from_vonorms(vnorms).lnf
    print(lnf)

    expected_thesis_vonorms = VonormList([8, 8, 8, 21, 15, 15, 15])
    assert lnf.vonorms.is_obtuse()
    # assert sb.is_obtuse(tol=1e-5)
    print(expected_thesis_vonorms)
    assert lnf.vonorms.has_same_members(expected_thesis_vonorms)
    assert vnorms.tuple >= expected_thesis_vonorms.tuple


# @pytest.mark.skip
def test_two_distinct_fcc_sublattice_lnfs(zr_fcc_lnfs_and_diff):
    print()
    for lnf_class, lnf_datas in zr_fcc_lnfs_and_diff.items():
        assert len(set([d['computed_lnf'] for d in lnf_datas])) == 1
        for idx, d in enumerate(lnf_datas):
            lnf = d['computed_lnf']
            actual_lnf = d['expected_lnf']
            diff = d['diff']

            print(f"{idx} Computed LNF: {lnf.coords}, Hoped for LNF: {actual_lnf} (class {lnf_class}), diff: {diff}")

def _trace_lnf(x, xi):
    computed_lnf = x['computed_lnf']
    actual_lnf = x['expected_lnf']
    diff = x['diff']
    vecs = x['generating_vecs']

    print()
    print(f"Case 1: {computed_lnf}")
    sb = Superbasis.from_generating_vecs(vecs)
    vnorms = sb.compute_vonorms()    
    lnf_constructor = LatticeNormalFormConstructor(xi, verbose_logging=True)
    lnf = lnf_constructor.build_lnf_from_vonorms(vnorms).lnf
    print(lnf)
    assert lnf == computed_lnf
# @pytest.mark.skip
def test_trace_example_fcc_lnf_construction(zr_fcc_lnfs_and_diff):
    xi=1.5

    d1 = zr_fcc_lnfs_and_diff[2][2]
    d2 = zr_fcc_lnfs_and_diff[2][3]
    assert d1["computed_lnf"] == d2["computed_lnf"]
    print()

    # Check if the undiscretized lattices are the same:
    def _get_selling_reduced_vonorms(vecs):
        sb = Superbasis.from_generating_vecs(vecs)
        vnorms = sb.compute_vonorms()
        reducer = VonormListSellingReducer()
        vnorms = reducer.reduce(vnorms).reduced_object
        return vnorms
    
    d1_undiscretized = _get_selling_reduced_vonorms(d1['generating_vecs'])
    d2_undiscretized = _get_selling_reduced_vonorms(d2['generating_vecs'])
    assert d1_undiscretized.has_same_members(d2_undiscretized)

    # _trace_lnf(d1, xi)
    # _trace_lnf(d2, xi)

def test_debug_fcc_class_one_distinct_members(zr_fcc_lnfs_and_diff):
    xi=1.5

    d1 = zr_fcc_lnfs_and_diff[1][0]
    d2 = zr_fcc_lnfs_and_diff[1][1]
    # assert d1["computed_lnf"] == d2["computed_lnf"]
    print()

    # Check if the undiscretized lattices are the same:
    def _get_selling_reduced_vonorms(vecs):
        sb = Superbasis.from_generating_vecs(vecs)
        vnorms = sb.compute_vonorms()
        reducer = VonormListSellingReducer()
        vnorms = reducer.reduce(vnorms).reduced_object
        return vnorms
    
    d1_undiscretized = _get_selling_reduced_vonorms(d1['generating_vecs'])
    d2_undiscretized = _get_selling_reduced_vonorms(d2['generating_vecs'])
    assert d1_undiscretized.has_same_members(d2_undiscretized)

    # _trace_lnf(d1, xi)
    # _trace_lnf(d2, xi)

