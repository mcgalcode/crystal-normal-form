import pytest
import helpers

from pymatgen.core.structure import Structure

from cnf.lattice.voronoi import VonormList
from cnf.lattice import Superbasis
from cnf.motif import FractionalMotif
from cnf.lattice.selling import VonormListSellingReducer
from cnf.lattice.selling import SuperbasisSellingReducer


@helpers.parameterized_by_mp_structs
def test_vn_selling_reduction_maintains_crystal(idx: int, struct: Structure):
    sb = Superbasis.from_pymatgen_structure(struct)
    vn = sb.compute_vonorms()
    reduction_result = VonormListSellingReducer(verbose_logging=True).reduce(vn)
    vn: VonormList = reduction_result.reduced_object
    motif = FractionalMotif.from_pymatgen_structure(struct)
    motif = motif.apply_unimodular(reduction_result.transform_matrix)

    recovered = Structure(vn.to_superbasis().generating_vecs(), motif.atoms, motif.positions)
    helpers.assert_identical_by_pdd_distance(struct, recovered, cutoff=0.02)


@helpers.parameterized_by_mp_structs
def test_sb_selling_reduction_maintains_crystal(idx: int, struct: Structure):
    sb = Superbasis.from_pymatgen_structure(struct)
    reduction_result = SuperbasisSellingReducer(verbose_logging=True).reduce(sb)
    sb: VonormList = reduction_result.reduced_object
    motif = FractionalMotif.from_pymatgen_structure(struct)
    motif = motif.apply_unimodular(reduction_result.transform_matrix)

    recovered = Structure(sb.generating_vecs(), motif.atoms, motif.positions)
    helpers.assert_identical_by_pdd_distance(struct, recovered, cutoff=0.00001)
