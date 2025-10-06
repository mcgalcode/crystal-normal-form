import pytest
import helpers

from pymatgen.core.structure import Structure

from cnf.lattice.voronoi import VonormList
from cnf.lattice import Superbasis
from cnf.motif import FractionalMotif
from cnf.lattice.selling import VonormListSellingReducer


def test_selling_reduction_maintains_crystal(mp_structures):
    for struct in mp_structures[::100]:
        sb = Superbasis.from_pymatgen_structure(struct)
        vn = sb.compute_vonorms()
        reduction_result = VonormListSellingReducer().reduce(vn)
        vn: VonormList = reduction_result.reduced_object
        motif = FractionalMotif.from_pymatgen_structure(struct)
        motif = motif.apply_unimodular(reduction_result.transform_matrix)

        recovered = Structure(vn.to_superbasis().generating_vecs(), motif.atoms, motif.positions)
        helpers.assert_identical_by_pdd_distance(struct, recovered, cutoff=0.00001)
