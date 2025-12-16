import helpers
from pymatgen.core.structure import Structure
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.navigation.motif_neighbor_finder import MotifNeighborFinder

@helpers.parameterized_by_mp_struct_idxs(every=20)
def test_lattice_neighbs_neighbors_are_close(idx, struct: Structure):
    verbose = True
    xi = 1.5
    delta = 30
    helpers.printif("", verbose)
    constructor = CNFConstructor(xi, delta, False) 

    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    nf = NeighborFinder.from_cnf(original_cnf)
    # helpers.printif(f"Original CNF: {original_cnf.coords}", verbose)
    neigb_set = nf.find_lattice_neighbor_cnfs(original_cnf)
    for n in neigb_set:
        # helpers.printif(f"Neighbor CNF: {n.point.coords}", verbose)
        pdd = helpers.assertions.pdd_for_cnfs(n, original_cnf, k=100)
        # print(pdd)
        # cond = 
        # if not cond:
        #     n.to_file("cnf1.json")
        #     original_cnf.to_file("cnf2.json")
        exact_geo_matches, reason = helpers.are_cnfs_geo_matches(n, original_cnf)
        assert pdd < (xi / 2) and not exact_geo_matches, reason

@helpers.parameterized_by_mp_struct_idxs(every=20)
def test_basis_neighbs_neighbors_are_close(idx, struct: Structure):
    verbose = True
    xi = 1.5
    delta = 30
    helpers.printif("", verbose)
    constructor = CNFConstructor(xi, delta, False) 

    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    nf = NeighborFinder.from_cnf(original_cnf)

    neigb_set = nf.find_motif_neighbor_cnfs(original_cnf)
    for n in neigb_set:
        pdd = helpers.assertions.pdd_for_cnfs(n, original_cnf, k=100)
        exact_geo_matches, reason = helpers.are_cnfs_geo_matches(n, original_cnf)
        assert pdd < (xi / 2) and not exact_geo_matches, reason