import helpers
from pymatgen.core.structure import Structure
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.lattice_neighbor_finder import LatticeNeighborFinder

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
def test_cnf_neighbors_are_close(idx, struct: Structure):
    verbose = True
    xi = 1.5
    delta = 30
    helpers.printif("", verbose)
    constructor = CNFConstructor(xi, delta, False) 

    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    # helpers.printif(f"Original CNF: {original_cnf.coords}", verbose)
    neigb_set = LatticeNeighborFinder(original_cnf, verbose_logging=True).find_cnf_neighbors()
    for n in neigb_set.neighbors:
        # helpers.printif(f"Neighbor CNF: {n.point.coords}", verbose)
        pdd = helpers.assertions.pdd_for_cnfs(n.point, original_cnf, k=100)
        # print(pdd)
        # cond = 
        # if not cond:
        #     n.point.to_file("cnf1.json")
        #     original_cnf.to_file("cnf2.json")
        exact_geo_matches, reason = helpers.are_cnfs_geo_matches(n.point, original_cnf)
        assert pdd < (xi / 2) and not exact_geo_matches, reason