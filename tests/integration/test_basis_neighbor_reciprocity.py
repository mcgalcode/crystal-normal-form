import pytest
import numpy as np
import helpers

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.basis_neighbor_finder import BasisStepResult, BasisNeighborFinder
from cnf.navigation.neighbor import Neighbor
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell

from pathlib import Path

STRUCT_SAMPLE_FREQ = 1

@helpers.parameterized_by_structs_with_num_sites_less_than(8)
def test_basis_neighbor_reciprocity(idx, struct: Structure):
    verbose = False
    xi = 1.5
    delta = 10

    helpers.printif("", verbose)
    helpers.printif(f"Attempting struct at idx {idx * STRUCT_SAMPLE_FREQ}", verbose)
    constructor = CNFConstructor(xi, delta, False) 

    struct = struct.to_primitive()
    
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    neighbor_set = BasisNeighborFinder(original_cnf).find_basis_neighbors()
    print(f"Structure had {len(struct)} sites")
    print(f"Found {len(neighbor_set)} basis neighbors!")
    recipricol_nbs = [] 
    nonreciprocal_nbs = []
    geo_rec_neighbs = []
    geo_matches = []
    for nb_idx, n in enumerate(neighbor_set.neighbors):
        assert n.point.lattice_normal_form == original_cnf.lattice_normal_form
        second_neighbors = BasisNeighborFinder(n.point).find_basis_neighbors()
        print(f"Found {len(neighbor_set)} basis SECOND neighbors!")
        if original_cnf not in second_neighbors:
            # helpers.save_cnfs_to_dir(f"patho_neighbor_pairs/mp_{idx}_nb_{nb_idx}", [original_cnf, n.point])
            nonreciprocal_nbs.append(n.point)
            num_geo_matches = 0

            for n2 in second_neighbors.neighbors:
                if helpers.are_cnfs_geo_matches(original_cnf, n2.point, tol=1e-7):
                    num_geo_matches += 1
                    geo_matches.append(n2.point)
            if num_geo_matches > 0:
                geo_rec_neighbs.append(n)
            
            helpers.printif(f"Found {num_geo_matches} geometrically identical second degree neighbs with same LNF!", verbose)
        else:
            recipricol_nbs.append(n)
    
    assert len(recipricol_nbs) == len(neighbor_set.neighbors)