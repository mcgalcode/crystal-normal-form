import json
import os
import numpy as np
import tqdm
import itertools
from pathlib import Path

from pymatgen.core.structure import Structure, Lattice
from cnf.navigation import find_neighbors, NeighborFinder
from cnf import UnitCell
from utils import n_rand_structs


data_root = Path(os.environ["CNF_ROOT"]) /"experiments" / "neighbor_stats" / "data" 

def main():
    print("Computing neighbor counts as a function of xi, delta")


    def get_nb_rings(pt, num_rings):
        nf = NeighborFinder.from_cnf(pt)
        rings = [[pt.coords]]
        already_explored = set([pt])
        for _ in range(num_rings):
            prev_ring = rings[-1]
            curr_ring = set([])
            for prev_pt in prev_ring:
                nbs = nf.find_neighbor_tuples(prev_pt)
                curr_ring = curr_ring.union(nbs)
            curr_ring = list(curr_ring.difference(already_explored))
            already_explored = already_explored.union(curr_ring)
            rings.append(curr_ring)
        return rings
    
    num_sites_values = range(1,2)

    delta = 1000
    xi = 0.5
    
    NUM_RINGS=8
    
    data_pts = []
    for num_sites in tqdm.tqdm(num_sites_values):
        structs = n_rand_structs(1, num_sites)
        ring_sets = []

        for s in structs:
            cnf = UnitCell.from_pymatgen_structure(s).to_cnf(xi, delta)
            ring_sets.append(get_nb_rings(cnf, NUM_RINGS))
        
        data_pts.append({
            "num_sites": num_sites,
            "ring_sets": ring_sets
        })

    with open(data_root / "rings.json", 'w+') as f:
        json.dump(data_pts, f)
 
if __name__ == "__main__":
    main()