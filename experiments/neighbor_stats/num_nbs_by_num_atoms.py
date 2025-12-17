import json
import os
import numpy as np
import tqdm
import itertools
from pathlib import Path

from pymatgen.core.structure import Structure, Lattice
from cnf.navigation import find_neighbors
from cnf import UnitCell
from utils import n_rand_structs


data_root = Path(os.environ["CNF_ROOT"]) /"experiments" / "neighbor_stats" / "data" 

def main():
    print("Computing neighbor counts as a function of xi, delta")

    structs = n_rand_structs(100)

    num_sites_values = range(1,20)

    delta = 1000
    xi = 0.5
    
    data_pts = []
    for num_sites in tqdm.tqdm(num_sites_values):
        structs = n_rand_structs(50, num_sites)
        nb_cts = []
        for s in structs:
            cnf = UnitCell.from_pymatgen_structure(s).to_cnf(xi, delta)
            nbs = find_neighbors(cnf)
            num_nbs = len(nbs)
            nb_cts.append(num_nbs)
        
        data_pts.append({
            "num_sites": num_sites,
            "num_nbs": np.mean(nb_cts)
        })
    with open(data_root / "nb_ct_vs_sites.json", 'w+') as f:
        json.dump(data_pts, f)
 
if __name__ == "__main__":
    main()