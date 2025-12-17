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

    xis = np.logspace(-2,0.2, 10, base=10)
    deltas = [int(i) for i in np.logspace(1,5, num=10,base=10)]
    
    data_pts = []
    for xi, delta in tqdm.tqdm(list(itertools.product(xis, deltas))):
        nb_cts = []
        for s in structs:
            cnf = UnitCell.from_pymatgen_structure(s).to_cnf(xi, delta)
            nbs = find_neighbors(cnf)
            num_nbs = len(nbs)
            nb_cts.append(num_nbs)
        
        data_pts.append({
            "xi": xi,
            "delta": delta,
            "num_nbs": np.mean(nb_cts)
        })
    with open(data_root / "disc_param_sweep.json", 'w+') as f:
        json.dump(data_pts, f)
 
if __name__ == "__main__":
    main()