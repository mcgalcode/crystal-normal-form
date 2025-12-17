import json
import os
import numpy as np
import tqdm

from pathlib import Path

from cnf import UnitCell
from cnf.utils.pdd import pdd
from utils import n_rand_structs
from itertools import product
from cnf.calculation.grace import GraceCalculator

data_root = Path(os.environ["CNF_ROOT"]) /"experiments" / "neighbor_stats" / "data" 

def main():
    print("Computing neighbor counts as a function of xi, delta")

    structs = n_rand_structs(100)

    xis = np.logspace(-2,0.2, 10, base=10)
    deltas = [int(i) for i in np.logspace(1,5, num=10,base=10)]
    data_pts = []

    calc = GraceCalculator()


    for xi, delta in tqdm.tqdm(list(product(xis, deltas))):
        pdds = []
        for struct in structs:
            uc = UnitCell.from_pymatgen_structure(struct)
    
            cnf = uc.to_cnf(xi, delta)
            cnf = calc.relax(cnf)
            recon = cnf.reconstruct()
            pdds.append(pdd(recon, struct))
        data_pts.append({
            "xi": xi,
            "delta": delta,
            "pdd": np.mean(pdds)
        })

    with open(data_root / "pdd_by_xi_delta.json", 'w+') as f:
        json.dump(data_pts, f)
 
if __name__ == "__main__":
    main()