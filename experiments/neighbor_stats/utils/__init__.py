import json
import os
import numpy as np
import tqdm

from pathlib import Path

from pymatgen.core.structure import Structure, Lattice

data_root = Path(os.environ["CNF_ROOT"]) /"experiments" / "neighbor_stats" / "data" 


def rand_lat_param():
    return np.random.random() * 7 + 5

def rand_lattice():
    roll = np.random.random()

    if roll < 0.33:
        return Lattice.cubic(rand_lat_param())
    elif roll < 0.67:
        return Lattice.tetragonal(rand_lat_param(), rand_lat_param())
    else:
        return Lattice.orthorhombic(rand_lat_param(), rand_lat_param(), rand_lat_param())

def rand_frac_coords():
    return np.random.random(3)

def rand_motif(num_sites = None):
    if num_sites is None:
        num_sites = np.random.randint(2,5)
    coords = [rand_frac_coords() for _ in range(num_sites)]
    atoms = ["Li" for _ in range(num_sites)]
    return atoms, coords

def rand_struct(num_sites = None):
    lat = rand_lattice()
    atoms, coords = rand_motif(num_sites=num_sites)
    return Structure(lat, atoms, coords)

def n_rand_structs(n, num_sites = None):
    return [rand_struct(num_sites=num_sites) for _ in range(n)]
