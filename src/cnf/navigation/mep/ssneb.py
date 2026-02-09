"""
SSNEB wrapper utilities for running solid-state NEB from within the CNF framework.

Thin wrapper around tsase's SSNEB implementation and the extracted utility
functions in tsase.neb.ssneb_utils. Provides convenience functions for
converting between pymatgen Structures and ASE Atoms, running SSNEB, and
retrieving results.
"""

import os
import numpy as np

from pymatgen.core import Structure, Lattice
from ase import Atoms

from tsase.neb.ssneb_utils import (
    compute_jacobian,
    interpolate_path,
    initialize_image_properties,
    image_distance_vector,
)
from tsase.neb.util import vmag

def cell_to_lower_triangular(cell):
    """Orient a cell matrix to lower-triangular form via QR decomposition.

    tsase requires cells in lower-triangular orientation (a along x, b on
    the xy-plane). This function achieves that via QR decomposition.

    Args:
        cell: 3x3 numpy array (cell matrix, rows = lattice vectors).

    Returns:
        3x3 lower-triangular cell matrix.
    """
    Q, R = np.linalg.qr(cell.T)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs
    R = R * signs[:, np.newaxis]
    lt = R.T
    lt[0, 1] = 0.0
    lt[0, 2] = 0.0
    lt[1, 2] = 0.0
    return lt


def prepare_ssneb_image(struct):
    """Convert a pymatgen Structure to an ASE Atoms with lower-triangular cell.

    Args:
        struct: A pymatgen Structure.

    Returns:
        An ASE Atoms object with lower-triangular cell orientation.
    """
    lt_cell = cell_to_lower_triangular(struct.lattice.matrix)
    return Atoms(
        symbols=[str(sp) for sp in struct.species],
        scaled_positions=struct.frac_coords,
        cell=lt_cell,
        pbc=True,
    )


def run_ssneb(images, calc, method='ci', spring_k=5.0, max_iters=1000,
              force_tol=0.05, ss=True, work_dir=None):
    """Run solid-state NEB using tsase.

    Args:
        images: List of ASE Atoms objects (endpoints + intermediates).
            Endpoints should already have the calculator set. Intermediate
            images will have calc assigned automatically.
        calc: ASE calculator instance for energy/force evaluation.
        method: NEB method — 'ci' for climbing image, 'normal' otherwise.
        spring_k: Spring force constant.
        max_iters: Maximum optimizer iterations.
        force_tol: Force convergence tolerance (eV/Angstrom).
        ss: If True, use solid-state NEB (cell deformation). If False,
            fixed-cell NEB.
        work_dir: Directory to run in. Created if it doesn't exist.
            If None, runs in current directory.

    Returns:
        List of energies (one per image, including endpoints).
    """
    from tsase import neb

    orig_dir = os.getcwd()
    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)

    try:
        p1 = images[0].copy()
        p2 = images[-1].copy()
        p1.calc = calc
        p2.calc = calc
        n = len(images)

        band = neb.ssneb(p1, p2, numImages=n, k=spring_k, method=method, ss=ss)

        # Replace linearly interpolated intermediates with our images
        for i in range(1, n - 1):
            band.path[i].set_cell(images[i].get_cell())
            band.path[i].set_positions(images[i].get_positions())
            band.path[i].set_calculator(calc)

        opt = neb.fire_ssneb(band, maxmove=0.1, dt=0.05, dtmax=0.5,
                             trajectory='ssneb.traj')
        opt.minimize(forceConverged=force_tol, maxIterations=max_iters)

        energies = []
        for img in band.path:
            energies.append(img.u if hasattr(img, 'u') else img.get_potential_energy())
        return energies
    finally:
        os.chdir(orig_dir)
