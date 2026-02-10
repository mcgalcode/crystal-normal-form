"""
SSNEB wrapper utilities for running solid-state NEB from within the CNF framework.

Thin wrapper around tsase's SSNEB implementation and the extracted utility
functions in tsase.neb.ssneb_utils. Provides convenience functions for
converting between pymatgen Structures and ASE Atoms, running SSNEB, and
retrieving results.
"""

import os
import math
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.core.trajectory import Trajectory

from ase import Atoms
from ...crystal_normal_form import CrystalNormalForm
from ..endpoints import get_endpoint_unit_cells

from .paths import (
    align_cnf_path,
    align_structure_to_reference,
    resample_path_by_distance,
)

from tsase.neb.ssneb_utils import (
    compute_jacobian,
    interpolate_path,
    initialize_image_properties,
    image_distance_vector,
)
from tsase import neb

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


def compute_ssneb_distances(structs: list[Structure], weight=1.0):
    """Compute pairwise SSNEB-metric distances between consecutive structures.

    Uses the tsase SSNEB distance metric: Jacobian-scaled cell strain +
    PBC-wrapped fractional coords in Cartesian, combined via Euclidean norm.

    Args:
        structs: List of pymatgen Structures (should be pre-aligned).
        weight: Relative weight of cell vs atomic degrees of freedom.

    Returns:
        List of floats, length len(structs) - 1. distances[i] is the
        SSNEB distance from structs[i] to structs[i+1].
    """
    # Convert to ASE Atoms
    atoms_list = [s.to_ase_atoms() for s in structs]

    # Compute Jacobian from endpoints
    vol1 = atoms_list[0].get_volume()
    vol2 = atoms_list[-1].get_volume()
    natom = len(atoms_list[0])
    jacobian = compute_jacobian(vol1, vol2, natom, weight)

    # Initialize image properties
    for a in atoms_list:
        initialize_image_properties(a, jacobian)

    # Compute consecutive distances
    distances = []
    for i in range(len(atoms_list) - 1):
        dv = image_distance_vector(atoms_list[i], atoms_list[i + 1])
        distances.append(float(vmag(dv)))

    return distances

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

def get_interpolated_images(endpt1, endpt2, num_images):
    start_cells, end_cells = get_endpoint_unit_cells(endpt1, endpt2)
    min_dist = math.inf
    selected_start_struct = None
    selected_end_struct = None

    for sc in start_cells:
        for ec in end_cells:
            sc_struct = sc.to_pymatgen_structure()
            ec_struct = ec.to_pymatgen_structure()
            ec_struct = align_structure_to_reference(sc_struct, ec_struct)
            dist = compute_ssneb_distances([sc_struct, ec_struct])[0]
            if dist < min_dist:
                selected_start_struct = sc_struct
                selected_end_struct= ec_struct

    img1 = prepare_ssneb_image(selected_start_struct)
    img2 = prepare_ssneb_image(selected_end_struct)
    images: list[Atoms] = interpolate_path(img1, img2, num_images)
    structs = [Structure.from_ase_atoms(i) for i in images]
    return structs

def _band_to_structures(band):
    """Convert a tsase ssneb band to a list of pymatgen Structures."""
    structs = []
    for img in band.path:
        structs.append(Structure(
            lattice=img.get_cell(),
            species=img.get_chemical_symbols(),
            coords=img.get_scaled_positions(),
            coords_are_cartesian=False,
        ))
    return structs


def run_ssneb(images, calc, method='ci', spring_k=5.0, max_iters=1000,
              force_tol=0.05, ss=True, work_dir=None, xdatcar_file=None):
    """Run solid-state NEB using tsase.

    Args:
        images: List of pymatgen structures.
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
        xdatcar_file: If provided, save the final band as an XDATCAR file
            at this path (absolute or relative to the original directory).

    Returns:
        Tuple of (energies, structures) where energies is a list of floats
        and structures is a list of pymatgen Structures for the final band.
    """
    images = [i.to_ase_atoms() for i in images]

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

        structures = _band_to_structures(band)

        if xdatcar_file is not None:
            xdatcar_path = os.path.join(orig_dir, xdatcar_file)
            t = Trajectory.from_structures(structures, constant_lattice=False)
            t.write_Xdatcar(xdatcar_path)

        return energies, structures
    finally:
        os.chdir(orig_dir)
