"""Relaxation utilities for crystal structures."""

from ..unit_cell import UnitCell


def relax_unit_cell(uc: UnitCell, calc, fmax=0.01, max_steps=500,
                    verbose=True, label=""):
    """Relax a UnitCell in continuous space (cell + positions) using ASE FIRE.

    Args:
        uc: UnitCell to relax.
        calc: ASE calculator (e.g. from grace_fm()).
        fmax: Force convergence threshold (eV/A).
        max_steps: Maximum optimizer steps.
        verbose: Print before/after energies.
        label: Label for printing (e.g. "start" or "end").

    Returns:
        Relaxed UnitCell.
    """
    from ase.filters import ExpCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = uc.to_pymatgen_structure()
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calc

    e_before = atoms.get_potential_energy()
    if verbose:
        print(f"  [{label}] Before: E = {e_before:.4f} eV, "
              f"vol = {atoms.get_volume():.2f} A^3")

    ecf = ExpCellFilter(atoms)
    opt = FIRE(ecf, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)

    e_after = atoms.get_potential_energy()
    if verbose:
        print(f"  [{label}] After:  E = {e_after:.4f} eV, "
              f"vol = {atoms.get_volume():.2f} A^3 "
              f"({opt.nsteps} steps)")

    relaxed_struct = AseAtomsAdaptor.get_structure(atoms)
    return UnitCell.from_pymatgen_structure(relaxed_struct)
