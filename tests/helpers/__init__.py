import numpy as np
import os
import pytest

from .assertions import *
from .data import ALL_MP_STRUCTURES, load_pathological_cifs
from cnf.motif.atomic_motif import FractionalMotif

IS_FAST = int(os.getenv("CNF_FAST_TEST", 0)) == 1

STRUCT_SAMPLE_FREQ = int(os.getenv("SSF", 10))

SPECIFIC_STRUCT_IDX = os.getenv("SSI")
if SPECIFIC_STRUCT_IDX is not None:
    SPECIFIC_STRUCT_IDX = int(SPECIFIC_STRUCT_IDX)

def skip_if_fast(func):
    return pytest.mark.skipif(IS_FAST, reason="Skipped because CNF_FAST_TEST env var was set to 1")(func)

def parameterized_by_mp_structs(func):
    if SPECIFIC_STRUCT_IDX is None:
        return pytest.mark.parametrize("idx, struct", zip(range(0, len(ALL_MP_STRUCTURES), STRUCT_SAMPLE_FREQ), ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ]))(func)
    else:
        return parameterized_by_mp_struct_idxs([SPECIFIC_STRUCT_IDX])(func)

def parameterized_by_mp_struct_idxs(idxs):
    structs = [ALL_MP_STRUCTURES[i] for i in idxs]
    def _wrapper(func):
        return pytest.mark.parametrize("idx, struct", zip(idxs, structs))(func)    
    return _wrapper

def printif(msg, flag):
    if flag:
        print(msg)

def are_cnfs_mirror_images(cnf1: CrystalNormalForm, cnf2: CrystalNormalForm, atol=1e-6):
    motif1 = FractionalMotif.from_pymatgen_structure(cnf1.reconstruct())
    motif2 = FractionalMotif.from_pymatgen_structure(cnf2.reconstruct())
    return are_mirror_images(motif1, motif2, atol)

def are_mirror_images(motif1: FractionalMotif, motif2: FractionalMotif, atol=1e-6):
    """
    Check if two structures are related by inversion, accounting for atom types.
    
    Parameters
    ----------
    coords1, coords2 : array-like, shape (n_atoms, 3)
        Fractional coordinates of atoms
    species1, species2 : list
        Element symbols or types for each atom
    atol : float
        Absolute tolerance for floating point comparison
        
    Returns
    -------
    bool
        True if structures are mirror images
        
    Examples
    --------
    >>> coords1 = np.array([[0.4, 0.8, 0.2], [0.0, 0.0, 0.0]])
    >>> coords2 = np.array([[0.6, 0.2, 0.8], [0.0, 0.0, 0.0]])
    >>> species = ['Mg', 'Nd']
    >>> are_mirror_images(coords1, coords2, species, species)
    True
    """

    assert isinstance(motif1, FractionalMotif)
    assert isinstance(motif2, FractionalMotif)
    species1, coords1 = motif1.to_elements_and_positions()
    species2, coords2 = motif2.to_elements_and_positions()


    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    
    if coords1.shape != coords2.shape:
        return False
    
    if len(species1) != len(species2):
        return False
    
    # Invert coords1 through (0.5, 0.5, 0.5)
    inverted_coords1 = (1.0 - coords1) % 1.0
    
    # Build a list of (species, coords) for coords2
    atoms2 = list(zip(species2, coords2))
    
    # For each inverted atom in coords1, find matching atom in coords2
    for i, inv_coord in enumerate(inverted_coords1):
        species_to_match = species1[i]
        found = False
        
        for j, (sp2, coord2) in enumerate(atoms2):
            if sp2 == species_to_match:
                # Check if coordinates match (with PBC)
                diff = np.abs(inv_coord - coord2)
                diff = np.minimum(diff, 1.0 - diff)  # Handle periodic boundaries
                
                if np.all(diff < atol):
                    found = True
                    break
        
        if not found:
            return False
    
    return True