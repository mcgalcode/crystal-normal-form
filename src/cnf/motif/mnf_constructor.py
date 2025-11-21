import numpy as np

from .atomic_motif import FractionalMotif, DiscretizedMotif
from ..linalg import MatrixTuple
from .motif_normal_form import MotifNormalForm
    
class MNFCandidate():

    def __init__(self,
                 mnf_coords: tuple,
                 motif: DiscretizedMotif | FractionalMotif,
                 unimodular: MatrixTuple,
                 shift: np.array):
        self.mnf_coords = mnf_coords
        self.motif = motif
        self.unimodular = unimodular
        self.shift = shift
    
    def __repr__(self):
        repr = ""
        repr += f"Stabilizer Matrix: {self.unimodular.tuple}"
        repr += "\n"
        repr += f"Shift: {self.shift}"
        return repr

class MNFConstructionResult():

    def __init__(self,
                 original_motif: DiscretizedMotif,
                 delta: int,
                 stabilizers: list[MatrixTuple],
                 sorted_mnf_candidates: list[MNFCandidate]):
        self.delta = delta
        self.original_motif = original_motif
        self.stabilizers = stabilizers
        self.sorted_mnf_candidates = sorted_mnf_candidates
    
    def print_details(self):
        print(f"Found MNF: {self.mnf.coord_list}")
        print(f"Applied the following stabilizers...")
        for p in self.stabilizers:
            print(p.tuple)

        print(f"Considered stablizers:")
        for m in self.stabilizers:
            print(f"Mat: {m.tuple}")
        
        print(f"Found phone-book first shift: {self.sorted_mnf_candidates[0].shift}")
    
    @property
    def canonical_candidate(self):
        return self.sorted_mnf_candidates[0]
    
    @property
    def canonical_motif(self):
        if isinstance(self.original_motif, FractionalMotif):
            return FractionalMotif.from_mnf_list(self.canonical_candidate.mnf_coords, self.original_motif.atoms)
        elif isinstance(self.original_motif, DiscretizedMotif):
            return DiscretizedMotif.from_mnf_list(self.canonical_candidate.mnf_coords, self.original_motif.atoms, delta=self.delta)
    
    @property
    def mnf(self):
        element_list, _ = self.canonical_motif.to_elements_and_positions()
        canonical_mnf_coords = self.canonical_candidate.mnf_coords
        if isinstance(self.original_motif, DiscretizedMotif):
            canonical_mnf_coords = tuple([int(c) for c in canonical_mnf_coords])

        return MotifNormalForm(canonical_mnf_coords, element_list, self.delta)

def get_all_shifted_motifs(m: FractionalMotif) -> tuple[list[FractionalMotif], list[np.ndarray]]:
    sorted_elements = m.sorted_elements
    origin_element = sorted_elements[0]

    origin_element_positions = m.get_element_positions(origin_element)
    # For each possible origin, compute the list
    shifted_motifs = []
    shifts = []
    for origin_candidate in origin_element_positions:
        shift = -origin_candidate
        shifted_origin = m.shift_origin(shift)
        shifted_motifs.append(shifted_origin)
        shifts.append(shift)
    return shifted_motifs, shifts

def get_stabilized_coord_mats(stabilizers, motif):
    original_motif_coords = motif.coord_matrix
    stabilizers_inverted = invert_unimods(stabilizers)
    new_motifs = stabilizers_inverted @ original_motif_coords
    return move_coords_into_bounds(new_motifs, motif._mod)

def invert_unimods(matrices):
    return np.linalg.inv(matrices)

def get_mnf_strs_from_coord_mats(coord_mats: np.ndarray):
    return [cm.T.reshape(-1)[3:] for cm in coord_mats]

def move_coords_into_bounds(coord_mats, mod):
    all_motifs = np.array(coord_mats).round(6)
    all_motifs = np.mod(all_motifs, mod) 
    return all_motifs

def get_all_shifted_coord_mats(coord_mat, num_origin_atoms, mod):
    motifs = []
    shift_vecs = -coord_mat.T[:num_origin_atoms]
    for v in shift_vecs:
        shifted_motif = v + coord_mat.T
        motifs.append(shifted_motif.T)
    return move_coords_into_bounds(motifs, mod)

def get_atom_labels(motif):
    atom_labels = []
    el_num = 0
    prev_el = motif.atoms[0]
    for el in motif.atoms:
        if el != prev_el:
            el_num += 1
            prev_el = el
        atom_labels.append(el_num)
    atom_labels = np.array(atom_labels)
    return atom_labels
    
def sort_motif_coord_arr(coord_mat, atom_labels):
    x_col, y_col, z_col = coord_mat
    sorted_indices = np.lexsort((z_col, y_col, x_col, atom_labels))
    sorted_coord_mat = coord_mat.T[sorted_indices]    
    return sorted_coord_mat.T

class MNFConstructor():
    """Implements methods for taking a list of atomic positions
    in fractional coordinates and producing the Motif Normal Form string
    as described in the section "Representation of Crystalline Atomic Bases" on
    pp. 52 of David Mrdjenovich's thesis.
    """
    
    def __init__(self,
                 delta: int,
                 stabilizer: list[MatrixTuple] = None,
                 verbose_logging = False):
        self.delta = delta
        if stabilizer is None:
            stabilizer = [MatrixTuple(np.eye(3))]
        self.stabilizer = stabilizer
        self.verbose_logging = verbose_logging
    
    def build_vectorized(self, original_motif: FractionalMotif):
        if len(original_motif.atoms) == 1:
            candidate = MNFCandidate(tuple([]), None, None, None)
            return MNFConstructionResult(
                original_motif,
                self.delta,
                self.stabilizer,
                [candidate]
            ) 

        # compute a list of element labels for help in later
        # lexicographic sorting
        np_stabilizer_mats = self.stabilizer
        atom_labels = get_atom_labels(original_motif)
        num_origin_atoms = original_motif.num_origin_atoms
        
        stabilized_coord_mats = get_stabilized_coord_mats(np_stabilizer_mats, original_motif)
        shifted_coord_mats = []
        for scm in stabilized_coord_mats:
            shifted = get_all_shifted_coord_mats(scm, num_origin_atoms, original_motif._mod)
            shifted_coord_mats.extend(shifted)
            
        sorted_coord_mats = []
        for scm in shifted_coord_mats:
            sorted_coord_mat = sort_motif_coord_arr(scm, atom_labels)
            sorted_coord_mats.append(sorted_coord_mat)

        sorted_mnfs = get_mnf_strs_from_coord_mats(sorted_coord_mats)
        
        stacked_mnfs = np.stack(sorted_mnfs)
        keys = stacked_mnfs.T
        sorted_indices = np.lexsort(keys[::-1])
        sorted_list = [stacked_mnfs[i] for i in sorted_indices]
        best_candidate = sorted_list[0]
        best_mnf_str = tuple([float(i) for i in best_candidate])
        candidate = MNFCandidate(best_mnf_str, None, None, None)

        return MNFConstructionResult(
            original_motif,
            self.delta,
            self.stabilizer,
            [candidate]
        ) 

    def build(self, original_motif: FractionalMotif):
        mnf_candidates: list[MNFCandidate] = []

        if self.verbose_logging:
            print(f"Initial motif positions:")
            original_motif.print_details()
        for mat in self.stabilizer:
            transformed_motif = original_motif.apply_unimodular(mat)

            if self.verbose_logging:
                print(f"Trying matrix: {mat}")
                transformed_motif.print_details()

            shifted_motifs, shifts = get_all_shifted_motifs(transformed_motif)

            for shifted_motif, shift in zip(shifted_motifs, shifts):
                mnf_list = shifted_motif.to_mnf_list(sort=True)
                candidate = MNFCandidate(mnf_list, shifted_motif, mat, shift)
                mnf_candidates.append(candidate)

        if self.verbose_logging:
            print("Found MNF candidates:")
            by_coords = {}
            for bc in mnf_candidates:
                if bc.mnf_coords not in by_coords:
                    by_coords[bc.mnf_coords] = [bc]
                else:
                    by_coords[bc.mnf_coords].append(bc)
            
            for coords, candidates in by_coords.items():
                print(f"Candidate: {coords}")
                print(f"Generated by: ", [c.unimodular for c in candidates])

        sorted_candidates = sorted(mnf_candidates, key=lambda c: c.mnf_coords)
        
        return MNFConstructionResult(
            original_motif,
            self.delta,
            self.stabilizer,
            sorted_candidates
        )