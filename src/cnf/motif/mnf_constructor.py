import numpy as np

from .atomic_motif import FractionalMotif, DiscretizedMotif
from ..linalg import MatrixTuple
from .motif_normal_form import MotifNormalForm
from ..utils.prof import maybe_profile
    
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
        # element_list, _ = self.canonical_motif.to_elements_and_positions()
        canonical_mnf_coords = self.canonical_candidate.mnf_coords
        if isinstance(self.original_motif, DiscretizedMotif):
            canonical_mnf_coords = tuple([int(c) for c in canonical_mnf_coords])

        return MotifNormalForm(canonical_mnf_coords, self.original_motif.atoms, self.delta)

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

@maybe_profile
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

@maybe_profile
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
    
@maybe_profile
def sort_motif_coord_arr(coord_mat, atom_labels):
    x_col, y_col, z_col = coord_mat
    sorted_indices = np.lexsort((z_col, y_col, x_col, atom_labels))
    sorted_coord_mat = coord_mat.T[sorted_indices]
    return sorted_coord_mat.T

def extract_coord_matrix_from_mnf_tuple(mnf_tuple, include_origin: bool = False):
    """
    Reconstruct coordinate matrix from MNF coord_list.

    Args:
        include_origin: If True, prepend origin atom at (0,0,0)

    Returns:
        coord_matrix: (3, N) or (3, N-1) array depending on include_origin
        n_atoms: Total number of atoms (including origin)
    """
    # Reconstruct coord_matrix from coord_list (which excludes origin atom)
    n_stored_atoms = len(mnf_tuple) // 3
    coords_array = np.array(mnf_tuple, dtype=np.int32).reshape(n_stored_atoms, 3)

    n_atoms = n_stored_atoms + 1  # +1 for implicit origin
    if include_origin:
        # Prepend origin atom for Rust (needs full structure for transformations)
        coords_array = np.vstack([np.array([[0, 0, 0]]), coords_array])
        
    motif_coord_matrix = coords_array.T  # (3, N-1) - no origin atom

    return motif_coord_matrix, n_atoms

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
    
    @maybe_profile
    def build_vectorized(self, original_motif: FractionalMotif, use_rust=False):
        if use_rust:
            return self._build_vectorized_rust(original_motif)

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

    def _build_vectorized_rust(self, original_motif: FractionalMotif):
        """Rust-accelerated MNF construction"""
        import rust_cnf

        if len(original_motif.atoms) == 1:
            candidate = MNFCandidate(tuple([]), None, None, None)
            return MNFConstructionResult(
                original_motif,
                self.delta,
                self.stabilizer,
                [candidate]
            )

        # Prepare inputs for Rust
        coords_flat = np.ascontiguousarray(original_motif.coord_matrix.T.flatten(), dtype=np.float64)
        atom_labels = np.ascontiguousarray(get_atom_labels(original_motif), dtype=np.int32)
        num_origin_atoms = original_motif.num_origin_atoms

        # Flatten stabilizer matrices
        stabilizers_flat = np.ascontiguousarray(
            np.array([mat.flatten() for mat in self.stabilizer]).flatten(),
            dtype=np.int32
        )

        # Call Rust implementation
        mnf_coords_arr = rust_cnf.build_mnf_vectorized_rust(
            coords_flat,
            atom_labels,
            num_origin_atoms,
            stabilizers_flat,
            float(original_motif._mod)
        )

        # Convert to tuple
        best_mnf_str = tuple([float(i) for i in mnf_coords_arr])
        candidate = MNFCandidate(best_mnf_str, None, None, None)

        return MNFConstructionResult(
            original_motif,
            self.delta,
            self.stabilizer,
            [candidate]
        )

    def build_many_from_raw_coords(self, coords_list: list[list[int]], atomic_numbers: list[int], use_rust=False):
        """
        Build MNFs for many coordinate sets at once with shared pre-computation.

        Optimized batch processing that pre-computes atom labels and inverted stabilizers once.

        Args:
            coords_list: List of coordinate lists, each [x1, y1, z1, x2, y2, z2, ...]
            atomic_numbers: Sorted list of atomic numbers (same for all coords)
            use_rust: Whether to use Rust implementation

        Returns:
            list[tuple]: List of canonical MNF coordinate tuples
        """
        if len(atomic_numbers) == 1:
            return [tuple([]) for _ in coords_list]

        # Pre-compute values once for all coords
        atom_labels, num_origin_atoms, stabilizers_inverted = self._precompute_mnf_data(atomic_numbers)

        if use_rust:
            # Use batch Rust implementation - single call for all coordinates
            import rust_cnf

            # Convert coords_list to list of float64 arrays for Rust
            coords_batch = []
            for coords_flat in coords_list:
                n_atoms = len(coords_flat) // 3
                coords_array = np.array(coords_flat, dtype=np.int32).reshape(n_atoms, 3)
                coords_batch.append(coords_array.flatten().astype(np.float64).tolist())

            # Prepare shared data
            atom_labels_contig = np.ascontiguousarray(atom_labels, dtype=np.int32)
            stabilizers_flat = np.ascontiguousarray(
                np.array([mat.flatten() for mat in self.stabilizer]).flatten(),
                dtype=np.int32
            )

            # Single Rust call for all coordinate sets
            mnf_arrays = rust_cnf.build_mnf_batch_rust(
                coords_batch,
                atom_labels_contig,
                num_origin_atoms,
                stabilizers_flat,
                float(self.delta)
            )

            # Convert numpy arrays back to tuples
            return [tuple([int(i) for i in mnf_arr]) for mnf_arr in mnf_arrays]
        else:
            # Python fallback - loop through each coordinate set
            results = []
            for coords_flat in coords_list:
                result = self.build_from_raw_coords(
                    coords_flat, atomic_numbers, False,
                    atom_labels, num_origin_atoms, stabilizers_inverted
                )
                results.append(result)
            return results

    def _precompute_mnf_data(self, atomic_numbers: list[int]):
        """Pre-compute data needed for MNF construction."""
        # Compute atom labels (group by atomic number)
        atom_labels = []
        el_num = 0
        prev_z = atomic_numbers[0]
        for z in atomic_numbers:
            if z != prev_z:
                el_num += 1
                prev_z = z
            atom_labels.append(el_num)
        atom_labels = np.array(atom_labels, dtype=np.int32)

        # Count origin atoms
        num_origin_atoms = sum(1 for z in atomic_numbers if z == atomic_numbers[0])

        # Pre-compute inverted stabilizers
        stabilizers_inverted = np.linalg.inv(np.array(self.stabilizer))

        return atom_labels, num_origin_atoms, stabilizers_inverted

    def build_from_raw_coords(self, coords_flat: list, atomic_numbers: list[int], use_rust=False,
                              atom_labels=None, num_origin_atoms=None, stabilizers_inverted=None):
        """
        Build MNF directly from raw coordinate list and atomic numbers.

        Bypasses DiscretizedMotif construction for performance.

        Args:
            coords_flat: Flat list of coordinates [x1, y1, z1, x2, y2, z2, ...]
            atomic_numbers: Sorted list of atomic numbers [22, 22, 8, 8, ...]
            use_rust: Whether to use Rust implementation
            atom_labels: Optional pre-computed atom labels (for batch processing)
            num_origin_atoms: Optional pre-computed origin atom count (for batch processing)
            stabilizers_inverted: Optional pre-computed inverted stabilizers (for batch processing)

        Returns:
            tuple: Canonical MNF coordinate tuple
        """
        # Handle single atom case
        if len(atomic_numbers) == 1:
            return tuple([])

        # Use pre-computed values if provided, otherwise compute them
        if atom_labels is None or num_origin_atoms is None or stabilizers_inverted is None:
            atom_labels, num_origin_atoms, stabilizers_inverted = self._precompute_mnf_data(atomic_numbers)

        # Reshape coords to (N, 3) then transpose to (3, N) for coord_matrix
        n_atoms = len(coords_flat) // 3
        coords_array = np.array(coords_flat, dtype=np.int32).reshape(n_atoms, 3)
        coord_matrix = coords_array.T  # Shape: (3, N)

        if use_rust:
            # Use Rust implementation
            import rust_cnf

            # Rust expects float64 for coordinates
            coords_flat_contig = np.ascontiguousarray(coords_array.flatten(), dtype=np.float64)
            atom_labels_contig = np.ascontiguousarray(atom_labels, dtype=np.int32)

            stabilizers_flat = np.ascontiguousarray(
                np.array([mat.flatten() for mat in self.stabilizer]).flatten(),
                dtype=np.int32
            )

            mnf_coords_arr = rust_cnf.build_mnf_vectorized_rust(
                coords_flat_contig,
                atom_labels_contig,
                num_origin_atoms,
                stabilizers_flat,
                float(self.delta)
            )

            return tuple([int(i) for i in mnf_coords_arr])
        else:
            # Python implementation using pre-computed inverted stabilizers
            stabilized_coords = stabilizers_inverted @ coord_matrix
            stabilized_coords = move_coords_into_bounds(stabilized_coords, self.delta)

            # Apply shifts
            shifted_coord_mats = []
            for scm in stabilized_coords:
                shifted = get_all_shifted_coord_mats(scm, num_origin_atoms, self.delta)
                shifted_coord_mats.extend(shifted)

            # Sort
            sorted_coord_mats = []
            for scm in shifted_coord_mats:
                sorted_coord_mat = sort_motif_coord_arr(scm, atom_labels)
                sorted_coord_mats.append(sorted_coord_mat)

            # Get MNF tuples
            sorted_mnfs = get_mnf_strs_from_coord_mats(sorted_coord_mats)

            # Find lexicographically smallest
            stacked_mnfs = np.stack(sorted_mnfs)
            keys = stacked_mnfs.T
            sorted_indices = np.lexsort(keys[::-1])
            sorted_list = [stacked_mnfs[i] for i in sorted_indices]
            best_candidate = sorted_list[0]
            return tuple([int(i) for i in best_candidate])

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