import numpy as np

from ..crystal_normal_form import CrystalNormalForm, get_canonical_bnf_from_stabilizers
from ..lattice import LatticeNormalForm
from ..lattice.lnf_constructor import VonormCanonicalizer
from ..lattice.voronoi import VonormList, ConormList, ConormListForm
from ..lattice.permutations import apply_permutation, UnimodPermMapper

def is_primary_idx(idx):
    return idx >= 0 and idx < 4

def is_secondary_idx(idx):
    return idx >= 4 and idx < 7

class LatticeStep():

    @classmethod
    def all_step_vecs(cls):
        steps = []
        for first_idx in range(7):
            for second_idx in range(first_idx + 1, 7):
                vec = np.zeros(7)
                vec[first_idx] = 1
                if is_primary_idx(first_idx) and is_primary_idx(second_idx):
                    vec[second_idx] = -1

                if is_primary_idx(first_idx) and is_secondary_idx(second_idx):
                    vec[second_idx] = 1

                if is_secondary_idx(first_idx):
                    vec[second_idx] = -1
                
                steps.append(cls([int(v) for v in vec]))
                steps.append(cls([-int(v) for v in vec]))
        
        return steps

    def __init__(self, vals, prereq_perm = None):
        self.vals = vals
        self.tuple = tuple(vals)
        self.prereq_perm = prereq_perm

        for idx, val in enumerate(vals):
            if np.abs(val) != 1 and val != 0:
                raise ValueError(f"LatticeStep instantiated with invalid element != 1 at pos {idx}: {tuple(vals)}")
    
    def __eq__(self, other: 'LatticeStep'):
        return self.tuple == other.tuple and self.prereq_perm == other.prereq_perm
    
    def __hash__(self):
        return (self.tuple, self.prereq_perm).__hash__()

class LatticeNeighborFinder():

    def __init__(self, verbose_logging=False):
        self.verbose_logging = verbose_logging
        pass

    def _log(self, msg):
        if self.verbose_logging:
            print(msg)

    def get_vonorm_neighbor(self, vonorms: VonormList, step: LatticeStep):
        canonicalizer = VonormCanonicalizer()
        old_vonorms = np.array(vonorms.vonorms)
        self._log(f"Original vonorms: {vonorms}")
        new_vonorms = VonormList(tuple([int(v) for v in old_vonorms + np.array(step.vals)]))
        self._log(f"Computed neighbor vonorms before canonicalization: {new_vonorms}")
        if new_vonorms.is_obtuse() and new_vonorms.is_superbasis():
            c_result = canonicalizer.get_canonicalized_vonorms(new_vonorms, skip_reduction=True)
            canonicalized_vonorms = c_result.canonical_vonorms        
            self._log(f"Canonicalized the neighbor vonorms: {canonicalized_vonorms}")
            self._log(f"Found stabilizing permutations: {[p.vonorm_permutation for p in c_result.stabilizer_permutations]}")
            return c_result
        else:
            if self.verbose_logging:
                if not new_vonorms.is_obtuse():
                    self._log(f"Neighbor was not obtuse.")

                if not new_vonorms.is_superbasis():
                    self._log(f"Neighbor was not a superbasis.")
            return None

    def find_lnf_neighbors(self, lnf_point: LatticeNormalForm) -> list[LatticeNormalForm]:
        grouped_vnorm_perms = lnf_point.vonorms.conorms.form.grouped_vonorm_permutations()

        neighbors = set()
        for _, perms in grouped_vnorm_perms.items():
            # Choose a representative perm - any of these will work
            # because the all_step_vecs chi vectors from David's thesis
            # are already designed to cover S4. So we just need any representative
            # that gets us to each of Kurlin's basis possibilities.
            vonorm_perm = perms[0].vonorm_permutation
            permuted_vonorms = lnf_point.vonorms.apply_permutation(vonorm_perm)

            for step in LatticeStep.all_step_vecs():
                canonical_result = self.get_vonorm_neighbor(permuted_vonorms, step)
                if canonical_result is not None:
                    neighbor_vonorms = canonical_result.canonical_vonorms
                    neighbor_lnf = LatticeNormalForm(neighbor_vonorms, lnf_point.lattice_step_size)
                    neighbors.add(neighbor_lnf)
        return list(neighbors)
    
    def find_cnf_neighbors(self, cnf_point: CrystalNormalForm) -> list[CrystalNormalForm]:
        grouped_vnorm_perms = cnf_point.lattice_normal_form.vonorms.conorms.form.grouped_vonorm_permutations()

        neighbors = set()
        for _, perms in grouped_vnorm_perms.items():
            # Choose a representative perm - any of these will work
            # because the all_step_vecs chi vectors from David's thesis
            # are already designed to cover S4. So we just need any representative
            # that gets us to each of Kurlin's basis possibilities.
            representative_perm = perms[0]

            vonorm_perm = representative_perm.vonorm_permutation

            original_vonorms = cnf_point.lattice_normal_form.vonorms
            original_motif = cnf_point.basis_normal_form.to_motif()

            permuted_vonorms = original_vonorms.apply_permutation(vonorm_perm)
            permuted_motif = original_motif.apply_unimodular(representative_perm.matrix)

            for step in LatticeStep.all_step_vecs():
                
                neighbor_canonicalization_result = self.get_vonorm_neighbor(permuted_vonorms, step)

                if neighbor_canonicalization_result is not None:
                    neighbor_vonorms = neighbor_canonicalization_result.canonical_vonorms
                    stabilizers = neighbor_canonicalization_result.stabilizer_permutations
                    stabilized_bnf = get_canonical_bnf_from_stabilizers(stabilizers, permuted_motif, cnf_point.delta)

                    new_point = CrystalNormalForm(
                        LatticeNormalForm(neighbor_vonorms, cnf_point.xi),
                        stabilized_bnf
                    )
                    neighbors.add(new_point)
        return list(neighbors)