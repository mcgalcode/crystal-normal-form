import numpy as np

from ..crystal_normal_form import CrystalNormalForm
from ..cnf_constructor import CNFConstructor, CNFConstructionResult
from ..lattice.lnf_constructor import LatticeNormalFormConstructor
from ..lattice.voronoi import VonormList
from .lattice_step import LatticeStep, LatticeStepResult
from .neighbor_set import NeighborSet

class LatticeNeighborFinder():

    def __init__(self, cnf_point: CrystalNormalForm, verbose_logging=False):
        self.verbose_logging = verbose_logging
        self.point = cnf_point

        if self._is_cnf_neighbor_finder():
            self.discretized_motif = cnf_point.basis_normal_form.to_discretized_motif()
            self.fractional_motif = cnf_point.basis_normal_form.to_motif()

    def _is_cnf_neighbor_finder(self):
        return isinstance(self.point, CrystalNormalForm)
    
    def _lnf(self):
        if self._is_cnf_neighbor_finder():
            return self.point.lattice_normal_form
        else:
            return self.point

    def _log(self, msg):
        if self.verbose_logging:
            print(msg)

    def possible_steps(self):
        vonorms = self._lnf().vonorms
        grouped_vnorm_perms = vonorms.conorms.form.grouped_vonorm_permutations()
        steps: list[LatticeStep] = []
        for _, perms in grouped_vnorm_perms.items():
            # Choose a representative perm - any of these will work
            # because the all_step_vecs chi vectors from David's thesis
            # are already designed to cover S4. So we just need any representative
            # that gets us to each of Kurlin's basis possibilities.
            repr_perm = perms[0]
            for step_vec in LatticeStep.all_step_vecs():
                steps.append(LatticeStep(step_vec, repr_perm))
        return steps

    def get_vonorm_neighbor(self, step: LatticeStep):
        lnf_point = self._lnf()
        vonorms = lnf_point.vonorms
        self._log(f"Original vonorms: {vonorms}")

        permuted_vonorms = vonorms.apply_permutation(step.prereq_perm.vonorm_permutation)
        self._log(f"Permuted vonorms: {permuted_vonorms} (after applying {step.prereq_perm.vonorm_permutation})")
        old_vonorms = np.array(permuted_vonorms.vonorms)
        
        new_vonorms = VonormList(tuple([int(v) for v in old_vonorms + np.array(step.vals)]))
        self._log(f"Computed neighbor vonorms before canonicalization: {new_vonorms}")

        if new_vonorms.is_obtuse() and new_vonorms.is_superbasis():
            return new_vonorms
        else:
            if self.verbose_logging:
                if not new_vonorms.is_obtuse():
                    self._log(f"Neighbor was not obtuse.")

                if not new_vonorms.is_superbasis():
                    self._log(f"Neighbor was not a superbasis.")
            return None
    
    def get_basis_neighbor(self, step: LatticeStep):
        matrix = step.prereq_perm.matrix
        return self.discretized_motif.apply_unimodular(matrix)

    def find_lnf_neighbor(self, step: LatticeStep):
        new_vonorms = self.get_vonorm_neighbor(step)
        if new_vonorms is None:
            return None
        
        lnf_constructor = LatticeNormalFormConstructor(self._lnf().lattice_step_size)
        construction_result = lnf_constructor.build_lnf_from_vonorms(new_vonorms, skip_reduction=True)
        neighbor_lnf = construction_result.lnf
            
        return LatticeStepResult(step, new_vonorms, construction_result, neighbor_lnf)
        

    def find_lnf_neighbors(self) -> NeighborSet:
        neighbors = NeighborSet()
        for step in self.possible_steps():
            result = self.find_lnf_neighbor(step)
            if result is not None:
                neighbors.add_neighbor(result)
        return neighbors
    
    def find_cnf_neighbor(self, step: LatticeStep) -> LatticeStepResult | None:
        neighbor_vonorms = self.get_vonorm_neighbor(step)
        if neighbor_vonorms is None:
            return None
        
        
        cnf_results: list[CNFConstructionResult] = []
        for mat in step.prereq_perm.all_matrices:
            neighbor_motif = self.discretized_motif.apply_unimodular(mat)
            cnf_constructor = CNFConstructor(
                self.point.xi,
                self.point.delta,
                verbose_logging=self.verbose_logging,
            )

            cnf_results.append(
                cnf_constructor.from_discretized_obtuse_vonorms_and_motif(neighbor_vonorms, neighbor_motif)
            )
        
        cnf_construction_result = sorted(cnf_results, key=lambda cnf_res: cnf_res.cnf.coords)[0]
    
        if cnf_construction_result.cnf == self.point:
            return None

        return LatticeStepResult(
            step,
            neighbor_vonorms,
            cnf_construction_result,
            cnf_construction_result.cnf,
            None
        )

    def find_cnf_neighbors(self) -> NeighborSet:
        neighbors = NeighborSet()
        self._log(f"Considering {len(self.possible_steps())} possible steps...")
        for step in self.possible_steps():
            self._log("")
            self._log(f"Step: {step.vals}, {step.prereq_perm.perm.perm}")
            result = self.find_cnf_neighbor(step)
            if result is not None:
                neighbors.add_neighbor(result)
        return neighbors