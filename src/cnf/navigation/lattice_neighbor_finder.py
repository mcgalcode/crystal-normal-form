import numpy as np
import itertools
from ..crystal_normal_form import CrystalNormalForm
from ..cnf_constructor import CNFConstructor
from ..lattice.lnf_constructor import LatticeNormalFormConstructor
from ..lattice.voronoi import VonormList
from .lattice_step import LatticeStep, LatticeStepResult
from .neighbor_set import NeighborSet
from ..utils.pdd import pdd_for_cnfs
from ..linalg.unimodular import combine_unimodular_matrices
from ..utils.prof import maybe_profile

class LatticeNeighborFinder():

    def __init__(self, cnf_point: CrystalNormalForm, verbose_logging=False):
        self.verbose_logging = verbose_logging
        self.point = cnf_point

        if self._is_cnf_neighbor_finder():
            self.discretized_motif = cnf_point.motif_normal_form.to_discretized_motif()
            self.fractional_motif = cnf_point.motif_normal_form.to_motif()

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
        steps: list[LatticeStep] = []
        current_stabilizer = self.point.lattice_normal_form.vonorms.stabilizer_matrices()
        for s4_idxs, data in vonorms.maximally_ascending_equivalence_class_members().items():           
            permuted_vonorms = data['maximal_permuted_list']
            transform_mats = data['transition_mats']

            mat_sequences = itertools.product(current_stabilizer, transform_mats, permuted_vonorms.stabilizer_matrices())
            unimodular_mats = set([combine_unimodular_matrices(mat_seq) for mat_seq in mat_sequences])
            for mat in unimodular_mats:
                transformed_motif = self.discretized_motif.apply_unimodular(mat)
                for step_vec in LatticeStep.all_step_vecs():
                    old_vonorms = np.array(permuted_vonorms.vonorms)                    
                    new_vonorms = VonormList(tuple([int(v) for v in old_vonorms + np.array(step_vec)]))                    
                    steps.append(LatticeStep(step_vec, new_vonorms, transformed_motif, mat))
        steps = list(set(steps))
        return steps


    def get_vonorm_neighbor(self, step: LatticeStep):
        permuted_vonorms = step.vonorms
        self._log(f"Permuted vonorms: {permuted_vonorms}")
        old_vonorms = np.array(permuted_vonorms.vonorms)
        
        new_vonorms = VonormList(tuple([int(v) for v in old_vonorms + np.array(step.vals)]))
        self._log(f"Computed neighbor vonorms before canonicalization: {new_vonorms}")

        if not new_vonorms.has_valid_conorms():
            self._log("Neighbor had invalid conorms")
            return None

        if new_vonorms.is_superbasis() and new_vonorms.is_obtuse():
            return new_vonorms
        else:
            if self.verbose_logging:
                if not new_vonorms.is_obtuse():
                    self._log(f"Neighbor was not obtuse.")

                if not new_vonorms.is_superbasis():
                    self._log(f"Neighbor was not a superbasis.")
            return None
    
    def find_lnf_neighbor(self, step: LatticeStep):
        new_vonorms = step.vonorms
        if new_vonorms is None:
            return None
        
        lnf_constructor = LatticeNormalFormConstructor(self._lnf().lattice_step_size)
        construction_result = lnf_constructor.build_lnf_from_vonorms(new_vonorms, skip_reduction=True)
        neighbor_lnf = construction_result.lnf
            
        return LatticeStepResult(step, new_vonorms, construction_result, neighbor_lnf, step.matrix)
        

    def find_lnf_neighbors(self) -> NeighborSet:
        neighbors = NeighborSet()
        for step in self.possible_steps():
            result = self.find_lnf_neighbor(step)
            if result is not None:
                neighbors.add_neighbor(result)
        return neighbors
    
    @maybe_profile
    def find_cnf_neighbor_results(self, step: LatticeStep) -> list[LatticeStepResult]:
        results = []

        if not step.vonorms.has_valid_conorms_exact():
            self._log("Neighbor had invalid conorms")
            return results

        is_obtuse = step.vonorms.is_obtuse()
        is_sb = step.vonorms.is_superbasis_exact()
        if not (is_obtuse and is_sb):
            if self.verbose_logging:
                if not step.vonorms.is_obtuse():
                    self._log(f"Neighbor was not obtuse.")

                if not step.vonorms.is_superbasis():
                    self._log(f"Neighbor was not a superbasis.")
            return results

        cnf_constructor = CNFConstructor(
            self.point.xi,
            self.point.delta,
            verbose_logging=False,
        )

        cnf_result = cnf_constructor._from_vonorms_and_motif_rust(step.vonorms, step.transformed_motif)

        if cnf_result.cnf != self.point:
            results.append(LatticeStepResult(
                step,
                cnf_result.cnf.lattice_normal_form.vonorms,
                cnf_result,
                cnf_result.cnf,
                step.matrix
            ))

        return results

    def find_cnf_neighbors(self) -> NeighborSet:
        neighbors = NeighborSet()
        steps = self.possible_steps()
        self._log(f"Considering {len(steps)} possible steps...")
        for step in steps:
            self._log("")
            self._log(f"Step: {step.vals}, {step.matrix}")
            results = self.find_cnf_neighbor_results(step)
            for r in results:
                neighbors.add_neighbor(r)
        return neighbors