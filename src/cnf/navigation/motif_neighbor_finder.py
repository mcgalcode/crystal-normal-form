from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.motif import MotifNormalForm
from cnf.motif.atomic_motif import DiscretizedMotif
from .neighbor import Neighbor
from .neighbor_set import NeighborSet
import copy

class MotifStepResult():

    def __init__(self, result: CrystalNormalForm, affected_idxs: list[int], adj: int):
        self.result = result
        self.affected_idxs = affected_idxs
        self.adj = adj

class MotifNeighborFinder():

    def __init__(self, point: CrystalNormalForm):
        self.point = point
    

    def find_motif_neighbors(self):


        neighbor_mnf_tuples = []

        for stabilizer in self.point.lattice_normal_form.vonorms.stabilizer_matrices(1e-4):
            current_mnf_tuple = self.point.motif_normal_form.to_discretized_motif().apply_unimodular(stabilizer).to_mnf_list(sort=True)
            current_mnf_tuple = (0,0,0) + current_mnf_tuple
            for idx in range(len(current_mnf_tuple)):
                for adj in [-1, +1]:
                    n = list(copy.copy(current_mnf_tuple))
                    n[idx] = n[idx] + adj
                    neighbor_mnf_tuples.append((tuple(n), [idx], adj))
            
            for idx in range(0, len(current_mnf_tuple), 3):
                for adj in [-1, +1]:
                    n = list(copy.copy(current_mnf_tuple))
                    n[idx] = n[idx] + adj
                    n[idx + 1] = n[idx + 1] + adj
                    n[idx + 2] = n[idx + 2] + adj
                    neighbor_mnf_tuples.append((tuple(n), [idx, idx + 1, idx + 2], adj))
        
        original_els = self.point.motif_normal_form.elements
        nbs = NeighborSet()
        cnf_constructor = CNFConstructor(self.point.xi, self.point.delta)

        for nb_mnf_tup in neighbor_mnf_tuples:
            nb_mnf, affected_idxs, adj = nb_mnf_tup
            positions = [list(nb_mnf[start_idx:start_idx + 3])for start_idx in range(0, len(nb_mnf), 3)]
            motif = DiscretizedMotif.from_elements_and_positions(original_els, positions, self.point.delta)
            nb_pt = cnf_constructor.from_vonorms_and_motif_fast(self.point.lattice_normal_form.vonorms, motif)
            result = MotifStepResult(nb_pt.cnf, affected_idxs, adj)
            nbs.add_neighbor(result)
        return nbs