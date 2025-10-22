import numpy as np
import tqdm
from cnf.unit_cell import UnitCell
from cnf.linalg.unimodular import get_unimodulars_col_max
from ..crystal_normal_form import CrystalNormalForm
from pymatgen.core.trajectory import Trajectory, Structure


class TrajectoryVisualizer():

    def __init__(self, filename, supercell_scaling=None):
        self.filename = filename
        self.supercell_scaling = supercell_scaling

    def save_trajectory_from_cnfs(self, cnfs: list[CrystalNormalForm]):
        structs: list[Structure] = [cnf.reconstruct() for cnf in cnfs]
        return self.save_trajectory_from_pmg_structs(structs)

    def save_trajectory_from_pmg_structs(self, structs: list[Structure]):
        if self.supercell_scaling is not None:
            structs = [s.make_supercell(self.supercell_scaling) for s in structs]

        ucs = [UnitCell.from_pymatgen_structure(s) for s in structs]
        return self.save_trajectory(ucs)

    def save_trajectory(self, unit_cells: list[UnitCell]):
        transformed_structs = [unit_cells[0]]

        def _get_com(unit_cell: UnitCell):
            cart_coords = unit_cell.motif.compute_cartesian_coords_in_basis(unit_cell.superbasis)
            pos_sum = np.zeros(3)
            for row in cart_coords.coord_matrix.T:
                pos_sum = pos_sum + row
            return pos_sum / len(unit_cell.motif.atoms)

        for struct in tqdm.tqdm(unit_cells[1:]):
            prev = transformed_structs[-1]
            prev_cols = prev.superbasis.generating_vecs().T
            prev_com = _get_com(prev)
            prev_inv = np.linalg.inv(prev_cols)
            min_score = (np.inf, np.inf)
            selected_cell = prev
            for m in get_unimodulars_col_max(2):
                ts = struct.apply_unimodular(m)
                tcols = ts.superbasis.generating_vecs().T
                t_com = _get_com(ts)
                com_dist = round(float(np.linalg.norm(t_com - prev_com)), 6)
                A = prev_inv @ tcols
                AtA = A.T @ A
                D = AtA - np.eye(3)
                DtD = D.T @ D
                dtd_tr = round(float(np.trace(DtD)), 6)
                score = (dtd_tr, com_dist)

                if score < min_score:
                    min_score = score
                    selected_cell = ts
                
            transformed_structs.append(selected_cell)
        transformed_structs = [t.to_pymatgen_structure() for t in transformed_structs]
        t = Trajectory.from_structures(transformed_structs, constant_lattice=False)
        t.write_Xdatcar(self.filename)
        return t