import random
import numpy as np
from cnf.linalg.unimodular import get_unimodulars_col_max
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.vis.structure_vtk import StructureVis
from ase.io import read, write
from cnf import UnitCell
import matplotlib.pyplot as plt
from cnf.viz.voronoi_cell import plot_wigner_seitz_cell, plot_lattice_vectors, plot_lattice_vectors_with_planes
from cnf.lattice.selling.superbasis_reducer import SuperbasisSellingReducer
from cnf.navigation.lattice_neighbor_finder import LatticeNeighborFinder
from cnf.navigation.motif_neighbor_finder import MotifNeighborFinder
from cnf.lattice.permutations import VONORM_PERMUTATION_TO_CONORM_PERMUTATION
from cnf.motif.mnf_constructor import MNFConstructor
from cnf.linalg import MatrixTuple

def main():
    mnf_cif = "SnO2.cif"
    mnf_uc = UnitCell.from_cif(mnf_cif)
    mnf_uc.motif.print_details()
    stab = [MatrixTuple.identity()]
    c = MNFConstructor(10, stab, verbose_logging=True)
    mnf = c.build(mnf_uc.motif.discretize(10))
    print("MFN: ", mnf.canonical_candidate.mnf_coords)


    fpath = "/Users/maxg/Documents/research/groupmeeting11625/TiO2.cif"
    fpath = "tests/data/specific_cifs/Zr_HCP.cif"
    # 1. Load the CIF file into a Structure object
    uc = UnitCell.from_cif(fpath)

    uc_cnf = uc.to_cnf(1.5, 10)
    lnff = LatticeNeighborFinder(uc_cnf)
    nbs = lnff.find_cnf_neighbors()
    for nb in nbs.neighbors:
        nb_str = "_".join([str(int(i)) for i in nb.point.coords])
        UnitCell.from_cnf(nb.point).to_cif(f"lat_nb_{nb_str}.cif")
    
    mnff = MotifNeighborFinder(uc_cnf)
    nbs = mnff.find_motif_neighbors()
    for nb in nbs.neighbors:
        nb_str = "_".join([str(int(i)) for i in nb.point.coords])
        UnitCell.from_cnf(nb.point).to_cif(f"mot_nb_{nb_str}.cif")

    plot_lattice_vectors(uc.superbasis.superbasis_vecs[1:], "lattice_basis.png", vector_labels=["v_{1}", "v_{2}", "v_{3}"])
    plot_lattice_vectors(uc.superbasis.superbasis_vecs, "lattice_superbasis.png", vector_labels=["v_{0}", "v_{1}", "v_{2}", "v_{3}"])

    ruc = uc.reduce()
    plot_lattice_vectors_with_planes(ruc.superbasis.superbasis_vecs, "lattice_superbasis_bisectors.png")
    sbvs = ruc.superbasis.superbasis_vecs
    all_superbasis_vecs_no_negs = list(ruc.superbasis.superbasis_vecs) + [sbvs[0] + sbvs[1], sbvs[0] + sbvs[2], sbvs[0] + sbvs[3]]
    all_superbasis_vecs = all_superbasis_vecs_no_negs + list(-np.array(all_superbasis_vecs_no_negs))
    plot_lattice_vectors(all_superbasis_vecs_no_negs, "lattice_voronoi_vecs.png", vector_labels=["v_{0}", "v_{1}", "v_{2}", "v_{3}", "v_{0,1}", "v_{0,2}", "v_{0,3}"])
    plot_lattice_vectors(all_superbasis_vecs, "lattice_voronoi_vecs_and_negatives.png", vector_labels=["v_{0}", "v_{1}", "v_{2}", "v_{3}", "v_{0,1}", "v_{0,2}", "-v_{0,3}", "-v_{0}", "-v_{1}", "-v_{2}", "-v_{3}", "-v_{0,1}", "-v_{0,2}", "-v_{0,3}"])

    plot_lattice_vectors_with_planes(all_superbasis_vecs, "all_sb_vecs_bisectors.png", uniform_plane_color='lightgrey',vector_labels=["" for _ in all_superbasis_vecs])
    plot_lattice_vectors_with_planes(all_superbasis_vecs, "all_sb_vecs_bisectors_cropped.png", voronoi_crop=True, uniform_plane_color='lightgrey', vector_labels=["" for _ in all_superbasis_vecs])

    dirname = "zr_hex_cells"
    allmats = get_unimodulars_col_max(3)
    mats = random.sample(allmats, 20)
    for m in mats[:5]:
        print(m.matrix)
    # for idx, m in enumerate(mats):
    #     tuc = uc.apply_unimodular(m)
    #     struct = tuc.to_pymatgen_structure()
    #     struct.to_file(f"{dirname}/{idx}_unit_cell.cif")
    #     truc = tuc.reduce()
    #     truc_struct = truc.to_pymatgen_structure()
    #     truc_struct.to_file(f"{dirname}/{idx}_unit_cell_reduced.cif")
    #     plot_wigner_seitz_cell(
    #         truc_struct.lattice, f"{dirname}/{idx}_voronoi_cell.png",
    #     )

    #     sb = tuc.superbasis
    #     r = SuperbasisSellingReducer()
    #     step = 0
    #     while not sb.is_obtuse(tol=1e-5):
    #         plot_lattice_vectors(sb.superbasis_vecs, f"{dirname}/mat_{idx}_sel_step_{step}_vecs.png")
    #         sb, _ = r.apply_selling_transform(sb)
    #         step += 1
    #     plot_lattice_vectors(sb.superbasis_vecs, f"{dirname}/mat_{idx}_sel_step_{step}_vecs.png")


    print(uc.to_cnf(1.5,10))

    selected_mats = allmats[:10000:2000]
    for s in selected_mats:
        tuc = uc.apply_unimodular(s)
        mat_str = "_".join([str(i) for i in s.tuple])
        tuc.to_cif(f"{dirname}/uni_mats/{mat_str}.cif")

    vnorms = ruc.to_cnf(1.5, 10).lattice_normal_form.vonorms
    print(vnorms)
    print(vnorms.conorms)
    print()
    for p in vnorms.permissible_perms:
        # print(p.vonorm_permutation)
        _vnorms = vnorms.apply_permutation(p.vonorm_permutation)
        cell = ruc.apply_unimodular(p.matrix)
        _cnorms = _vnorms.conorms
        print()
        print(_vnorms,p.vonorm_permutation)
        print(_cnorms,  p.conorm_permutation)
        for m in p.all_matrices:
            print(m)

        sb = _vnorms.to_superbasis(1.5)
        perm_str = '_'.join([str(int(i)) for i in p.vonorm_permutation])
        cell.to_cif(f"{dirname}/perms/{perm_str}.cif")
        plot_lattice_vectors(sb.superbasis_vecs, f"{dirname}/perms/{perm_str}_perm_sb.png", vector_labels=["v_{0}", "v_{1}", "v_{2}", "v_{3}"])
    
    for m in vnorms.stabilizer_matrices():
        # print(p.vonorm_permutation)
        cell = ruc.apply_unimodular(m)
        mat_str = "_".join([str(i) for i in m.tuple])
        cell.to_cif(f"{dirname}/stabs/{mat_str}.cif")


    UnitCell.from_cnf(uc.to_cnf(0.001,100)).to_cif(f"{dirname}/cnf_cell.cif")
    print(f"{len(VONORM_PERMUTATION_TO_CONORM_PERMUTATION)} valid permutations!")



if __name__ == "__main__":
    main()