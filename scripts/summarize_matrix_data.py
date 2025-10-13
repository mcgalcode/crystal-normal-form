from cnf.lattice.voronoi.conorm_list_form import ConormListForm

import sys

def main():
    if len(sys.argv) > 1:
        vclasses = [int(i) for i in sys.argv[1].split(",")]
    else:
        vclasses = range(1,6)
    for voronoi_class in vclasses:
        print()
        print(f"========== Summarizing Voronoi Class {voronoi_class} =============")
        for cf in ConormListForm.get_coforms_of_voronoi_class(voronoi_class):
            print()
            print(f"Coform with Zero Conorm IDXS: {cf.zero_indices}")
            print(f"TOTAL matrices: {len(cf.all_matrices())}")
            print(f"TOTAL permutations: {len(cf.permissible_permutations())}")
            print()
            counts = {}
            for perm in cf.permissible_permutations():
                mcount = len(cf.matrices_for_perm(perm.conorm_permutation))
                prev_count = counts.get(mcount, 0)
                counts[mcount] = prev_count + 1
            
            for matrix_count, occurences in counts.items():
                print(f"{occurences} permutations had {matrix_count} matching matrices")
                # print(f"Perm: {perm.vonorm_permutation}, matrix count: {len(cf.matrices_for_perm(perm.conorm_permutation))}")


if __name__ == "__main__":
    main()