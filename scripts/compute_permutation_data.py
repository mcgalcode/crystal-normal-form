import tqdm
import json

from cnf.lattice.voronoi import ConormListForm
from cnf.lattice.permutations import ConormPermutation
from cnf.linalg.unimodular import UNIMODULAR_MATRICES
from cnf.lattice.voronoi.math import ConormCalculator, Transformation

def map_unimod_to_conorm_perms():
    mat_to_perms = []
    all_valid_perms = ConormPermutation.all_conorm_perm_tuples()
    for conorm_form in tqdm.tqdm(ConormListForm.all_coforms()):
        for u in UNIMODULAR_MATRICES:
            t = Transformation(u)
            calc = ConormCalculator(t, conorm_form.zero_conorms())
            try:
                perms = calc.get_permutations()
                filtered_perms = list(set(perms).intersection(all_valid_perms))
                mat_to_perms.append([conorm_form.zero_indices, u.tuple, filtered_perms])
            except:
                pass

    with open("unimod_mats_to_perms.json", 'w+') as f:
        json.dump(mat_to_perms, f)

def read_matching():
    with open("unimod_mats_to_perms.json", 'r') as f:
        values = json.load(f)
    
    results = {}
    for item in values:
        
        zeros = tuple(item[0])
        if zeros not in results:
            results[zeros] = {}
        
        mat = tuple(item[1])
        perms = [tuple(p) for p in item[2]]
        for perm in perms:
            if perm not in results[zeros]:
                results[zeros][perm] = []

            results[zeros][perm].append(mat)

    for zero_set, perm_map in results.items():
        print(f"Number of zero conorms: {len(zero_set)}")
        perms_with_mats = set([p for p, mats in perm_map.items()])
        mats = set([mat for p, mats in perm_map.items() for mat in mats])
        print(f"Number of permutations with matching matrix: {len(perms_with_mats)}")
        print(f"Number of matrices mapped to by these perms: {len(mats)}")



if __name__ == '__main__':
    # map_unimod_to_conorm_perms()
    read_matching()