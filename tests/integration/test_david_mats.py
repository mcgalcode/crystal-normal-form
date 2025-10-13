import pytest
import helpers
import json
import numpy as np

from cnf.linalg import MatrixTuple
from cnf.lattice.permutations import UnimodPermMapper

def test_david_and_max_mats_are_the_same():
    david_data_path = helpers.data.get_data_file_path("david_parsed.json")
    
    with open(david_data_path, 'r+') as f:
        david_data = json.load(f)

        all_david_zero_sets = [tuple(i['zero_idxs']) for i in david_data]
        all_max_zero_sets = UnimodPermMapper.all_zero_sets()
        impossible_zero_conorm_sets = set([
            (0,1,2),
            (2,4,5),
            (0,3,4),
            (1,3,5)
        ])
        assert set(all_david_zero_sets) - impossible_zero_conorm_sets == set(all_max_zero_sets)
        for item in david_data:
            zero_idxs = tuple(sorted(item["zero_idxs"]))
            if zero_idxs in impossible_zero_conorm_sets:
                continue
            
            print(f"Zeros: {zero_idxs}")
            david_perms = [tuple(entry['conorm_permutation']) for entry in item['entries']]

            max_perms = UnimodPermMapper.get_perms_for_zero_set(zero_idxs)

            assert set(max_perms) == set(david_perms)
            for entry in item['entries']:
                
                perm = tuple(entry['conorm_permutation'])
                print(f"    Perm: {perm}")
                david_mats = []
                for mat_arr in entry['transforms']:
                    np_arr = np.array([
                        mat_arr[0],
                        mat_arr[1],
                        mat_arr[2]
                    ])
                    tup = MatrixTuple(np_arr)
                    david_mats.append(tup)
                
                max_mats = UnimodPermMapper.get_matrices_for_zero_set_and_perm(zero_idxs, perm)

                print("         Maxs Mats:")
                for m in sorted(max_mats, key=lambda t: t.tuple):
                    print("             ", m.tuple)
                
                print("         Davids Mats:")
                for m in sorted(david_mats, key=lambda t: t.tuple):
                    print("             ", m.tuple)

                assert set(max_mats) == set(david_mats)