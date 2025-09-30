import amd
import tempfile
from pymatgen.core.structure import Structure


def assert_identical_by_pdd_distance(struct1: Structure, struct2: Structure, cutoff = 0.015, verbose=False):
        
    with tempfile.NamedTemporaryFile(suffix=".cif") as struct_1_file:
        with tempfile.NamedTemporaryFile(suffix=".cif") as struct_2_file:
            struct1.to_file(struct_1_file.file.name)
            struct2.to_file(struct_2_file.file.name)

            amd_struct_1 = amd.CifReader(struct_1_file.file.name).read()
            amd_struct_2 = amd.CifReader(struct_2_file.file.name).read()

            # calculate PDDs
            k = max(100, struct1.num_sites * 3, struct2.num_sites * 3)
            pdd1 = amd.PDD(amd_struct_1, k)
            pdd2 = amd.PDD(amd_struct_2, k)

            distance = amd.EMD(pdd1, pdd2)
            if verbose:
                print(f"Got {distance} for structs with {struct1.num_sites} and {struct2.num_sites} sites respectively")
            assert distance < cutoff
