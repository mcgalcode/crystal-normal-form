import amd
import tempfile
from pymatgen.core.structure import Structure
from ..crystal_normal_form import CrystalNormalForm

def pdd(struct1: Structure, struct2: Structure, k=None):
    with tempfile.NamedTemporaryFile(suffix=".cif") as struct_1_file:
        with tempfile.NamedTemporaryFile(suffix=".cif") as struct_2_file:
            struct1.to_file(struct_1_file.file.name)
            struct2.to_file(struct_2_file.file.name)

            amd_struct_1 = amd.CifReader(struct_1_file.file.name).read()
            amd_struct_2 = amd.CifReader(struct_2_file.file.name).read()

            # calculate PDDs
            if k is None:
                k = max(100, struct1.num_sites * 3, struct2.num_sites * 3)
    
            pdd1 = amd.PDD(amd_struct_1, k)
            pdd2 = amd.PDD(amd_struct_2, k)

            return amd.EMD(pdd1, pdd2)

def pdd_for_cnfs(cnf1: CrystalNormalForm, cnf2: CrystalNormalForm, k=100):
    return pdd(cnf1.reconstruct(), cnf2.reconstruct(), k=k)