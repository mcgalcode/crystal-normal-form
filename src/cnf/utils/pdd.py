import amd
import tempfile
from pymatgen.core.structure import Structure
from ..crystal_normal_form import CrystalNormalForm

def pdd(struct1: Structure, struct2: Structure, k=None):
    amd_struct_1 = amd.periodicset_from_pymatgen_structure(struct1)
    amd_struct_2 = amd.periodicset_from_pymatgen_structure(struct2)

    # calculate PDDs
    if k is None:
        k = max(100, struct1.num_sites * 3, struct2.num_sites * 3)

    pdd1 = amd.PDD(amd_struct_1, k)
    pdd2 = amd.PDD(amd_struct_2, k)

    return amd.EMD(pdd1, pdd2)

def pdd_amd(struct1: Structure, struct2: Structure, k=None):
    amd_struct_1 = amd.periodicset_from_pymatgen_structure(struct1)
    amd_struct_2 = amd.periodicset_from_pymatgen_structure(struct2)

    # calculate PDDs
    if k is None:
        k = max(100, struct1.num_sites * 3, struct2.num_sites * 3)

    amd1 = amd.AMD(amd_struct_1, k)
    amd2 = amd.AMD(amd_struct_2, k)

    return amd.AMD_cdist([amd1], [amd2])[0][0]

def amd_from_cnf(cnf: CrystalNormalForm, k=None):
    struct = cnf.reconstruct()
    amd_struct_1 = amd.periodicset_from_pymatgen_structure(struct)

    # calculate PDDs
    if k is None:
        k = max(100, struct.num_sites * 3)

    amd1 = amd.AMD(amd_struct_1, k)
    return amd1




def pdd_for_cnfs(cnf1: CrystalNormalForm, cnf2: CrystalNormalForm, k=100):
    return pdd(cnf1.reconstruct(), cnf2.reconstruct(), k=k)



def pdd_amd_for_cnfs(cnf1: CrystalNormalForm, cnf2: CrystalNormalForm, k=100):
    return pdd_amd(cnf1.reconstruct(), cnf2.reconstruct(), k=k)