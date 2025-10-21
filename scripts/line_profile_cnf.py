from cnf import CrystalNormalForm
from pymatgen.core.structure import Structure

import sys

def main():
    cif_path = sys.argv[1]
    struct = Structure.from_file(cif_path)
    cnf = CrystalNormalForm.from_pmg_struct(struct, 1.5, 10)
    print(cnf.voronoi_class)
    print(cnf.coords)

if __name__ == "__main__":
    main()