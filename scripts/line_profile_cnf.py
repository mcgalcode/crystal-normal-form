from cnf import CrystalNormalForm
from pymatgen.core.structure import Structure
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.utils.prof import maybe_profile

import sys

@maybe_profile
def main():
    cif_path = sys.argv[1]
    struct = Structure.from_file(cif_path)
    cnf = CrystalNormalForm.from_pmg_struct(struct, 1.5, 10)
    nf = NeighborFinder(cnf)
    cnfs = nf.find_neighbors()
    # print(cnf.voronoi_class)
    # print(cnf.coords)

if __name__ == "__main__":
    main()