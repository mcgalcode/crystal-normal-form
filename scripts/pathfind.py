from cnf.db.setup import setup_cnf_db
from cnf.search import instantiate_search, continue_search
from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.search_store import SearchProcessStore
from cnf.navigation.search_filters import VolumeLimitFilter
from cnf.calculation.grace import GraceCalculator
from cnf import UnitCell, CrystalNormalForm

XI = 1.5
DELTA = 10

zr_bcc = UnitCell.from_cif("./tests/data/specific_cifs/Zr_BCC.cif")
zr_hcp = UnitCell.from_cif("./tests/data/specific_cifs/Zr_HCP.cif")

start_cnfs = [uc.to_cnf(XI, DELTA) for uc in zr_bcc.supercells(2)]
start_cnfs = list(set(start_cnfs))
end_cnfs = [uc.to_cnf(XI, DELTA) for uc in zr_hcp.supercells(2)]
end_cnfs = list(set(end_cnfs))

DB_FNAME = "test_search_db"
def setup():

    setup_cnf_db(DB_FNAME, XI, DELTA, start_cnfs[0].elements)
    instantiate_search("test search", start_cnfs, end_cnfs, DB_FNAME)


if __name__ == "__main__":
    # setup()
    print("Start Points:")
    for cnf in start_cnfs:
        print(cnf.coords)
    print()

    print("End Points:")
    for cnf in end_cnfs:
        print(cnf.coords)
    print()

    explore_filter = VolumeLimitFilter.from_endpoint_structs(
        [cnf.reconstruct() for cnf in start_cnfs + end_cnfs],
        0.7,
        1.3
    )

    continue_search(1, DB_FNAME, GraceCalculator(), [explore_filter])