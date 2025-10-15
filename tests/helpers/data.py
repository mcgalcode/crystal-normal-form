import os

from cnf import CrystalNormalForm
from pathlib import Path
from pymatgen.core.structure import Structure

def get_data_file_path(path_in_data_dir):
    return Path(__file__).parent / ".." / "data" / path_in_data_dir

all_mp_cif_dir = Path(__file__).parent / ".." / "data" / "mp_cifs"
_ALL_MP_STRUCTURES = []
for cif_path in all_mp_cif_dir.iterdir():
    _ALL_MP_STRUCTURES.append(Structure.from_file(cif_path))

def load_pathological_cifs(dir_name):
    patho_dir = Path(__file__).parent / ".." / "data" / "patho_pairs" / dir_name
    structs = []
    for cif_path in patho_dir.iterdir():
        structs.append(Structure.from_file(cif_path))
    return structs

def load_pathological_neighbors(dir_name):
    patho_dir = Path(__file__).parent / ".." / "data" / "patho_neighbor_pairs" / dir_name
    cnfs = []
    for cnf_path in patho_dir.iterdir():
        cnfs.append(CrystalNormalForm.from_file(cnf_path))
    return cnfs    

def save_cnfs_to_dir(dirname, cnfs: list[CrystalNormalForm]):
    for idx, cnf in enumerate(cnfs):
        data_dir = get_data_file_path(Path(dirname))
        os.makedirs(data_dir, exist_ok=True)
        fpath = get_data_file_path(Path(dirname) / f"pt{idx}.json")
        cnf.to_file(fpath)