import os

from cnf import CrystalNormalForm
from pathlib import Path
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell

def get_data_file_path(path_in_data_dir):
    return Path(__file__).parent / ".." / "data" / path_in_data_dir

def load_specific_cif(cif_name: str):
    p = Path(__file__).parent / ".." / "data" / "specific_cifs" / cif_name
    return Structure.from_file(p)

all_mp_cif_dir = Path(__file__).parent / ".." / "data" / "mp_cifs"
_ALL_MP_STRUCTURES = []
for cif_path in sorted(all_mp_cif_dir.iterdir()):
    _ALL_MP_STRUCTURES.append(Structure.from_file(cif_path))

def mp_structs_with_voronoi_class(vclass):
    filtered = []
    for struct in _ALL_MP_STRUCTURES:
        if UnitCell.from_pymatgen_structure(struct).voronoi_class == vclass:
            filtered.append(struct)
    return filtered

def list_data_dir(path_in_data_dir):
    return get_data_file_path(path_in_data_dir).iterdir()

def load_pathological_cifs(dir_name):
    patho_dir = Path(__file__).parent / ".." / "data" / "patho_pairs" / dir_name
    structs = []
    for cif_path in patho_dir.iterdir():
        structs.append(Structure.from_file(cif_path))
    return structs

def load_cnfs(dir_name):
    patho_dir = Path(__file__).parent / ".." / "data" /  dir_name
    unitcells = []
    for cnf_path in patho_dir.iterdir():
        unitcells.append(CrystalNormalForm.from_file(cnf_path))
    return unitcells

def load_cifs(dir_name):
    patho_dir = Path(__file__).parent / ".." / "data" / dir_name
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

def save_cifs_to_dir(dirname, structs: list[Structure]):
    for idx, struct in enumerate(structs):
        data_dir = get_data_file_path(Path(dirname))
        os.makedirs(data_dir, exist_ok=True)
        fpath = get_data_file_path(Path(dirname) / f"pt{idx}.cif")
        struct.to_file(fpath)