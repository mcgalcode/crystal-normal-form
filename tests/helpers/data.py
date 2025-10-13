from pathlib import Path
from pymatgen.core.structure import Structure

def get_data_file_path(path_in_data_dir):
    return Path(__file__).parent / ".." / "data" / path_in_data_dir

all_mp_cif_dir = Path(__file__).parent / ".." / "data" / "mp_cifs"
ALL_MP_STRUCTURES = []
for cif_path in all_mp_cif_dir.iterdir():
    ALL_MP_STRUCTURES.append(Structure.from_file(cif_path))

def load_pathological_cifs(dir_name):
    patho_dir = Path(__file__).parent / ".." / "data" / "patho_pairs" / dir_name
    structs = []
    for cif_path in patho_dir.iterdir():
        structs.append(Structure.from_file(cif_path))
    return structs