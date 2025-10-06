from pathlib import Path
from pymatgen.core.structure import Structure

cif_dir = Path(__file__).parent / ".." / "data" / "mp_cifs"
ALL_MP_STRUCTURES = []
for cif_path in cif_dir.iterdir():
    ALL_MP_STRUCTURES.append(Structure.from_file(cif_path))