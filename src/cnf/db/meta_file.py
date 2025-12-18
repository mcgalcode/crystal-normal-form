import json
from pathlib import Path
from datetime import datetime

from ..crystal_normal_form import CrystalNormalForm

META_FILE_NAME = 'search_metadata.json'

def write_meta_file(location: str,
                    xi: float,
                    delta: int,
                    atom_list: list[str],
                    calculator_model: str,
                    start_cnfs: list[CrystalNormalForm],
                    end_cnfs: list[CrystalNormalForm],
                    description: str = ""):
    timestamp = datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "description": description,
        "xi": xi,
        "delta": delta,
        "atom_list": atom_list,
        "calculator_model": calculator_model,
        "start_cnfs": [cnf.coords for cnf in start_cnfs],
        "end_cnfs": [cnf.coords for cnf in end_cnfs],
        "time_created": timestamp
    }
    full_loc = Path(location) / META_FILE_NAME
    with open(full_loc, 'w+') as f:
        json.dump(metadata, f)