import json
import dataclasses
from pathlib import Path
from datetime import datetime

from ..crystal_normal_form import CrystalNormalForm

META_FILE_NAME = 'search_metadata.json'

@dataclasses.dataclass
class SearchProcess():

    search_id: int
    start_cnfs: list[list[int]]
    end_cnfs: list[list[int]]
    time_created: str

@dataclasses.dataclass
class SearchMetadata():

    description: str
    xi: float
    delta: int
    atom_list: list[str]
    calculator_model: str
    time_created: str
    search_processes: list[SearchProcess]


def write_meta_file(location: str,
                    xi: float,
                    delta: int,
                    atom_list: list[str],
                    calculator_model: str,
                    description: str = ""):

    timestamp = datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    sm = SearchMetadata(
        description=description,
        xi=xi,
        delta=delta,
        atom_list=atom_list,
        calculator_model=calculator_model,
        time_created=timestamp,
        search_processes=[]
    )

    _write_metafile(location, sm)

def _write_metafile(location: str, search_metadata: SearchMetadata):
    metadata = dataclasses.asdict(search_metadata)
    full_loc = Path(location) / META_FILE_NAME
    with open(full_loc, 'w+') as f:
        json.dump(metadata, f)
        return f.name

def add_search_process(search_dir, sid, start_cnfs: list[CrystalNormalForm], end_cnfs: list[CrystalNormalForm]):
    metadata = load_meta_file(search_dir)
    timestamp = datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")    
    new_sp = SearchProcess(search_id=sid,
                           start_cnfs=[cnf.coords for cnf in start_cnfs],
                           end_cnfs=[cnf.coords for cnf in end_cnfs],
                           time_created=timestamp)
    metadata.search_processes.append(new_sp)
    _write_metafile(search_dir, metadata)

def load_meta_file(search_dir):
    full_loc = Path(search_dir) / META_FILE_NAME
    with open(full_loc, 'r+') as f:
        metadata = json.load(f)
        search_procs = [SearchProcess(**sp) for sp in metadata["search_processes"]]
        metadata["search_processes"] = search_procs
        return SearchMetadata(**metadata)