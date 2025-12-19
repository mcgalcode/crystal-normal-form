import os
import json
import tqdm
from .setup import setup_cnf_db, instantiate_search, setup_meta_db
from .meta_file import write_meta_file, add_search_process
from ..crystal_normal_form import CrystalNormalForm
from ..calculation.base_calculator import BaseCalculator
from .partitioned_db import PartitionedDB
from .constants import PARTITION_SUFFIX, META_DB_NAME



def _meta_params_from_cnfs(cnfs: list[CrystalNormalForm]):
    xi = cnfs[0].xi
    delta = cnfs[0].delta
    element_list = cnfs[0].elements
    return xi, delta, element_list

def setup_partitioned_db(location,
                         description: str,
                         num_partitions,
                         start_cnfs: list[CrystalNormalForm],
                         end_cnfs: list[CrystalNormalForm],
                         calculator: BaseCalculator):
    xi, delta, element_list = _meta_params_from_cnfs(start_cnfs)
    print(f"\n")
    os.makedirs(location, exist_ok=True)
    sids = []
    start_energies = [calculator.calculate_energy(c) for c in start_cnfs]
    end_energies = [calculator.calculate_energy(c) for c in end_cnfs]
    for i in tqdm.tqdm(range(num_partitions), total=num_partitions, desc=f"Creating database partitions in: {location}"):
        store_file = f"{location}/{i}{PARTITION_SUFFIX}" 
        setup_cnf_db(store_file, xi, delta, element_list)
        new_sid = instantiate_search(description,
                                     start_cnfs,
                                     end_cnfs,
                                     store_file,
                                     calculator,
                                     start_energies,
                                     end_energies)
        sids.append(new_sid)
    
    if len(set(sids)) > 1:
        raise RuntimeError("Inconsistent search IDs encountered while setting up search in partitioned DB")
    sid = sids[0]

    meta_db = setup_meta_db(os.path.join(location, META_DB_NAME))

    for i in range(num_partitions):
        meta_db.create_partition_entry(sid, i)

    return sid
    

def setup_search_dir(location,
                     description: str,
                     num_partitions: int,
                     start_cnfs: list[CrystalNormalForm],
                     end_cnfs: list[CrystalNormalForm],
                     calculator: BaseCalculator):
    xi, delta, element_list = _meta_params_from_cnfs(start_cnfs)
    calc_identifier = calculator.identifier()
    sid = setup_partitioned_db(location, description, num_partitions, start_cnfs, end_cnfs, calculator)
    write_meta_file(location, xi, delta, element_list, calc_identifier, description)
    add_search_process(location, sid, start_cnfs, end_cnfs)
    PartitionedDB(location, sid).sync_control_water_level()
    return sid
