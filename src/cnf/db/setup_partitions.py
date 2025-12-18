import os
import json
import tqdm
from .setup import setup_cnf_db, instantiate_search
from .meta_file import write_meta_file
from ..crystal_normal_form import CrystalNormalForm
from ..calculation.base_calculator import BaseCalculator

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
    for i in tqdm.tqdm(range(num_partitions), total=num_partitions, desc=f"Creating database partitions in: {location}"):
        store_file = f"{location}/graph_partition_{i}.db" 
        setup_cnf_db(store_file, xi, delta, element_list)
        instantiate_search(description, start_cnfs, end_cnfs, store_file, calculator)

def setup_search_dir(location,
                     description: str,
                     num_partitions: int,
                     start_cnfs: list[CrystalNormalForm],
                     end_cnfs: list[CrystalNormalForm],
                     calculator: BaseCalculator):
    xi, delta, element_list = _meta_params_from_cnfs(start_cnfs)
    calc_identifier = calculator.identifier()
    setup_partitioned_db(location, description, num_partitions, start_cnfs, end_cnfs, calculator)
    write_meta_file(location, xi, delta, element_list, calc_identifier, start_cnfs, end_cnfs, description)