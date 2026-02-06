import os
import shutil
import random
import dataclasses
import json
import multiprocessing as mp

from pathlib import Path
from typing import Iterable

from ...crystal_normal_form import CrystalNormalForm
from .utilities import get_energies
from ..astar import pathfind_and_save
from ..astar.search_result import PathSearchResult

# The MEP path sampling algorithm is going to be as follows:
#
# Step 0: Select pathfinding parameters
# Step 1: Generate many paths between endpoints (forward and backward)
# Step 2: Filter paths by energy (iterative algorithm)
#         - Choose low resolution: 10 images per path
#         - Evaluate all images, then write results
#         - Choose maximum(num_desired_paths, maximum_retention_rate)
#           - The idea here is that if we want 10 paths at the end, we keep either the top 10 or,
#             if we are keeping 50% of the paths each time, we keep that 50%. Whichever forms the larger
#             group.
#         - Choose higher resolution: 50 images per path, and repeat

def random_string():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))

def get_sampled_path(path: list, num_pts: int):
    total_pts = len(path)
    num_chunks = num_pts - 1
    chunk_size = int(total_pts / num_chunks)
    sampled_path = path[::chunk_size]
    if (total_pts % chunk_size) / chunk_size > 0.5:
        sampled_path.append(path[-1])
    return sampled_path


@dataclasses.dataclass
class PathFindingParameters():

    xi: float
    delta: int
    min_distance: float
    max_iterations: int

    beam_width: int = 1000
    greedy: bool = False

def _save_path(inputs: tuple[str, str, str, PathFindingParameters]):
    start_cif, end_cif, output_file, params = inputs
    dropout = random.uniform(0, 0.8)
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return

    pathfind_and_save(start_cif,
                    end_cif,
                    output_file,
                    xi=params.xi,
                    delta=params.delta,
                    min_distance=params.min_distance,
                    max_iterations=params.max_iterations,
                    use_python=False,
                    bidirectional=False,
                    greedy=params.greedy,
                    beam_width=params.beam_width,
                    dropout=dropout,
                    speak_freq=1000,
                    verbose=True)

def _get_energy_key_str(cnf: CrystalNormalForm):
    return f"{cnf.coords.__repr__()}-{cnf.elements.__repr__()}-{cnf.xi}-{cnf.delta}"

class PathSampler():

    PATHS_RESULT_DIR = "path_results"
    SELECTED_PATHS = "selected_paths"

    ENERGY_MANIFEST_FILE = "path_energies.json"

    ENDPOINT_ONE_CIF = "endpoint1.cif"
    ENDPOINT_TWO_CIF = "endpoint2.cif"

    FW_PATH_PREFIX = "fw_path_"
    BW_PATH_PREFIX = "bw_path_"

    _path_results: dict[str, PathSearchResult]
    _energies: dict[str, float]

    @classmethod
    def setup(cls, working_dir: str, endpt1_cif_path: str, endpt2_cif_path: str):
        working_dir = Path(working_dir)
        if os.path.exists(working_dir):
            raise RuntimeError(f"Tried to setup path sampling workdir in an existing directory: {working_dir}")
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(working_dir / cls.PATHS_RESULT_DIR, exist_ok=True)
        os.makedirs(working_dir / cls.SELECTED_PATHS, exist_ok=True)
        shutil.copyfile(endpt1_cif_path, working_dir / cls.ENDPOINT_ONE_CIF)
        shutil.copyfile(endpt2_cif_path, working_dir / cls.ENDPOINT_TWO_CIF)

    def __init__(self,
                 working_dir: str,
                 parallel: bool = False):
        if not os.path.exists(working_dir):
            raise RuntimeError("Can't instantiate path sampler with non-existant working dir.")
        
        self.working_dir = Path(working_dir)
        self.paths_results_dir = self.working_dir / self.PATHS_RESULT_DIR
        self.energy_manifest_path = self.working_dir / self.ENERGY_MANIFEST_FILE
        self.endpoint_1_path = self.working_dir / self.ENDPOINT_ONE_CIF
        self.endpoint_2_path = self.working_dir / self.ENDPOINT_TWO_CIF
        self.parallel = parallel

        self._energies = {}

        if not os.path.exists(self.energy_manifest_path):
            self.write_energies({})
        
        self.reload_energies()
        self.reload_paths()

    def sample_paths(self,
                     pathfinding_params: PathFindingParameters,
                     num_attempts: int = None,
                     max_path_results: int = None):
        
        if max_path_results is None and num_attempts is None:
            raise ValueError(f"Must provide either num_attempts or max_path_results to PathSampler.sample_paths")
        
        current_num_paths = len(self._existing_path_files)

        if num_attempts is None:
            num_attempts = max(max_path_results - current_num_paths, 0)

        if max_path_results is not None and max_path_results > current_num_paths + num_attempts:
            raise ValueError(f"Incompatible parameters max_path_results and num_attempts: desired number of attempts would overrun max_path_results given current number of results: {current_num_paths}")
        
        parameter_sets = []
        for _ in range(num_attempts):
            if random.random() < 0.5:
                start = self.endpoint_1_path
                end = self.endpoint_2_path
                pref = self.FW_PATH_PREFIX
            else:
                start = self.endpoint_2_path
                end = self.endpoint_1_path
                pref = self.BW_PATH_PREFIX
            
            output_path = self.paths_results_dir / f"{pref}{random_string()}"
            parameter_sets.append(
                (start, end, output_path, pathfinding_params)
            )
        print(f"Finished {num_attempts} path-finding attempts!")
        
        if self.parallel:
            with mp.Pool(processes=6) as pool:
                pool.map(_save_path, parameter_sets)
        else:
            for pset in parameter_sets:
                _save_path(pset)
        self.reload_paths()

    @property
    def _existing_path_files(self):
        return [self.paths_results_dir / fobj.name for fobj in list(self.paths_results_dir.iterdir())]
    
    def reload_paths(self):
        loaded_paths = {}
        for path_file in self._existing_path_files:
            loaded_paths[str(path_file)] = PathSearchResult.from_json_file(path_file)
        self._path_results = loaded_paths

    def reload_energies(self):
        with open(self.energy_manifest_path, 'r+') as f:
            self._energies = json.load(f)
        
    def write_energies(self, energy_map: dict[int, float]):
        new_energies = { **self._energies, **energy_map }
        with open(self.energy_manifest_path, 'w+') as f:
            json.dump(new_energies, f)

    def compute_new_energies(self, cnfs: Iterable[CrystalNormalForm]):
        filtered_cnfs = [cnf for cnf in cnfs if _get_energy_key_str(cnf) not in self._energies]
        
        if len(filtered_cnfs) == 0:
            print(f"Already computed all these energies!")
            return
        
        energies = get_energies(filtered_cnfs)
        print(f"Finished computing {len(energies)} energies")
        new_entries = { _get_energy_key_str(cnf): e for e, cnf in zip(energies, filtered_cnfs) }
        self.write_energies(new_entries)
        self.reload_energies()
    
    def compute_path_energies(self, num_path_pts: int, specific_path_files: list[str] = None):
        all_reqd_cnfs = []
        for fname, path in self.completed_path_results.items():
            if specific_path_files is not None and fname not in specific_path_files:
                continue

            cnfs = path.get_cnfs_on_path()
            if num_path_pts is not None:
                sampled_path = get_sampled_path(cnfs, num_path_pts)
            else:
                sampled_path = cnfs

            all_reqd_cnfs.extend(sampled_path)
            
        self.compute_new_energies(set(all_reqd_cnfs))

    @property
    def completed_path_results(self):
        return { k: v for k, v in self._path_results.items() if v.path is not None}

    def get_path_max_energies(self):
        r = {}
        for fname, path in self.completed_path_results.items():
            path_keys = [_get_energy_key_str(cnf) for cnf in path.get_cnfs_on_path()]
            energies = [self._energies.get(k, -math.inf) for k in path_keys]
            r[fname] = max(energies)
        return r
    
    def refine_paths(self, num_pts_per_path, top_k=10):
        """
        Given the current energy evaluations, identify the `top_k` paths with the
        lowest maximum energies (i.e. the most promising paths for truly having a low
        maximum energy), then refine those paths with new energy computations until they
        have `num_pts_per_path` computed for them. Designed to be used iteratively. 
        
        :param self: Description
        :param num_pts_per_path: Description
        :param best: Description
        """
        curr_max_path_energies = self.get_path_max_energies()
        sorted_path_files = sorted(curr_max_path_energies.keys(), key=lambda p_file: curr_max_path_energies[p_file])
        top_k_paths = sorted_path_files[:top_k]
        self.compute_path_energies(num_pts_per_path, top_k_paths)



    
