import os
import shutil
import random
import dataclasses

from math import lcm

from pathlib import Path

from ..endpoints import get_endpoint_cnfs
from ..astar import pathfind_and_save

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

class PathSampler():

    PATHS_RESULT_DIR = "path_results"
    SELECTED_PATHS = "selected_paths"

    ENERGY_MANIFEST_FILE = "path_energies.json"

    ENDPOINT_ONE_CIF = "endpoint1.cif"
    ENDPOINT_TWO_CIF = "endpoint2.cif"

    FW_PATH_PREFIX = "fw_path_"
    BW_PATH_PREFIX = "bw_path_"

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

    def sample_paths(self,
                     pathfinding_params: PathFindingParameters,
                     num_attempts: int = None,
                     max_path_results: int = None):
        
        if max_path_results is None and num_attempts is None:
            raise ValueError(f"Must provide either num_attempts or max_path_results to PathSampler.sample_paths")
        
        current_num_paths = len(self._existing_path_files)

        if num_attempts is None:
            num_attempts = max_path_results - current_num_paths

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
        
        if self.parallel:
            pass
        else:
            for pset in parameter_sets:
                _save_path(pset)
        


    @property
    def _existing_path_files(self):
        return list(self.paths_results_dir.iterdir())


    
