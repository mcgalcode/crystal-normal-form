import json
import tqdm

from cnf.linalg.matrix_tuple import MatrixTuple
from itertools import product
from importlib.resources import files

import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
import time


def get_valid_columns(max_norm):
    """
    Pre-compute all valid columns (those satisfying the norm constraint).
    """
    r = range(-max_norm, max_norm + 1)
    valid_columns = []
    
    for col in product(r, repeat=3):
        if sum([abs(col[0]), abs(col[1]), abs(col[2])]) <= max_norm:
            valid_columns.append(col)
    
    return valid_columns

def check_chunk(args):
    """
    Worker function to check a chunk of column combinations.
    Returns list of valid matrices.
    """
    col1_list, valid_columns = args
    matrices = []
    
    for a1, a2, a3 in col1_list:
        # Iterate through all valid second columns
        for b1, b2, b3 in valid_columns:
            # Iterate through all valid third columns
            for c1, c2, c3 in valid_columns:
                mat = np.array([
                    [a1, b1, c1],
                    [a2, b2, c2],
                    [a3, b3, c3],
                ])

                if np.isclose(np.linalg.det(mat), 1):
                    matrices.append(mat)
    
    return matrices

def generate_unimodular_matrices_3x3(max_norm=5, num_processes=None):
    """
    Generate unimodular matrices using multiprocessing with pre-filtered valid columns.
    
    Args:
        max_norm: Maximum norm for columns
        num_processes: Number of processes to use (defaults to CPU count)
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes")
    print(f"Generating matrices with column max norm ≤ {max_norm}...")
    
    # Pre-compute all valid columns
    print("\nPre-computing valid columns...")
    valid_columns = get_valid_columns(max_norm)
    num_valid = len(valid_columns)
    
    print(f"Found {num_valid:,} valid columns (out of {(2*max_norm+1)**3:,} total)")
    print(f"Total combinations to check: {num_valid**3:,}\n")
    
    # Split first columns into chunks for parallel processing
    chunk_size = max(1, num_valid // num_processes)
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = num_valid if i == num_processes - 1 else (i + 1) * chunk_size
        chunks.append((valid_columns[start:end], valid_columns))
    
    # Process in parallel
    start_time = time.time()
    
    print("Processing...")
    with Pool(processes=num_processes) as pool:
        results = pool.map(check_chunk, chunks)
    
    # Combine results
    all_matrices = []
    for result in results:
        all_matrices.extend(result)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Found {len(all_matrices):,} unimodular matrices")
    
    # Convert to MatrixTuple and save
    all_mats = [MatrixTuple(m) for m in all_matrices]
    with open(f"unimodular_{max_norm}_det_1.json", 'w') as f:
        json.dump([m.to_list() for m in all_mats], f)
    
    print(f"Saved to unimodular_{max_norm}_det_1.json")
    
    return all_mats


def build_unimodular_matrices(max_entry_val, det = 1):
    entry_choices = range(-max_entry_val, max_entry_val + 1)
    print(f"I expect {len(entry_choices) ** 9} matrices")
    combinations = list(product(entry_choices, repeat=9))

    all_matrices = set([MatrixTuple.from_tuple(tuple(c)) for c in tqdm.tqdm(combinations, "Constructing matrix tuples...")])
    print(f"Found {len(all_matrices)} distinct matrices")

    all_unimodular = [m for m in tqdm.tqdm(all_matrices, "Testing unimodularity...") if m.is_unimodular()]
    print(f"Of which, {len(all_unimodular)} were unimodular")

    if det is not None:
        det_one = [m for m in tqdm.tqdm(all_unimodular, f"Testing det={det}") if m.determinant() == det]
        print(f"Of which, {len(det_one)} had determinant == {det}")
    
        for m in det_one:
            assert m.determinant() == det
    else:
        det_one = all_unimodular

    if det is None:
        det = "NA"
    with open(f"unimodular_{max_entry_val}_det_{det}.json", 'w') as f:
        json.dump([m.to_list() for m in det_one], f)
    return det_one

def load_unimodular(fname = "unimodular.json"):
    data = files("cnf.linalg").joinpath("data", fname).read_text()
    matrix_lists = json.loads(data)
    matrices = [MatrixTuple.from_tuple(tuple(l)) for l in matrix_lists]
    return matrices

_UNIMODULARS = {}

def get_unimodulars_col_max(col_max):
    if col_max not in _UNIMODULARS:
        unis = load_unimodular(f"unimodular_{col_max}_det_1.json")
        _UNIMODULARS[col_max] = unis
    return _UNIMODULARS[col_max]

UNIMODULAR_MATRICES = load_unimodular("unimodular_6_det_1.json")
UNIMODULAR_MATRICES_COL_MAX_NORM_1 = load_unimodular("unimodular.json")
