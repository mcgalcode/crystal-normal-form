import json
from ..crystal_normal_form import CrystalNormalForm

def cnf_to_str(cnf: CrystalNormalForm):
    return json.dumps([c for c in cnf.coords])

def cnf_from_str(cnf_string: str, xi: float, delta: int, els: list[str]):
    cnf_list = json.loads(cnf_string)
    return CrystalNormalForm.from_tuple(tuple(cnf_list), els, xi, delta)