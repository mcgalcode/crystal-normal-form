from dataclasses import dataclass

import json
from ..crystal_normal_form import CrystalNormalForm

def cnf_to_str(cnf: CrystalNormalForm):
    return json.dumps([c for c in cnf.coords])

def cnf_from_str(cnf_string: str, xi: float, delta: int, els: list[str]):
    cnf_list = json.loads(cnf_string)
    return CrystalNormalForm.from_tuple(tuple(cnf_list), els, xi, delta)

@dataclass
class CNFPoint():

    id: int
    cnf: CrystalNormalForm
    explored: bool
    external_id: str
    value: float
    partition: int = None


def cnf_pt_from_row(row: tuple, delta: int, xi: float, elements: list[str]):
    cnf = cnf_from_str(row[1], xi, delta, elements)
    return CNFPoint(
        id=row[0],
        cnf=cnf,
        external_id=row[2],
        value=row[3],
        explored=row[4],
    )