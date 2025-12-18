import pytest
import tempfile

from cnf.db.meta_file import write_meta_file, load_meta_file, add_search_process
from cnf.navigation.endpoints import get_endpoint_cnfs


def test_can_write_and_read_metafile():

    xi = 1.5
    delta = 10
    elements = ["Zr", "Zr"]
    calculator_model = "testcalc"
    description = "test meta"

    with tempfile.TemporaryDirectory() as tmp:
        write_meta_file(tmp, xi, delta, elements, calculator_model, description)
        loaded = load_meta_file(tmp)
        assert loaded.atom_list == elements
        assert loaded.xi == xi
        assert loaded.delta == delta
        assert loaded.calculator_model == calculator_model
        assert loaded.description == description
        assert loaded.search_processes == []
        assert loaded.time_created is not None

def test_can_add_search_process(zr_bcc_mp, zr_hcp_mp):

    xi = 1.5
    delta = 10
    elements = ["Zr", "Zr"]
    calculator_model = "testcalc"
    description = "test meta"

    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi=1.5, delta=10)

    with tempfile.TemporaryDirectory() as tmp:
        write_meta_file(tmp, xi, delta, elements, calculator_model, description)

        sid = 1
        add_search_process(tmp, sid, sps, eps)
        loaded = load_meta_file(tmp)
        proc = loaded.search_processes[0]
        assert proc.search_id == 1
        assert set([tuple(cnf) for cnf in proc.start_cnfs]) == set([c.coords for c in sps])
        assert set([tuple(cnf) for cnf in proc.end_cnfs]) == set([c.coords for c in eps])

        sid = 2
        add_search_process(tmp, sid, sps, eps)
        loaded = load_meta_file(tmp)
        assert len(loaded.search_processes) == 2
        proc = loaded.search_processes[1]
        assert proc.search_id == 2
        assert set([tuple(cnf) for cnf in proc.start_cnfs]) == set([c.coords for c in sps])
        assert set([tuple(cnf) for cnf in proc.end_cnfs]) == set([c.coords for c in eps])
