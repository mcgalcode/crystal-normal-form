import pytest

import cnf.lattice_normal_form.sorting as srt

@pytest.fixture
def antimony_vonorms_unsorted():
    return [
        19.2,
        21.3,
        19.2,
        21.3,
        40.5,
        19.2,
        21.3
    ]

def test_is_primary_vonorm():
    assert srt.is_primary_vonorm_idx(0)
    assert srt.is_primary_vonorm_idx(1)
    assert srt.is_primary_vonorm_idx(2)
    assert srt.is_primary_vonorm_idx(3)

    assert not srt.is_primary_vonorm_idx(4)
    assert not srt.is_primary_vonorm_idx(5)
    assert not srt.is_primary_vonorm_idx(6)

def test_is_secondary_vonorm():
    assert not srt.is_secondary_vonorm_idx(0)
    assert not srt.is_secondary_vonorm_idx(1)
    assert not srt.is_secondary_vonorm_idx(2)
    assert not srt.is_secondary_vonorm_idx(3)

    assert srt.is_secondary_vonorm_idx(4)
    assert srt.is_secondary_vonorm_idx(5)
    assert srt.is_secondary_vonorm_idx(6)

def test_check_primary_sorted(antimony_vonorms_unsorted):
    is_sorted, out_of_order_pair = srt.check_primary_vonorms_sorted(antimony_vonorms_unsorted)
    assert not is_sorted
    assert out_of_order_pair == (1,2)

def test_check_secondary_sorted(antimony_vonorms_unsorted):
    is_sorted, out_of_order_pair = srt.check_secondary_vonorms_sorted(antimony_vonorms_unsorted)
    assert not is_sorted
    assert out_of_order_pair == (4,5)

def test_swap_list_items_in_place():
    test_list = [4, 3, 2, 1]
    srt.swap_list_items_in_place(0, 3, test_list)
    assert test_list[0] == 1
    assert test_list[3] == 4

def test_swap_vonorm_idxs_in_place(antimony_vonorms_unsorted):
    with pytest.raises(RuntimeError) as order_err:
        srt.swap_vonorm_idxs_in_place(3, 2, antimony_vonorms_unsorted)
    assert "Out-of-order" in str(order_err.value)
    
    with pytest.raises(RuntimeError) as primary_secondary_err:
        srt.swap_vonorm_idxs_in_place(3,5, antimony_vonorms_unsorted)
    assert "primary vonorms" in str(primary_secondary_err.value)

    srt.swap_vonorm_idxs_in_place(2,3, antimony_vonorms_unsorted)
    assert antimony_vonorms_unsorted[2] == 21.3
    assert antimony_vonorms_unsorted[3] == 19.2

    assert antimony_vonorms_unsorted[5] == 21.3
    assert antimony_vonorms_unsorted[6] == 19.2

def test_sort_vonorm_list(antimony_vonorms_unsorted):
    swaps = srt.sort_vonorms(antimony_vonorms_unsorted)

    assert antimony_vonorms_unsorted[0] == 19.2
    assert antimony_vonorms_unsorted[1] == 19.2
    assert antimony_vonorms_unsorted[2] == 21.3
    assert antimony_vonorms_unsorted[3] == 21.3

    assert antimony_vonorms_unsorted[4] == 19.2
    assert antimony_vonorms_unsorted[5] == 21.3
    assert antimony_vonorms_unsorted[6] == 40.5
    

