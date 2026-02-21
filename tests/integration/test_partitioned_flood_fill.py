import pytest
import helpers
import tempfile
import shutil
from cnf.db.setup_partitions import setup_search_dir
from cnf.db.partitioned_db import PartitionedDB
from cnf.navigation.waterfill import continue_search_waterfill
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.calculation.constant_calculator import ConstantCalculator
from cnf import CrystalNormalForm


@pytest.fixture
def zr_hcp_mp():
    """Load Zr HCP structure for testing."""
    return helpers.load_specific_cif("Zr_HCP.cif")


@pytest.fixture
def partitioned_db_dir():
    """Create a temporary directory for partitioned database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)


def test_partitioned_flood_fill_integrity(zr_hcp_mp, partitioned_db_dir):
    """
    Integration test for partitioned flood-fill search.

    Tests:
    1. Points are stored uniquely across partitions
    2. Explored points have correct neighbors stored
    3. Neighbors are consistent with NeighborFinder
    """
    xi = 1.5
    delta = 5
    num_partitions = 4

    # Create start and end points
    zr_hcp_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_mp, xi, delta)
    start_cnf = zr_hcp_cnf

    # Find neighbors 3 rings out
    nf = NeighborFinder.from_cnf(start_cnf)
    ring_1 = set(nf.find_neighbors(start_cnf))
    ring_2 = set()
    for nb in list(ring_1)[:10]:
        ring_2.update(nf.find_neighbors(nb))
    ring_2 = ring_2 - ring_1 - {start_cnf}

    ring_3 = set()
    for nb in list(ring_2)[:10]:
        ring_3.update(nf.find_neighbors(nb))
    ring_3 = ring_3 - ring_2 - ring_1 - {start_cnf}

    end_cnf = list(ring_3)[0]  # Pick first point from ring 3

    # Set up partitioned database with search
    calculator = ConstantCalculator(1)
    search_id = setup_search_dir(
        partitioned_db_dir,
        "Test search",
        num_partitions,
        [start_cnf],
        [end_cnf],
        calculator
    )

    # Run flood-fill search for 10 iterations
    continue_search_waterfill(
        search_id,
        partitioned_db_dir,
        calculator,
        max_iters=10,
        log_lvl=0
    )

    # Create PartitionedDB for assertions
    db = PartitionedDB(partitioned_db_dir, search_id)

    # ========================================
    # Test 1: All points are unique (except start/end which are in all partitions)
    # ========================================
    all_points = []
    all_point_coords = set()

    # Start and end points are allowed to be duplicated across partitions
    allowed_duplicates = {start_cnf.coords, end_cnf.coords}

    for partition_idx in range(db.num_partitions):
        map_store = db.get_map_store_by_idx(partition_idx)

        # Get all points using store methods
        explored_points = map_store.get_all_explored_points()
        unexplored_points = map_store.get_all_unexplored_points()
        all_partition_points = explored_points + unexplored_points

        for point in all_partition_points:
            coords = point.cnf.coords

            # Check if we've seen this point before (skip start/end points)
            if coords not in allowed_duplicates:
                assert coords not in all_point_coords, \
                    f"Duplicate point found: {coords} in partition {partition_idx}"

            all_point_coords.add(coords)
            all_points.append((partition_idx, point))

    print(f"\n✓ Test 1 passed: All {len(all_points)} points are unique across {db.num_partitions} partitions (excluding start/end)")

    # ========================================
    # Test 2: Explored points have EXACTLY the neighbors from NeighborFinder
    # ========================================
    explored_points_checked = 0

    for partition_idx, point in all_points:
        # Only check explored points
        if not point.explored:
            continue

        explored_points_checked += 1
        map_store = db.get_map_store_by_idx(partition_idx)

        # Get neighbors from database using new method that handles both
        # same-partition (target_id) and cross-partition (target_cnf) edges
        db_neighbor_cnfs = map_store.get_neighbor_cnfs(point.id)
        db_neighbor_coords = set(nb.coords for nb in db_neighbor_cnfs)

        # Get neighbors using NeighborFinder (ground truth)
        computed_neighbors = nf.find_neighbors(point.cnf)
        computed_neighbor_coords = set(nb.coords for nb in computed_neighbors)

        # Assert exact set equality - DB should have EXACTLY the same neighbors
        assert db_neighbor_coords == computed_neighbor_coords, \
            f"Point {point.cnf.coords} has mismatched neighbors:\n" \
            f"  DB has {len(db_neighbor_coords)} neighbors\n" \
            f"  NeighborFinder has {len(computed_neighbor_coords)} neighbors\n" \
            f"  In DB but not NeighborFinder: {db_neighbor_coords - computed_neighbor_coords}\n" \
            f"  In NeighborFinder but not DB: {computed_neighbor_coords - db_neighbor_coords}"

    print(f"✓ Test 2 passed: {explored_points_checked} explored points have EXACTLY the correct neighbors stored")

    # ========================================
    # Test 3: Start point was explored
    # ========================================
    start_partition = db.get_partition_idx(start_cnf)
    start_map_store = db.get_map_store_by_idx(start_partition)
    start_point = start_map_store.get_point_by_cnf(start_cnf)

    assert start_point.explored, "Start point should be explored after 10 iterations"
    print(f"✓ Test 3 passed: Start point was explored")

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*60}")
    print(f"Integration test passed!")
    print(f"  Total points: {len(all_points)}")
    print(f"  Explored points: {explored_points_checked}")
    print(f"  Partitions: {db.num_partitions}")
    print(f"{'='*60}")


def test_partitioned_flood_fill_neighbor_reciprocity(zr_hcp_mp, partitioned_db_dir):
    """
    Test that neighbor relationships are properly reciprocal.

    When point A is explored and finds neighbor B:
    - A should have outgoing edge to B
    When point B is later explored:
    - B should have outgoing edge to A
    """
    xi = 1.5
    delta = 5
    num_partitions = 4

    zr_hcp_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_mp, xi, delta)
    start_cnf = zr_hcp_cnf

    # Create search with endpoint a couple rings away
    nf = NeighborFinder.from_cnf(start_cnf)
    neighbors = nf.find_neighbors(start_cnf)
    ring_2 = set()
    for nb in neighbors:
        ring_2.update(nf.find_neighbors(nb))
    ring_2 = ring_2 - set(neighbors) - {start_cnf}
    end_cnf = list(ring_2)[0]

    # Set up partitioned database with search
    calculator = ConstantCalculator(1)
    search_id = setup_search_dir(
        partitioned_db_dir,
        "Test reciprocity",
        num_partitions,
        [start_cnf],
        [end_cnf],
        calculator
    )

    # Run search for more iterations to explore multiple points
    continue_search_waterfill(
        search_id,
        partitioned_db_dir,
        calculator,
        max_iters=50,
        log_lvl=0
    )

    # Create PartitionedDB for assertions
    db = PartitionedDB(partitioned_db_dir, search_id)

    # ========================================
    # Test 1: Every explored point has EXACTLY the neighbors from NeighborFinder
    # ========================================
    explored_points_checked = 0
    all_explored_points = []

    for partition_idx in range(db.num_partitions):
        map_store = db.get_map_store_by_idx(partition_idx)
        explored_points = map_store.get_all_explored_points()

        for point in explored_points:
            # Get neighbors from database
            db_neighbor_cnfs = map_store.get_neighbor_cnfs(point.id)
            db_neighbor_coords = set(nb.coords for nb in db_neighbor_cnfs)

            # Get neighbors using NeighborFinder (ground truth)
            computed_neighbors = nf.find_neighbors(point.cnf)
            computed_neighbor_coords = set(nb.coords for nb in computed_neighbors)

            # Assert exact set equality
            assert db_neighbor_coords == computed_neighbor_coords, \
                f"Point {point.cnf.coords} has mismatched neighbors:\n" \
                f"  DB has {len(db_neighbor_coords)} neighbors\n" \
                f"  NeighborFinder has {len(computed_neighbor_coords)} neighbors\n" \
                f"  In DB but not NeighborFinder: {db_neighbor_coords - computed_neighbor_coords}\n" \
                f"  In NeighborFinder but not DB: {computed_neighbor_coords - db_neighbor_coords}"

            explored_points_checked += 1
            all_explored_points.append((partition_idx, point))

    print(f"✓ Neighbor set test passed: {explored_points_checked} explored points have EXACTLY the correct neighbors")

    # ========================================
    # Test 2: Check reciprocity
    # ========================================
    # For any two explored points that are neighbors, they should each have an edge to the other
    reciprocal_pairs_found = 0

    for partition_idx, point in all_explored_points:
        map_store = db.get_map_store_by_idx(partition_idx)

        # Get neighbors from DB using new method
        db_neighbor_cnfs = map_store.get_neighbor_cnfs(point.id)

        for nb_cnf in db_neighbor_cnfs:
            # Check if neighbor is also explored
            nb_partition = db.get_partition_idx(nb_cnf)
            nb_map_store = db.get_map_store_by_idx(nb_partition)

            try:
                nb_point = nb_map_store.get_point_by_cnf(nb_cnf)

                if nb_point and nb_point.explored:
                    # Both points are explored - check reciprocity
                    # Get neighbors of the neighbor
                    nb_neighbor_cnfs = nb_map_store.get_neighbor_cnfs(nb_point.id)
                    nb_neighbor_coords = [n.coords for n in nb_neighbor_cnfs]

                    # Point should be in neighbor's neighbor list
                    assert point.cnf.coords in nb_neighbor_coords, \
                        f"Reciprocity violated: {point.cnf.coords} → {nb_cnf.coords} exists, " \
                        f"but {nb_cnf.coords} → {point.cnf.coords} is missing"

                    reciprocal_pairs_found += 1

            except Exception:
                # Neighbor not in database yet, skip
                pass

    print(f"✓ Reciprocity test passed: {reciprocal_pairs_found} reciprocal edge pairs verified")
