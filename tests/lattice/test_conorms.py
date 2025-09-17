from cnf.lattice.superbasis import Superbasis
from cnf.linalg.matrix_tuple import MatrixTuple
from pymatgen.core.lattice import Lattice

def test_v5_case():
    cuboid = Lattice.cubic(1.0)
    sb = Superbasis.from_pymatgen_lattice(cuboid)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.zero_indices) == 3
    assert conorms.voronoi_class == 5
    print(f"{conorms.voronoi_class}: {len(conorms.permissible_permutations)}")

def test_v4_case():
    hexagonal_prism = Lattice.hexagonal(1.0, 2.0)
    sb = Superbasis.from_pymatgen_lattice(hexagonal_prism)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.zero_indices) == 2
    assert conorms.voronoi_class == 4
    print(f"{conorms.voronoi_class}: {len(conorms.permissible_permutations)}")

def test_v3_case():
    rhombic_dodecahedron = Lattice([
        [1, 1, 0],
        [1, -1, 0],
        [-1, 0, 1],
    ])
    sb = Superbasis.from_pymatgen_lattice(rhombic_dodecahedron)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.zero_indices) == 2
    assert conorms.voronoi_class == 3
    print(f"{conorms.voronoi_class}: {len(conorms.permissible_permutations)}")

def test_v2_case():
    hexarhombic_dodecahedron = Lattice([
        [1, 1, 0],
        [1, -1, 0],
        [-1, 0.2, 1],
    ])
    sb = Superbasis.from_pymatgen_lattice(hexarhombic_dodecahedron)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.zero_indices) == 1
    assert conorms.voronoi_class == 2
    print(f"{conorms.voronoi_class}: {len(conorms.permissible_permutations)}")
    

def test_v1_case():
    truncated_octahedron = Lattice([
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ])
    sb = Superbasis.from_pymatgen_lattice(truncated_octahedron)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.zero_indices) == 0
    assert conorms.voronoi_class == 1
    print(f"{conorms.voronoi_class}: {len(conorms.permissible_permutations)}")

    unimodular_matrices = []
    for p in conorms.permissible_permutations:
        unimodular_matrices.append(MatrixTuple(p.to_unimodular_matrix()))
    
    mat_tuples = [m.tuple for m in unimodular_matrices]
    print(f"{len(conorms.permissible_permutations)} distinct permutations")
    print(f"{len(set(mat_tuples))} distinct unimodular matrices")
    for m in unimodular_matrices:
        print(m.matrix)
