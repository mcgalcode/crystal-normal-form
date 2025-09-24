from cnf.lattice.superbasis import Superbasis
from cnf.lattice.selling import SuperbasisSellingReducer
from cnf.lattice.permutations import is_permutation_set_closed
from pymatgen.core.lattice import Lattice
from cnf.lattice.voronoi import ConormListForm

def test_v5_case():
    cuboid = Lattice.cubic(1.0)
    sb = Superbasis.from_pymatgen_lattice(cuboid)
    r = SuperbasisSellingReducer()
    sb: Superbasis = r.reduce(sb).reduced_object

    conorms = sb.compute_vonorms().conorms
    assert len(conorms.form) == 3
    assert conorms.form.voronoi_class == 5
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")



    good = []
    bad = []
    for cperm in conorms.permissible_permutations:
        perm = cperm.to_vonorm_permutation()
        # print(mat, cperm.to_vonorm_permutation())
        permuted = sb.apply_permutation(perm)
        if permuted.is_superbasis():
            good.append(perm)
        else:
            bad.append(perm)
        # assert permuted.is_superbasis()
        # assert permuted.compute_vonorms().has_same_members(sb.compute_vonorms())
    
    print(len(good))


def test_v4_case():
    hexagonal_prism = Lattice.hexagonal(1.0, 2.0)
    sb = Superbasis.from_pymatgen_lattice(hexagonal_prism)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.form) == 2
    assert conorms.form.voronoi_class == 4
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")

def test_v3_case():
    rhombic_dodecahedron = Lattice([
        [1, 1, 0],
        [1, -1, 0],
        [-1, 0, 1],
    ])
    sb = Superbasis.from_pymatgen_lattice(rhombic_dodecahedron)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.form) == 2
    assert conorms.form.voronoi_class == 3
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")

def test_v2_case():
    hexarhombic_dodecahedron = Lattice([
        [1, 1, 0],
        [1, -1, 0],
        [-1, 0.2, 1],
    ])
    sb = Superbasis.from_pymatgen_lattice(hexarhombic_dodecahedron)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.form) == 1
    assert conorms.form.voronoi_class == 2
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    

def test_v1_case():
    truncated_octahedron = Lattice([
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ])
    sb = Superbasis.from_pymatgen_lattice(truncated_octahedron)
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.form) == 0
    assert conorms.form.voronoi_class == 1
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")

    unimodular_matrices = []
    for p in conorms.permissible_permutations:
        unimodular_matrices.append(p.to_unimodular_matrix())
    
    mat_tuples = [m.tuple for m in unimodular_matrices]
    print(f"{len(conorms.permissible_permutations)} distinct permutations")
    print(f"{len(set(mat_tuples))} distinct unimodular matrices")

def test_build_all_conorm_lists():
    all_lists = ConormListForm.all_coforms()
    assert len(all_lists) == 42
    