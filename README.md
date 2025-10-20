# Crystal Normal Form (CNF)

A Python implementation of the crystallographic mapping algorithm described in [Mrdjenovich & Persson (2024)](https://doi.org/10.1103/PhysRevMaterials.8.033401), enabling efficient and discretized exploration of crystallographic phase space.

## What is CNF?

**Crystal Normal Form** provides a unique, canonical representation of crystal structures as integer coordinates. This enables:

- **Systematic phase space exploration**: Navigate through crystallographic configurations via small, incremental structural perturbations
- **Transformation pathway discovery**: Find low-energy pathways between crystal phases by connecting neighboring structures
- **Materials stability analysis**: Study how crystal energy varies across the full configuration space
- **Phase transformation studies**: Investigate martensitic transformations, piezoelectricity, and other structural phenomena

### The Algorithm

The CNF algorithm represents any 3D crystal structure using two components:

1. **Lattice Normal Form (LNF)**: Uses the Selling reduction to represent the lattice via 7 squared "vonorms" (vector norms from an obtuse superbasis), discretized to integers
2. **Basis Normal Form (BNF)**: Represents atomic positions in fractional coordinates, discretized to integer intervals

Together, these form a unique integer coordinate string that:
- Eliminates redundancies (origin choice, cell choice, atom labeling)
- Creates an implicit graph structure over all possible crystals
- Enables generation of "neighbor" crystals that differ by small strains (~�) or phonon displacements (~1/�)
- Guarantees any two crystals can be connected via a series of neighbors

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crystal-normal-form.git
cd crystal-normal-form

# Install with uv (recommended) or pip
uv pip install -e .
# or
pip install -e .
```

## Quick Start

### Basic Usage: CIF File to CNF

```python
from pymatgen.core import Structure
from cnf.cnf_constructor import CNFConstructor

# Load a crystal structure from a CIF file
structure = Structure.from_file("my_crystal.cif")

# Create a CNF constructor with discretization parameters
# xi: lattice discretization (Ų) - controls lattice neighbor resolution
# delta: basis discretization (divisions of [0,1)) - controls atomic position resolution
constructor = CNFConstructor(xi=1.5, delta=30)

# Convert structure to CNF
result = constructor.from_pymatgen_structure(structure)
cnf = result.cnf

# The CNF is now a unique integer representation
print(f"CNF coordinates: {cnf.coords}")
print(f"Lattice part: {cnf.coords[:7]}")  # First 7 integers
print(f"Basis part: {cnf.coords[7:]}")    # Remaining 3(n-1) integers
```

### Working with pymatgen Structures

```python
from pymatgen.core import Structure, Lattice
from cnf.cnf_constructor import CNFConstructor

# Create a structure programmatically
lattice = Lattice.cubic(4.0)
structure = Structure(
    lattice,
    ["Fe", "Fe"],
    [[0, 0, 0], [0.5, 0.5, 0.5]]
)

# Convert to CNF
constructor = CNFConstructor(xi=0.1, delta=100)
cnf_result = constructor.from_pymatgen_structure(structure)
cnf = cnf_result.cnf

# Access CNF properties
print(f"Discretization parameters: xi={cnf.xi}, delta={cnf.delta}")
print(f"Voronoi class: {cnf.voronoi_class}")

# Reconstruct the original crystal structure
reconstructed = cnf.reconstruct()
print(f"Reconstructed structure: {reconstructed}")
```

### Round-trip Conversion

```python
from pymatgen.core import Structure
from cnf.cnf_constructor import CNFConstructor

# Load structure
original = Structure.from_file("diamond.cif")

# Convert to CNF and back
constructor = CNFConstructor(xi=0.01, delta=1000)
cnf = constructor.from_pymatgen_structure(original).cnf
reconstructed = cnf.reconstruct()

# The reconstructed structure should match the original
# (within the discretization tolerance)
```

### Saving and Loading CNF

```python
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from pymatgen.core import Structure

# Create CNF from structure
structure = Structure.from_file("crystal.cif")
constructor = CNFConstructor(xi=1.5, delta=30)
cnf = constructor.from_pymatgen_structure(structure).cnf

# Save to JSON file
cnf.to_file("crystal_cnf.json")

# Load from file
loaded_cnf = CrystalNormalForm.from_file("crystal_cnf.json")

# Verify they're identical
assert cnf == loaded_cnf
```

## Understanding Discretization Parameters

### `xi` (lattice step size, in Ų)

Controls the resolution of lattice variations. Neighboring lattices differ by strains of magnitude ~(�/v�), where v is a typical lattice vector length.

- **Smaller xi** (e.g., 0.01): Fine-grained lattice space, many neighbors, slower but more thorough exploration
- **Larger xi** (e.g., 1.5): Coarse-grained, fewer neighbors, faster exploration but may miss fine features
- **Typical values**: 0.01-2.0 r

### `delta` (basis divisions)

Controls the resolution of atomic positions by dividing the unit cell interval [0,1) into `delta` equal parts.

- **Smaller delta** (e.g., 10): Coarse atomic positions, larger phonon mode amplitudes (~1/�)
- **Larger delta** (e.g., 100-1000): Fine atomic positions, small atomic displacements
- **Typical values**: 30-1000

### Choosing Parameters

For **phase transformation studies**: Use moderate values (xiH0.1-1.0, deltaH30-100) to balance thoroughness and computational cost.

For **high-precision comparisons**: Use fine discretization (xiH0.01, deltaH1000) to minimize rounding errors.

For **exploratory analysis**: Use coarse discretization (xiH1-2, deltaH10-30) for rapid surveying of phase space.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mrdjenovich2024crystallographic,
  title={Crystallographic map: A general lattice and basis formalism enabling efficient and discretized exploration of crystallographic phase space},
  author={Mrdjenovich, David and Persson, Kristin A.},
  journal={Physical Review Materials},
  volume={8},
  number={3},
  pages={033401},
  year={2024},
  publisher={American Physical Society},
  doi={10.1103/PhysRevMaterials.8.033401}
}
```

