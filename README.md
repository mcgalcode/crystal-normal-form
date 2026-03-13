# Crystal Normal Form

A Python implementation of the crystallographic mapping algorithm described in [Mrdjenovich & Persson (2024)](https://doi.org/10.1103/PhysRevMaterials.8.033401).

## Introduction

Crystal structures are traditionally described using three lattice vectors combined with fractional atomic coordinates. While intuitive, this representation is highly redundant: infinitely many choices of unit cell, origin, and atom labeling can describe the same physical crystal. This redundancy complicates crystal comparison, database storage, and systematic exploration of structural phase space.

**Crystal Normal Form (CNF)** provides a unique, canonical integer representation for any 3D crystal structure, eliminating all representational ambiguities and creating an implicit graph structure connecting every possible crystal configuration.

The algorithm decomposes crystal geometry into two components:

1. **Lattice Normal Form (LNF)**: The periodic lattice is represented using the Selling reduction, which transforms three lattice vectors into an "obtuse superbasis" of four vectors. From this, seven "vonorms" (squared vector norms) uniquely characterize the lattice geometry, discretized to integers.

2. **Motif Normal Form (MNF)**: Atomic positions within the unit cell are represented as fractional coordinates discretized to integer multiples of 1/δ, with one atom fixed at the origin to eliminate translational freedom.

Together, these form a unique integer coordinate string for any crystal structure. Two crystals have identical CNF coordinates if and only if they are geometrically equivalent.

This representation has several useful properties:

- Every crystal has well-defined "neighbors" that differ by small strains or phonon displacements, creating a navigable graph over all possible crystal structures.

- Any two crystal structures with the same composition can be connected through a series of neighbor steps, enabling search for transformation pathways between phases.

- By exploring the energy landscape over this graph, one can find saddle points representing minimum energy barriers between phases.

## Citation

This implementation is based on the theoretical framework developed by David Mrdjenovich. If you use this code in your research, please cite his work:

```bibtex
@article{mrdjenovich2024crystallographic,
  title={Crystallographic map: A general lattice and basis formalism enabling
         efficient and discretized exploration of crystallographic phase space},
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

For the complete theoretical development, see David Mrdjenovich's doctoral dissertation:

```bibtex
@phdthesis{mrdjenovich2022algorithm,
  title={An Algorithm for Exploring Crystallographic Configuration Space},
  author={Mrdjenovich, David James},
  year={2022},
  school={University of California, Berkeley},
  type={PhD Dissertation}
}
```
