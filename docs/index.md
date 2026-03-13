# Crystal Normal Form

Welcome to the documentation for **Crystal Normal Form (CNF)**, a Python implementation of the crystallographic mapping algorithm described in [Mrdjenovich & Persson (2024)](https://doi.org/10.1103/PhysRevMaterials.8.033401).

## What is CNF?

Crystal structures are traditionally described using three lattice vectors combined with fractional atomic coordinates. While intuitive, this representation is highly redundant: infinitely many choices of unit cell, origin, and atom labeling can describe the same physical crystal.

**Crystal Normal Form** provides a unique, canonical integer representation for any 3D crystal structure, eliminating all representational ambiguities. Two crystals have identical CNF coordinates if and only if they are geometrically equivalent.

This representation creates an implicit graph structure where every crystal has well-defined "neighbors" that differ by small strains or atomic displacements—enabling systematic exploration of crystallographic phase space and search for transformation pathways between phases.

## Getting Started

```bash
# Install with uv
uv pip install -e .

# Verify installation
cnf --help
```

## Learn More

- [Mrdjenovich & Persson (2024)](https://doi.org/10.1103/PhysRevMaterials.8.033401) - The original paper describing the algorithm
- [Mrdjenovich (2022)](https://escholarship.org/uc/item/4r2181tt) - PhD dissertation with complete theoretical development
