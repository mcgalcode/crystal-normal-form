import numpy as np
import tqdm

from cnf.unit_cell import UnitCell
from cnf.linalg.unimodular import get_unimodulars_col_max
from cnf.navigation.mep.paths import align_structure_to_reference, align_path
from ..crystal_normal_form import CrystalNormalForm
from pymatgen.core.trajectory import Trajectory
from pymatgen.core import Structure


class TrajectoryVisualizer():

    def __init__(self, filename, supercell_scaling=None):
        self.filename = filename
        self.supercell_scaling = supercell_scaling
        self._aligned_structs = None

    def save_trajectory_from_cnfs(self, cnfs: list[CrystalNormalForm]):
        structs: list[Structure] = [cnf.reconstruct() for cnf in cnfs]
        return self.save_trajectory_from_pmg_structs(structs)

    def save_trajectory_from_pmg_structs(self, structs: list[Structure]):
        if self.supercell_scaling is not None:
            structs = [s.make_supercell(self.supercell_scaling) for s in structs]

        aligned = align_path(structs, verbose=True)
        self._aligned_structs = aligned

        t = Trajectory.from_structures(aligned, constant_lattice=False)
        t.write_Xdatcar(self.filename)
        return t

    def save_trajectory(self, unit_cells: list[UnitCell]):
        """Legacy method — converts to Structures and uses new alignment."""
        structs = [uc.to_pymatgen_structure() for uc in unit_cells]
        return self.save_trajectory_from_pmg_structs(structs)

    def save_gif(self, gif_filename, structs=None, rotation="10x,-80y",
                 radii=0.5, scale=50, fps=15):
        """Render aligned structures to an animated GIF using ASE.

        Args:
            gif_filename: Output path for the GIF.
            structs: List of pymatgen Structures (uses cached aligned structs if None).
            rotation: ASE rotation string for viewing angle.
            radii: Atom radii for rendering.
            scale: Pixels per Angstrom.
            fps: Frames per second in the GIF.
        """
        from ase import Atoms
        from ase.io import write as ase_write

        if structs is None:
            structs = self._aligned_structs
        if structs is None:
            raise ValueError("No aligned structures available. Run save_trajectory_from_* first.")

        atoms_list = []
        for s in structs:
            atoms = Atoms(
                symbols=[str(sp) for sp in s.species],
                scaled_positions=s.frac_coords,
                cell=s.lattice.matrix,
                pbc=True,
            )
            atoms_list.append(atoms)

        ase_write(
            gif_filename, atoms_list,
            rotation=rotation, radii=radii, scale=scale,
        )
        # If ase_write doesn't handle fps for GIF, try with imageio
        if gif_filename.endswith(".gif"):
            try:
                import imageio.v2 as imageio
                import tempfile, os, glob

                with tempfile.TemporaryDirectory() as tmpdir:
                    png_files = []
                    for i, atoms in enumerate(atoms_list):
                        png_path = os.path.join(tmpdir, f"frame_{i:05d}.png")
                        ase_write(png_path, atoms, rotation=rotation, radii=radii, scale=scale)
                        png_files.append(png_path)

                    images = [imageio.imread(f) for f in png_files]
                    duration = 1.0 / fps
                    imageio.mimsave(gif_filename, images, duration=duration, loop=0)
            except ImportError:
                print("imageio not available — GIF saved via ASE (may not have custom fps)")

        print(f"Saved GIF: {gif_filename} ({len(atoms_list)} frames)")
