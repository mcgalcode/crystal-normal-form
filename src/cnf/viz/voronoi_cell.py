#!/usr/bin/env python3
"""
Wigner-Seitz Cell Visualizer

This script creates a 3D visualization of the Wigner-Seitz cell for a given lattice
using pymatgen's get_wigner_seitz_cell method and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymatgen.core.lattice import Lattice

default_colors = ['purple', 'cyan', 'brown', 'magenta', 'pink', 'lime', 'orange', 'red', 'green', 'blue'] + ['purple', 'cyan', 'brown', 'magenta', 'pink', 'lime', 'orange', 'red', 'green', 'blue']


def plot_lattice_vectors_with_planes(vectors, output_file="lattice_vectors_planes.png", 
                                    figsize=(10, 10), show_labels=True, 
                                    colors=None, linewidth=3, vector_labels=None,
                                    plane_size=8.0, plane_alpha=0.3, uniform_plane_color=None,
                                    voronoi_crop=False):
    """
    Plot lattice generating vectors with perpendicular bisecting planes.
    
    For each vector, this function plots a plane that:
    - Passes through the midpoint of the vector
    - Is perpendicular to the vector
    
    Parameters:
    -----------
    vectors : list or np.ndarray
        List of lattice vectors. Can be 3 vectors (typical basis) or 4 (superbasis).
        Each vector should be a 3D array-like object [x, y, z].
    output_file : str
        Output filename for the PNG image
    figsize : tuple
        Figure size (width, height) in inches
    show_labels : bool
        Whether to show vector labels (v0, v1, v2, v3)
    colors : list
        List of colors for each vector. If None, uses default colors.
    linewidth : float
        Width of the vector arrows
    vector_labels : list of str, optional
        Custom labels for each vector. If None, uses v0, v1, v2, etc.
    plane_size : float
        Size of the rectangular plane sections (width/height). Default: 8.0
    plane_alpha : float
        Transparency of the planes (0=transparent, 1=opaque)
    uniform_plane_color : str, optional
        If provided, all planes will use this color instead of matching vector colors.
        Example: 'gray', 'lightblue', 'silver', etc.
    voronoi_crop : bool
        If True, crop each plane to only show the portion that forms the Voronoi cell
        (envelope around the origin). If False, show full rectangular planes.
    """
    # Convert to numpy array if not already
    vectors = np.array(vectors)
    n_vectors = len(vectors)
    
    # Default labels if not provided
    if vector_labels is None:
        vector_labels = [f'v_{{{i}}}' for i in range(n_vectors)]
    else:
        # Format labels for LaTeX if they don't already contain braces
        vector_labels = [label if '{' in label else f'{label}' for label in vector_labels]
    
    # Default colors (reversed so last vectors get consistent colors)
    if colors is None:
        # Assign colors from the end of the list backward
        colors = default_colors[len(default_colors) - n_vectors:]
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Origin point
    origin = np.array([0, 0, 0])
    
    # If using Voronoi cropping, we need to compute the intersections
    if voronoi_crop:
        from scipy.spatial import HalfspaceIntersection, ConvexHull
        
        # Create half-spaces for each bisecting plane
        # A half-space is defined as: a·x + b·y + c·z <= d
        # Represented as [a, b, c, -d]
        # For our bisecting planes, we want the half-space containing the origin
        halfspaces = []
        for vec in vectors:
            normal = vec / np.linalg.norm(vec)
            midpoint = vec / 2.0
            # Distance from origin to plane along normal
            d = np.dot(normal, midpoint)
            # Half-space: normal · x <= d
            halfspaces.append(np.append(normal, -d))
        
        halfspaces = np.array(halfspaces)
        
        # The interior point should be the origin (or very close to it)
        # Make sure it's strictly inside all halfspaces
        interior_point = np.array([0.0, 0.0, 0.0])
        
        try:
            hs = HalfspaceIntersection(halfspaces, interior_point)
            voronoi_vertices = hs.intersections
            
            # Check if the region is bounded by verifying vertices aren't too far
            max_vertex_dist = np.max(np.linalg.norm(voronoi_vertices, axis=1))
            max_vec_dist = np.max(np.linalg.norm(vectors, axis=1))
            
            # If vertices are much farther than the vectors, the region is likely unbounded
            if max_vertex_dist > max_vec_dist * 100:
                raise ValueError(
                    "The bisecting planes do not form a bounded envelope around the origin. "
                    "This typically happens when you don't have enough vectors in different directions. "
                    "For a proper Voronoi cell, you need vectors that surround the origin."
                )
            
        except ValueError as e:
            # Re-raise ValueError with our message
            raise e
        except Exception as e:
            # For other errors, raise a more informative error
            raise ValueError(
                f"Failed to compute Voronoi cell from bisecting planes: {e}. "
                "The planes may not form a bounded envelope around the origin."
            )
    
    # Plot each vector and its perpendicular bisecting plane
    for i, vec in enumerate(vectors):
        # Plot the vector
        ax.quiver(origin[0], origin[1], origin[2], 
                 vec[0], vec[1], vec[2],
                 color=colors[i],
                 arrow_length_ratio=0.15, 
                 linewidth=linewidth)
        
        # Add text annotation offset from the arrow tip if labels are requested
        if show_labels:
            # Calculate offset to move label away from arrow tip
            vec_normalized = vec / np.linalg.norm(vec)
            offset = vec_normalized * 0.2 * np.linalg.norm(vec)  # 20% of vector length
            label_pos = vec + offset

            if len(vector_labels[i]) > 0:
                lab = f'${vector_labels[i]}$'
            else:
                lab = ""
            
            ax.text(label_pos[0], label_pos[1], label_pos[2], 
                   lab, 
                   fontsize=18,
                   color=colors[i],
                   fontweight='bold',
                   ha='center', 
                   va='center')
        
        # Create perpendicular bisecting plane
        # Midpoint of the vector
        midpoint = vec / 2.0
        
        # Normal vector (the vector itself, normalized)
        normal = vec / np.linalg.norm(vec)
        
        # Determine plane color
        plane_color = uniform_plane_color if uniform_plane_color is not None else colors[i]
        
        if voronoi_crop:
            # Find the facet vertices that lie on this plane
            # A point is on the plane if: normal · (point - midpoint) ≈ 0
            tolerance = 1e-6
            distances = np.abs(np.dot(voronoi_vertices - midpoint, normal))
            on_plane_mask = distances < tolerance
            
            if np.sum(on_plane_mask) >= 3:  # Need at least 3 points for a facet
                facet_vertices = voronoi_vertices[on_plane_mask]
                
                # Order vertices by angle around the centroid for proper polygon plotting
                centroid = np.mean(facet_vertices, axis=0)
                
                # Project vertices onto the plane's 2D coordinate system
                # Create two perpendicular vectors in the plane
                if abs(normal[0]) < 0.9:
                    temp = np.array([1, 0, 0])
                else:
                    temp = np.array([0, 1, 0])
                
                u = np.cross(normal, temp)
                u = u / np.linalg.norm(u)
                v = np.cross(normal, u)
                
                # Project vertices to 2D
                vertices_2d = np.array([
                    [np.dot(vertex - centroid, u), np.dot(vertex - centroid, v)]
                    for vertex in facet_vertices
                ])
                
                # Sort by angle
                angles = np.arctan2(vertices_2d[:, 1], vertices_2d[:, 0])
                sorted_indices = np.argsort(angles)
                ordered_vertices = facet_vertices[sorted_indices]
                
                # Plot the cropped facet
                plane = Poly3DCollection([ordered_vertices], alpha=plane_alpha, 
                                        facecolors=plane_color, 
                                        edgecolors=plane_color, 
                                        linewidths=1.5)
                ax.add_collection3d(plane)
        else:
            # Create full rectangular plane (original behavior)
            # Create two perpendicular vectors in the plane
            if abs(normal[0]) < 0.9:
                temp = np.array([1, 0, 0])
            else:
                temp = np.array([0, 1, 0])
            
            # Use cross product to get perpendicular vectors
            perp1 = np.cross(normal, temp)
            perp1 = perp1 / np.linalg.norm(perp1) * plane_size
            
            perp2 = np.cross(normal, perp1)
            perp2 = perp2 / np.linalg.norm(perp2) * plane_size
            
            # Create rectangular plane
            plane_corners = np.array([
                midpoint - perp1/2 - perp2/2,
                midpoint + perp1/2 - perp2/2,
                midpoint + perp1/2 + perp2/2,
                midpoint - perp1/2 + perp2/2
            ])
            
            # Plot the plane
            plane = Poly3DCollection([plane_corners], alpha=plane_alpha, 
                                    facecolors=plane_color, 
                                    edgecolors=plane_color, 
                                    linewidths=1)
            ax.add_collection3d(plane)
    
    # Hide all axes, labels, and decorations
    ax.set_axis_off()
    
    # Set equal aspect ratio for all axes
    # Calculate bounds based on the vectors and planes
    all_coords = np.vstack([origin, vectors])
    max_range = np.array([all_coords[:, 0].max() - all_coords[:, 0].min(),
                         all_coords[:, 1].max() - all_coords[:, 1].min(),
                         all_coords[:, 2].max() - all_coords[:, 2].min()]).max() / 2.0
    
    # Add some padding
    max_range *= 1.2
    
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Lattice vectors with planes saved to {output_file}")
    
    return fig, ax


def plot_lattice_vectors(vectors, output_file="lattice_vectors.png", 
                        figsize=(10, 10), show_labels=True, 
                        colors=None, linewidth=3, vector_labels=None):
    """
    Plot lattice generating vectors.
    
    Parameters:
    -----------
    vectors : list or np.ndarray
        List of lattice vectors. Can be 3 vectors (typical basis) or 4 (superbasis).
        Each vector should be a 3D array-like object [x, y, z].
    output_file : str
        Output filename for the PNG image
    figsize : tuple
        Figure size (width, height) in inches
    show_labels : bool
        Whether to show vector labels (v0, v1, v2, v3)
    colors : list
        List of colors for each vector. If None, uses default colors.
    linewidth : float
        Width of the vector arrows
    vector_labels : list of str, optional
        Custom labels for each vector. If None, uses v0, v1, v2, etc.
        Example: ['v1', 'v2', 'v3'] or ['a', 'b', 'c']
    """
    # Convert to numpy array if not already
    vectors = np.array(vectors)
    n_vectors = len(vectors)
    
    # Default labels if not provided
    if vector_labels is None:
        vector_labels = [f'v_{{{i}}}' for i in range(n_vectors)]
    else:
        # Format labels for LaTeX if they don't already contain braces
        vector_labels = [label if '{' in label else f'{label}' for label in vector_labels]
    
    # Default colors (reversed so last vectors get consistent colors)
    if colors is None:
        # Assign colors from the end of the list backward
        colors = default_colors[len(default_colors) - n_vectors:]
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Origin point
    origin = np.array([0, 0, 0])
    
    # Plot each vector
    for i, vec in enumerate(vectors):
        ax.quiver(origin[0], origin[1], origin[2], 
                 vec[0], vec[1], vec[2],
                 color=colors[i],
                 arrow_length_ratio=0.15, 
                 linewidth=linewidth)
        
        # Add text annotation offset from the arrow tip if labels are requested
        if show_labels:
            # Calculate offset to move label away from arrow tip
            vec_normalized = vec / np.linalg.norm(vec)
            offset = vec_normalized * 0.2 * np.linalg.norm(vec)  # 20% of vector length
            label_pos = vec + offset
            
            ax.text(label_pos[0], label_pos[1], label_pos[2], 
                   f'${vector_labels[i]}$', 
                   fontsize=18,  # Increased from 14
                   color=colors[i],
                   fontweight='bold',
                   ha='center', 
                   va='center')  # Changed from 'bottom' to 'center'
    
    # Hide all axes
    ax.set_axis_off()
    
    # Set equal aspect ratio and appropriate bounds
    all_points = np.vstack([np.array([0, 0, 0]), vectors])
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                         all_points[:, 1].max() - all_points[:, 1].min(),
                         all_points[:, 2].max() - all_points[:, 2].min()]).max() / 1.5
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Lattice vectors saved to {output_file}")
    
    return fig, ax


def plot_wigner_seitz_cell(lattice, output_file="wigner_seitz_cell.png", 
                           figsize=(10, 10), alpha=0.3, edge_color='black',
                           face_color='cyan', show_lattice_vectors=False):
    """
    Plot the Wigner-Seitz cell for a given lattice.
    
    Parameters:
    -----------
    lattice : pymatgen.core.lattice.Lattice
        The lattice object for which to compute and plot the Wigner-Seitz cell
    output_file : str
        Output filename for the PNG image
    figsize : tuple
        Figure size (width, height) in inches
    alpha : float
        Transparency of the cell facets (0=transparent, 1=opaque)
    edge_color : str
        Color of the edges
    face_color : str
        Color of the facets
    show_lattice_vectors : bool
        Whether to show the lattice vectors (default: False)
    """
    # Get the Wigner-Seitz cell facets
    ws_cell = lattice.get_wigner_seitz_cell()
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each facet
    for facet in ws_cell:
        # Convert facet to numpy array
        facet_array = np.array(facet)
        
        # Create a Poly3DCollection for this facet
        poly = Poly3DCollection([facet_array], alpha=alpha, 
                               facecolors=face_color, 
                               edgecolors=edge_color, 
                               linewidths=1.5)
        ax.add_collection3d(poly)
    
    # Plot the lattice vectors from origin (optional)
    if show_lattice_vectors:
        origin = np.array([0, 0, 0])
        matrix = lattice.matrix
        
        for i, vec in enumerate(matrix):
            ax.quiver(origin[0], origin[1], origin[2], 
                     vec[0], vec[1], vec[2],
                     color=['red', 'green', 'blue'][i],
                     arrow_length_ratio=0.1, linewidth=2,
                     label=f'$\\mathbf{{a}}_{i+1}$')
        
        # Add legend only if showing vectors
        ax.legend(loc='upper right')
    
    # Hide all axes, labels, and decorations
    ax.set_axis_off()
    
    # Set equal aspect ratio for all axes
    # Get the limits of all facets to set proper bounds
    all_points = np.vstack([np.array(facet) for facet in ws_cell])
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                         all_points[:, 1].max() - all_points[:, 1].min(),
                         all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Wigner-Seitz cell saved to {output_file}")
    
    return fig, ax


def main():
    """
    Example usage with different lattice types.
    """
    
    # Example 1: Simple Cubic
    print("Creating Wigner-Seitz cell for Simple Cubic lattice...")
    cubic_lattice = Lattice.cubic(5.0)
    plot_wigner_seitz_cell(cubic_lattice, 
                          output_file="ws_cubic.png",
                          face_color='lightblue')
    
    # Example 2: FCC (Face-Centered Cubic)
    print("Creating Wigner-Seitz cell for FCC lattice...")
    fcc_lattice = Lattice.from_parameters(a=5.0, b=5.0, c=5.0, 
                                         alpha=60, beta=60, gamma=60)
    plot_wigner_seitz_cell(fcc_lattice, 
                          output_file="ws_fcc.png",
                          face_color='lightgreen')
    
    # Example 3: BCC (Body-Centered Cubic)
    print("Creating Wigner-Seitz cell for BCC lattice...")
    # BCC lattice vectors
    a = 5.0
    bcc_matrix = np.array([
        [-a/2, a/2, a/2],
        [a/2, -a/2, a/2],
        [a/2, a/2, -a/2]
    ])
    bcc_lattice = Lattice(bcc_matrix)
    plot_wigner_seitz_cell(bcc_lattice, 
                          output_file="ws_bcc.png",
                          face_color='lightyellow')
    
    # Example 4: Hexagonal
    print("Creating Wigner-Seitz cell for Hexagonal lattice...")
    hex_lattice = Lattice.hexagonal(a=5.0, c=8.0)
    plot_wigner_seitz_cell(hex_lattice, 
                          output_file="ws_hexagonal.png",
                          face_color='lightcoral')
    
    print("\nAll Wigner-Seitz cells have been generated!")
    
    # Example 5: Plot 3 lattice vectors with labels v1, v2, v3
    print("\nCreating visualization of 3 lattice vectors (v1, v2, v3)...")
    vectors_3 = [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ]
    plot_lattice_vectors(vectors_3, 
                        output_file="vectors_3_basis.png",
                        vector_labels=['v_{1}', 'v_{2}', 'v_{3}'])
    
    # Example 6: Plot 4 lattice vectors (superbasis) with v0 prepended
    print("Creating visualization of 4 lattice vectors (v0, v1, v2, v3)...")
    vectors_4 = [
        [2.0, 2.0, 2.0],  # v0
        [5.0, 0.0, 0.0],  # v1 (same as first vector in 3-basis)
        [0.0, 5.0, 0.0],  # v2 (same as second vector in 3-basis)
        [0.0, 0.0, 5.0]   # v3 (same as third vector in 3-basis)
    ]
    plot_lattice_vectors(vectors_4, 
                        output_file="vectors_4_superbasis.png",
                        vector_labels=['v_{0}', 'v_{1}', 'v_{2}', 'v_{3}'])
    
    # Example 7: Plot BCC lattice vectors with default labels
    print("Creating visualization of BCC lattice vectors...")
    bcc_vectors = [
        [-2.5, 2.5, 2.5],
        [2.5, -2.5, 2.5],
        [2.5, 2.5, -2.5]
    ]
    plot_lattice_vectors(bcc_vectors, output_file="vectors_bcc.png")
    
    print("\nAll visualizations have been generated!")
    
    # Example 8: Plot 3 lattice vectors with perpendicular bisecting planes (colored)
    print("\nCreating visualization of 3 lattice vectors with colored bisecting planes...")
    vectors_3_planes = [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ]
    plot_lattice_vectors_with_planes(vectors_3_planes, 
                                    output_file="vectors_3_with_planes_colored.png",
                                    vector_labels=['v_{1}', 'v_{2}', 'v_{3}'],
                                    plane_size=10.0)
    
    # Example 9: Plot 3 lattice vectors with uniform gray planes
    print("Creating visualization of 3 lattice vectors with uniform gray planes...")
    plot_lattice_vectors_with_planes(vectors_3_planes, 
                                    output_file="vectors_3_with_planes_uniform.png",
                                    vector_labels=['v_{1}', 'v_{2}', 'v_{3}'],
                                    uniform_plane_color='silver',
                                    plane_alpha=0.4,
                                    plane_size=10.0)
    
    # Note: 3 orthogonal vectors don't form a bounded envelope, so voronoi_crop would fail
    
    # Example 10: Plot 4 lattice vectors with perpendicular bisecting planes (colored)
    print("Creating visualization of 4 lattice vectors with colored bisecting planes...")
    vectors_4_planes = [
        [2.0, 2.0, 2.0],  # v0
        [5.0, 0.0, 0.0],  # v1
        [0.0, 5.0, 0.0],  # v2
        [0.0, 0.0, 5.0]   # v3
    ]
    plot_lattice_vectors_with_planes(vectors_4_planes, 
                                    output_file="vectors_4_with_planes_colored.png",
                                    vector_labels=['v_{0}', 'v_{1}', 'v_{2}', 'v_{3}'],
                                    plane_size=8.0)
    
    # Example 11: Plot 4 lattice vectors with uniform light blue planes
    print("Creating visualization of 4 lattice vectors with uniform planes...")
    plot_lattice_vectors_with_planes(vectors_4_planes, 
                                    output_file="vectors_4_with_planes_uniform.png",
                                    vector_labels=['v_{0}', 'v_{1}', 'v_{2}', 'v_{3}'],
                                    uniform_plane_color='lightblue',
                                    plane_alpha=0.35,
                                    plane_size=8.0)
    
    # Example 12: Plot 4 lattice vectors with Voronoi-cropped planes
    # This works because 4 vectors form a bounded envelope around the origin
    print("Creating visualization of 4 lattice vectors with Voronoi-cropped planes...")
    plot_lattice_vectors_with_planes(vectors_4_planes, 
                                    output_file="vectors_4_with_planes_voronoi.png",
                                    vector_labels=['v_{0}', 'v_{1}', 'v_{2}', 'v_{3}'],
                                    uniform_plane_color='silver',
                                    voronoi_crop=True,
                                    plane_alpha=0.5)
    
    # Example 13: Another 4-vector example with different geometry
    print("Creating visualization of another 4-vector superbasis with Voronoi facets...")
    vectors_4_alt = [
        [3.0, 3.0, 0.0],
        [3.0, -3.0, 0.0],
        [0.0, 3.0, 3.0],
        [0.0, 3.0, -3.0]
    ]
    plot_lattice_vectors_with_planes(vectors_4_alt, 
                                    output_file="vectors_4_alt_with_voronoi.png",
                                    voronoi_crop=True,
                                    uniform_plane_color='lightcyan',
                                    plane_alpha=0.5)
    
    print("\nAll visualizations including planes have been generated!")
    print("Note: voronoi_crop only works when vectors form a bounded envelope around the origin")
    
    # Show the plots (optional - comment out if running non-interactively)
    # plt.show()


if __name__ == "__main__":
    main()