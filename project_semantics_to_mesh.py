import os
import argparse
import numpy as np
import numpy.lib.recfunctions as rfn
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser(description="KD-Tree Semantic Mesh Projection")
parser.add_argument("--mesh", type=str, required=True, help="Path to the uncolored master TSDF mesh")
parser.add_argument("--semantics", type=str, required=True, help="Path to the 7000-iteration painted point cloud")
parser.add_argument("--out", type=str, required=True, help="Output path for the Blender-ready segmented mesh")
args = parser.parse_args()

print("Loading semantic point cloud into system memory...")
sem_ply = PlyData.read(args.semantics)
sem_data = sem_ply.elements[0].data

print("Extracting spatial coordinates and Spherical Harmonic colors...")
sem_xyz = np.vstack([sem_data['x'], sem_data['y'], sem_data['z']]).T
sem_fdc = np.vstack([sem_data['f_dc_0'], sem_data['f_dc_1'], sem_data['f_dc_2']]).T

# Mathematically convert Spherical Harmonics Degree 0 back to standard RGB
sem_rgb = (sem_fdc * 0.28209) + 0.5
sem_rgb = np.clip(sem_rgb, 0.0, 1.0) * 255.0
sem_rgb = sem_rgb.astype(np.uint8)

print("Building KD-Tree structure...")
kdtree = cKDTree(sem_xyz)

print("Loading master TSDF mesh...")
mesh_ply = PlyData.read(args.mesh)
mesh_vertices = mesh_ply['vertex'].data
mesh_xyz = np.vstack([mesh_vertices['x'], mesh_vertices['y'], mesh_vertices['z']]).T

print(f"Projecting semantic IDs onto {len(mesh_xyz)} polygonal vertices...")
# k=1 mathematically guarantees mapping to the single closest semantic Gaussian
distances, indices = kdtree.query(mesh_xyz, k=1)
vertex_colors = sem_rgb[indices]

print("Appending RGB data to mesh topology...")
# Safely inject the new color properties into the vertex array without destroying normals
if 'red' not in mesh_vertices.dtype.names:
    new_vertices = rfn.append_fields(
        mesh_vertices, 
        ['red', 'green', 'blue'], 
        [vertex_colors[:, 0], vertex_colors[:, 1], vertex_colors[:, 2]], 
        dtypes=['u1', 'u1', 'u1'], 
        usemask=False, 
        asrecarray=False
    )
else:
    new_vertices = mesh_vertices.copy()
    new_vertices['red'] = vertex_colors[:, 0]
    new_vertices['green'] = vertex_colors[:, 1]
    new_vertices['blue'] = vertex_colors[:, 2]

vertex_element = PlyElement.describe(new_vertices, 'vertex')
elements = [vertex_element]

# Retain the polygonal faces if they exist
if any(el.name == 'face' for el in mesh_ply.elements):
    elements.append(mesh_ply['face'])

print("Writing Blender-compatible semantic mesh to disk...")
PlyData(elements, text=False).write(args.out)

print(f"Success! Semantic mesh saved to {args.out}")
