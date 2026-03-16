import os
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans

# --- 1. Configuration ---
PLY_PATH = "/workspace/output/garden_fast_5k/point_cloud/iteration_7000_semantic/point_cloud.ply"
OUT_DIR = "/workspace/output/garden_fast_5k/point_cloud/iteration_7000_semantic/isolated_objects"
os.makedirs(OUT_DIR, exist_ok=True)

# 15 SAM Objects + 1 Background = 16 distinct mathematical colors
NUM_CLUSTERS = 16

# --- 2. Load the Point Cloud ---
print(f"Loading painted geometry from {PLY_PATH}...")
plydata = PlyData.read(PLY_PATH)
vertex_data = plydata.elements[0].data

print(f"Loaded {len(vertex_data)} 3D primitives.")

# --- 3. Extract the Color Arrays ---
# 2DGS stores base colors in the f_dc_0, f_dc_1, and f_dc_2 arrays
print("Extracting Spherical Harmonic feature dimensions...")
f_dc = np.vstack([
    vertex_data['f_dc_0'], 
    vertex_data['f_dc_1'], 
    vertex_data['f_dc_2']
]).T

# --- 4. Execute K-Means Clustering ---
print(f"Executing K-Means clustering to isolate the {NUM_CLUSTERS} semantic objects... (This uses CPU RAM)")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(f_dc)

# --- 5. Physically Split and Save ---
print("Ripping point cloud arrays and saving independent 3D models...")
for i in range(NUM_CLUSTERS):
    # Create a boolean mask for the current object
    cluster_mask = (labels == i)
    cluster_data = vertex_data[cluster_mask]
    
    point_count = len(cluster_data)
    
    # Filter out mathematically anomalous micro-clusters
    if point_count < 1000:
        print(f"  -> Skipping Cluster {i} (Noise fragment: {point_count} points)")
        continue
        
    out_file = os.path.join(OUT_DIR, f"object_{i}_points_{point_count}.ply")
    
    # Construct a new standalone PLY element and write it to disk
    el = PlyElement.describe(cluster_data, 'vertex')
    PlyData([el]).write(out_file)
    print(f"  -> Saved Cluster {i} : {point_count} points to {out_file}")

print(f"\nSuccess! All independent 3D models are ready in {OUT_DIR}")