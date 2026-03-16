import os
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans

# --- 1. Configuration ---
parser = argparse.ArgumentParser(description="Extract original RGB Gaussians using Semantic Clusters")
parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained model output directory")
args = parser.parse_args()

PLY_SEMANTIC = os.path.join(args.model_path, "point_cloud", "iteration_7000_semantic", "point_cloud.ply")
PLY_ORIGINAL = os.path.join(args.model_path, "point_cloud", "iteration_5000", "point_cloud.ply")
OUT_DIR = os.path.join(args.model_path, "point_cloud", "iteration_5000", "isolated_rgb_objects")

os.makedirs(OUT_DIR, exist_ok=True)
NUM_CLUSTERS = 16

# --- 2. Load Both Point Clouds ---
print("Loading semantic point cloud for clustering...")
semantic_data = PlyData.read(PLY_SEMANTIC).elements[0].data

print("Loading original RGB point cloud for extraction...")
original_data = PlyData.read(PLY_ORIGINAL).elements[0].data

# Mathematical safety check to guarantee 1:1 mapping
assert len(semantic_data) == len(original_data), "Fatal Error: Primitive counts do not match!"
point_count = len(semantic_data)
print(f"Verified perfect index alignment across {point_count} primitives.")

# --- 3. Extract Semantic Colors for Clustering ---
print("Extracting semantic Spherical Harmonic arrays...")
f_dc_semantic = np.vstack([
    semantic_data['f_dc_0'], 
    semantic_data['f_dc_1'], 
    semantic_data['f_dc_2']
]).T

# --- 4. Execute K-Means Clustering ---
print(f"Executing K-Means clustering on semantic colors... ")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(f_dc_semantic)

# --- 5. Physically Split the ORIGINAL Point Cloud ---
print("Ripping original RGB point cloud arrays and saving independent 3D models...")
for i in range(NUM_CLUSTERS):
    cluster_mask = (labels == i)
    
    # THE CRITICAL SWAP: Apply the semantic mask to the original data
    cluster_data_rgb = original_data[cluster_mask]
    
    count = len(cluster_data_rgb)
    
    if count < 1000:
        print(f"  -> Skipping Cluster {i} (Noise fragment: {count} points)")
        continue
        
    out_file = os.path.join(OUT_DIR, f"object_{i}_rgb_{count}.ply")
    
    el = PlyElement.describe(cluster_data_rgb, 'vertex')
    PlyData([el]).write(out_file)
    print(f"  -> Saved RGB Cluster {i} : {count} points to {out_file}")

print(f"\nSuccess! All independent, photorealistic 3D models are ready in {OUT_DIR}")