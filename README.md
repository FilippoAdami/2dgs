# 2DGS-AMD: VRAM-Optimized Open-Vocabulary 3D Segmentation

This repository implements a lightweight, hardware-aware pipeline for extracting fully interactive, open-vocabulary 3D objects from 2D Gaussian Splatting (2DGS) models.

Inspired by the mathematical principles of *Segment then Splat*, this architecture pivots away from heavy C++ rasterizer modifications. Instead, it utilizes a "Semantic Geometry Freeze" and K-Means array clustering to achieve true 3D segmentation within a strict 16GB VRAM budget on AMD ROCm infrastructure.

## Architectural Contributions & Engineering Choices

1.  **Consumer VRAM SAM 2 Tracking:** Extracting exhaustive 2D masks typically causes Out-of-Memory (OOM) failures on consumer GPUs. This pipeline throttles the `SAM2AutomaticMaskGenerator` (`points_per_batch=16`) and enforces `bfloat16` mixed precision, successfully executing Meta's `sam2_hiera_large` model within a 16GB VRAM ceiling.
2.  **The Symlink Sandbox:** Bypasses a rigid integer-casting bug within the native SAM 2 Video Predictor API by dynamically building integer-named symlinks (`00000.jpg`) mapped back to original dataset filenames (e.g., `DSC07956.JPG`), preserving COLMAP camera parameters.
3.  **The Semantic Geometry Freeze:** Directly injecting semantic channels into the 2DGS C++ backend breaks native HIP compilation on AMD GPUs. This pipeline bypasses the rasterizer barrier. It loads a fully trained structural model, executes `requires_grad_(False)` on spatial coordinates ($X, Y, Z$, Scale, Rotation, Opacity), drops Spherical Harmonics to degree 0, and mathematically forces the optimizer to "paint" the Gaussians using SAM 2 masks using a pure L1 loss.
4.  **Dual-Load K-Means Extraction:** PyTorch gradient descent causes mathematical drift in color representation. To extract the objects, a CPU-bound K-Means clustering algorithm maps the exact semantic centroids, identifies the 3D primitives, and applies those spatial indices to the *original* RGB point cloud. This physically splits the scene into isolated, photorealistic `.ply` objects without geometric degradation.
5.  **KD-Tree Mesh Projection:** To support traditional polygonal workflows (e.g., Blender), a spatial KD-Tree algorithm projects the clustered semantic IDs onto a fused master TSDF mesh. This guarantees watertight topological boundaries while maintaining $O(N \log M)$ computational efficiency on system RAM.

## System Requirements

  * **OS:** Pop\!\_OS / Ubuntu 24.04
  * **Compute:** AMD GPU (e.g., Radeon RX 9060 XT) with 16GB+ VRAM
  * **Memory:** 64GB System RAM (required for dual-load point cloud clustering and KD-Tree arrays)
  * **Backend:** ROCm 7.x

## Setup & Installation

**1. Clone the Repository and Submodules**

```bash
git clone https://github.com/FilippoAdami/2dgs.git
cd 2dgs
git submodule update --init --recursive
```

**2. Install Core Dependencies**
Ensure your Python virtual environment is active, then install the required masking and clustering engines:

```bash
pip install opencv-python matplotlib scikit-learn scipy
pip install git+https://github.com/facebookresearch/sam2.git
```

**3. Download SAM 2 Weights**

```bash
mkdir -p /workspace/weights
cd /workspace/weights
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd /workspace/2dgs
```

## Execution Pipeline

The entire extraction process is fully parameterized. To process a new dataset (e.g., `garden`), set your environment variables and execute the pipeline sequentially.

```bash
# Define your dataset and output paths
export SCENE_DIR="/path/to/mip_nerf_360_data/garden"
export OUT_DIR="/path/to/output/garden_fast_5k"

# 1. Base Geometry Reconstruction
# Trains the fast structural framework. 
# Note: -r 4 is required to downscale high-res images to fit the 16GB VRAM budget.
# Note: 5000 iterations were chosen as baseline to have a decent enough result (PSNR>22) while keeping training time short (t<20 min)
python3 train.py -s $SCENE_DIR -m $OUT_DIR --iterations 5000 -r 4

# 2. Automatic 2D Semantic Tracking
# Drops a 16x16 mathematical grid and tracks the 15 largest geometric objects using SAM 2
# Note: 15 was chosen based on performance compromise on the garden scene.
python3 generate_auto_masks.py -s $SCENE_DIR

# 3. Semantic Geometry Freeze
# Locks the spatial graph and paints the Gaussians to match the SAM 2 masks.
python3 train_semantic.py -s $SCENE_DIR -m $OUT_DIR --sam_masks_dir $SCENE_DIR/masks_auto -r 4

# 4. Physical RGB Extraction (For Native WebGL Viewers)
# Executes K-Means clustering on the semantic model to rip the original RGB geometry into standalone objects.
python3 split_original_rgb.py -m $OUT_DIR

# 5. Batch TSDF Mesh Extraction (For Traditional 3D Software)
# Automates the native C++ TSDF backend to generate clean, independent polygon meshes for each isolated object, preventing boundary hallucination.
python3 extract_isolated_meshes.py -s $SCENE_DIR -m $OUT_DIR
```

## Visualization & Manipulation

This architecture supports two fundamentally different mathematical representations of the 3D scene, depending on the target workflow.

**1. Native Volumetric Surfels (SuperSplat)**

  * **Output Location:** `[OUT_DIR]/point_cloud/iteration_5000/isolated_rgb_objects/`
  * **Mechanism:** 2DGS outputs flat, oriented 2D surfels characterized by scale, rotation, opacity, and Spherical Harmonics. The extraction script mathematically rips these arrays into up to 16 completely isolated `.ply` files.
  * **Workflow:** Drag and drop the multiple `.ply` files simultaneously into native Gaussian web viewers like [SuperSplat](https://www.google.com/search?q=https://playcanvas.com/supersplat). Because they share a unified global coordinate system, the scene perfectly reassembles. You can independently toggle visibility, rotate, or translate specific objects without affecting surrounding geometries.

**2. Traditional Polygonal Mesh (Blender / Unity)**

  * **Output Location:** `[OUT_DIR]/isolated_meshes/`
  * **Mechanism:** Generating a single master mesh causes the TSDF algorithm to hallucinate continuous geometric "webs" between touching objects, leading to semantic color bleeding at the boundaries. Instead, Step 5 iterates through the isolated point clouds and computes a watertight TSDF mesh for each object independently.
  * **Workflow:** Import the generated `*_mesh.ply` files directly into standard 3D software like Blender (`File > Import > Stanford (.ply)`). Because they are standard polygonal meshes with native vertex colors, they render immediately without any complex node setups or material filters. You can import the specific objects you want, and they will slot perfectly into their exact physical coordinates.

<img width="877" height="701" alt="image" src="https://github.com/user-attachments/assets/89486b84-cc30-432c-820e-be7ac127cfe7" />

