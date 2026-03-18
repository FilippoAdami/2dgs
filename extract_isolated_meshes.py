import os
import shutil
import subprocess
import argparse
from glob import glob

def batch_extract_meshes():
    parser = argparse.ArgumentParser(description="Batch TSDF Mesh Extraction for Isolated Objects")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to the original COLMAP dataset")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the 5K trained model directory")
    args = parser.parse_args()

    ISOLATED_DIR = os.path.join(args.model_path, "point_cloud", "iteration_5000", "isolated_rgb_objects")
    MESH_OUT_DIR = os.path.join(args.model_path, "isolated_meshes")
    os.makedirs(MESH_OUT_DIR, exist_ok=True)

    ply_files = glob(os.path.join(ISOLATED_DIR, "*.ply"))
    if not ply_files:
        print(f"Error: No isolated .ply files found in {ISOLATED_DIR}")
        return

    print(f"Found {len(ply_files)} isolated point clouds. Commencing batch TSDF extraction...")

    for ply_path in ply_files:
        obj_name = os.path.basename(ply_path).replace(".ply", "")
        print(f"\n--- Processing {obj_name} ---")

        # 1. Build the Mock Directory Architecture
        mock_dir = os.path.join(args.model_path, f"mock_{obj_name}")
        mock_pc_dir = os.path.join(mock_dir, "point_cloud", "iteration_5000")
        os.makedirs(mock_pc_dir, exist_ok=True)

        # 2. Copy the critical configuration files so the C++ backend doesn't crash
        try:
            shutil.copy(os.path.join(args.model_path, "cameras.json"), mock_dir)
            if os.path.exists(os.path.join(args.model_path, "cfg_args")):
                shutil.copy(os.path.join(args.model_path, "cfg_args"), mock_dir)
        except Exception as e:
            print(f"Failed to copy config files: {e}")
            continue

        # 3. Symlink the isolated object as the "master" point cloud
        os.symlink(ply_path, os.path.join(mock_pc_dir, "point_cloud.ply"))

        # 4. Trigger the native ROCm TSDF extraction
        cmd = [
            "python3", "render.py",
            "-m", mock_dir,
            "-s", args.source_path,
            "-r", "4",
            "--skip_train", "--skip_test"
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            # 5. Harvest the watertight topology
            generated_mesh = os.path.join(mock_dir, "train", "ours_5000", "fuse_post.ply")
            if os.path.exists(generated_mesh):
                dest_mesh = os.path.join(MESH_OUT_DIR, f"{obj_name}_mesh.ply")
                shutil.move(generated_mesh, dest_mesh)
                print(f"Success! Mesh saved to {dest_mesh}")
            else:
                print("TSDF fusion failed to generate a manifold for this object (likely too sparse).")
                
        except subprocess.CalledProcessError:
            print("C++ backend crashed during TSDF integration for this object.")
            
        finally:
            # 6. Destroy the mock directory to preserve storage
            shutil.rmtree(mock_dir)

    print(f"\nBatch extraction complete. All valid meshes are located in: {MESH_OUT_DIR}")

if __name__ == "__main__":
    batch_extract_meshes()
    