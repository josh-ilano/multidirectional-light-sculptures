import numpy as np

from image_io import load_binary_image, save_mask
from voxel_ops import make_voxel_centers, voxel_pitch
from shadow_hull import compute_shadow_hull
from config import ShadowSource
from export_mesh import export_voxels_to_stl, export_voxels_to_glb
from carve import carve_hollow_shell_strict
from simulate import simulate_and_save
from debug_slices import save_voxel_slices
from optimize_consistency import optimize_silhouettes
from reset_output import reset_output_dirs

def print_view_metrics(name, summaries):
    print(f"\n{name}")
    for i, m in enumerate(summaries):
        print(
            f"  view {i}: "
            f"IoU={m['iou']:.4f}, "
            f"target={m['target_pixels']}, "
            f"actual={m['actual_pixels']}, "
            f"missing={m['missing_pixels']}, "
            f"extra={m['extra_pixels']}"
        )

def main():
    # world and image parameters
    world_size = 1.0
    nx, ny, nz = 96, 96, 96
    image_size = (128, 128)

    # reset the output directory for new run
    reset_output_dirs()

    # Load silhouettes as binary matrices
    img0 = load_binary_image("inputs/view0.png", size=image_size)
    img1 = load_binary_image("inputs/view1.png", size=image_size)

    print("img0 true pixels:", img0.sum(), "of", img0.size, "ratio:", img0.mean())
    print("img1 true pixels:", img1.sum(), "of", img1.size, "ratio:", img1.mean())

    save_mask(img0, "outputs/debug/view0_mask.png")
    save_mask(img1, "outputs/debug/view1_mask.png")

    sources = [
        ShadowSource(
            image=img0,
            direction=np.array([1, 0, 0], dtype=float),
            up=np.array([0, 1, 0], dtype=float),
            world_center=np.array([0, 0, 0], dtype=float),
            world_size=world_size
        ),
        ShadowSource(
            image=img1,
            direction=np.array([0, 0, 1], dtype=float),
            up=np.array([0, 1, 0], dtype=float),
            world_center=np.array([0, 0, 0], dtype=float),
            world_size=world_size
        ),
    ]

    print("\nShadow setup")
    for i, src in enumerate(sources):
        print(f"  source {i}: direction={src.direction}, up={src.up}")

    voxel_centers = make_voxel_centers(nx, ny, nz, world_size)

    # Optimize silhouettes to allow arbitrary silhouette inputs
    optimized_sources = optimize_silhouettes(
        sources,
        voxel_centers,
        iterations=6,
        alpha=0.15,
        sigma=4.0,
        sample_per_view=300,
        growth_radius=2,
        verbose=True
    )

    # Save optimized silhouettes for debug
    for i, src in enumerate(optimized_sources):
        save_mask(src.image, f"outputs/debug/view{i}_optimized_mask.png")

    # Use optimized silhouettes for hull construction
    sources = optimized_sources

    # 1) Conservative hull
    hull = compute_shadow_hull(sources, voxel_centers)
    print("\nInitial hull")
    print("  hull voxels:", int(hull.sum()))
    print("  hull occupancy ratio:", float(hull.mean()))

    hull_summaries = simulate_and_save(
        hull,
        voxel_centers,
        sources,
        out_dir="outputs/sim",
        prefix="hull"
    )
    print_view_metrics("Hull shadow simulation", hull_summaries)

    # Export raw hull mesh
    pitch = voxel_pitch(world_size, nx, ny, nz)

    try:
        mesh = export_voxels_to_stl(hull, pitch, "outputs/meshes/shadow_hull.stl")
        export_voxels_to_glb(hull, pitch, "outputs/meshes/shadow_hull.glb")
        print("\nSaved raw hull mesh:")
        print("  outputs/meshes/shadow_hull.stl")
        print("  outputs/meshes/shadow_hull.glb")
        print("  vertices:", len(mesh.vertices))
        print("  faces:", len(mesh.faces))
        print("  watertight:", mesh.is_watertight)
    except Exception as e:
        print("\nRaw hull export failed:", e)

    # 2) Material optimization by greedy carving
    carved, carve_stats = carve_hollow_shell_strict(
        hull,
        voxel_centers,
        sources,
        shell_thickness_voxels=3,
        max_passes=1,
        random_seed=0,
        protect_endcaps=True,
        cleanup_components=True,
        min_component_size=150,
        verbose=True
    )

    print("\nCarved sculpture")
    print("  voxels:", int(carved.sum()))
    print("  occupancy ratio:", float(carved.mean()))
    print("  removed:", carve_stats["removed"])
    print("  reduction ratio:", carve_stats["reduction_ratio"])

    carved_summaries = simulate_and_save(
        carved,
        voxel_centers,
        sources,
        out_dir="outputs/sim",
        prefix="carved"
    )
    print_view_metrics("Carved shadow simulation", carved_summaries)

    # print summary of optimization
    pitch = voxel_pitch(world_size, nx, ny, nz)
    voxel_volume = pitch ** 3

    hull_voxels = int(hull.sum())
    carved_voxels = int(carved.sum())

    hull_volume = hull_voxels * voxel_volume
    carved_volume = carved_voxels * voxel_volume

    print("\nOptimization summary")
    print(f"  hull voxels:   {hull_voxels}")
    print(f"  carved voxels: {carved_voxels}")
    print(f"  removed voxels:{hull_voxels - carved_voxels}")
    print(f"  saved percent: {100.0 * (1.0 - carved_voxels / hull_voxels):.2f}%")
    print(f"  hull volume:   {hull_volume:.6f}")
    print(f"  carved volume: {carved_volume:.6f}")
    print(f"  saved volume:  {hull_volume - carved_volume:.6f}")

    # save slices to confirm optimization
    save_voxel_slices(hull, "outputs/debug/slices", "hull")
    save_voxel_slices(carved, "outputs/debug/slices", "carved")

    # Export carved mesh
    try:
        mesh = export_voxels_to_stl(carved, pitch, "outputs/meshes/shadow_carved.stl")
        export_voxels_to_glb(carved, pitch, "outputs/meshes/shadow_carved.glb")
        print("\nSaved carved mesh:")
        print("  outputs/meshes/shadow_carved.stl")
        print("  outputs/meshes/shadow_carved.glb")
        print("  vertices:", len(mesh.vertices))
        print("  faces:", len(mesh.faces))
        print("  watertight:", mesh.is_watertight)
    except Exception as e:
        print("\nCarved export failed:", e)

if __name__ == "__main__":
    main()