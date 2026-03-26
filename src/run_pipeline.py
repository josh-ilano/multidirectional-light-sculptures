import argparse
import numpy as np

from image_io import load_binary_image, save_mask
from voxel_ops import make_voxel_centers, voxel_pitch
from shadow_hull import compute_shadow_hull
from shadow_source import ShadowSource, build_sources
from export_mesh import export_voxels_to_stl
from carve import carve_hollow_shell_strict
from simulate import simulate_and_save
from debug_slices import save_voxel_slices
from optimize_consistency import optimize_silhouettes
from reset_output import reset_output_dirs

# helper function to print out metrics per silhouette input view
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

# helper function for parsing command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a shadow hull from 1, 2, or 3 silhouette images."
    )

    parser.add_argument(
        "views",
        nargs="+",
        help="1 to 3 silhouette image paths"
    )

    parser.add_argument(
        "--world-size",
        type=float,
        default=1.0,
        help="Physical size of the voxel world"
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=350,
        help="Voxel resolution used for nx=ny=nz"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=350,
        help="Square size used to resize input silhouettes"
    )

    parser.add_argument(
        "--optimize-material",
        action="store_true",
        help="Enable hollow-shell carving"
    )

    args = parser.parse_args()

    if not (1 <= len(args.views) <= 3):
        parser.error("You must provide between 1 and 3 silhouette image paths.")

    return args


def main():
    # parse command line args
    args = parse_args()

    # world and image parameters
    world_size = args.world_size
    nx = ny = nz = args.grid
    image_size = (args.image_size, args.image_size)
    optimize_material = args.optimize_material

    # reset the output directory for new run
    reset_output_dirs()

    # load silhouettes from command line
    images = []
    for i, path in enumerate(args.views):
        img = load_binary_image(path, size=image_size)
        images.append(img)
        print(f"[PIPELINE] img{i} silhouette pixels:", img.sum(), "of", img.size)

    # save the masks to debug folder
    for i, img in enumerate(images):
        save_mask(img, f"outputs/debug/masks/base/view{i}_mask.png")

    # define the shadow sources, one per image
    sources = build_sources(images, world_size)

    # print the shadow sources for debug
    for i, src in enumerate(sources):
        print(f"[PIPELINE] shadow source {i}: direction={src.direction}, up={src.up}")

    # calculate the voxel centers for the given params
    voxel_centers = make_voxel_centers(nx, ny, nz, world_size)

    # optimize silhouettes to allow arbitrary silhouette inputs
    optimized_sources = optimize_silhouettes(
        sources,
        voxel_centers,
        iterations=4,
        alpha=0.10,
        sigma=3.0,
        sample_per_view=80,
        growth_radius=1,
        max_ray_samples=8,
        save_debug_masks=False,
        verbose=True,
        num_workers=4,
    )

    # save optimized silhouettes for debug
    for i, src in enumerate(optimized_sources):
        save_mask(src.image, f"outputs/debug/masks/opt/view{i}_optimized_mask.png")

    # use optimized silhouettes for hull construction
    sources = optimized_sources

    # 1) Conservative hull
    hull = compute_shadow_hull(
        optimized_sources,
        voxel_centers,
        enforce_connectivity=True,
        min_component_size=3,
        verbose=True,
    )
    print("[PIPELINE] initial hull voxel count:", int(hull.sum()))
    print("[PIPELINE] initial hull occupancy ratio:", float(hull.mean()))

    # simulate the shadows and save to debug
    hull_summaries = simulate_and_save(
        hull,
        voxel_centers,
        sources,
        out_dir="outputs/sim",
        prefix="hull"
    )
    print_view_metrics("Hull shadow simulation", hull_summaries)

    # export raw hull mesh as stl
    pitch = voxel_pitch(world_size, nx, ny, nz)

    try:
        export_voxels_to_stl(hull, pitch, "outputs/meshes/shadow_hull.stl")
        print("[PIPELINE] saved raw hull mesh: outputs/meshes/shadow_hull.stl")
    except Exception as e:
        print("[ERROR] Raw hull export failed:", e)

    # 2) Material optimization only if flag is true
    if optimize_material:
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

        print("[PIPELINE] carved sculpture voxels:", int(carved.sum()))
        print("[PIPELINE] carved sculpture occupancy ratio:", float(carved.mean()))
        print("[PIPELINE] carved sculpture voxels removed:", carve_stats["removed"])
        print("[PIPELINE] carved sculpture reduction ratio:", carve_stats["reduction_ratio"])

        # simulate the shadows of the carved sculpture and save
        carved_summaries = simulate_and_save(
            carved,
            voxel_centers,
            sources,
            out_dir="outputs/sim",
            prefix="carved"
        )
        print_view_metrics("Carved shadow simulation", carved_summaries)

        # save slices to confirm optimization
        save_voxel_slices(hull, "outputs/debug/slices", "hull")
        save_voxel_slices(carved, "outputs/debug/slices", "carved")

if __name__ == "__main__":
    main()