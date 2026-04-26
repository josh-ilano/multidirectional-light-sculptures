import argparse
import numpy as np

from image_io import load_binary_image, save_mask
from voxel_ops import make_voxel_centers, voxel_pitch
from shadow_hull import compute_shadow_hull
from config import ShadowSource
from export_mesh import export_voxels_to_stl
from carve import carve_hollow_shell_strict
from simulate import simulate_and_save
from debug_slices import save_voxel_slices
from optimize_consistency import optimize_silhouettes
from reset_output import reset_output_dirs
from postprocess_prune import fast_projection_prune
import time

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


def build_sources(images, world_size):
    """
    Assign a default orthographic direction/up pair for up to 3 views.
    View 0: +X
    View 1: +Z
    View 2: +Y
    """
    view_configs = [
        {
            "direction": np.array([1, 0, 0], dtype=float),
            "up": np.array([0, 1, 0], dtype=float),
        },
        {
            "direction": np.array([0, 0, 1], dtype=float),
            "up": np.array([0, 1, 0], dtype=float),
        },
        {
            "direction": np.array([0, 1, 0], dtype=float),
            "up": np.array([0, 0, 1], dtype=float),
        },
    ]

    sources = []
    for i, img in enumerate(images):
        cfg = view_configs[i]
        sources.append(
            ShadowSource(
                image=img,
                direction=cfg["direction"],
                up=cfg["up"],
                world_center=np.array([0, 0, 0], dtype=float),
                world_size=world_size,
            )
        )

    return sources


def run_pipeline(
    view_paths,
    world_size=1.0,
    grid=350,
    image_size_value=350,
    optimize_material=False,
    log=print,
):
    t0 = time.time()

    reset_output_dirs()

    nx = ny = nz = grid
    image_size = (image_size_value, image_size_value)

    images = []
    for i, path in enumerate(view_paths):
        img = load_binary_image(path, size=image_size)
        images.append(img)
        log(f"[PIPELINE] img{i} silhouette pixels: {img.sum()} of {img.size}")

    for i, img in enumerate(images):
        save_mask(img, f"outputs/debug/masks/base/view{i}_mask.png")

    sources = build_sources(images, world_size)
    original_sources = sources

    for i, src in enumerate(sources):
        log(f"[PIPELINE] shadow source {i}: direction={src.direction}, up={src.up}")

    voxel_centers = make_voxel_centers(nx, ny, nz, world_size)

    log("[PIPELINE] optimizing silhouettes...")
    optimized_sources = optimize_silhouettes(
        sources,
        voxel_centers,
        iterations=6,
        alpha=0.15,
        sigma=4.0,
        sample_per_view=300,
        growth_radius=2,
        verbose=True,
    )

    for i, src in enumerate(optimized_sources):
        save_mask(src.image, f"outputs/debug/masks/opt/view{i}_optimized_mask.png")

    sources = optimized_sources

    log("[PIPELINE] computing shadow hull...")
    hull = compute_shadow_hull(sources, voxel_centers)

    log(f"[PIPELINE] initial hull voxel count: {int(hull.sum())}")
    log(f"[PIPELINE] initial hull occupancy ratio: {float(hull.mean())}")

    log("[PIPELINE] pruning hull...")
    hull, post_stats = fast_projection_prune(
        hull,
        voxel_centers,
        optimized_sources=optimized_sources,
        original_sources=original_sources,
        max_passes=6,
        max_remove_fraction_per_pass=0.15,
        min_face_neighbors=2,
        redundancy_threshold=2.0,
        cleanup_each_pass=True,
        verbose=True,
    )

    log(f"[PIPELINE] fast prune bulk removed: {post_stats['bulk_removed']}")
    log(f"[PIPELINE] fast prune cc removed: {post_stats['cc_removed']}")
    log(f"[PIPELINE] fast prune final hull voxel count: {post_stats['final_voxels']}")

    log("[PIPELINE] simulating shadows...")
    hull_summaries = simulate_and_save(
        hull,
        voxel_centers,
        sources,
        out_dir="outputs/sim",
        prefix="hull",
    )

    pitch = voxel_pitch(world_size, nx, ny, nz)

    hull_stl_path = "outputs/meshes/shadow_hull.stl"

    log("[PIPELINE] exporting STL...")
    export_voxels_to_stl(hull, pitch, hull_stl_path)
    log(f"[PIPELINE] saved raw hull mesh: {hull_stl_path}")

    carved_stl_path = None

    if optimize_material:
        log("[PIPELINE] carving material...")
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
            verbose=True,
        )

        log(f"[PIPELINE] carved sculpture voxels: {int(carved.sum())}")
        log(f"[PIPELINE] carved sculpture occupancy ratio: {float(carved.mean())}")
        log(f"[PIPELINE] carved sculpture voxels removed: {carve_stats['removed']}")
        log(f"[PIPELINE] carved sculpture reduction ratio: {carve_stats['reduction_ratio']}")

        simulate_and_save(
            carved,
            voxel_centers,
            sources,
            out_dir="outputs/sim",
            prefix="carved",
        )

        save_voxel_slices(hull, "outputs/debug/slices", "hull")
        save_voxel_slices(carved, "outputs/debug/slices", "carved")

        carved_stl_path = "outputs/meshes/shadow_carved.stl"
        export_voxels_to_stl(carved, pitch, carved_stl_path)
        log(f"[PIPELINE] saved carved mesh: {carved_stl_path}")

    log(f"[PIPELINE] completed in {time.time() - t0:.2f} seconds")

    return {
        "hull_stl_path": hull_stl_path,
        "carved_stl_path": carved_stl_path,
        "hull_summaries": hull_summaries,
    }


def main():
    args = parse_args()

    run_pipeline(
        view_paths=args.views,
        world_size=args.world_size,
        grid=args.grid,
        image_size_value=args.image_size,
        optimize_material=args.optimize_material,
    )


if __name__ == "__main__":
    main()