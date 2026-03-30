import argparse
import numpy as np
import time

from image_io import load_binary_image, save_mask
from voxel_ops import make_voxel_centers, voxel_pitch
from shadow_hull import compute_shadow_hull
from shadow_source import build_sources
from export_mesh import export_voxels_to_stl
from carve import carve_hollow_shell_strict
from simulate import simulate_and_save
from debug_slices import save_voxel_slices
from optimize_consistency import optimize_silhouettes
from reset_output import reset_output_dirs


def print_view_metrics(name, summaries):
    print(f"\n{name}")
    ious = []
    for i, m in enumerate(summaries):
        ious.append(m["iou"])
        print(
            f"  view {i}: "
            f"IoU={m['iou']:.4f}, "
            f"target={m['target_pixels']}, "
            f"actual={m['actual_pixels']}, "
            f"missing={m['missing_pixels']}, "
            f"extra={m['extra_pixels']}"
        )
    if len(ious) > 0:
        print(f"  avg IoU = {float(np.mean(ious)):.4f}")
        print(f"  min IoU = {float(np.min(ious)):.4f}")


def parse_direction_string(s: str):
    """
    Example:
      "1,0,0;0,0,1"
    """
    dirs = []
    parts = s.strip().split(";")
    for p in parts:
        vals = [float(x) for x in p.split(",")]
        if len(vals) != 3:
            raise ValueError("Each direction must have 3 values.")
        dirs.append(np.array(vals, dtype=float))
    return dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a multi-view shadow sculpture from 2 or more silhouette images."
    )

    parser.add_argument(
        "views",
        nargs="+",
        help="2 or more silhouette image paths"
    )

    parser.add_argument(
        "--world-size",
        type=float,
        default=80.0,
        help="Physical size of the voxel world in mm"
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=160,
        help="Voxel resolution used for nx=ny=nz"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square size used to resize input silhouettes"
    )

    parser.add_argument(
        "--directions",
        type=str,
        default=None,
        help='Semicolon-separated direction vectors, e.g. "1,0,0;0,0,1"'
    )

    parser.add_argument(
        "--optimize-material",
        action="store_true",
        help="Enable hollow-shell carving"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binarization threshold for input image"
    )

    parser.add_argument(
        "--close-iters",
        type=int,
        default=0,
        help="Binary closing iterations for silhouette cleanup"
    )

    parser.add_argument(
        "--open-iters",
        type=int,
        default=0,
        help="Binary opening iterations for silhouette cleanup"
    )

    parser.add_argument(
        "--dilate-iters",
        type=int,
        default=0,
        help="Optional binary dilation iterations"
    )

    parser.add_argument(
        "--opt-iters",
        type=int,
        default=4,
        help="Silhouette optimization iterations"
    )

    args = parser.parse_args()

    if len(args.views) < 2:
        parser.error("Please provide at least 2 silhouette images.")

    return args


def main():
    args = parse_args()
    t0 = time.time()

    world_size = args.world_size
    nx = ny = nz = args.grid
    image_size = (args.image_size, args.image_size)
    optimize_material = args.optimize_material

    reset_output_dirs()

    directions = None
    if args.directions is not None:
        directions = parse_direction_string(args.directions)
        if len(directions) != len(args.views):
            raise ValueError("Number of directions must match number of input images.")

    images = []
    for i, path in enumerate(args.views):
        img = load_binary_image(
            path,
            size=image_size,
            threshold=args.threshold,
            do_cleanup=True,
            close_iters=args.close_iters,
            open_iters=args.open_iters,
            dilate_iters=args.dilate_iters,
        )
        images.append(img)
        print(f"[PIPELINE] img{i} silhouette pixels: {int(img.sum())} of {img.size}")
        save_mask(img, f"outputs/debug/masks/base/view{i}_mask.png")

    sources = build_sources(images, world_size, directions=directions)

    for i, src in enumerate(sources):
        print(f"[PIPELINE] shadow source {i}: direction={src.direction}, up={src.up}")

    voxel_centers = make_voxel_centers(nx, ny, nz, world_size)

    print("\n[PIPELINE] skipping silhouette optimization for hole preservation...")
    sources = build_sources(images, world_size, directions=directions)

    print("\n[PIPELINE] computing shadow hull...")
    hull = compute_shadow_hull(
        sources,
        voxel_centers,
        enforce_connectivity=True,
        min_component_size=8,
        verbose=True,
    )

    print("[PIPELINE] initial hull voxel count:", int(hull.sum()))
    print("[PIPELINE] initial hull occupancy ratio:", float(hull.mean()))

    hull_summaries = simulate_and_save(
        hull,
        voxel_centers,
        sources,
        out_dir="outputs/sim",
        prefix="hull"
    )
    print_view_metrics("Hull shadow simulation", hull_summaries)

    pitch = voxel_pitch(world_size, nx, ny, nz)

    try:
        export_voxels_to_stl(hull, pitch, "outputs/meshes/shadow_hull.stl")
        print("[PIPELINE] saved raw hull mesh: outputs/meshes/shadow_hull.stl")
    except Exception as e:
        print("[ERROR] Raw hull export failed:", e)

    save_voxel_slices(hull, "outputs/debug/slices", "hull")

    if optimize_material:
        print("\n[PIPELINE] carving hollow shell...")
        carved, carve_stats = carve_hollow_shell_strict(
            hull,
            voxel_centers,
            sources,
            shell_thickness_voxels=2,
            max_passes=4,
            random_seed=0,
            protect_endcaps=True,
            cleanup_components=True,
            min_component_size=120,
            verbose=True
        )

        print("[PIPELINE] carved sculpture voxels:", int(carved.sum()))
        print("[PIPELINE] carved sculpture occupancy ratio:", float(carved.mean()))
        print("[PIPELINE] carved sculpture voxels removed:", carve_stats["removed"])
        print("[PIPELINE] carved sculpture reduction ratio:", carve_stats["reduction_ratio"])

        carved_summaries = simulate_and_save(
            carved,
            voxel_centers,
            sources,
            out_dir="outputs/sim",
            prefix="carved"
        )
        print_view_metrics("Carved shadow simulation", carved_summaries)

        save_voxel_slices(carved, "outputs/debug/slices", "carved")

        try:
            export_voxels_to_stl(carved, pitch, "outputs/meshes/shadow_carved.stl")
            print("[PIPELINE] saved carved mesh: outputs/meshes/shadow_carved.stl")
        except Exception as e:
            print("[ERROR] Carved mesh export failed:", e)

    print(f"\n[PIPELINE] total runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()