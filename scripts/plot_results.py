#!/usr/bin/env python3

# casadi_env 环境 python3 scripts/plot_results.py results/unicycle_demo_cpu.h5 --plot-mode sets3d
#python3 scripts/plot_results.py results/unicycle_demo_cpu.h5 --plot-mode sets3d --speed-mode fixed --speed-idx 1

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a symbolic-control HDF5 result file and render 2D/3D visualizations."
    )
    parser.add_argument("result_file", help="Path to the .h5 result file.")
    parser.add_argument(
        "--run",
        default="cpu",
        help="Run label inside /result/<run>. Default: cpu",
    )
    parser.add_argument(
        "--plot-mode",
        default="slice2d",
        choices=["slice2d", "sets3d"],
        help="Visualization mode. Default: slice2d",
    )
    parser.add_argument(
        "--field",
        default="reachable",
        choices=["valid", "candidate", "reachable", "controller", "reach_step"],
        help="Dataset to visualize in 2D slice mode. Default: reachable",
    )
    parser.add_argument(
        "--theta-idx",
        type=int,
        default=0,
        help="Fixed theta index for 2D slice mode. Default: 0",
    )
    parser.add_argument(
        "--speed-mode",
        default="project",
        choices=["project", "fixed"],
        help="How to reduce the speed dimension in 3D mode. Default: project",
    )
    parser.add_argument(
        "--speed-idx",
        type=int,
        default=0,
        help="Fixed speed index for 2D mode, or for 3D mode when --speed-mode fixed. Default: 0",
    )
    parser.add_argument(
        "--candidate-color",
        default="tab:red",
        help="Surface color for the candidate set in 3D mode. Default: tab:red",
    )
    parser.add_argument(
        "--reachable-color",
        default="tab:blue",
        help="Surface color for the reachable set in 3D mode. Default: tab:blue",
    )
    parser.add_argument(
        "--candidate-alpha",
        type=float,
        default=0.18,
        help="Surface opacity for the candidate set in 3D mode. Default: 0.18",
    )
    parser.add_argument(
        "--reachable-alpha",
        type=float,
        default=0.35,
        help="Surface opacity for the reachable set in 3D mode. Default: 0.35",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=100.0,
        help="Azimuth angle for 3D mode. Default: 100",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=16.0,
        help="Elevation angle for 3D mode. Default: 16",
    )
    parser.add_argument(
        "--output",
        help="Optional output PNG path. Default: derived from the HDF5 filename and mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive matplotlib window.",
    )
    return parser.parse_args()


def import_plotting_modules(show: bool, need_skimage: bool):
    try:
        import h5py
        import matplotlib
        import numpy as np

        if not show:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        measure = None
        if need_skimage:
            from skimage import measure
    except ImportError as exc:
        missing = getattr(exc, "name", "dependency")
        install = "python3 -m pip install h5py numpy matplotlib"
        if need_skimage:
            install += " scikit-image"
        raise SystemExit(
            "Missing Python package: "
            f"{missing}\n"
            "Install with:\n"
            f"  {install}"
        ) from exc

    return h5py, np, plt, measure


def default_output_path(args: argparse.Namespace) -> Path:
    stem = Path(args.result_file).stem
    if args.plot_mode == "slice2d":
        name = f"{stem}_{args.run}_{args.field}_th{args.theta_idx}_v{args.speed_idx}.png"
    else:
        speed_token = "projv" if args.speed_mode == "project" else f"v{args.speed_idx}"
        name = f"{stem}_{args.run}_candidate_vs_reachable_{speed_token}.png"
    # Create plots directory in current working directory if it doesn't exist
    plots_dir = Path.cwd() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir / name


def load_run_data(h5py, result_path: Path, run: str):
    with h5py.File(result_path, "r") as f:
        shape = f["/state/shape"][:].astype(int)
        lb = f["/state/lb"][:]
        eta = f["/state/eta"][:]
        invalid_input = int(f["/meta/invalid_input"][()])
        unreachable_step = int(f["/meta/unreachable_step"][()])
        available_runs = list(f["/result"].keys())

        run_group_path = f"/result/{run}"
        if run_group_path not in f:
            raise SystemExit(f"Run not found: {run}. Available runs: {available_runs}")

        datasets = {
            "valid": f[f"{run_group_path}/valid_mask"][:],
            "candidate": f[f"{run_group_path}/candidate_mask"][:],
            "reachable": f[f"{run_group_path}/reachable_mask"][:],
            "controller": f[f"{run_group_path}/controller"][:],
            "reach_step": f[f"{run_group_path}/reach_step"][:],
        }

    return {
        "shape": shape,
        "lb": lb,
        "eta": eta,
        "invalid_input": invalid_input,
        "unreachable_step": unreachable_step,
        "datasets": datasets,
    }


def reshape_field(np, data, shape):
    return np.asarray(data).reshape(shape, order="F")


def render_slice2d(args: argparse.Namespace, np, plt, loaded) -> Path:
    shape = loaded["shape"]
    lb = loaded["lb"]
    eta = loaded["eta"]
    invalid_input = loaded["invalid_input"]
    unreachable_step = loaded["unreachable_step"]

    if not (0 <= args.theta_idx < shape[2]):
        raise SystemExit(f"--theta-idx out of range: 0 <= idx < {shape[2]}")
    if not (0 <= args.speed_idx < shape[3]):
        raise SystemExit(f"--speed-idx out of range: 0 <= idx < {shape[3]}")

    grid4d = reshape_field(np, loaded["datasets"][args.field], shape)
    image = grid4d[:, :, args.theta_idx, args.speed_idx].T.astype(float)

    cmap = "viridis"
    colorbar_label = args.field

    if args.field in {"valid", "candidate", "reachable"}:
        colorbar_label = f"{args.field} mask"
    elif args.field == "controller":
        image[image == float(invalid_input)] = np.nan
        cmap = "tab10"
        colorbar_label = "input index"
    elif args.field == "reach_step":
        image[image == float(unreachable_step)] = np.nan
        cmap = "plasma"
        colorbar_label = "reach step"

    x = lb[0] + (np.arange(shape[0]) + 0.5) * eta[0]
    y = lb[1] + (np.arange(shape[1]) + 0.5) * eta[1]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        image,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"{args.field} | run={args.run} | theta_idx={args.theta_idx} | speed_idx={args.speed_idx}"
    )
    fig.tight_layout()

    output_path = Path(args.output) if args.output else default_output_path(args)
    fig.savefig(output_path, dpi=180)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def reduce_to_xyz(np, grid4d, speed_mode: str, speed_idx: int):
    if speed_mode == "project":
        return np.any(grid4d != 0, axis=3).astype(float), "speed-projected"

    if not (0 <= speed_idx < grid4d.shape[3]):
        raise SystemExit(f"--speed-idx out of range: 0 <= idx < {grid4d.shape[3]}")
    return grid4d[:, :, :, speed_idx].astype(float), f"speed_idx={speed_idx}"


def extract_isosurface(measure, np, volume_xyz, lb, eta):
    volume = np.transpose(volume_xyz, (1, 0, 2))
    if np.count_nonzero(volume > 0.5) == 0:
        return None, None

    verts, faces, _, _ = measure.marching_cubes(
        volume,
        level=0.5,
        spacing=(float(eta[1]), float(eta[0]), float(eta[2])),
    )
    verts[:, 0] += lb[1] + 0.5 * eta[1]
    verts[:, 1] += lb[0] + 0.5 * eta[0]
    verts[:, 2] += lb[2] + 0.5 * eta[2]
    verts = verts[:, [1, 0, 2]]
    return verts, faces


def set_equal_box_aspect(ax, x_range, y_range, z_range):
    ax.set_box_aspect(
        (
            max(x_range[1] - x_range[0], 1e-9),
            max(y_range[1] - y_range[0], 1e-9),
            max(z_range[1] - z_range[0], 1e-9),
        )
    )


def render_sets3d(args: argparse.Namespace, np, plt, measure, loaded) -> Path:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    shape = loaded["shape"]
    lb = loaded["lb"]
    eta = loaded["eta"]

    candidate_4d = reshape_field(np, loaded["datasets"]["candidate"], shape)
    reachable_4d = reshape_field(np, loaded["datasets"]["reachable"], shape)

    candidate_xyz, speed_label = reduce_to_xyz(np, candidate_4d, args.speed_mode, args.speed_idx)
    reachable_xyz, _ = reduce_to_xyz(np, reachable_4d, args.speed_mode, args.speed_idx)

    candidate_verts, candidate_faces = extract_isosurface(measure, np, candidate_xyz, lb, eta)
    reachable_verts, reachable_faces = extract_isosurface(measure, np, reachable_xyz, lb, eta)

    if candidate_faces is None and reachable_faces is None:
        raise SystemExit("No non-empty candidate/reachable set found for the requested 3D view.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Candidate set
    if candidate_faces is not None:
        candidate_mesh = Poly3DCollection(
            candidate_verts[candidate_faces],
            facecolor=args.candidate_color,
            edgecolor="none",
            alpha=args.candidate_alpha,
        )
        ax.add_collection3d(candidate_mesh)

    # Reachable set
    if reachable_faces is not None:
        reachable_mesh = Poly3DCollection(
            reachable_verts[reachable_faces],
            facecolor=args.reachable_color,
            edgecolor="none",
            alpha=args.reachable_alpha,
        )
        ax.add_collection3d(reachable_mesh)

    x_range = (lb[0], lb[0] + shape[0] * eta[0])
    y_range = (lb[1], lb[1] + shape[1] * eta[1])
    z_range = (lb[2], lb[2] + shape[2] * eta[2])
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(*z_range)
    set_equal_box_aspect(ax, x_range, y_range, z_range)

    # Draw hyperbolic constraint curves at theta = -pi, 0, +pi
    # Hyperbolic constraint: x1^2 - x2^2 <= a AND b*x2^2 - x1^2 <= c
    # We draw the boundary curves where equality holds
    theta_values = [-np.pi, 0, np.pi]
    x_line = np.linspace(x_range[0], x_range[1], 200)
    
    for theta_val in theta_values:
        # Constraint 1: x1^2 - x2^2 = a  =>  x2 = ±sqrt(x1^2 - a)
        # 将左半支和右半支分开画，避免 matplotlib 强行跨越无效区域连线
        
        # 左半支: x <= -2
        x1_left = x_line[x_line <= -2.0]
        if len(x1_left) > 0:
            x2_pos_left = np.sqrt(x1_left**2 - 4.0)
            x2_neg_left = -x2_pos_left
            ax.plot(x1_left, x2_pos_left, [theta_val]*len(x1_left), 'r-', linewidth=2.5, zorder=100)
            ax.plot(x1_left, x2_neg_left, [theta_val]*len(x1_left), 'r-', linewidth=2.5, zorder=100)

        # 右半支: x >= 2
        x1_right = x_line[x_line >= 2.0]
        if len(x1_right) > 0:
            x2_pos_right = np.sqrt(x1_right**2 - 4.0)
            x2_neg_right = -x2_pos_right
            ax.plot(x1_right, x2_pos_right, [theta_val]*len(x1_right), 'r-', linewidth=2.5, zorder=100)
            ax.plot(x1_right, x2_neg_right, [theta_val]*len(x1_right), 'r-', linewidth=2.5, zorder=100)
        
        # For constraint 2: b*x2^2 - x1^2 = c  =>  x2 = ±sqrt((x1^2 + c) / b)
        # 这个约束没有定义域截断，可以直接整体画
        # b = 4.0, c = 16.0
        x2_pos_c2 = np.sqrt((x_line**2 + 16.0) / 4.0)
        x2_neg_c2 = -x2_pos_c2
        ax.plot(x_line, x2_pos_c2, [theta_val]*len(x_line), 'r-', linewidth=2.5, zorder=100)
        ax.plot(x_line, x2_neg_c2, [theta_val]*len(x_line), 'r-', linewidth=2.5, zorder=100)
    # Set labels with bold and larger font
    ax.set_xlabel("x", fontsize=14, fontweight='bold')
    ax.set_ylabel("y", fontsize=14, fontweight='bold')
    ax.set_zlabel("θ (theta)", fontsize=14, fontweight='bold')
    
    # Set tick labels to bold and larger
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_zticklabels():
        label.set_fontweight('bold')
    
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(True, linewidth=0.5)
    
    # Format velocity slice text with real velocity value
    if args.speed_mode == "project":
        velocity_text = "Velocity: Projected"
    else:
        velocity_value = lb[3] + (args.speed_idx + 0.5) * eta[3]
        velocity_text = f"Velocity: {velocity_value:.3f}"
    
    title_text = f"Candidate vs Reachable | {velocity_text}"
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)

    # Create legend with bold text (without transparency descriptions)
    legend_handles = [
        plt.Line2D([0], [0], color=args.candidate_color, lw=10, alpha=args.candidate_alpha, label="Candidate"),
        plt.Line2D([0], [0], color=args.reachable_color, lw=10, alpha=args.reachable_alpha, label="Reachable"),
        plt.Line2D([0], [0], color='red', lw=3, label="Hyperbolic Constraint (θ = -π, 0, +π)"),
    ]
    legend = ax.legend(handles=legend_handles, loc="upper right", fontsize=11, framealpha=0.95)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    fig.tight_layout()

    output_path = Path(args.output) if args.output else default_output_path(args)
    fig.savefig(output_path, dpi=180)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main() -> int:
    args = parse_args()
    need_skimage = args.plot_mode == "sets3d"
    h5py, np, plt, measure = import_plotting_modules(args.show, need_skimage=need_skimage)

    result_path = Path(args.result_file)
    loaded = load_run_data(h5py, result_path, args.run)

    if args.plot_mode == "slice2d":
        output_path = render_slice2d(args, np, plt, loaded)
    else:
        output_path = render_sets3d(args, np, plt, measure, loaded)

    print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
