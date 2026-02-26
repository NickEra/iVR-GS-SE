"""
generate_review.py
读取统一 PLY + 语义字典，生成审查可视化：
1) turntable.gif
2) 六方向静态图
3) review_sheet.png
"""

import os
import json
import argparse
import numpy as np
from plyfile import PlyData
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import imageio


def load_colored_points(ply_path, dict_path, subsample=None):
    with open(dict_path, "r", encoding="utf-8") as f:
        sem_dict = json.load(f)

    color_map = {}
    for comp in sem_dict["components"]:
        color_map[comp["semantic_id"]] = comp["default_visual"].get("color", [0.5, 0.5, 0.5])

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"] if "vertex" in plydata else plydata.elements[0]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    sem_ids = np.array(vertex["semantic_id"], dtype=np.int16)
    n = len(xyz)

    colors = np.zeros((n, 3), dtype=np.float32)
    for sid, color in color_map.items():
        colors[sem_ids == sid] = color
    colors[np.all(colors == 0, axis=1)] = [0.5, 0.5, 0.5]

    if subsample and subsample < n:
        idx = np.random.choice(n, subsample, replace=False)
        xyz, colors, sem_ids = xyz[idx], colors[idx], sem_ids[idx]

    return xyz, colors, sem_ids, sem_dict


def _set_equal_axes(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 + 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_view(ax, xyz, colors, elev, azim, title, point_size=0.4):
    ax.clear()
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=point_size, alpha=0.65, edgecolors="none")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    _set_equal_axes(ax, xyz)
    ax.grid(True, alpha=0.2)


def fig_to_image(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    return buf


def generate_gif(xyz, colors, output_path, n_frames=60, resolution=800, point_size=0.4):
    print(f"Generating turntable GIF ({n_frames} frames)...")
    dpi = 100
    fig = plt.figure(figsize=(resolution / dpi, resolution / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    frames = []
    for i in range(n_frames):
        azim = (360.0 / n_frames) * i
        render_view(ax, xyz, colors, elev=25, azim=azim, title="", point_size=point_size)
        ax.set_title("")
        frame = fig_to_image(fig)
        frames.append(frame)
    plt.close(fig)

    imageio.mimsave(output_path, frames, fps=15, loop=0)
    print(f"  Saved: {output_path}")


def generate_direction_images(xyz, colors, output_dir, resolution=800, point_size=0.4):
    views = {
        "front": (0, 0),
        "back": (0, 180),
        "left": (0, 90),
        "right": (0, -90),
        "top": (90, 0),
        "bottom": (-90, 0),
    }
    dpi = 100
    paths = {}
    for name, (elev, azim) in views.items():
        fig = plt.figure(figsize=(resolution / dpi, resolution / dpi), dpi=dpi, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        render_view(ax, xyz, colors, elev=elev, azim=azim, title=name.title(), point_size=point_size)
        out = os.path.join(output_dir, f"{name}.png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[name] = out
    return paths


def generate_review_sheet(xyz, colors, sem_dict, output_path, point_size=0.35):
    views = [
        ("Front", 0, 0),
        ("Back", 0, 180),
        ("Left", 0, 90),
        ("Right", 0, -90),
        ("Top", 90, 0),
        ("Bottom", -90, 0),
    ]
    fig = plt.figure(figsize=(18, 12), dpi=100, facecolor="white")

    positions = [1, 2, 3, 5, 6, 7]
    for idx, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 4, positions[idx], projection="3d")
        render_view(ax, xyz, colors, elev=elev, azim=azim, title=title, point_size=point_size)

    ax_legend = fig.add_subplot(1, 4, 4)
    ax_legend.set_axis_off()
    ax_legend.set_title("Semantic Components", fontsize=12, fontweight="bold")
    patches = []
    for comp in sem_dict["components"]:
        patches.append(
            Patch(
                facecolor=comp["default_visual"].get("color", [0.5, 0.5, 0.5]),
                edgecolor="gray",
                label=f"[{comp['semantic_id']}] {comp['name']}",
            )
        )
    ax_legend.legend(handles=patches, loc="center", fontsize=9, frameon=True)

    fig.suptitle(
        f"Review: {sem_dict['dataset']} ({sem_dict['total_gaussians']:,} gaussians)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--dict", required=True)
    parser.add_argument("--output_dir", default="review")
    parser.add_argument("--gif_frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--subsample", type=int, default=50000)
    parser.add_argument("--point_size", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    subsample = args.subsample if args.subsample > 0 else None
    xyz, colors, _, sem_dict = load_colored_points(args.ply, args.dict, subsample=subsample)
    print(f"Loaded {len(xyz):,} points for review")

    gif_path = os.path.join(args.output_dir, "turntable.gif")
    generate_gif(
        xyz,
        colors,
        gif_path,
        n_frames=args.gif_frames,
        resolution=args.resolution,
        point_size=args.point_size,
    )
    generate_direction_images(
        xyz,
        colors,
        args.output_dir,
        resolution=args.resolution,
        point_size=args.point_size,
    )

    review_sheet = os.path.join(args.output_dir, "review_sheet.png")
    generate_review_sheet(xyz, colors, sem_dict, review_sheet, point_size=args.point_size)
    print(f"Review saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
