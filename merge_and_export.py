"""
merge_and_export.py
将多个 iVR-GS 子模型合并为统一 PLY，注入 semantic_id，
计算空间关系并导出增强语义字典。
"""

import os
import sys
import json
import argparse
import io as _io
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image as PILImage

from semantic_config import SemanticConfig


def _get_vertex_element(plydata):
    if "vertex" in plydata:
        return plydata["vertex"]
    if len(plydata.elements) > 0:
        return plydata.elements[0]
    raise ValueError("No PLY elements found.")


def find_ply_path(model_dir, iteration):
    candidates = [
        os.path.join(model_dir, f"point_cloud/iteration_{iteration}/point_cloud.ply"),
        os.path.join(model_dir, f"point_cloud/iteration_{iteration:05d}/point_cloud.ply"),
        os.path.join(model_dir, "point_cloud.ply"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    for root, _, files in os.walk(model_dir):
        if "point_cloud.ply" in files:
            return os.path.join(root, "point_cloud.ply")
    return None


def resolve_component_model_dir(model_root, component):
    name = component["name"]
    tf_id = component["tf_id"]
    candidates = [
        os.path.join(model_root, name),
        os.path.join(model_root, name, "3dgs"),
        os.path.join(model_root, name, "neilf"),
        os.path.join(model_root, tf_id),
        os.path.join(model_root, tf_id, "3dgs"),
        os.path.join(model_root, tf_id, "neilf"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_ply_raw(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = _get_vertex_element(plydata)
    n = len(vertex)
    print(f"  Loaded {n:,} gaussians from {ply_path}")
    return vertex, n


def compute_spatial_relations(comp_stats):
    relations = []
    directions = [("right_of", "left_of"), ("above", "below"), ("behind", "in_front_of")]
    for i, a in enumerate(comp_stats):
        for j, b in enumerate(comp_stats):
            if i >= j:
                continue

            ca = np.array(a["centroid"], dtype=np.float32)
            cb = np.array(b["centroid"], dtype=np.float32)
            diff = cb - ca
            dist = float(np.linalg.norm(diff))
            axis = int(np.argmax(np.abs(diff)))
            rel_a_to_b = directions[axis][0] if diff[axis] > 0 else directions[axis][1]

            a_min = np.array(a["bbox_min"], dtype=np.float32)
            a_max = np.array(a["bbox_max"], dtype=np.float32)
            b_min = np.array(b["bbox_min"], dtype=np.float32)
            b_max = np.array(b["bbox_max"], dtype=np.float32)
            overlap = bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))

            relations.append(
                {
                    "from": a["name"],
                    "to": b["name"],
                    "distance": round(dist, 4),
                    "adjacent": overlap,
                    "relative_position": rel_a_to_b,
                }
            )
    return relations


def merge_gaussians(config, model_root, iteration):
    all_vertex_arrays = []
    all_semantic_ids = []
    comp_stats = []
    offset = 0

    for comp in config.components:
        name = comp["name"]
        sem_id = comp["semantic_id"]

        model_dir = resolve_component_model_dir(model_root, comp)
        if model_dir is None:
            print(f"  [WARN] Not found: {name}, skipping")
            continue

        ply_path = find_ply_path(model_dir, iteration)
        if ply_path is None:
            print(f"  [WARN] No PLY in {model_dir}, skipping")
            continue

        print(f"\nProcessing: {name} (id={sem_id})")
        vertex, n = load_ply_raw(ply_path)

        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
        centroid = xyz.mean(axis=0).tolist()
        bbox_min = xyz.min(axis=0).tolist()
        bbox_max = xyz.max(axis=0).tolist()

        all_vertex_arrays.append(vertex.data)
        all_semantic_ids.append(np.full(n, sem_id, dtype=np.int16))
        comp_stats.append(
            {
                "semantic_id": sem_id,
                "name": name,
                "gaussian_count": n,
                "index_start": offset,
                "index_end": offset + n - 1,
                "centroid": centroid,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            }
        )
        offset += n

    print(f"\n=== Total: {offset:,} gaussians ===")
    return all_vertex_arrays, all_semantic_ids, comp_stats


def export_unified_ply(all_vertex_arrays, all_semantic_ids, output_path):
    merged_raw = np.concatenate(all_vertex_arrays)
    sem_ids = np.concatenate(all_semantic_ids)

    old_dtype = merged_raw.dtype
    new_dtype = np.dtype(old_dtype.descr + [("semantic_id", "<i2")])
    n = len(merged_raw)
    new_arr = np.zeros(n, dtype=new_dtype)

    for field in old_dtype.names:
        new_arr[field] = merged_raw[field]
    new_arr["semantic_id"] = sem_ids

    element = PlyElement.describe(new_arr, "vertex")
    PlyData([element]).write(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported: {output_path} ({size_mb:.1f} MB, {n:,} gaussians)")


def export_semantic_dict(config, comp_stats, spatial_relations, output_path):
    stats_by_id = {s["semantic_id"]: s for s in comp_stats}
    total = sum(s["gaussian_count"] for s in comp_stats)

    components = []
    for comp in config.components:
        sid = comp["semantic_id"]
        st = stats_by_id.get(sid, {})
        components.append(
            {
                "semantic_id": sid,
                "name": comp["name"],
                "parent": comp.get("parent", None),
                "aliases": comp.get("aliases", []),
                "description": comp.get("description", ""),
                "gaussian_count": st.get("gaussian_count", 0),
                "index_range": [st.get("index_start", 0), st.get("index_end", 0)],
                "centroid": st.get("centroid", [0, 0, 0]),
                "bbox": {
                    "min": st.get("bbox_min", [0, 0, 0]),
                    "max": st.get("bbox_max", [0, 0, 0]),
                },
                "default_visual": comp.get("default_visual", {"color": [1, 1, 1], "opacity": 1.0}),
                "custom_visual": None,
            }
        )

    groups = []
    for g in config.groups:
        children_ids = [
            config.get_by_name(n)["semantic_id"] for n in g.get("children", []) if config.get_by_name(n)
        ]
        groups.append(
            {
                "name": g["name"],
                "children_ids": children_ids,
                "children_names": g.get("children", []),
                "aliases": g.get("aliases", []),
                "description": g.get("description", ""),
            }
        )

    semantic_dict = {
        "dataset": config.dataset,
        "description": config.description,
        "version": "1.0",
        "total_gaussians": total,
        "components": components,
        "groups": groups,
        "spatial_relations": spatial_relations,
        "guided_tours": config.guided_tours,
        "action_schema": {
            "supported_actions": [
                "show",
                "hide",
                "highlight",
                "set_opacity",
                "set_color",
                "isolate",
                "reset",
                "explain",
                "guided_tour",
            ]
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(semantic_dict, f, indent=2, ensure_ascii=False)
    print(f"Exported dict: {output_path} ({len(components)} components, {len(groups)} groups)")


def generate_review(output_dir, ply_path, semantic_dict, max_points=60000):
    """Render 6-direction PNGs and a rotating GIF for the merged point cloud."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    review_dir = os.path.join(output_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    # Load PLY
    plydata = PlyData.read(ply_path)
    vertex = _get_vertex_element(plydata)
    n = len(vertex)
    print(f"[Review] Loaded {n:,} gaussians from PLY")

    # Build semantic_id -> RGB color map from default_visual
    color_map = {}
    for comp in semantic_dict.get("components", []):
        sid = comp["semantic_id"]
        c = comp.get("default_visual", {}).get("color", [0.65, 0.65, 0.65])
        color_map[sid] = np.clip(np.array(c, dtype=np.float32), 0.0, 1.0)

    # Extract xyz + semantic_id
    xyz = np.stack(
        [np.array(vertex["x"], dtype=np.float32),
         np.array(vertex["y"], dtype=np.float32),
         np.array(vertex["z"], dtype=np.float32)], axis=1)

    prop_names = vertex.dtype.names if hasattr(vertex.dtype, "names") else []
    if "semantic_id" in prop_names:
        sids = np.array(vertex["semantic_id"], dtype=np.int32)
    else:
        sids = np.zeros(n, dtype=np.int32)

    # Subsample
    if n > max_points:
        idx = np.random.default_rng(42).choice(n, max_points, replace=False)
        xyz = xyz[idx]
        sids = sids[idx]
        print(f"[Review] Subsampled to {max_points:,} points for rendering")

    # Assign per-point RGB
    colors = np.full((len(xyz), 3), 0.5, dtype=np.float32)
    for sid, color in color_map.items():
        mask = sids == sid
        if mask.any():
            colors[mask] = color

    # Center and normalise to [-1, 1]
    center = xyz.mean(axis=0)
    xyz = xyz - center
    scale = np.abs(xyz).max()
    if scale > 0:
        xyz /= scale

    BG = "#111111"
    marker_size = max(0.3, 3000.0 / len(xyz))

    def make_frame(elev, azim):
        fig = plt.figure(figsize=(5, 5), facecolor=BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=BG)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=colors, s=marker_size, alpha=0.75,
                   linewidths=0, rasterized=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        lim = 0.92
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        plt.tight_layout(pad=0)
        return fig

    def fig_to_pil(fig, dpi):
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi,
                    bbox_inches="tight", facecolor=BG, pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        return PILImage.open(buf).convert("RGB")

    # 6-direction stills
    directions = [
        ("front",    0,   0),
        ("back",     0, 180),
        ("left",     0,  90),
        ("right",    0, 270),
        ("top",     90,   0),
        ("bottom",  -90,  0),
    ]
    for name, elev, azim in directions:
        img = fig_to_pil(make_frame(elev, azim), dpi=150)
        out_path = os.path.join(review_dir, f"{name}.png")
        img.save(out_path)
        print(f"[Review] Saved {name}.png")

    # Rotating GIF (360° orbit at 25° elevation, 24 frames, 12 fps)
    n_frames = 24
    gif_frames = []
    for i in range(n_frames):
        azim = i * (360.0 / n_frames)
        gif_frames.append(fig_to_pil(make_frame(elev=25, azim=azim), dpi=80))
        if (i + 1) % 8 == 0:
            print(f"[Review] GIF progress: {i + 1}/{n_frames} frames")

    gif_path = os.path.join(review_dir, "rotating.gif")
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        optimize=False,
        duration=int(1000 / 12),
        loop=0,
    )
    print(f"[Review] Saved rotating.gif ({n_frames} frames @ 12 fps)")
    print(f"[Review] Review assets → {review_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--output_dir", default="unity_export")
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()

    config = SemanticConfig(args.config)
    print(f"Config: {config}")
    os.makedirs(args.output_dir, exist_ok=True)

    verts, sids, stats = merge_gaussians(config, args.model_root, args.iteration)
    if not stats:
        print("[ERROR] No models loaded.")
        sys.exit(1)

    ply_path = os.path.join(args.output_dir, "unified_scene.ply")
    export_unified_ply(verts, sids, ply_path)

    spatial = compute_spatial_relations(stats)
    json_path = os.path.join(args.output_dir, "semantic_dict.json")
    export_semantic_dict(config, stats, spatial, json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        semantic_dict = json.load(f)
    generate_review(args.output_dir, ply_path, semantic_dict)

    print("\n" + "=" * 50)
    print(f"  PLY:  {ply_path}")
    print(f"  JSON: {json_path}")
    print(f"  Review: {os.path.join(args.output_dir, 'review')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
