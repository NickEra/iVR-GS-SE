"""
merge_and_export.py
将多个 iVR-GS 子模型合并为统一 PLY，注入 semantic_id，
计算空间关系并导出增强语义字典。
"""

import os
import sys
import json
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

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

    print("\n" + "=" * 50)
    print(f"  PLY:  {ply_path}")
    print(f"  JSON: {json_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
