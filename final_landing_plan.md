# 二次开发落地执行手册（含优化方案 + 可视化审查输出）

---

## 第一部分：完整文件清单

以下是基于 iVR-GS 二次开发需要**修改**和**新增**的所有文件，已整合全部优化方案。

```
iVR-GS/
│
│  ===== 修改的文件 =====
├── requirements.txt                          # [修改] 新增依赖
├── arguments/__init__.py                     # [修改] +10行 新增语义参数
├── scene/gaussian_model.py                   # [修改] +30行 新增 semantic_id 属性
├── train.py                                  # [修改] +15行 语义标签+元数据+缩略图
│
│  ===== 新增：语义配置 =====
├── semantic_config.py                        # [新增] 语义配置解析器（含分层本体）
├── configs/
│   ├── combustion.yaml                       # [新增] combustion 语义配置
│   └── carp.yaml                             # [新增] carp 语义配置
│
│  ===== 新增：合并导出 =====
├── merge_and_export.py                       # [新增] ★核心：合并+注入+空间关系+字典
├── generate_review.py                        # [新增] ★审查输出：GIF动图+6方向预览图
├── generate_prompt.py                        # [新增] 从字典自动生成LLM系统提示
├── export_pipeline.sh                        # [新增] 一键流水线
│
│  ===== 新增：LLM交互测试 =====
├── test_llm_interaction.py                   # [新增] 命令行LLM交互测试
│
│  ===== 输出产物 =====
└── unity_export/
    └── <dataset>/
        ├── unified_scene.ply                 # 统一PLY（含semantic_id）
        ├── semantic_dict.json                # 语义字典（含空间关系、分组、导览）
        ├── system_prompt.txt                 # 自动生成的LLM系统提示
        ├── thumbnails/                       # 各组件缩略图
        │   ├── dorsal_fin.png
        │   └── ...
        └── review/                           # ★审查文件夹
            ├── turntable.gif                 # 旋转动图（默认颜色着色）
            ├── front.png                     # 前视图
            ├── back.png                      # 后视图
            ├── left.png                      # 左视图
            ├── right.png                     # 右视图
            ├── top.png                       # 顶视图
            ├── bottom.png                    # 底视图
            └── review_sheet.png             # 拼接总览图（6视图+颜色图例）
```

---

## 第二部分：执行步骤

### Step 0：环境搭建（约30分钟）

```bash
git clone https://github.com/TouKaienn/iVR-GS.git
cd iVR-GS
conda create --name iVRGS python=3.9
conda activate iVRGS

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e ./submodules/diff-gaussian-rasterization
pip install -e ./submodules/simple-knn
pip install -r requirements.txt
```

从 Google Drive 下载 combustion 数据集 + 预训练 ckpt，分别放入 `Data/` 和 `output/`。

### Step 1：代码改动（约2小时）

按本文档第三部分的逐文件说明操作。

### Step 2：验证原始功能（约15分钟）

```bash
# 确认预训练模型可用
bash exp_scripts/gui.sh
# 确认渲染评估正常
bash exp_scripts/render.sh
```

### Step 3：运行合并导出（约5分钟）

```bash
python merge_and_export.py \
    --config configs/combustion.yaml \
    --model_root output/combustion \
    --output_dir unity_export/combustion \
    --iteration 30000
```

检查输出：`unity_export/combustion/unified_scene.ply` + `semantic_dict.json`

### Step 4：生成审查可视化（约3分钟）

```bash
python generate_review.py \
    --ply unity_export/combustion/unified_scene.ply \
    --dict unity_export/combustion/semantic_dict.json \
    --output_dir unity_export/combustion/review
```

检查输出：`review/turntable.gif` + 6张方向图 + `review_sheet.png`

### Step 5：LLM 交互测试（约15分钟）

```bash
python generate_prompt.py \
    --dict unity_export/combustion/semantic_dict.json \
    --output unity_export/combustion/system_prompt.txt

python test_llm_interaction.py \
    --dict unity_export/combustion/semantic_dict.json
```

### Step 6（可选）：从头训练

```bash
bash export_pipeline.sh combustion
```

---

## 第三部分：所有新增/修改文件的完整代码

### 3.1 `requirements.txt`（修改）

```txt
# === iVR-GS 原始依赖 ===
plyfile
tqdm
Pillow
scipy
lpips
opencv-python
dearpygui

# === 二次开发新增 ===
pyyaml>=6.0
numpy>=1.24.0
trimesh>=4.0.0
matplotlib>=3.7.0
imageio>=2.31.0
openai>=1.0.0
requests>=2.31.0
```

### 3.2 `arguments/__init__.py`（修改，+10行）

在 `ModelParams` 类的参数定义区追加：

```python
# ===== 语义配置参数（追加到已有参数之后）=====
parser.add_argument("--semantic_config", type=str, default="",
                    help="Path to semantic config YAML")
parser.add_argument("--semantic_name", type=str, default="",
                    help="Semantic name of current training component")
parser.add_argument("--semantic_id", type=int, default=-1,
                    help="Semantic ID of current training component")
```

### 3.3 `scene/gaussian_model.py`（修改，+30行，4处插入）

**插入点 1** — `__init__` 方法中，在 `self._opacity = torch.empty(0)` 附近追加：

```python
self._semantic_id = torch.empty(0, dtype=torch.uint8)
self.semantic_label = 255  # 255 = unassigned (uint8 sentinel)
```

**插入点 2** — `create_from_pcd` 方法末尾追加：

```python
self._semantic_id = torch.full(
    (fused_point_cloud.shape[0],), self.semantic_label,
    dtype=torch.uint8, device="cuda"
)
```

**插入点 3** — `save_ply` 方法中，在构建 `dtype_full` 列表和 `attributes` 拼接处追加：

```python
# dtype_full 列表末尾追加
dtype_full.append(('semantic_id', 'u1'))

# attributes 拼接时末尾追加
semantic_col = self._semantic_id.detach().cpu().numpy().reshape(-1, 1).astype(np.uint8)
# 然后把 semantic_col 加入 np.concatenate 的列表中
```

**插入点 4** — `load_ply` 方法末尾追加：

```python
try:
    sem_ids = np.array(plydata.elements[0]["semantic_id"], dtype=np.uint8)
    self._semantic_id = torch.tensor(sem_ids, dtype=torch.uint8, device="cuda")
except (ValueError, KeyError):
    n = self._xyz.shape[0]
    self._semantic_id = torch.full((n,), 255, dtype=torch.uint8, device="cuda")
```

### 3.4 `train.py`（修改，+15行）

**插入点 1** — 在 `gaussians` 初始化完成后（scene 创建之后）：

```python
if args.semantic_id >= 0:
    gaussians.semantic_label = args.semantic_id
    print(f"[Semantic] Training: {args.semantic_name} (id={args.semantic_id})")
```

**插入点 2** — 训练循环结束后、最终保存之前：

```python
if args.semantic_config:
    import json
    meta = {
        "semantic_id": args.semantic_id,
        "semantic_name": args.semantic_name,
        "num_gaussians": gaussians.get_xyz.shape[0],
    }
    meta_path = os.path.join(args.model_path, "semantic_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[Semantic] Metadata saved: {meta_path}")
```

### 3.5 `semantic_config.py`（新增）

```python
"""
semantic_config.py — 语义配置解析器（支持分层本体）
"""
import yaml

class SemanticConfig:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.raw = yaml.safe_load(f)
        self.dataset = self.raw['dataset']
        self.description = self.raw.get('description', '')
        self.components = self.raw['components']
        self.groups = self.raw.get('groups', [])
        self.guided_tours = self.raw.get('guided_tours', [])
        self._by_id = {c['semantic_id']: c for c in self.components}
        self._by_tf = {c['tf_id']: c for c in self.components}
        self._by_name = {c['name']: c for c in self.components}

    def get_by_id(self, sid):        return self._by_id.get(sid)
    def get_by_tf(self, tf):         return self._by_tf.get(tf)
    def get_by_name(self, name):     return self._by_name.get(name)
    def get_all_tf_ids(self):        return [c['tf_id'] for c in self.components]
    def get_all_names(self):         return [c['name'] for c in self.components]
    def num_components(self):        return len(self.components)

    def resolve_group(self, group_name):
        """展开分组为子组件 semantic_id 列表"""
        for g in self.groups:
            if g['name'] == group_name or group_name in g.get('aliases', []):
                return [self._by_name[n]['semantic_id'] for n in g['children'] if n in self._by_name]
        return []

    def __repr__(self):
        return f"SemanticConfig({self.dataset}, {self.num_components()} components, {len(self.groups)} groups)"

def load_config(path):
    return SemanticConfig(path)

if __name__ == "__main__":
    import sys
    cfg = load_config(sys.argv[1])
    print(cfg)
    for c in cfg.components:
        print(f"  {c['tf_id']} -> id={c['semantic_id']}, name={c['name']}")
    for g in cfg.groups:
        print(f"  [group] {g['name']}: {g['children']}")
```

### 3.6 `configs/combustion.yaml`（新增）

```yaml
dataset: combustion
description: "Combustion simulation with 10 transfer function regions"

components:
  - tf_id: "TF01"
    semantic_id: 1
    name: "low_temperature"
    parent: "cool_zones"
    aliases: ["低温区", "cool region", "cold zone"]
    description: "Low temperature region of the combustion field"
    default_visual: { color: [0.0, 0.2, 0.8], opacity: 1.0 }

  - tf_id: "TF02"
    semantic_id: 2
    name: "medium_low_temperature"
    parent: "cool_zones"
    aliases: ["中低温区", "warm region"]
    description: "Medium-low temperature transition zone"
    default_visual: { color: [0.0, 0.6, 0.6], opacity: 1.0 }

  - tf_id: "TF03"
    semantic_id: 3
    name: "medium_temperature"
    parent: "transition_zones"
    aliases: ["中温区", "mid temperature"]
    description: "Medium temperature region"
    default_visual: { color: [0.2, 0.8, 0.2], opacity: 1.0 }

  - tf_id: "TF04"
    semantic_id: 4
    name: "medium_high_temperature"
    parent: "transition_zones"
    aliases: ["中高温区"]
    description: "Medium-high temperature zone approaching combustion core"
    default_visual: { color: [0.6, 0.8, 0.0], opacity: 1.0 }

  - tf_id: "TF05"
    semantic_id: 5
    name: "high_temperature"
    parent: "hot_zones"
    aliases: ["高温区", "hot zone"]
    description: "High temperature combustion region"
    default_visual: { color: [0.9, 0.6, 0.0], opacity: 1.0 }

  - tf_id: "TF06"
    semantic_id: 6
    name: "flame_front"
    parent: "hot_zones"
    aliases: ["火焰前沿", "flame boundary"]
    description: "Flame front where rapid oxidation occurs"
    default_visual: { color: [1.0, 0.4, 0.0], opacity: 1.0 }

  - tf_id: "TF07"
    semantic_id: 7
    name: "combustion_core"
    parent: "hot_zones"
    aliases: ["燃烧核心", "fire core", "hottest zone"]
    description: "Core combustion region with highest temperature"
    default_visual: { color: [1.0, 0.1, 0.0], opacity: 1.0 }

  - tf_id: "TF08"
    semantic_id: 8
    name: "exhaust_plume"
    parent: "byproducts"
    aliases: ["排气羽流", "exhaust", "smoke plume"]
    description: "Exhaust plume of combustion products"
    default_visual: { color: [0.6, 0.3, 0.1], opacity: 0.8 }

  - tf_id: "TF09"
    semantic_id: 9
    name: "turbulent_mixing"
    parent: "byproducts"
    aliases: ["湍流混合区", "turbulence zone"]
    description: "Turbulent mixing region between fuel and oxidizer"
    default_visual: { color: [0.5, 0.5, 0.5], opacity: 0.6 }

  - tf_id: "TF10"
    semantic_id: 10
    name: "ambient_gas"
    aliases: ["环境气体", "surrounding gas", "background"]
    description: "Ambient gas surrounding the combustion region"
    default_visual: { color: [0.2, 0.2, 0.3], opacity: 0.3 }

groups:
  - name: "cool_zones"
    children: ["low_temperature", "medium_low_temperature"]
    aliases: ["冷区", "cool regions", "低温区域"]
    description: "All low-temperature regions"
  - name: "transition_zones"
    children: ["medium_temperature", "medium_high_temperature"]
    aliases: ["过渡区", "middle regions"]
    description: "Temperature transition regions"
  - name: "hot_zones"
    children: ["high_temperature", "flame_front", "combustion_core"]
    aliases: ["热区", "hot regions", "高温区域", "火焰区"]
    description: "All high-temperature and combustion regions"
  - name: "byproducts"
    children: ["exhaust_plume", "turbulent_mixing"]
    aliases: ["副产物", "byproducts", "废气"]
    description: "Combustion byproducts and mixing regions"

guided_tours:
  - name: "temperature_gradient"
    description: "从低温到高温逐步展示"
    steps:
      - { action_type: "show", target: "all", narration: "这是一个燃烧模拟数据集，包含10个温度区域。" }
      - { action_type: "isolate", target: "cool_zones", narration: "首先是低温区域，包括远离燃烧核心的冷区。" }
      - { action_type: "isolate", target: "transition_zones", narration: "接着是过渡区域，温度逐渐升高。" }
      - { action_type: "isolate", target: "hot_zones", narration: "最核心的高温区域，包括火焰前沿和燃烧核心。" }
      - { action_type: "highlight", target: "combustion_core", narration: "燃烧核心是温度最高的区域。" }
      - { action_type: "reset", target: "all" }
```

### 3.7 `configs/carp.yaml`（新增）

```yaml
dataset: carp
description: "CT scan of a carp fish (carp_boneRGBa_sags_class7) with 7 semantic components"

components:
  - tf_id: "TF00"
    semantic_id: 0
    name: "head"
    parent: "body_structure"
    aliases: ["头部", "fish head", "skull", "鱼头"]
    description: "Protects the brain and sensory organs"
    default_visual: { color: [0.85, 0.75, 0.55], opacity: 1.0 }

  - tf_id: "TF01"
    semantic_id: 1
    name: "pectoral_fin"
    parent: "fins"
    aliases: ["胸鳍", "pectoral fin", "side fin"]
    description: "Used for steering, braking and hovering"
    default_visual: { color: [0.9, 0.5, 0.1], opacity: 1.0 }

  - tf_id: "TF02"
    semantic_id: 2
    name: "spinal_cord_and_ribs"
    parent: "skeleton"
    aliases: ["脊椎和肋骨", "spine", "ribs", "backbone", "骨骼"]
    description: "Main structural support of the skeleton"
    default_visual: { color: [0.95, 0.95, 0.85], opacity: 1.0 }

  - tf_id: "TF03"
    semantic_id: 3
    name: "anal_fin"
    parent: "fins"
    aliases: ["臀鳍", "anal fin", "bottom fin"]
    description: "Helps with stability during swimming"
    default_visual: { color: [0.9, 0.2, 0.2], opacity: 1.0 }

  - tf_id: "TF04"
    semantic_id: 4
    name: "ventral_fin"
    parent: "fins"
    aliases: ["腹鳍", "ventral fin", "pelvic fin", "belly fin"]
    description: "Used for steering, braking and hovering"
    default_visual: { color: [0.7, 0.2, 0.7], opacity: 1.0 }

  - tf_id: "TF05"
    semantic_id: 5
    name: "dorsal_fin"
    parent: "fins"
    aliases: ["背鳍", "dorsal fin", "top fin"]
    description: "Prevents rolling, provides stability"
    default_visual: { color: [0.2, 0.4, 0.9], opacity: 1.0 }

  - tf_id: "TF06"
    semantic_id: 6
    name: "caudal_fin"
    parent: "fins"
    aliases: ["尾鳍", "tail fin", "caudal fin", "尾巴"]
    description: "Primary fin for propulsion and steering"
    default_visual: { color: [0.2, 0.8, 0.2], opacity: 1.0 }

groups:
  - name: "fins"
    children: ["pectoral_fin", "anal_fin", "ventral_fin", "dorsal_fin", "caudal_fin"]
    aliases: ["鱼鳍", "all fins", "所有鳍"]
    description: "All fin structures of the carp"
  - name: "skeleton"
    children: ["spinal_cord_and_ribs"]
    aliases: ["骨骼", "bones", "骨架"]
    description: "Skeletal structure"
  - name: "body_structure"
    children: ["head", "spinal_cord_and_ribs"]
    aliases: ["身体结构", "body", "主体"]
    description: "Main body and skeletal structures"

guided_tours:
  - name: "anatomy_overview"
    description: "鲤鱼解剖结构概览"
    steps:
      - { action_type: "show", target: "all", narration: "这是一条鲤鱼的CT扫描骨骼三维重建，包含7个语义组件。" }
      - { action_type: "isolate", target: "body_structure", narration: "鲤鱼的身体结构包括头部和脊椎肋骨。" }
      - { action_type: "highlight", target: "head", narration: "头部保护大脑和感觉器官。" }
      - { action_type: "isolate", target: "fins", narration: "鲤鱼有五种鳍，各有不同功能。" }
      - { action_type: "highlight", target: "caudal_fin", narration: "尾鳍是推进和转向的主要动力。" }
      - { action_type: "highlight", target: "pectoral_fin", narration: "胸鳍用于转向、制动和悬停。" }
      - { action_type: "reset", target: "all" }
```

### 3.8 `merge_and_export.py`（新增 ★核心）

```python
"""
merge_and_export.py
将多个 iVR-GS 子模型合并为统一 PLY，注入 semantic_id，
计算空间关系，导出语义字典。

用法:
    python merge_and_export.py \
        --config configs/combustion.yaml \
        --model_root output/combustion \
        --output_dir unity_export/combustion \
        --iteration 30000
"""
import os, sys, json, argparse
import numpy as np
from plyfile import PlyData, PlyElement
from semantic_config import SemanticConfig


def find_ply_path(model_dir, iteration):
    candidates = [
        os.path.join(model_dir, f"point_cloud/iteration_{iteration}/point_cloud.ply"),
        os.path.join(model_dir, f"point_cloud/iteration_{iteration:05d}/point_cloud.ply"),
        os.path.join(model_dir, "point_cloud.ply"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f == "point_cloud.ply":
                return os.path.join(root, f)
    return None


def load_ply_raw(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    n = len(vertex)
    print(f"  Loaded {n:,} gaussians from {ply_path}")
    return vertex, n


def compute_spatial_relations(comp_stats):
    """计算组件间空间关系"""
    relations = []
    for i, a in enumerate(comp_stats):
        for j, b in enumerate(comp_stats):
            if i >= j:
                continue
            ca = np.array(a['centroid'])
            cb = np.array(b['centroid'])
            dist = float(np.linalg.norm(ca - cb))

            # 相对方向
            diff = cb - ca
            abs_diff = np.abs(diff)
            axis = int(np.argmax(abs_diff))
            directions = [
                ("right_of", "left_of"),   # x
                ("above", "below"),         # y
                ("behind", "in_front_of"),  # z
            ]
            if diff[axis] > 0:
                rel_a_to_b = directions[axis][0]
            else:
                rel_a_to_b = directions[axis][1]

            # bbox 重叠判定
            a_min, a_max = np.array(a['bbox_min']), np.array(a['bbox_max'])
            b_min, b_max = np.array(b['bbox_min']), np.array(b['bbox_max'])
            overlap = np.all(a_min <= b_max) and np.all(b_min <= a_max)

            relations.append({
                "from": a['name'], "to": b['name'],
                "distance": round(dist, 4),
                "adjacent": bool(overlap),
                "relative_position": rel_a_to_b,
            })
    return relations


def merge_gaussians(config, model_root, iteration):
    all_vertex_arrays = []
    all_semantic_ids = []
    comp_stats = []
    offset = 0

    for comp in config.components:
        name = comp['name']
        sem_id = comp['semantic_id']

        model_dir = os.path.join(model_root, name)
        if not os.path.exists(model_dir):
            model_dir = os.path.join(model_root, comp['tf_id'])
        if not os.path.exists(model_dir):
            print(f"  [WARN] Not found: {name}, skipping")
            continue

        ply_path = find_ply_path(model_dir, iteration)
        if ply_path is None:
            print(f"  [WARN] No PLY in {model_dir}, skipping")
            continue

        print(f"\nProcessing: {name} (id={sem_id})")
        vertex, n = load_ply_raw(ply_path)

        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        centroid = xyz.mean(axis=0).tolist()
        bbox_min = xyz.min(axis=0).tolist()
        bbox_max = xyz.max(axis=0).tolist()

        all_vertex_arrays.append(vertex.data)
        all_semantic_ids.append(np.full(n, sem_id, dtype=np.uint8))

        comp_stats.append({
            "semantic_id": sem_id,
            "name": name,
            "gaussian_count": n,
            "index_start": offset,
            "index_end": offset + n - 1,
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        })
        offset += n
        print(f"  n={n:,}, centroid={[round(v,3) for v in centroid]}")

    print(f"\n=== Total: {offset:,} gaussians ===")
    return all_vertex_arrays, all_semantic_ids, comp_stats


def export_unified_ply(all_vertex_arrays, all_semantic_ids, output_path):
    """导出统一 PLY，追加 semantic_id 属性"""
    merged_raw = np.concatenate(all_vertex_arrays)
    sem_ids = np.concatenate(all_semantic_ids)

    # 构建新的 dtype：在原有基础上追加 semantic_id
    old_dtype = merged_raw.dtype
    new_dtype = np.dtype(old_dtype.descr + [('semantic_id', 'u1')])

    # 构建新数组
    n = len(merged_raw)
    new_arr = np.zeros(n, dtype=new_dtype)
    for field in old_dtype.names:
        new_arr[field] = merged_raw[field]
    new_arr['semantic_id'] = sem_ids

    el = PlyElement.describe(new_arr, 'vertex')
    PlyData([el]).write(output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExported: {output_path} ({size_mb:.1f} MB, {n:,} gaussians)")


def export_semantic_dict(config, comp_stats, spatial_relations, output_path):
    stats_by_id = {s['semantic_id']: s for s in comp_stats}
    total = sum(s['gaussian_count'] for s in comp_stats)

    components = []
    for comp in config.components:
        sid = comp['semantic_id']
        st = stats_by_id.get(sid, {})
        components.append({
            "semantic_id": sid,
            "name": comp['name'],
            "parent": comp.get('parent', None),
            "aliases": comp.get('aliases', []),
            "description": comp.get('description', ''),
            "gaussian_count": st.get('gaussian_count', 0),
            "index_range": [st.get('index_start', 0), st.get('index_end', 0)],
            "centroid": st.get('centroid', [0, 0, 0]),
            "bbox": {"min": st.get('bbox_min', [0,0,0]), "max": st.get('bbox_max', [0,0,0])},
            "default_visual": comp.get('default_visual', {"color": [1,1,1], "opacity": 1.0}),
            "custom_visual": None,
        })

    # 分组
    groups = []
    for g in config.groups:
        children_ids = [config.get_by_name(n)['semantic_id']
                        for n in g['children'] if config.get_by_name(n)]
        groups.append({
            "name": g['name'],
            "children_ids": children_ids,
            "children_names": g['children'],
            "aliases": g.get('aliases', []),
            "description": g.get('description', ''),
        })

    # 导览
    tours = config.guided_tours

    d = {
        "dataset": config.dataset,
        "description": config.description,
        "version": "1.0",
        "total_gaussians": total,
        "components": components,
        "groups": groups,
        "spatial_relations": spatial_relations,
        "guided_tours": tours,
        "action_schema": {
            "supported_actions": [
                "show", "hide", "highlight", "set_opacity",
                "set_color", "isolate", "reset", "explain", "guided_tour"
            ],
        },
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
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
        print("[ERROR] No models loaded."); sys.exit(1)

    ply_path = os.path.join(args.output_dir, "unified_scene.ply")
    export_unified_ply(verts, sids, ply_path)

    spatial = compute_spatial_relations(stats)
    json_path = os.path.join(args.output_dir, "semantic_dict.json")
    export_semantic_dict(config, stats, spatial, json_path)

    print("\n" + "="*50)
    print(f"  PLY:  {ply_path}")
    print(f"  JSON: {json_path}")
    print("="*50)


if __name__ == "__main__":
    main()
```

### 3.9 `generate_review.py`（新增 ★审查可视化）

```python
"""
generate_review.py
读取统一 PLY + 语义字典，生成：
  1. turntable.gif — 360° 旋转动图（按语义默认颜色着色）
  2. 6 个方向的静态预览图
  3. review_sheet.png — 拼接总览图（含颜色图例）

用法:
    python generate_review.py \
        --ply unity_export/combustion/unified_scene.ply \
        --dict unity_export/combustion/semantic_dict.json \
        --output_dir unity_export/combustion/review \
        --gif_frames 60 \
        --resolution 800
"""
import os, json, argparse
import numpy as np
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import imageio


def load_colored_points(ply_path, dict_path, subsample=None):
    """加载 PLY 并按语义字典分配默认颜色"""
    # 读取字典
    with open(dict_path, 'r', encoding='utf-8') as f:
        sem_dict = json.load(f)

    # 建立 semantic_id -> 颜色 的映射
    color_map = {}
    name_map = {}
    for comp in sem_dict['components']:
        sid = comp['semantic_id']
        c = comp['default_visual'].get('color', [0.5, 0.5, 0.5])
        color_map[sid] = c
        name_map[sid] = comp['name']

    # 读取 PLY
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    n = len(vertex)

    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    sem_ids = np.array(vertex['semantic_id'], dtype=np.uint8)

    # 分配颜色
    colors = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        sid = sem_ids[i]
        colors[i] = color_map.get(sid, [0.5, 0.5, 0.5])

    # 可选下采样（加速渲染）
    if subsample and subsample < n:
        indices = np.random.choice(n, subsample, replace=False)
        xyz = xyz[indices]
        colors = colors[indices]
        sem_ids = sem_ids[indices]

    return xyz, colors, sem_ids, sem_dict, color_map, name_map


def render_view(ax, xyz, colors, elev, azim, title, point_size=0.3):
    """在给定角度渲染点云到 matplotlib 3D 轴"""
    ax.clear()
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=colors, s=point_size, alpha=0.6, edgecolors='none')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_aspect('equal')
    # 隐藏网格让点云更突出
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.2)


def fig_to_image(fig):
    """matplotlib figure 转 numpy 图像数组"""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    return buf


def generate_gif(xyz, colors, output_path, n_frames=60, resolution=800, point_size=0.3):
    """生成 360° 旋转 GIF"""
    print(f"Generating turntable GIF ({n_frames} frames)...")
    dpi = 100
    figsize = (resolution / dpi, resolution / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    frames = []
    for i in range(n_frames):
        azim = (360 / n_frames) * i
        render_view(ax, xyz, colors, elev=25, azim=azim, title='', point_size=point_size)
        ax.set_title('')
        # 黑色背景
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')

        frame = fig_to_image(fig)
        frames.append(frame)

        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=15, loop=0)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def generate_direction_images(xyz, colors, output_dir, resolution=800, point_size=0.3):
    """生成 6 个方向的静态预览图"""
    views = {
        'front':       (0, 0),
        'back':        (0, 180),
        'left':        (0, 90),
        'right':       (0, -90),
        'top':         (90, 0),
        'bottom':      (-90, 0),
    }

    dpi = 100
    figsize = (resolution / dpi, resolution / dpi)
    paths = {}

    print(f"Generating {len(views)} direction images...")
    for name, (elev, azim) in views.items():
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        render_view(ax, xyz, colors, elev=elev, azim=azim,
                    title=name.replace('_', ' ').title(), point_size=point_size)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        paths[name] = path
        print(f"  Saved: {path}")

    return paths


def generate_review_sheet(xyz, colors, sem_dict, color_map, name_map, output_path, resolution=800, point_size=0.3):
    """生成拼接总览图：3x2 方向图 + 颜色图例"""
    views = [
        ('Front',       0,   0),
        ('Back',        0, 180),
        ('Left',        0,  90),
        ('Right',       0, -90),
        ('Top',        90,   0),
        ('Bottom',    -90,   0),
    ]

    fig = plt.figure(figsize=(18, 12), dpi=100, facecolor='white')

    # 左侧 3x2 方向视图
    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 4, [1,2,3,5,6,7][i], projection='3d')
        render_view(ax, xyz, colors, elev=elev, azim=azim, title=title, point_size=point_size)

    # 右侧颜色图例
    ax_legend = fig.add_subplot(1, 4, 4)
    ax_legend.set_axis_off()
    ax_legend.set_title('Semantic Components', fontsize=12, fontweight='bold')

    legend_patches = []
    for comp in sem_dict['components']:
        sid = comp['semantic_id']
        c = comp['default_visual'].get('color', [0.5, 0.5, 0.5])
        label = f"[{sid}] {comp['name']}"
        legend_patches.append(Patch(facecolor=c, edgecolor='gray', label=label))

    ax_legend.legend(handles=legend_patches, loc='center', fontsize=9,
                     frameon=True, fancybox=True, shadow=True)

    fig.suptitle(f"Review: {sem_dict['dataset']} ({sem_dict['total_gaussians']:,} gaussians)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved review sheet: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate review visualizations for unified PLY")
    parser.add_argument("--ply", required=True, help="Path to unified_scene.ply")
    parser.add_argument("--dict", required=True, help="Path to semantic_dict.json")
    parser.add_argument("--output_dir", default="review", help="Output directory")
    parser.add_argument("--gif_frames", type=int, default=60, help="Number of GIF frames")
    parser.add_argument("--resolution", type=int, default=800, help="Image resolution")
    parser.add_argument("--subsample", type=int, default=50000,
                        help="Max points to render (for speed). 0=all")
    parser.add_argument("--point_size", type=float, default=0.5, help="Point size in scatter plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subsample = args.subsample if args.subsample > 0 else None
    xyz, colors, sem_ids, sem_dict, color_map, name_map = \
        load_colored_points(args.ply, args.dict, subsample=subsample)

    print(f"\nLoaded {len(xyz):,} points ({len(sem_dict['components'])} components)")
    print(f"Output: {args.output_dir}\n")

    # 1. 旋转 GIF
    gif_path = os.path.join(args.output_dir, "turntable.gif")
    generate_gif(xyz, colors, gif_path,
                 n_frames=args.gif_frames,
                 resolution=args.resolution,
                 point_size=args.point_size)

    # 2. 6 方向图
    generate_direction_images(xyz, colors, args.output_dir,
                              resolution=args.resolution,
                              point_size=args.point_size)

    # 3. 拼接总览图
    sheet_path = os.path.join(args.output_dir, "review_sheet.png")
    generate_review_sheet(xyz, colors, sem_dict, color_map, name_map, sheet_path,
                          resolution=args.resolution,
                          point_size=args.point_size)

    print(f"\n{'='*50}")
    print(f"Review generation complete!")
    print(f"  GIF:    {gif_path}")
    print(f"  Images: {args.output_dir}/ (6 directions)")
    print(f"  Sheet:  {sheet_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
```

### 3.10 `generate_prompt.py`（新增）

```python
"""
generate_prompt.py — 从语义字典自动生成 LLM 系统提示
"""
import json, argparse

def dict_to_prompt(sem_dict):
    comps = []
    for c in sem_dict['components']:
        comps.append({
            "id": c['semantic_id'], "name": c['name'],
            "aliases": c['aliases'], "description": c['description'],
        })

    groups = sem_dict.get('groups', [])
    spatial = sem_dict.get('spatial_relations', [])
    tours = sem_dict.get('guided_tours', [])
    actions = sem_dict['action_schema']['supported_actions']

    prompt = f"""你是沉浸式数据可视化助手。用户在VR环境中探索三维体数据。

当前数据集：{sem_dict['dataset']}
{sem_dict.get('description', '')}

语义组件：
{json.dumps(comps, ensure_ascii=False, indent=2)}

{('分组：' + json.dumps([{"name":g["name"],"aliases":g["aliases"],"children":g["children_names"]} for g in groups], ensure_ascii=False, indent=2)) if groups else ''}

{('空间关系：' + json.dumps(spatial[:10], ensure_ascii=False, indent=2)) if spatial else ''}

{('预置导览：' + json.dumps([{"name":t["name"],"description":t["description"]} for t in tours], ensure_ascii=False, indent=2)) if tours else ''}

输出规则：
1. 仅操作已知对象，输出严格JSON
2. aliases匹配口语化表达到正确semantic_id
3. 分组名可作target（如"fins"表示所有鳍）
4. 支持的action_type: {actions}
5. 用户说"导览/介绍"时，若有对应预置guided_tour则直接调用

JSON输出格式：
{{"actions": [{{"action_type":"...", "target":"name或id或all", "parameters":{{}}, "narration":"可选中文解说"}}]}}
"""
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", required=True)
    parser.add_argument("--output", default="system_prompt.txt")
    args = parser.parse_args()

    with open(args.dict, 'r', encoding='utf-8') as f:
        d = json.load(f)
    prompt = dict_to_prompt(d)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Prompt saved: {args.output} ({len(prompt)} chars)")

if __name__ == "__main__":
    main()
```

### 3.11 `export_pipeline.sh`（新增）

```bash
#!/bin/bash
# 一键：训练全部子模型 → 合并导出 → 生成审查可视化 → 生成LLM提示
DATASET=${1:-"combustion"}
CONFIG="configs/${DATASET}.yaml"
DATA_ROOT="Data/${DATASET}"
OUTPUT_ROOT="output/${DATASET}"
EXPORT_DIR="unity_export/${DATASET}"
ITERATION=30000

echo "========================================="
echo "Pipeline: $DATASET"
echo "========================================="

# Step 1: 训练
echo "[1/4] Training..."
python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
for c in cfg['components']:
    print(f\"{c['tf_id']}|{c['semantic_id']}|{c['name']}\")
" | while IFS='|' read tf_id sem_id sem_name; do
    echo "--- $sem_name ---"
    python train.py \
        -s "$DATA_ROOT/$tf_id" \
        -m "$OUTPUT_ROOT/$sem_name" \
        --semantic_config "$CONFIG" \
        --semantic_name "$sem_name" \
        --semantic_id "$sem_id" \
        --resolution 512 \
        --iterations $ITERATION \
        --save_iterations $ITERATION \
        --test_iterations $ITERATION
done

# Step 2: 合并导出
echo "[2/4] Merging..."
python merge_and_export.py \
    --config "$CONFIG" \
    --model_root "$OUTPUT_ROOT" \
    --output_dir "$EXPORT_DIR" \
    --iteration $ITERATION

# Step 3: 审查可视化
echo "[3/4] Generating review..."
python generate_review.py \
    --ply "$EXPORT_DIR/unified_scene.ply" \
    --dict "$EXPORT_DIR/semantic_dict.json" \
    --output_dir "$EXPORT_DIR/review" \
    --gif_frames 60 --resolution 800 --subsample 50000

# Step 4: LLM 提示
echo "[4/4] Generating prompt..."
python generate_prompt.py \
    --dict "$EXPORT_DIR/semantic_dict.json" \
    --output "$EXPORT_DIR/system_prompt.txt"

echo ""
echo "========================================="
echo "DONE! Output:"
ls -la "$EXPORT_DIR/"
echo ""
ls -la "$EXPORT_DIR/review/"
echo "========================================="
```

### 3.12 `test_llm_interaction.py`（新增）

```python
"""
test_llm_interaction.py — 测试LLM基于语义字典生成动作序列
支持本地规则匹配（快速路径）+ 云端LLM（复杂指令）

用法:
    python test_llm_interaction.py --dict unity_export/combustion/semantic_dict.json
"""
import json, argparse, re

# ===== 本地规则匹配层（P1优化）=====
class LocalMatcher:
    PATTERNS = [
        (r"(?:显示|展示|show)\s*(.+)",       "show"),
        (r"(?:隐藏|关闭|hide)\s*(.+)",       "hide"),
        (r"(?:高亮|强调|highlight)\s*(.+)",   "highlight"),
        (r"(?:只看|只显示|isolate)\s*(.+)",   "isolate"),
        (r"(?:重置|reset|恢复)",              "reset"),
    ]

    def __init__(self, sem_dict):
        self.components = sem_dict['components']
        self.groups = sem_dict.get('groups', [])
        self._alias_map = {}
        for c in self.components:
            for alias in [c['name']] + c.get('aliases', []):
                self._alias_map[alias.lower()] = c['name']
        for g in self.groups:
            for alias in [g['name']] + g.get('aliases', []):
                self._alias_map[alias.lower()] = g['name']

    def resolve(self, text):
        text_lower = text.lower().strip()
        for alias, name in self._alias_map.items():
            if alias in text_lower:
                return name
        return None

    def try_match(self, user_input):
        for pattern, action_type in self.PATTERNS:
            m = re.search(pattern, user_input)
            if m:
                if action_type == "reset":
                    return {"actions": [{"action_type": "reset", "target": "all", "parameters": {}}]}
                target_text = m.group(1).strip()
                target = self.resolve(target_text)
                if target:
                    return {"actions": [{"action_type": action_type, "target": target, "parameters": {}}]}
        return None


# ===== 场景状态追踪（P0优化）=====
class SceneState:
    def __init__(self, sem_dict):
        self.visible = {c['name']: True for c in sem_dict['components']}
        self.highlighted = set()
        self.opacity = {c['name']: c['default_visual'].get('opacity', 1.0) for c in sem_dict['components']}

    def apply(self, actions):
        for a in actions.get('actions', []):
            t = a.get('target', 'all')
            at = a['action_type']
            if at == 'show':
                if t == 'all':
                    for k in self.visible: self.visible[k] = True
                else:
                    self.visible[t] = True
            elif at == 'hide':
                if t == 'all':
                    for k in self.visible: self.visible[k] = False
                else:
                    self.visible[t] = False
            elif at == 'reset':
                for k in self.visible: self.visible[k] = True
                self.highlighted.clear()

    def summary(self):
        vis = [k for k, v in self.visible.items() if v]
        hid = [k for k, v in self.visible.items() if not v]
        return {"visible": vis, "hidden": hid, "highlighted": list(self.highlighted)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", required=True)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="")
    args = parser.parse_args()

    with open(args.dict, 'r', encoding='utf-8') as f:
        sem_dict = json.load(f)

    matcher = LocalMatcher(sem_dict)
    state = SceneState(sem_dict)

    print(f"Dataset: {sem_dict['dataset']} ({len(sem_dict['components'])} components)")
    print(f"Groups: {[g['name'] for g in sem_dict.get('groups', [])]}")
    print(f"\nType commands (local matching mode). 'quit' to exit.\n")

    # 尝试加载 OpenAI（可选）
    llm_client = None
    if args.api_key or args.base_url:
        try:
            from openai import OpenAI
            kwargs = {}
            if args.api_key: kwargs['api_key'] = args.api_key
            if args.base_url: kwargs['base_url'] = args.base_url
            llm_client = OpenAI(**kwargs)

            with open(args.dict.replace('semantic_dict.json', 'system_prompt.txt'), 'r') as f:
                system_prompt = f.read()
            print("[LLM mode enabled: local match → cloud fallback]\n")
        except Exception as e:
            print(f"[LLM not available: {e}. Local-only mode.]\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # 1. 本地快速匹配
        result = matcher.try_match(user_input)
        if result:
            print(f"[LOCAL MATCH] {json.dumps(result, ensure_ascii=False, indent=2)}")
            state.apply(result)
            print(f"[STATE] {json.dumps(state.summary(), ensure_ascii=False)}\n")
            continue

        # 2. 云端 LLM（如果可用）
        if llm_client:
            print("[Calling LLM...]")
            messages = [{"role": "system", "content": system_prompt}]
            # 附带场景状态
            state_msg = f"当前场景状态: {json.dumps(state.summary(), ensure_ascii=False)}"
            messages.append({"role": "system", "content": state_msg})
            messages.extend(history)
            messages.append({"role": "user", "content": user_input})

            try:
                resp = llm_client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages,
                    temperature=0.3, response_format={"type": "json_object"},
                )
                result_text = resp.choices[0].message.content
                result = json.loads(result_text)
                print(f"[LLM] {json.dumps(result, ensure_ascii=False, indent=2)}")
                state.apply(result)
                print(f"[STATE] {json.dumps(state.summary(), ensure_ascii=False)}\n")
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": result_text})
            except Exception as e:
                print(f"[LLM ERROR] {e}\n")
        else:
            print("[No local match, LLM not available. Try simpler command.]\n")


if __name__ == "__main__":
    main()
```

---

## 第四部分：最终输出产物检查清单

运行 `export_pipeline.sh combustion` 后，完整输出如下：

```
unity_export/combustion/
├── unified_scene.ply           ← 统一PLY（含semantic_id），标准3DGS+扩展
├── semantic_dict.json          ← 语义字典（组件+分组+空间关系+导览+schema）
├── system_prompt.txt           ← 自动生成的LLM系统提示
│
└── review/                     ← ★审查可视化文件夹
    ├── turntable.gif           ← 360°旋转动图，按语义默认颜色着色
    ├── front.png               ← 正前方视图
    ├── back.png                ← 正后方视图
    ├── left.png                ← 左侧视图
    ├── right.png               ← 右侧视图
    ├── top.png                 ← 顶部俯视图
    ├── bottom.png              ← 底部仰视图
    └── review_sheet.png        ← 拼接总览（6视图+颜色图例）
```

每个文件的验收标准：

| 产物 | 验收标准 |
|------|---------|
| `unified_scene.ply` | PLY header 中包含 `property uint8 semantic_id`；总高斯数 = 各子模型之和 |
| `semantic_dict.json` | 含 components/groups/spatial_relations/guided_tours/action_schema 全部字段 |
| `system_prompt.txt` | 包含完整组件表、分组、空间关系、输出格式说明 |
| `turntable.gif` | 60帧流畅旋转，各组件颜色与字典 default_visual.color 一致 |
| `review_sheet.png` | 6个视图清晰可辨，右侧图例颜色与点云着色对应 |
| 6张方向图 | 覆盖前/后/左/右/上/下，标题标注正确 |
