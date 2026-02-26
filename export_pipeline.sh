#!/bin/bash
set -o errexit
set -o nounset

# 一键：训练全部子模型 -> 合并导出 -> 生成审查可视化 -> 生成LLM提示词
DATASET=${1:-"combustion"}
CONFIG="configs/${DATASET}.yaml"
DATA_ROOT="Data/${DATASET}"
OUTPUT_ROOT="output/${DATASET}"
EXPORT_DIR="unity_export/${DATASET}"
ITERATION=30000

echo "========================================="
echo "Pipeline: $DATASET"
echo "========================================="

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

echo ""
echo "[1/4] Training semantic sub-models..."
python -c "
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
for c in cfg['components']:
    print(f\"{c['tf_id']}|{c['semantic_id']}|{c['name']}\")
" | while IFS='|' read -r tf_id sem_id sem_name; do
    echo "--- Training: $sem_name ---"
    python train.py \
        --eval \
        -s "$DATA_ROOT/$tf_id" \
        -m "$OUTPUT_ROOT/$sem_name" \
        --semantic_config "$CONFIG" \
        --semantic_name "$sem_name" \
        --semantic_id "$sem_id" \
        --resolution 512 \
        --iterations "$ITERATION" \
        --save_interval "$ITERATION" \
        --test_interval "$ITERATION" \
        --checkpoint_interval "$ITERATION"
done

echo ""
echo "[2/4] Merging and exporting..."
python merge_and_export.py \
    --config "$CONFIG" \
    --model_root "$OUTPUT_ROOT" \
    --output_dir "$EXPORT_DIR" \
    --iteration "$ITERATION"

echo ""
echo "[3/4] Generating review assets..."
python generate_review.py \
    --ply "$EXPORT_DIR/unified_scene.ply" \
    --dict "$EXPORT_DIR/semantic_dict.json" \
    --output_dir "$EXPORT_DIR/review" \
    --gif_frames 60 \
    --resolution 800 \
    --subsample 50000

echo ""
echo "[4/4] Generating prompt..."
python generate_prompt.py \
    --dict "$EXPORT_DIR/semantic_dict.json" \
    --output "$EXPORT_DIR/system_prompt.txt"

echo ""
echo "Validating outputs..."
if [ -f "$EXPORT_DIR/unified_scene.ply" ] && \
   [ -f "$EXPORT_DIR/semantic_dict.json" ] && \
   [ -f "$EXPORT_DIR/system_prompt.txt" ] && \
   [ -f "$EXPORT_DIR/review/review_sheet.png" ]; then
    echo "SUCCESS: all outputs generated"
    ls -la "$EXPORT_DIR/"
    echo ""
    ls -la "$EXPORT_DIR/review/"
else
    echo "FAILED: output files missing"
    exit 1
fi
