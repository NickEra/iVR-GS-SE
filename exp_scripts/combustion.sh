#!/bin/bash
set -o errexit
set -o nounset

# Train combustion components with semantic metadata
DATASET="combustion"
CONFIG="configs/${DATASET}.yaml"
DATA_ROOT="Data/${DATASET}"
OUTPUT_ROOT="output/${DATASET}"
ITERATION=30000

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Missing config: $CONFIG"
    exit 1
fi

python -c "
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
for c in cfg['components']:
    print(f\"{c['tf_id']}|{c['semantic_id']}|{c['name']}\")
" | while IFS='|' read -r tf_id sem_id sem_name; do
    echo "========================================"
    echo "Training: $sem_name (TF=$tf_id, semantic_id=$sem_id)"
    echo "========================================"

    python train.py \
        --eval \
        -s "$DATA_ROOT/$tf_id" \
        -m "$OUTPUT_ROOT/$sem_name/3dgs" \
        --semantic_config "$CONFIG" \
        --semantic_name "$sem_name" \
        --semantic_id "$sem_id" \
        --resolution 512 \
        --iterations "$ITERATION" \
        --save_interval "$ITERATION" \
        --test_interval "$ITERATION" \
        --checkpoint_interval "$ITERATION" \
        --lambda_normal_render_depth 0.01 \
        --lambda_opacity 0.1 \
        --densification_interval 500 \
        --densify_grad_normal_threshold 0.000004
done

echo "All semantic components finished."
echo "Next step:"
echo "python merge_and_export.py --config $CONFIG --model_root $OUTPUT_ROOT --output_dir unity_export/$DATASET --iteration $ITERATION"