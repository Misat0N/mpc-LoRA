#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${TASK_NAME:-sst2}"
GPU_IDS="${GPU_IDS:-0,1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-512}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-256}"
MAX_LENGTH="${MAX_LENGTH:-64}"
LEN_DATA="${LEN_DATA:-64}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-64}"
STEPS_LIST="${STEPS_LIST:-10 30 50}"

BASE_OUT="eval_private/${TASK_NAME}/convergence_sweep"
mkdir -p "${BASE_OUT}"

for steps in ${STEPS_LIST}; do
  out_dir="${BASE_OUT}/steps_${steps}"
  echo "[sweep] running max_train_steps=${steps} -> ${out_dir}"

  python run_glue_private_light_train.py \
    --model_name_or_path andeskyl/bert-base-cased-${TASK_NAME} \
    --task_name ${TASK_NAME} \
    --gpu_ids ${GPU_IDS} \
    --pad_to_max_length \
    --len_data ${LEN_DATA} \
    --max_length ${MAX_LENGTH} \
    --train_max_samples ${TRAIN_MAX_SAMPLES} \
    --eval_max_samples ${EVAL_MAX_SAMPLES} \
    --max_train_steps ${steps} \
    --log_every_steps 1 \
    --eval_max_steps ${EVAL_MAX_STEPS} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --train_classifier_only \
    --output_dir "${out_dir}"
done

echo "[sweep] summarizing results from ${BASE_OUT}"
python summarize_convergence.py --base_dir "${BASE_OUT}" --steps ${STEPS_LIST}
