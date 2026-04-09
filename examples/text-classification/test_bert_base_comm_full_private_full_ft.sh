#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${TASK_NAME:-sst2}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
GPU_IDS="${GPU_IDS:-0,1}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-64}"
LEN_DATA="${LEN_DATA:-64}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-300}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-5}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-256}"

TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:--1}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"

RUN_TAG="${RUN_TAG:-full_private_full_ft_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="eval_private/${TASK_NAME}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo "[full-private-full-ft] output_dir=${OUT_DIR}"

python run_glue_private_full_ft_baseline.py \
  --model_name_or_path "${MODEL_NAME}" \
  --task_name "${TASK_NAME}" \
  --gpu_ids "${GPU_IDS}" \
  --seed "${SEED}" \
  --pad_to_max_length \
  --len_data "${LEN_DATA}" \
  --max_length "${MAX_LENGTH}" \
  --train_max_samples "${TRAIN_MAX_SAMPLES}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --log_every_steps "${LOG_EVERY_STEPS}" \
  --eval_max_steps "${EVAL_MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --output_dir "${OUT_DIR}"

echo "[full-private-full-ft] done: ${OUT_DIR}"
