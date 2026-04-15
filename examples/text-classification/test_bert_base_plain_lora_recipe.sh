#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${TASK_NAME:-sst2}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-128}"
LEN_DATA="${LEN_DATA:-128}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-600}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-256}"

TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:--1}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
CLASSIFIER_LEARNING_RATE="${CLASSIFIER_LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"

RUN_TAG="${RUN_TAG:-plain_lora_recipe_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="eval_plain/${TASK_NAME}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ "${SKIP_MODEL_SAVE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_model_save)
fi

python run_glue_plain_lora_recipe_train.py \
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
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --learning_rate "${LEARNING_RATE}" \
  --classifier_learning_rate "${CLASSIFIER_LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --lr_scheduler_type linear \
  --warmup_ratio "${WARMUP_RATIO}" \
  --output_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[plain-lora-recipe] done: ${OUT_DIR}"
