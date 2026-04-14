#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${TASK_NAME:-sst2}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
GPU_IDS="${GPU_IDS:-0,1}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-64}"
LEN_DATA="${LEN_DATA:-64}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-120}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-5}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-128}"

TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:--1}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
CLASSIFIER_LEARNING_RATE="${CLASSIFIER_LEARNING_RATE:-}"
MOMENTUM="${MOMENTUM:-0.9}"

RUN_TAG="${RUN_TAG:-minimal_low_rank_ablation_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="eval_compare/${TASK_NAME}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${CLASSIFIER_LEARNING_RATE}" ]]; then
  EXTRA_ARGS+=(--classifier_learning_rate "${CLASSIFIER_LEARNING_RATE}")
fi

python run_bert_minimal_low_rank_ablation.py \
  --task_name "${TASK_NAME}" \
  --model_name_or_path "${MODEL_NAME}" \
  --gpu_ids "${GPU_IDS}" \
  --seed "${SEED}" \
  --max_length "${MAX_LENGTH}" \
  --len_data "${LEN_DATA}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --log_every_steps "${LOG_EVERY_STEPS}" \
  --eval_max_steps "${EVAL_MAX_STEPS}" \
  --train_max_samples "${TRAIN_MAX_SAMPLES}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --learning_rate "${LEARNING_RATE}" \
  --momentum "${MOMENTUM}" \
  --output_root "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[minimal-low-rank-ablation] done: ${OUT_DIR}"
