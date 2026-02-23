#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${TASK_NAME:-sst2}"
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

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-query,value}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
MOMENTUM="${MOMENTUM:-0.9}"

RUN_TAG="${RUN_TAG:-mpc_lora_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="eval_private/${TASK_NAME}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ "${FREEZE_CLASSIFIER_HEAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze_classifier_head)
fi
if [[ "${SKIP_PRIVATE_EVAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_private_eval)
fi
if [[ "${SKIP_PLAIN_EVAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_plain_eval)
fi
if [[ "${PRINT_COMM_COST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--print_comm_cost)
fi
if [[ "${ALLOW_SPAM_LOGS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_spam_logs)
fi

echo "[mpc-lora] output_dir=${OUT_DIR}"
echo "[mpc-lora] task=${TASK_NAME} gpu_ids=${GPU_IDS} steps=${MAX_TRAIN_STEPS} len=${LEN_DATA} max_length=${MAX_LENGTH}"
echo "[mpc-lora] lora_r=${LORA_R} lora_alpha=${LORA_ALPHA} lora_dropout=${LORA_DROPOUT} targets=${LORA_TARGET_MODULES}"

python run_glue_private_mpc_lora_train.py \
  --model_name_or_path andeskyl/bert-base-cased-${TASK_NAME} \
  --task_name ${TASK_NAME} \
  --gpu_ids ${GPU_IDS} \
  --seed ${SEED} \
  --pad_to_max_length \
  --len_data ${LEN_DATA} \
  --max_length ${MAX_LENGTH} \
  --train_max_samples ${TRAIN_MAX_SAMPLES} \
  --eval_max_samples ${EVAL_MAX_SAMPLES} \
  --max_train_steps ${MAX_TRAIN_STEPS} \
  --log_every_steps ${LOG_EVERY_STEPS} \
  --eval_max_steps ${EVAL_MAX_STEPS} \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --learning_rate ${LEARNING_RATE} \
  --momentum ${MOMENTUM} \
  --output_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[mpc-lora] done: ${OUT_DIR}"
