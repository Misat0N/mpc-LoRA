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
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-query,value,attention.output.dense,output.dense}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
CLASSIFIER_LEARNING_RATE="${CLASSIFIER_LEARNING_RATE:-2e-4}"
MOMENTUM="${MOMENTUM:-0.9}"
ENCRYPTED_PARAM_KEYWORDS="${ENCRYPTED_PARAM_KEYWORDS:-lora_A.,lora_B.,classifier.,score.}"

RUN_TAG="${RUN_TAG:-lora_recipe_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="eval_private/${TASK_NAME}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
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
if [[ "${REUSE_PROFILE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse_profile)
fi

echo "[lora-recipe] output_dir=${OUT_DIR}"

python run_glue_private_loraxs_recipe_lora_train.py \
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
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --learning_rate "${LEARNING_RATE}" \
  --classifier_learning_rate "${CLASSIFIER_LEARNING_RATE}" \
  --momentum "${MOMENTUM}" \
  --public_non_lora_weights \
  --encrypted_param_keywords "${ENCRYPTED_PARAM_KEYWORDS}" \
  --output_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[lora-recipe] done: ${OUT_DIR}"
