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

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-query,key,value}"
LORA_B_EXPORT_SENTINEL_SCALE="${LORA_B_EXPORT_SENTINEL_SCALE:-1e-9}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
MOMENTUM="${MOMENTUM:-0.9}"
ENCRYPTED_PARAM_KEYWORDS="${ENCRYPTED_PARAM_KEYWORDS:-lora_A.,lora_B.,classifier.,score.}"

RUN_TAG="${RUN_TAG:-loraxs_lora_$(date +%Y%m%d_%H%M%S)}"
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
if [[ "${EXPERIMENTAL_REUSE_MASK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--experimental_reuse_mask)
fi
if [[ "${REUSE_PROFILE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse_profile)
fi
if [[ "${MATMUL_PROFILE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--matmul_profile)
fi
if [[ -n "${MATMUL_LOG_EVERY_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--matmul_log_every_steps "${MATMUL_LOG_EVERY_STEPS}")
fi
if [[ -n "${REUSE_MODE:-}" ]]; then
  EXTRA_ARGS+=(--reuse_mode "${REUSE_MODE}")
fi
if [[ -n "${SHARED_LEFT_MIN_FANOUT:-}" ]]; then
  EXTRA_ARGS+=(--shared_left_min_fanout "${SHARED_LEFT_MIN_FANOUT}")
fi
if [[ -n "${SHARED_LEFT_LOG_GROUPS:-}" ]]; then
  EXTRA_ARGS+=(--shared_left_log_groups "${SHARED_LEFT_LOG_GROUPS}")
fi
if [[ -n "${REUSE_LOG_EVERY_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--reuse_log_every_steps "${REUSE_LOG_EVERY_STEPS}")
fi

echo "[loraxs-lora] output_dir=${OUT_DIR}"
echo "[loraxs-lora] task=${TASK_NAME} model=${MODEL_NAME} gpu_ids=${GPU_IDS} steps=${MAX_TRAIN_STEPS}"
echo "[loraxs-lora] targets=${LORA_TARGET_MODULES} lora_r=${LORA_R} lora_alpha=${LORA_ALPHA} lora_dropout=${LORA_DROPOUT}"
echo "[loraxs-lora] lora_b_export_sentinel_scale=${LORA_B_EXPORT_SENTINEL_SCALE}"
echo "[loraxs-lora] encrypted_keywords=${ENCRYPTED_PARAM_KEYWORDS}"

python run_glue_private_loraxs_lora_train.py \
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
  --lora_b_export_sentinel_scale "${LORA_B_EXPORT_SENTINEL_SCALE}" \
  --learning_rate "${LEARNING_RATE}" \
  --momentum "${MOMENTUM}" \
  --public_non_lora_weights \
  --encrypted_param_keywords "${ENCRYPTED_PARAM_KEYWORDS}" \
  --output_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[loraxs-lora] done: ${OUT_DIR}"
