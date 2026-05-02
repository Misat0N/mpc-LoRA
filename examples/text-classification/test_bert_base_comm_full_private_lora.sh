#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${TASK_NAME:-sst2}"
MODEL_NAME="${MODEL_NAME:-prajjwal1/bert-tiny}"
GPU_IDS="${GPU_IDS:-0,1}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-64}"
LEN_DATA="${LEN_DATA:--1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1250}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:--1}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-2000}"

TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-10000}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:--1}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-query,value}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
MOMENTUM="${MOMENTUM:-0.9}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"

RUN_TAG="${RUN_TAG:-full_private_lora_$(date +%Y%m%d_%H%M%S)}"
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
if [[ "${EXPERIMENTAL_REUSE_MASK:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--experimental_reuse_mask --reuse_mode "${REUSE_MODE:-SHARED_LEFT}")
fi

echo "[full-private-lora] output_dir=${OUT_DIR}"

"${PYTHON_BIN:-python}" run_glue_private_full_private_lora_train.py \
  --model_name_or_path "${MODEL_NAME}" \
  --task_name "${TASK_NAME}" \
  --gpu_ids "${GPU_IDS}" \
  --seed "${SEED}" \
  --len_data "${LEN_DATA}" \
  --max_length "${MAX_LENGTH}" \
  --train_max_samples "${TRAIN_MAX_SAMPLES}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --eval_every_steps "${EVAL_EVERY_STEPS}" \
  --log_every_steps "${LOG_EVERY_STEPS}" \
  --eval_max_steps "${EVAL_MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --optimizer sgd \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --learning_rate "${LEARNING_RATE}" \
  --momentum "${MOMENTUM}" \
  --lr_scheduler_type linear \
  --warmup_ratio "${WARMUP_RATIO}" \
  --trainable_scope lora \
  --private_loss_mode manual \
  --softmax_method ode \
  --softmax_ode_iter_num 16 \
  --softmax_ode_iter_source fixed \
  --softmax_ode_clip true \
  --softmax_ode_center_by_max true \
  --softmax_ode_zero_masked true \
  --softmax_ode_mask_margin 100.0 \
  --ce_softmax_method ideal \
  --sqrt_method NR \
  --sqrt_nr_iters 10 \
  --sqrt_nr_initial_exp_iterations 7 \
  --sqrt_nr_linear_divisor 1536.0 \
  --shuffle_train \
  --skip_plain_eval \
  --output_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[full-private-lora] done: ${OUT_DIR}"
