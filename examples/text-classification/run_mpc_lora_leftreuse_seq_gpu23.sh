#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK_NAME="${TASK_NAME:-sst2}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
GPU_IDS="${GPU_IDS:-2,3}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-64}"
LEN_DATA="${LEN_DATA:-64}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"

REUSE_LOG_EVERY_STEPS="${REUSE_LOG_EVERY_STEPS:-1}"
SHARED_LEFT_MIN_FANOUT="${SHARED_LEFT_MIN_FANOUT:-3}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_PREFIX="${RUN_PREFIX:-mpc_lora_seq_gpu23_${TASK_NAME}_${STAMP}}"
LOG_DIR="logs/${RUN_PREFIX}"
mkdir -p "${LOG_DIR}"

run_case() {
  local case_name="$1"
  local reuse_mask="$2"
  local log_file="${LOG_DIR}/${case_name}.log"
  local run_tag="${RUN_PREFIX}_${case_name}"

  echo "[run] case=${case_name} run_tag=${run_tag} gpu_ids=${GPU_IDS} steps=${MAX_TRAIN_STEPS}"
  echo "[run] log=${log_file}"

  local -a env_args=(
    "RUN_TAG=${run_tag}"
    "TASK_NAME=${TASK_NAME}"
    "MODEL_NAME=${MODEL_NAME}"
    "GPU_IDS=${GPU_IDS}"
    "SEED=${SEED}"
    "MAX_LENGTH=${MAX_LENGTH}"
    "LEN_DATA=${LEN_DATA}"
    "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
    "LOG_EVERY_STEPS=${LOG_EVERY_STEPS}"
    "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
    "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
    "SKIP_PRIVATE_EVAL=1"
    "SKIP_PLAIN_EVAL=1"
    "REUSE_PROFILE=1"
    "REUSE_LOG_EVERY_STEPS=${REUSE_LOG_EVERY_STEPS}"
    "EXPERIMENTAL_REUSE_MASK=${reuse_mask}"
  )

  if [[ "${reuse_mask}" == "1" ]]; then
    env_args+=(
      "REUSE_MODE=SHARED_LEFT"
      "SHARED_LEFT_MIN_FANOUT=${SHARED_LEFT_MIN_FANOUT}"
    )
  fi

  env "${env_args[@]}" \
    bash test_bert_base_comm_loraxs_lora.sh 2>&1 | tee "${log_file}"

  echo "[done] case=${case_name}"
  echo "[done] summary=eval_private/${TASK_NAME}/${run_tag}/train_eval_summary.json"
}

run_case "no_leftreuse_10step" "0"
run_case "leftreuse_10step" "1"

echo "[all-done] logs=${LOG_DIR}"
echo "[all-done] no-reuse summary=eval_private/${TASK_NAME}/${RUN_PREFIX}_no_leftreuse_10step/train_eval_summary.json"
echo "[all-done] left-reuse summary=eval_private/${TASK_NAME}/${RUN_PREFIX}_leftreuse_10step/train_eval_summary.json"
