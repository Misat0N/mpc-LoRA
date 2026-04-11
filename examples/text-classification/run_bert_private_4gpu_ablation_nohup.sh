#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK_NAME="${TASK_NAME:-sst2}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-bert_private_4gpu_ablation_${TASK_NAME}_${TS}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-eval_private/${TASK_NAME}/${RUN_NAME}}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

LOG_FILE="${LOG_DIR}/driver.log"
PID_FILE="${LOG_DIR}/driver.pid"

nohup env RUN_NAME="${RUN_NAME}" LOG_DIR="${LOG_DIR}" OUT_ROOT="${OUT_ROOT}" \
  TASK_NAME="${TASK_NAME}" \
  MODEL_NAME="${MODEL_NAME:-bert-base-uncased}" \
  SEED="${SEED:-42}" \
  GPU_PAIR_A="${GPU_PAIR_A:-0,1}" \
  GPU_PAIR_B="${GPU_PAIR_B:-2,3}" \
  MAX_LENGTH="${MAX_LENGTH:-64}" \
  LEN_DATA="${LEN_DATA:-64}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-300}" \
  LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-5}" \
  EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-256}" \
  TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}" \
  EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:--1}" \
  PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  LORA_R="${LORA_R:-8}" \
  LORA_ALPHA="${LORA_ALPHA:-16}" \
  LORA_DROPOUT="${LORA_DROPOUT:-0.0}" \
  LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-query,key,value}" \
  LEARNING_RATE="${LEARNING_RATE:-5e-4}" \
  MOMENTUM="${MOMENTUM:-0.9}" \
  PRINT_COMM_COST="${PRINT_COMM_COST:-0}" \
  ALLOW_SPAM_LOGS="${ALLOW_SPAM_LOGS:-0}" \
  REUSE_PROFILE="${REUSE_PROFILE:-1}" \
  SKIP_PRIVATE_EVAL="${SKIP_PRIVATE_EVAL:-0}" \
  SKIP_PLAIN_EVAL="${SKIP_PLAIN_EVAL:-0}" \
  bash run_bert_private_4gpu_ablation.sh > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "[nohup] started 4-group ablation with 2 GPUs per group"
echo "  pid: $(cat "${PID_FILE}")"
echo "  run_name: ${RUN_NAME}"
echo "  driver_log: ${LOG_FILE}"
echo "  log_dir: ${LOG_DIR}"
echo "  out_root: ${OUT_ROOT}"
echo ""
echo "gpu allocation:"
echo "  wave1: full_private_lora on ${GPU_PAIR_A:-0,1}"
echo "  wave1: public_backbone_lora on ${GPU_PAIR_B:-2,3}"
echo "  wave2: public_backbone_loraxs on ${GPU_PAIR_A:-0,1}"
echo "  wave2: public_backbone_loraxs_shared_left on ${GPU_PAIR_B:-2,3}"
echo ""
echo "follow logs:"
echo "  tail -f ${LOG_FILE}"
echo "  tail -f ${LOG_DIR}/full_private_lora.log"
echo "  tail -f ${LOG_DIR}/public_backbone_lora.log"
echo "  tail -f ${LOG_DIR}/public_backbone_loraxs.log"
echo "  tail -f ${LOG_DIR}/public_backbone_loraxs_shared_left.log"
echo ""
echo "stop:"
echo "  kill $(cat "${PID_FILE}")"
