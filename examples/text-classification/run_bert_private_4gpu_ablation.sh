#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK_NAME="${TASK_NAME:-sst2}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
SEED="${SEED:-42}"

GPU_PAIR_A="${GPU_PAIR_A:-0,1}"
GPU_PAIR_B="${GPU_PAIR_B:-2,3}"

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
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
MOMENTUM="${MOMENTUM:-0.9}"

PRINT_COMM_COST="${PRINT_COMM_COST:-0}"
ALLOW_SPAM_LOGS="${ALLOW_SPAM_LOGS:-0}"
REUSE_PROFILE="${REUSE_PROFILE:-1}"
SKIP_PRIVATE_EVAL="${SKIP_PRIVATE_EVAL:-0}"
SKIP_PLAIN_EVAL="${SKIP_PLAIN_EVAL:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-bert_private_4gpu_ablation_${TASK_NAME}_${TS}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-eval_private/${TASK_NAME}/${RUN_NAME}}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

LAUNCH_TSV="${LOG_DIR}/launch_manifest.tsv"
{
  echo -e "wave\tgroup\tgpu_pair\tpid_file\tlog_path\toutput_dir\tentrypoint"
} > "${LAUNCH_TSV}"

cleanup_running_groups() {
  local groups=(
    "full_private_lora"
    "public_backbone_lora"
    "public_backbone_loraxs"
    "public_backbone_loraxs_shared_left"
  )
  for group in "${groups[@]}"; do
    local pid_file="${LOG_DIR}/${group}.pid"
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    fi
  done
}

on_term() {
  echo "[driver] received termination signal, stopping active groups" >&2
  cleanup_running_groups
  exit 1
}

trap on_term INT TERM

append_optional_flags() {
  local -n _cmd_ref=$1
  if [[ "${PRINT_COMM_COST}" == "1" ]]; then
    _cmd_ref+=(--print_comm_cost)
  fi
  if [[ "${ALLOW_SPAM_LOGS}" == "1" ]]; then
    _cmd_ref+=(--allow_spam_logs)
  fi
  if [[ "${REUSE_PROFILE}" == "1" ]]; then
    _cmd_ref+=(--reuse_profile)
  fi
  if [[ "${SKIP_PRIVATE_EVAL}" == "1" ]]; then
    _cmd_ref+=(--skip_private_eval)
  fi
  if [[ "${SKIP_PLAIN_EVAL}" == "1" ]]; then
    _cmd_ref+=(--skip_plain_eval)
  fi
}

start_group() {
  local wave_name="$1"
  local group_name="$2"
  local gpu_pair="$3"
  local entrypoint="$4"
  local encrypted_keywords="$5"

  local output_dir="${OUT_ROOT}/${group_name}"
  local log_path="${LOG_DIR}/${group_name}.log"
  local pid_file="${LOG_DIR}/${group_name}.pid"
  mkdir -p "${output_dir}"

  local -a cmd=(
    python "${entrypoint}"
    --model_name_or_path "${MODEL_NAME}"
    --task_name "${TASK_NAME}"
    --gpu_ids "0,1"
    --seed "${SEED}"
    --pad_to_max_length
    --len_data "${LEN_DATA}"
    --max_length "${MAX_LENGTH}"
    --train_max_samples "${TRAIN_MAX_SAMPLES}"
    --eval_max_samples "${EVAL_MAX_SAMPLES}"
    --max_train_steps "${MAX_TRAIN_STEPS}"
    --log_every_steps "${LOG_EVERY_STEPS}"
    --eval_max_steps "${EVAL_MAX_STEPS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --lora_target_modules "${LORA_TARGET_MODULES}"
    --learning_rate "${LEARNING_RATE}"
    --momentum "${MOMENTUM}"
    --output_dir "${output_dir}"
  )

  if [[ -n "${encrypted_keywords}" ]]; then
    cmd+=(--public_non_lora_weights --encrypted_param_keywords "${encrypted_keywords}")
  fi

  append_optional_flags cmd

  env CUDA_VISIBLE_DEVICES="${gpu_pair}" PYTHONUNBUFFERED=1 "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${pid_file}"
  echo -e "${wave_name}\t${group_name}\t${gpu_pair}\t${pid_file}\t${log_path}\t${output_dir}\t${entrypoint}" >> "${LAUNCH_TSV}"

  echo "[launch] wave=${wave_name} group=${group_name} gpus=${gpu_pair} pid=${pid}"
  echo "         log=${log_path}"
  echo "         out=${output_dir}"
}

wait_group() {
  local group_name="$1"
  local pid_file="${LOG_DIR}/${group_name}.pid"
  local pid
  pid="$(cat "${pid_file}")"
  if wait "${pid}"; then
    echo "[done] group=${group_name} pid=${pid}"
    return 0
  fi

  local status=$?
  echo "[failed] group=${group_name} pid=${pid} exit_code=${status}" >&2
  return "${status}"
}

run_wave() {
  local wave_name="$1"
  local group_a="$2"
  local gpu_pair_a="$3"
  local entry_a="$4"
  local encrypt_a="$5"
  local group_b="$6"
  local gpu_pair_b="$7"
  local entry_b="$8"
  local encrypt_b="$9"

  echo "[wave] start ${wave_name}"
  start_group "${wave_name}" "${group_a}" "${gpu_pair_a}" "${entry_a}" "${encrypt_a}"
  start_group "${wave_name}" "${group_b}" "${gpu_pair_b}" "${entry_b}" "${encrypt_b}"

  local status=0
  wait_group "${group_a}" || status=1
  wait_group "${group_b}" || status=1

  if [[ "${status}" -ne 0 ]]; then
    echo "[wave] ${wave_name} failed" >&2
    exit 1
  fi

  echo "[wave] done ${wave_name}"
}

echo "[driver] run_name=${RUN_NAME}"
echo "[driver] log_dir=${LOG_DIR}"
echo "[driver] out_root=${OUT_ROOT}"
echo "[driver] gpu_pair_a=${GPU_PAIR_A} gpu_pair_b=${GPU_PAIR_B}"
echo "[driver] task=${TASK_NAME} model=${MODEL_NAME} steps=${MAX_TRAIN_STEPS}"

run_wave \
  "wave1" \
  "full_private_lora" \
  "${GPU_PAIR_A}" \
  "run_glue_private_full_private_lora_train.py" \
  "" \
  "public_backbone_lora" \
  "${GPU_PAIR_B}" \
  "run_glue_private_loraxs_lora_train.py" \
  "lora_A.,lora_B.,classifier.,score."

run_wave \
  "wave2" \
  "public_backbone_loraxs" \
  "${GPU_PAIR_A}" \
  "run_glue_private_loraxs_train.py" \
  "lora_latent.,classifier.,score." \
  "public_backbone_loraxs_shared_left" \
  "${GPU_PAIR_B}" \
  "run_glue_private_loraxs_shared_left_train.py" \
  "lora_latent.,classifier.,score."

python summarize_parallel_bert_private_ablation.py --output_root "${OUT_ROOT}" --log_dir "${LOG_DIR}"

echo "[driver] summary saved"
echo "  ${OUT_ROOT}/ablation_summary.json"
echo "  ${OUT_ROOT}/ablation_summary.tsv"
