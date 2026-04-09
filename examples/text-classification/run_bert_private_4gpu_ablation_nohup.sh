#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK_NAME="${TASK_NAME:-sst2}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
SEED="${SEED:-42}"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=', ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_LIST must contain exactly 4 GPU ids, got: ${GPU_LIST}" >&2
  exit 1
fi

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
POLL_SECONDS="${POLL_SECONDS:-60}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-bert_private_4gpu_ablation_${TASK_NAME}_${TS}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-eval_private/${TASK_NAME}/${RUN_NAME}}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

LAUNCH_TSV="${LOG_DIR}/launch_manifest.tsv"
{
  echo -e "group\tgpu\tpid_file\tlog_path\toutput_dir\tentrypoint"
} > "${LAUNCH_TSV}"

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
  local group_name="$1"
  local gpu_id="$2"
  local entrypoint="$3"
  local encrypted_keywords="$4"

  local output_dir="${OUT_ROOT}/${group_name}"
  local log_path="${LOG_DIR}/${group_name}.log"
  local pid_file="${LOG_DIR}/${group_name}.pid"
  mkdir -p "${output_dir}"

  local -a cmd=(
    python "${entrypoint}"
    --model_name_or_path "${MODEL_NAME}"
    --task_name "${TASK_NAME}"
    --gpu_ids "0"
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

  (
    nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${cmd[@]}" > "${log_path}" 2>&1 &
    echo $! > "${pid_file}"
  )

  local pid
  pid="$(cat "${pid_file}")"
  echo -e "${group_name}\t${gpu_id}\t${pid_file}\t${log_path}\t${output_dir}\t${entrypoint}" >> "${LAUNCH_TSV}"

  echo "[launch] group=${group_name} gpu=${gpu_id} pid=${pid}"
  echo "         log=${log_path}"
  echo "         out=${output_dir}"
}

start_group "full_private_lora" "${GPUS[0]}" "run_glue_private_full_private_lora_train.py" ""
start_group "public_backbone_lora" "${GPUS[1]}" "run_glue_private_loraxs_lora_train.py" "lora_A.,lora_B.,classifier.,score."
start_group "public_backbone_loraxs" "${GPUS[2]}" "run_glue_private_loraxs_train.py" "lora_latent.,classifier.,score."
start_group "public_backbone_loraxs_shared_left" "${GPUS[3]}" "run_glue_private_loraxs_shared_left_train.py" "lora_latent.,classifier.,score."

COLLECTOR_SCRIPT="${LOG_DIR}/collect_when_done.sh"
cat > "${COLLECTOR_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${SCRIPT_DIR}"
OUT_ROOT="${OUT_ROOT}"
LOG_DIR="${LOG_DIR}"
POLL_SECONDS="${POLL_SECONDS}"
GROUPS=(
  "full_private_lora"
  "public_backbone_lora"
  "public_backbone_loraxs"
  "public_backbone_loraxs_shared_left"
)

while true; do
  all_done=1
  for group in "\${GROUPS[@]}"; do
    pid_file="\${LOG_DIR}/\${group}.pid"
    if [[ ! -f "\${pid_file}" ]]; then
      all_done=0
      break
    fi
    pid="\$(cat "\${pid_file}")"
    if kill -0 "\${pid}" 2>/dev/null; then
      all_done=0
      break
    fi
  done

  if [[ "\${all_done}" -eq 1 ]]; then
    cd "\${SCRIPT_DIR}"
    python summarize_parallel_bert_private_ablation.py --output_root "\${OUT_ROOT}" --log_dir "\${LOG_DIR}"
    exit 0
  fi

  sleep "\${POLL_SECONDS}"
done
EOF
chmod +x "${COLLECTOR_SCRIPT}"

nohup bash "${COLLECTOR_SCRIPT}" > "${LOG_DIR}/collector.log" 2>&1 &
echo $! > "${LOG_DIR}/collector.pid"

echo ""
echo "[nohup] started 4-group parallel ablation"
echo "  run_name: ${RUN_NAME}"
echo "  log_dir: ${LOG_DIR}"
echo "  out_root: ${OUT_ROOT}"
echo "  manifest: ${LAUNCH_TSV}"
echo "  collector_pid: $(cat "${LOG_DIR}/collector.pid")"
echo ""
echo "follow logs:"
for group in full_private_lora public_backbone_lora public_backbone_loraxs public_backbone_loraxs_shared_left; do
  echo "  tail -f ${LOG_DIR}/${group}.log"
done
echo ""
echo "stop:"
for group in full_private_lora public_backbone_lora public_backbone_loraxs public_backbone_loraxs_shared_left; do
  echo "  kill \$(cat ${LOG_DIR}/${group}.pid)"
done
echo "  kill \$(cat ${LOG_DIR}/collector.pid)"
