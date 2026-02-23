#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/mpc_lora_overnight_${TS}.log"
PID_FILE="${LOG_DIR}/mpc_lora_overnight_${TS}.pid"

nohup bash test_bert_base_comm_mpc_lora_overnight.sh > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "[nohup] started"
echo "  pid: $(cat "${PID_FILE}")"
echo "  log: ${LOG_FILE}"
echo "  pid_file: ${PID_FILE}"
echo ""
echo "follow log:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "stop:"
echo "  kill $(cat "${PID_FILE}")"
