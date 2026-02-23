#!/usr/bin/env bash
set -euo pipefail

# Overnight preset for true MPC-LoRA finetuning.
# You can override any variable from environment when launching.
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1200}"
export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
export EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-512}"
export MAX_LENGTH="${MAX_LENGTH:-64}"
export LEN_DATA="${LEN_DATA:-64}"
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-query,value}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export MOMENTUM="${MOMENTUM:-0.9}"
export RUN_TAG="${RUN_TAG:-mpc_lora_overnight_$(date +%Y%m%d_%H%M%S)}"

bash test_bert_base_comm_mpc_lora.sh
