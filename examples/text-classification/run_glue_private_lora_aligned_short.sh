#!/bin/bash

set -euo pipefail

TASK_NAME=${TASK_NAME:-sst2}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-prajjwal1/bert-tiny}
GPU_IDS=${GPU_IDS:-0,1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-50}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_DIR=${OUTPUT_DIR:-eval_private_control_bert_tiny_lora_sgd_t10000_e872_step${MAX_TRAIN_STEPS}_ode_ceideal_nr10_initexp7_div1536_qv/${TASK_NAME}/}
SKIP_EVAL=${SKIP_EVAL:-0}
MAX_EVAL_STEPS=${MAX_EVAL_STEPS:--1}

HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE
export TRANSFORMERS_OFFLINE

EXTRA_ARGS=()
if [[ "$SKIP_EVAL" == "1" || "$SKIP_EVAL" == "true" ]]; then
  EXTRA_ARGS+=(--skip_eval)
fi
if [[ "$MAX_EVAL_STEPS" != "-1" ]]; then
  EXTRA_ARGS+=(--max_eval_steps "$MAX_EVAL_STEPS")
fi

"$PYTHON_BIN" run_glue_private_mpc_control_no_lora_eval.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --task_name "$TASK_NAME" \
  --max_length 64 \
  --pad_to_max_length \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --optimizer sgd \
  --learning_rate 0.003 \
  --momentum 0.9 \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --warmup_ratio 0.06 \
  --eval_every_steps 2000 \
  --log_every_steps 10 \
  --train_max_samples 10000 \
  --shuffle_train \
  --seed 42 \
  --trainable_scope lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --lora_target_modules query,value \
  --private_loss_mode manual \
  --softmax_method ode \
  --softmax_ode_iter_num 16 \
  --softmax_ode_clip true \
  --softmax_ode_center_by_max true \
  --softmax_ode_zero_masked true \
  --softmax_ode_mask_margin 100 \
  --ce_softmax_method ideal \
  --sqrt_method NR \
  --sqrt_nr_iters 10 \
  --sqrt_nr_initial_exp_iterations 7 \
  --sqrt_nr_linear_divisor 1536 \
  --gpu_ids "$GPU_IDS" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
