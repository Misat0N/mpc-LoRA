# export TASK_NAME=qnli
export TASK_NAME=sst2

EXPERIMENTAL_REUSE_MASK="${EXPERIMENTAL_REUSE_MASK:-1}"
REUSE_MODE="${REUSE_MODE:-SHARED_LEFT}"
SHARED_LEFT_MIN_FANOUT="${SHARED_LEFT_MIN_FANOUT:-2}"
SHARED_LEFT_LOG_GROUPS="${SHARED_LEFT_LOG_GROUPS:-12}"

EXTRA_ARGS=()
if [[ "${EXPERIMENTAL_REUSE_MASK}" == "1" ]]; then
  EXTRA_ARGS+=(
    --experimental_reuse_mask
    --reuse_mode "${REUSE_MODE}"
    --shared_left_min_fanout "${SHARED_LEFT_MIN_FANOUT}"
    --shared_left_log_groups "${SHARED_LEFT_LOG_GROUPS}"
  )
fi

python run_glue_private_train_smoke.py \
  --model_name_or_path andeskyl/bert-base-cased-$TASK_NAME \
  --task_name $TASK_NAME \
  --len_data 128 \
  --num_data -1 \
  --max_length 128 \
  --per_device_eval_batch_size 1 \
  --output_dir eval_private/$TASK_NAME/ \
  "${EXTRA_ARGS[@]}"
