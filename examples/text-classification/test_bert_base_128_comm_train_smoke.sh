# export TASK_NAME=qnli
export TASK_NAME=sst2


python run_glue_private_train_smoke.py \
  --model_name_or_path andeskyl/bert-base-cased-$TASK_NAME \
  --task_name $TASK_NAME \
  --len_data 128 \
  --num_data -1 \
  --max_train_steps 100 \
  --log_every_steps 10 \
  --eval_max_steps -1 \
  --per_device_train_batch_size 1 \
  --max_length 128 \
  --per_device_eval_batch_size 1 \
  --output_dir eval_private/$TASK_NAME/
