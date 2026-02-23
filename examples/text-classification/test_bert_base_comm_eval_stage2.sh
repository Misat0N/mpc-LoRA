# export TASK_NAME=qnli
export TASK_NAME=sst2


python run_glue_private_light_train.py \
  --model_name_or_path andeskyl/bert-base-cased-$TASK_NAME \
  --task_name $TASK_NAME \
  --gpu_ids 0,1 \
  --pad_to_max_length \
  --len_data 64 \
  --max_length 64 \
  --train_max_samples 512 \
  --eval_max_samples 256 \
  --max_train_steps 10 \
  --log_every_steps 1 \
  --eval_max_steps 64 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --train_classifier_only \
  --output_dir eval_private/$TASK_NAME/stage2_eval/
