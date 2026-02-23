# export TASK_NAME=qnli
export TASK_NAME=sst2


python run_glue_private_light_train.py \
  --model_name_or_path andeskyl/bert-base-cased-$TASK_NAME \
  --task_name $TASK_NAME \
  --gpu_ids 0,1 \
  --quick_run \
  --output_dir eval_private/$TASK_NAME/ultra_light/
