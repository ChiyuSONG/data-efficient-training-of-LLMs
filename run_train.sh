#!/bin/bash

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --train_dir)
      train_dir="$2"
      shift
      shift
      ;;
    --valid_dir)
      valid_dir="$2"
      shift
      shift
      ;;
    --model_name_or_path)
      model_name_or_path="$2"
      shift
      shift
      ;;
    --batch_size)
      batch_size="$2"
      shift
      shift
      ;;
    --gradient_accumulation)
      gradient_accumulation="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
    esac
done


seed=0
deepspeed_config_file=configs/deepspeed_config.json

TIME_STAMP=`date "+%Y%m%d-%H%M"`
output_path=./runs

mkdir -p $output_path
output_dir=$output_path/$TIME_STAMP
log_file=$output_path/$TIME_STAMP.log


num_train_epochs=15

nohup deepspeed --num_gpus 8 \
    train_sft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${model_name_or_path} \
    --train_dir ${train_dir} \
    --valid_dir ${valid_dir} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation} \
    --num_train_epochs ${num_train_epochs} \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --save_total_limit 15 \
    --learning_rate 1e-5 \
    --end_learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --logging_steps 10 \
    --fp16 \
    --seed ${seed} \
    --output_dir ${output_dir} \
     > $log_file 2>&1