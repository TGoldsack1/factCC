#! /bin/bash
# Fine-tune FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH=/root/factCC/modeling # absolute path to modeling directory
export DATA_PATH=/root/autodl-tmp/data # absolute path to data directory
export OUTPUT_PATH=/root/autodl-tmp/output-long4 # absolute path to model checkpoint

export TASK_NAME=factcc_generated
export MODEL_NAME=/root/autodl-tmp/data/bert-base-8192

# export MODEL_NAME=/root/autodl-tmp/output-long2/root/autodl-tmp/output-long/root/autodl-tmp/data/bert-base-8192-factcc_generated-finetune-31160/-factcc_generated-finetune-14537/

python3 $CODE_PATH/run_long.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_lower_case \
  --max_seq_length 8192 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --gradient_accumulation_steps 12 \
  --warmup_steps 1000 \
  --fp16 \
  --learning_rate 1.25e-06 \
  --num_train_epochs 10.0 \
  --data_dir $DATA_PATH \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT_PATH/$MODEL_NAME-$TASK_NAME-finetune-$RANDOM/

#--fp16 \
#  --learning_rate 2e-5 \
