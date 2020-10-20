export MODEL_PATH=../model/pytorch_bert_base_uncased/
export TASK_NAME=QQP
cd bert_finetune && python run_glue.py \
  --model_type bert \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../glue_data/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --save_steps 100 \
  --output_dir ../model/$TASK_NAME/teacher/ \
  --evaluate_during_training \
  --overwrite_output_dir