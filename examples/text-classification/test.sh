#!/bin/bash

export GLUE_DIR=~/data/glue/glue_data
export TASK_NAME=CoLA
export TRANSFORMERS_VERBOSITY=info

# warmup
python run_glue_bert_split.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --per_device_eval_batch_size 2 \
  --max_seq_length 16

python run_glue_bert_split.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --per_device_eval_batch_size 2 \
  --max_seq_length 16 \
  --no_cuda 

rm -r ~/horovod_logs/model_log/

# no split：GPU
# encoder还没de好，希望不是因为什么根本性错误......
# strategys=("no_split" "head_mask" "embedding" "single_hidden" "encoder")
strategys=("no_split" "head_mask" "embedding" "single_hidden")
batch_size_l=("2" "4" "8" "16" "32" "64" "128" "256" "512" "1024")
 
echo "max_seq_length 16"
for strategy in ${strategys[@]}
do
  echo $strategy
    for batch_size in ${batch_size_l[@]}
    do
        echo "batch size " $batch_size
        python run_glue_bert_split.py \
        --model_name_or_path bert-base-cased \
        --task_name $TASK_NAME \
        --do_eval \
        --model_split $strategy \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --output_dir /tmp/$TASK_NAME/ \
        --overwrite_output_dir \
        --per_device_eval_batch_size $batch_size \
        --max_seq_length 16
    done
done

echo "max_seq_length 64"
for strategy in ${strategys[@]}
do
  echo $strategy
    for batch_size in ${batch_size_l[@]}
    do
        echo "batch size " $batch_size
        python run_glue_bert_split.py \
        --model_name_or_path bert-base-cased \
        --task_name $TASK_NAME \
        --do_eval \
        --model_split $strategy \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --output_dir /tmp/$TASK_NAME/ \
        --overwrite_output_dir \
        --per_device_eval_batch_size $batch_size \
        --max_seq_length 64
    done
done

echo "CPU"

echo "max_seq_length 16"
for batch_size in ${batch_size_l[@]}
do
  echo "batch size " $batch_size
  python run_glue_bert_split.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --per_device_eval_batch_size $batch_size \
  --max_seq_length 16 \
  --no_cuda
done

echo "max_seq_length 64"
for batch_size in ${batch_size_l[@]}
do
  echo "batch size " $batch_size
  python run_glue_bert_split.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --per_device_eval_batch_size $batch_size \
  --max_seq_length 64 \
  --no_cuda
done