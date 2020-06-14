#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.3

data_dir=$2
ckpt_dir=$3
predictfile=$4

python sequence_label.py --num_epoch 30 \
    --learning_rate 3e-5 \
    --data_dir ${data_dir} \
    --schema_path ${data_dir}/event_schema.json \
    --train_data ${data_dir}/train.json \
    --dev_data ${data_dir}/dev.json \
    --test_data ${data_dir}/dev.json \
    --predict_data ${data_dir}/${predictfile} \
    --do_train False \
    --do_predict False \
    --do_predict2 True \
    --add_gru True \
    --do_model role1 \
    --max_seq_len 256 \
    --batch_size 8 \
    --model_save_step 3000 \
    --eval_step 200 \
    --checkpoint_dir ${ckpt_dir}

