#!/bin/bash

BASE_MODEL_DIR= ... ### Specify the base model directory 
OUTPUT_DIR= ...     ### Specify where your model is going to be saved
DATA_DIR= ...       ### Specify the directory of your RLDS-format data
RUN_NAME= ...       ### Specify your run name

torchrun --nnodes 4 --nproc-per-node 8 \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK \
            train_phi3.py \
            --tracevla \
            --model_name_or_path $MODEL_PATH \
            --data_mix bridge_orig \
            --output_dir $OUTPUT_DIR \
            --batch_size 1920 \
            --per_device_batch_size 60 \
            --data_root_dir $DATA_DIR \
            --shuffle_buffer_size 100000 \
            --learning_rate 1e-5 \
            --run_name $RUN_NAME \
            --num_train_epochs 20 \
            --bf16  \
            --use_flash_attention