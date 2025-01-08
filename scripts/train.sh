#!/bin/bash

BASE_MODEL_DIR= ... ### Specify the base model directory 
OUTPUT_DIR= ...     ### Specify where your model is going to be saved
PRETRAINED_CKPT_DIR= ... ### Specify where your base checkpoint is saved
DATA_DIR= ...       ### Specify the directory of your RLDS-format data
RUN_NAME= ...       ### Specify your run name
WANDB_PROJECT= ...  ### Specify Wandb Project Name
WANDB_ENTITY= ...   ### Specify Wandb Entity Name
HF_TOKEN= ...       ### Specify HF Token

torchrun --nnodes 8 --nproc-per-node 8 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK \
              vla-scripts/train.py \
              --pretrained_checkpoint $PRETRAINED_CKPT_DIR \
              --vla.type 64gpu-tracevla \
              --data_root_dir $DATA_DIR \
              --run_root_dir $OUTPUT_DIR \
              --run_id $RUN_NAME \
              --wandb_project $WANDB_PROJECT \
              --wandb_entity $WANDB_ENTITY \
              --hf_token $HF_TOKEN
