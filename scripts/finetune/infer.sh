#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_PORT=6666
RANK=0

PRO_DIR=./
llama_ckpt_path=$PRO_DIR/model-hub/Llama-2-7b-chat-hf

CKPT_DIR=$PRO_DIR/model-ckpt/imod_finetune/finetune/llama_music/checkpoint-387

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune
RUN_NAME=test

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'


########### Arguments ###########
#################################
# lora_r: 4 means LoRA rank is $4$.
# ckpt_dir: the path of the checkpoint to be evaluated.
# vit_ckpt_path: the path of the pre-trained ViT checkpoint.
# BEATs_ckpt_path: the path of the pre-trained BEATs checkpoint.
# avqa_task: set True to train on AVQA task
#################################

torchrun --nproc_per_node=8 scripts/finetune/inference_parallel.py \
    --llm_name llama \
    --reserved_modality None \
    --loramethod test \
    --cut_folds 0 \
    --model_name_or_path $llama_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 False \
    --tf32 False \
    --fp16 False \
    --ckpt_dir $CKPT_DIR \
    --avqa_task True \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path $PRO_DIR/model-hub/clip-vit-large-patch14 \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path $PRO_DIR/model-hub/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir 'not_used' > "${CKPT_DIR}/log.txt" 2>&1

