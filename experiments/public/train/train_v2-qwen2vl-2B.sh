#!/bin/bash
echo "Python location: $(which python)"
echo -e "Python version: $(python --version)\n"

PATH_TO_VLM2VEC_REPO="/home/ubuntu/porialab-us-midwest-1/varun/vlm2vec2"
PATH_TO_VLM2VEC_NFS="/lambda/nfs/poria-cvpr-2026/varun/vlm2vec2"

export HF_DATASETS_CACHE="${PATH_TO_VLM2VEC_NFS}/hf_ds_cache"
export WANDB_PROJECT="multimodal-embeddings"
export EXP_NAME="Qwen2vl_2B.image+visdoc+video.autoresize.lora1.BS1024.IB64.GCq8p8.NormTemp002.lr5e-5.step5kwarm100.auxenc.3layer.rf64.ide8"

export WANDB_NAME=$EXP_NAME
export EXP_DIR=${PATH_TO_VLM2VEC_REPO}/$EXP_NAME
export WANDB_DIR=$EXP_DIR

echo $EXP_DIR
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd $PATH_TO_VLM2VEC_REPO

## Original Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=54321 --max_restarts=0 train.py \
        --bf16 \
        --lora \
        --lora_r 16 \
        --pooling eos \
        --normalize True \
        --temperature 0.02 \
        --dataloader_num_workers 8 \
        --model_name Qwen/Qwen2-VL-2B-Instruct \
        --dataset_config experiments/public/train/train_alltasks.yaml \
        --run_name $EXP_NAME \
        --output_dir $EXP_DIR \
        --grad_cache True \
        --per_device_train_batch_size 128 \
        --gc_q_chunk_size 8 \
        --gc_p_chunk_size 8 \
        --interleave_batch_size 64 \
        --lr_scheduler_type linear \
        --learning_rate 5e-5 \
        --max_steps 5000 \
        --warmup_steps 100 \
        --save_steps 50 \
        --logging_steps 1 \
        --save_safetensors True \
        --remove_unused_columns False \
        --resume_from auto \
        --report_to wandb 2>&1 | tee $EXP_DIR/train.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=54321 --max_restarts=0 train.py \
        --bf16 \
        --lora \
        --lora_r 1 \
        --lora_alpha 2 \
        --lora_dropout 0.0 \
        --pooling mean \
        --normalize True \
        --temperature 0.02 \
        --dataloader_num_workers 24 \
        --model_name Qwen/Qwen2-VL-2B-Instruct \
        --dataset_config ${PATH_TO_VLM2VEC_REPO}/experiments/public/train/train_alltasks.yaml \
        --run_name $EXP_NAME \
        --output_dir $EXP_DIR \
        --grad_cache True \
        --per_device_train_batch_size 128 \
        --gc_q_chunk_size 8 \
        --gc_p_chunk_size 8 \
        --interleave_batch_size 64 \
        --lr_scheduler_type linear \
        --learning_rate 5e-5 \
        --max_steps 5000 \
        --warmup_steps 100 \
        --save_steps 50 \
        --logging_steps 1 \
        --save_safetensors True \
        --remove_unused_columns False \
        --resume_from auto \
        --report_to wandb \
        --add_aux_encoder True \
        --reduction_factor 16 \
        --inter_dim_expansion_factor 8 \
        --num_layers 3 2>&1 | tee $EXP_DIR/train.log