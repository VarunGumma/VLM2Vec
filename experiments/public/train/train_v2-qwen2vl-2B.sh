#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,6

echo "Python location: $(which python)"
echo -e "Python version: $(python --version)\n"

PATH_TO_VLM2VEC_REPO="/home/ubuntu/porialab-us-midwest-1/varun/vlm2vec2"
PATH_TO_VLM2VEC_NFS="/lambda/nfs/poria-cvpr-2026/varun/vlm2vec2"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
GRAD_ACC=$((8 / NUM_GPUS))

MASTER_PORT=54321
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
CONFIG_YAML="experiments/public/train/train_alltasks.yaml"

echo "Using ${NUM_GPUS} GPUs with gradient accumulation steps ${GRAD_ACC}"
echo "Effective batch size: $((NUM_GPUS * 128 * GRAD_ACC))"

export HF_DATASETS_CACHE="${PATH_TO_VLM2VEC_NFS}/hf_ds_cache"
export WANDB_PROJECT="multimodal-embeddings"
export EXP_NAME="Qwen2vl_2B.image+visdoc+video.autoresize.lora1.BS1024.IB64.GCq8p8.NormTemp002.lr5e-5.step5kwarm100.auxenc.parallel"
export WANDB_NAME=$EXP_NAME
export EXP_DIR=${PATH_TO_VLM2VEC_REPO}/outputs/${EXP_NAME}
export WANDB_DIR=$EXP_DIR

echo $EXP_DIR
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*
cd $PATH_TO_VLM2VEC_REPO


# Original Baseline
# torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT --max_restarts=0 train.py \
#         --bf16 \
#         --lora \
#         --lora_r 16 \
#         --lora_alpha 32 \
#         --lora_dropout 0.1 \
#         --pooling last \
#         --normalize True \
#         --temperature 0.02 \
#         --dataloader_num_workers 16 \
#         --model_name $MODEL_NAME \
#         --dataset_config $CONFIG_YAML \
#         --run_name $EXP_NAME \
#         --output_dir $EXP_DIR \
#         --grad_cache True \
#         --per_device_train_batch_size 128 \
#         --gradient_accumulation_steps $GRAD_ACC \
#         --gc_q_chunk_size 8 \
#         --gc_p_chunk_size 8 \
#         --interleave_batch_size 64 \
#         --lr_scheduler_type linear \
#         --learning_rate 5e-5 \
#         --max_steps 5000 \
#         --warmup_steps 100 \
#         --save_steps 50 \
#         --logging_steps 1 \
#         --save_safetensors True \
#         --remove_unused_columns False \
#         --resume_from auto \
#         --report_to wandb 2>&1 | tee $EXP_DIR/train.log


torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT --max_restarts=0 train.py \
        --bf16 \
        --lora \
        --lora_r 1 \
        --lora_alpha 2 \
        --lora_dropout 0.0 \
        --pooling mean \
        --normalize True \
        --temperature 0.02 \
        --dataloader_num_workers 16 \
        --model_name $MODEL_NAME \
        --dataset_config $CONFIG_YAML \
        --run_name $EXP_NAME \
        --output_dir $EXP_DIR \
        --grad_cache True \
        --ddp_find_unused_parameters False \
        --per_device_train_batch_size 128 \
        --gradient_accumulation_steps $GRAD_ACC \
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
        --parallel_encoder True \
        --hidden_size 512 \
        --num_layers 28 \
        --use_gqa True \
        --attn_qk_norm True \
        --num_attn_heads 8 \
        --num_kv_attn_heads 4 \
        --intermediate_size 2048 \
        --backbone_model_hidden_size 1536 2>&1 | tee $EXP_DIR/train.log