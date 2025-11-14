#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "==> Environment"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

PATH_TO_VLM2VEC_REPO="/home/ubuntu/porialab-us-midwest-1/varun/vlm2vec2"
PATH_TO_VLM2VEC_NFS="/lambda/nfs/poria-cvpr-2026/varun/vlm2vec2"
CONFIG_DIR="experiments/public/eval"

CKPT_ID="Qwen2vl_2B.image+visdoc+video.autoresize.lora1.BS1024.IB64.GCq8p8.NormTemp002.lr5e-5.step5kwarm100.auxenc.parallel.hidden512.layers28.gqa.attnqknorm.heads8.kvheads4.intsize2048/checkpoint-550" 

cd $PATH_TO_VLM2VEC_REPO || exit

# ==============================================================================
# Configuration
# ==============================================================================

MASTER_PORT=54321
BATCH_SIZE=16
MODALITIES=("image" "video" "visdoc")
DATA_BASEDIR=$PATH_TO_VLM2VEC_NFS/data/vlm2vec_eval
OUTPUT_BASEDIR=$PATH_TO_VLM2VEC_NFS/exps/$CKPT_ID
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# ==> Define models and their base output paths here
# Format: "Model_Name;Model_Backbone;Base_Output_Path"
declare -a MODEL_SPECS
MODEL_SPECS+=( "$PATH_TO_VLM2VEC_NFS/outputs/$CKPT_ID;qwen2_vl;$OUTPUT_BASEDIR" )

# MODEL_SPECS+=( "VLM2Vec/VLM2Vec-V2.0;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct" )
# MODEL_SPECS+=( "code-kunkun/LamRA-Ret;lamra;$OUTPUT_BASEDIR/LamRA-Ret" )
# MODEL_SPECS+=( "vidore/colpali-v1.3;colpali;$OUTPUT_BASEDIR/colpali-v1.3" )
# MODEL_SPECS+=( "royokong/e5-v;llava_next;$OUTPUT_BASEDIR/e5-v" )
# MODEL_SPECS+=( "src/model/vlm_backbone/internvideo2/;internvideo2;$OUTPUT_BASEDIR/internvideo2" )
# MODEL_SPECS+=( "code-kunkun/LamRA-Ret-Qwen2.5VL-7b;lamra-qwen25;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct" )

# ==============================================================================
# Main Execution Loop
# ==============================================================================
# Loop through each model specification
for spec in "${MODEL_SPECS[@]}"; do
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH <<< "$spec"

  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "================================================="

  # Loop through each modality for the current model
  for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="$CONFIG_DIR/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY"

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    mkdir -p "$OUTPUT_PATH"

    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT --max_restarts=0 eval.py \
      --pooling mean \
      --normalize true \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone $MODEL_BACKBONE \
      --model_name $MODEL_NAME \
      --dataset_config $DATA_CONFIG_PATH \
      --encode_output_path $OUTPUT_PATH \
      --data_basedir $DATA_BASEDIR

    echo "  - Done."
    echo "-------------------------------------------------"
  done
done

echo "✅ All jobs completed."