export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# regular 
# bash experiments/public/train/train_v2-qwen2vl-2B-parallel-M.sh 
# bash experiments/public/train/train_v2-qwen2vl-2B-parallel-S.sh 
# bash experiments/public/train/train_v2-qwen2vl-2B-series-M.sh 
# bash experiments/public/train/train_v2-qwen2vl-2B-series-S.sh 

# MOE
bash experiments/public/train/train_v2-qwen2vl-2B-parallel-moe-M.sh 
# bash experiments/public/train/train_v2-qwen2vl-2B-parallel-moe-S.sh 
# bash experiments/public/train/train_v2-qwen2vl-2B-series-moe-M.sh 
# bash experiments/public/train/train_v2-qwen2vl-2B-series-moe-S.sh 