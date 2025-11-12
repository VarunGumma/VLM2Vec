export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# regular 
bash experiments/public/train/train_v2-qwen2vl-2B-parallel-M.sh 
bash experiments/public/train/train_v2-qwen2vl-2B-parallel-S.sh 
# bash experiments/public/train/train_v2-qwen2vl-2B-series-M.sh 
# bash experiments/public/train/train_v2-qwen2vl-2B-series-S.sh 