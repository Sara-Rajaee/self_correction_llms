#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

MODEL_NAME_OR_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B
OUTPUT_DIR=./outputs

mkdir -p $OUTPUT_DIR

for DATA_NAME in "aime24" "aime25"; do

SPLIT="test"
NUM_TEST_SAMPLE=-1
PROMPT_TYPE="deepseek-r1"
THRES=1

python3 -u first_reasoning_generation.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_dir "./data" \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call 4096 \
    --max_model_len 8192 \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 1.0 \
    --top_p 0.9 \
    --min_p 0.05 \
    --n_sampling 8 \
    --max_num_seqs 64 \


#MODEL_NAME_OR_PATH=/mnt/nushare2/data/baliao/PLLMs/deepseek/DeepSeek-R1-Distill-Qwen-7B
DATA_PATH=${OUTPUT_DIR}/${DATA_NAME}/test_${MODEL_NAME}_seed0_t1.0_len4096_num${NUM_TEST_SAMPLE}s0e-1_first_reasoning.json
python3 -u answer_sampling.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_paths ${DATA_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --max_tokens_per_call 4096 \
    --max_model_len 8192 \
    --seed 0 \
    --temperature 1.0 \
    --top_p 0.9 \
    --min_p 0.05 \
    --n_sampling 4 \
    --max_num_seqs 64 \


DATA_PATH=${OUTPUT_DIR}/${DATA_NAME}/test_${MODEL_NAME}_seed0_t1.0_len4096_num${NUM_TEST_SAMPLE}s0e-1_first_reasoning_${MODEL_NAME}_prediction.json
python3 -u judge_answer.py \
     --data_paths ${DATA_PATH} \
     --data_names ${DATA_NAME} \


DATA_PATH=${OUTPUT_DIR}/${DATA_NAME}/test_${MODEL_NAME}_seed0_t1.0_len4096_num${NUM_TEST_SAMPLE}s0e-1_first_reasoning_${MODEL_NAME}_prediction_judge.json
#MODEL_NAME_OR_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
python3 -u answer_sampling.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_paths ${DATA_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --max_tokens_per_call 16384 \
    --max_model_len 20000 \
    --seed 0 \
    --temperature 1.0 \
    --top_p 0.9 \
    --min_p 0.05 \
    --n_sampling 4 \
    --eval_mode \
    --score_threshold $THRES \
    --max_num_seqs 32


done