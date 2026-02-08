#!/bin/sh

export CUDA_VISIBLE_DEVICES=1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
#vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 7878
# vllm serve ibm-granite/granite-3.3-2b-instruct  --port 7878
# vllm serve meta-llama/Llama-3.1-8B  --port 7878

# /net/storage149/autofs/css22/nmg/models

# ├── amd
# │   ├── Llama-3.1-405B-Instruct-FP8-KV
# │   ├── Llama-3.1-70B-Instruct-FP8-KV
# │   ├── Llama-3.1-8B-Instruct-FP8-KV
# │   ├── Llama-3.2-1B-Instruct-FP8-KV
# │   ├── Llama-3.2-3B-Instruct-FP8-KV
# │   ├── Llama-3.3-70B-Instruct-FP8-KV
# │   ├── Mistral-7B-v0.1-FP8-KV
# │   ├── Mixtral-8x22B-Instruct-v0.1-FP8-KV
# │   └── Mixtral-8x7B-Instruct-v0.1-FP8-KV
# ├── ibm-ai-platform
# │   └── Bamba-9B-v1
# ├── ibm-granite
# │   ├── granite-20b-code-instruct-8k
# │   ├── granite-3.0-1b-a400m-instruct
# │   ├── granite-3.0-2b-instruct
# │   ├── granite-3.0-3b-a800m-instruct
# │   ├── granite-3.0-8b-instruct
# │   ├── granite-3.1-1b-a400m-instruct
# │   ├── granite-3.1-2b-instruct
# │   ├── granite-3.1-3b-a800m-instruct
# │   ├── granite-3.1-8b-instruct
# │   ├── granite-3.2-2b-instruct
# │   ├── granite-3.2-8b-instruct
# │   ├── granite-3.3-2b-instruct
# │   ├── granite-3.3-8b-instruct
# │   ├── granite-34b-code-instruct-8k
# │   ├── granite-3b-code-instruct-128k
# │   ├── granite-4.0-tiny-preview
# │   ├── granite-8b-code-instruct-128k
# │   ├── granite-embedding-125m-english
# │   └── granite-embedding-30m-english
# ├── meta-llama
# │   ├── Llama-2-13b-hf
# │   ├── Llama-2-70b-hf
# │   ├── Llama-2-7b-hf
# │   ├── Llama-3.1-405B-Instruct
# │   ├── Llama-3.1-405B-Instruct-FP8
# │   ├── Llama-3.1-70B-Instruct
# │   ├── Llama-3.1-8B-Instruct
# │   ├── Llama-3.2-1B-Instruct
# │   ├── Llama-3.2-3B-Instruct
# │   ├── Llama-3.3-70B-Instruct
# │   ├── Llama-4-Maverick-17B-128E-Instruct
# │   ├── Llama-4-Maverick-17B-128E-Instruct-FP8
# │   └── Llama-4-Scout-17B-16E-Instruct
# ├── mistralai
# │   ├── Mamba-Codestral-7B-v0.1
# │   ├── Mistral-7B-Instruct-v0.3
# │   ├── Mistral-Large-Instruct-2407
# │   ├── Mistral-Large-Instruct-2411
# │   ├── Mistral-Small-24B-Instruct-2501
# │   ├── Mistral-Small-Instruct-2409
# │   ├── Mixtral-8x22B-Instruct-v0.1
# │   └── Mixtral-8x7B-Instruct-v0.1
# └── Zyphra
#     └── Zamba2-2.7B

base_folder_path="/net/storage149/autofs/css22/nmg/models/hf"

# model_path="amd/Llama-3.1-8B-Instruct-FP8-KV"
# model_path="meta-llama/Llama-3.1-8B-Instruct"
# model_path="meta-llama/Llama-3.1-8B"
# model_path="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
model_path="Qwen/Qwen3-32B"

full_path="${base_folder_path}/${model_path}/main"

python -m vllm.entrypoints.openai.api_server \
    --model $full_path \
    --port 7878 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    2>&1 | tee vllm_output.log
    # --gpu_memory_utilization 0.95 \
    # --max_model_len 32768 \
    # --max_seq_len 

# vllm serve --model $full_path \
#     --port 7878