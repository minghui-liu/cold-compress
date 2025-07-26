#!/bin/bash

export OPENAI_API_KEY=

python parallelize_evals.py --command_file experiments/neurips/llama_govreport.txt --num_gpus 8
python parallelize_evals.py --command_file experiments/neurips/llama_multinews.txt --num_gpus 8
python parallelize_evals.py --command_file experiments/neurips/qwen_gsm8k.txt --num_gpus 8
python parallelize_evals.py --command_file experiments/neurips/qwen_medqa.txt --num_gpus 8
python parallelize_evals.py --command_file experiments/neurips/qwen_multinews.txt --num_gpus 8
python parallelize_evals.py --command_file experiments/neurips/qwen_govreport.txt --num_gpus 8
python parallelize_evals.py --command_file experiments/neurips/ruler_niah131k.txt --num_gpus 8
python parallelize_evals.py --command_file experiments/neurips/ruler_vt131k.txt --num_gpus 8
python parallelize_evals.py --command_file experiments/neurips/ruler_cwe131k.txt --num_gpus 8
