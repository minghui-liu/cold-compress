#!/bin/bash

export OPENAI_API_KEY=

command_files=(
    experiments/neurips/llama_govreport.txt
    experiments/neurips/llama_multinews.txt
    experiments/neurips/qwen_gsm8k.txt
    experiments/neurips/qwen_medqa.txt
    experiments/neurips/qwen_multinews.txt
    experiments/neurips/qwen_govreport.txt
    experiments/neurips/ruler_niah_long.txt
    experiments/neurips/ruler_qa_long.txt
    experiments/neurips/ruler_vt_long.txt
    experiments/neurips/ruler_cwe_long.txt
)

# cat command files into a single file called all_commands.txt
cat "${command_files[@]}" > experiments/neurips/all_commands.txt
# Run the parallelize_evals.py script with the combined command file
python parallelize_evals.py --command_file experiments/neurips/all_commands.txt --num_gpus 8

# python parallelize_evals.py --command_file experiments/neurips/llama_govreport.txt --num_gpus 8
# python parallelize_evals.py --command_file experiments/neurips/llama_multinews.txt --num_gpus 8
# python parallelize_evals.py --command_file experiments/neurips/qwen_gsm8k.txt --num_gpus 8
# python parallelize_evals.py --command_file experiments/neurips/qwen_medqa.txt --num_gpus 8
# python parallelize_evals.py --command_file experiments/neurips/qwen_multinews.txt --num_gpus 8
# python parallelize_evals.py --command_file experiments/neurips/qwen_govreport.txt --num_gpus 8
# python parallelize_evals.py --command_file experiments/neurips/ruler_niah131k.txt --num_gpus 8
# python parallelize_evals.py --command_file experiments/neurips/ruler_vt131k.txt --num_gpus 8
# python parallelize_evals.py --command_file experiments/neurips/ruler_cwe131k.txt --num_gpus 8
