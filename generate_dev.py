# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
import contextlib
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo.config
import torch._inductor.config
import logging

from cache import add_cache_arguments
from generation_utils import (
    add_generation_arguments,
    compile_funcs,
    compute_max_seq_length,
    device_sync,
    print_stats,
)

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
DEBUG_COMPILE = False
if DEBUG_COMPILE:
    import logging

    level = logging.DEBUG
    torch._logging.set_logs(dynamo=level, inductor=level)
    torch._dynamo.config.verbose = True

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from tokenizer import get_tokenizer, encode
from generation_utils import (
    generate,
    get_model_size,
    load_model,
    merge_cache_config,
    setup_caches,
)
from cache import add_cache_arguments, cache_compatibility


def main(
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 100,
    checkpoint_path: Path = Path(
        "checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct/model.pth"
    ),
    compile: bool = True,
    feed_long_prompts: bool = False,
    attn_top_k: float = 1.0,
    profile: Optional[Path] = None,
    device=default_device,
    cache_kwargs: dict = {},
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    # pytorch_logs_to_file()

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        # If there's no tokenizer.model, try to load the tokenizer from the parent directory
        # NOTE: We assume the tokenizer in the parent directory is compatible with huggingface transformers
        tokenizer_path = checkpoint_path.parent

    global print
    from tp import maybe_init_dist

    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_chat = (
        "chat" in str(checkpoint_path).lower()
        or "instruct" in str(checkpoint_path).lower()
    )

    print("Loading model ...")
    t0 = time.time()
    model = load_model(checkpoint_path, device, precision, use_tp)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path, is_chat=is_chat)

    inputs = [encode(tokenizer, prompt, device=device, is_chat=is_chat)]

    terminator_ids = tokenizer.get_terminator_ids()

    torch.manual_seed(1234)
    model_size = get_model_size(model)
    print(f"{model_size / 1e9:.02f} billion parameters in model.")

    prefill, decode_one_token = compile_funcs(compile)

    device_sync(device=device)  # MKG

    max_prompt_length, max_seq_length = compute_max_seq_length(
        model, inputs, None, max_new_tokens
    )
    max_new_tokens = min(max_new_tokens, max_seq_length - max_prompt_length)
    setup_caches(model, tokenizer, inputs[0].device, max_seq_length, cache_kwargs)

    y, _, perf_stats = generate(
        model,
        inputs[0],
        prefill,
        decode_one_token,
        max_new_tokens=max_new_tokens,
        terminator_ids=terminator_ids,
        attn_top_k=attn_top_k,
        feed_long_prompts=feed_long_prompts,
    )

    device_sync(device=device)  # MKG
    print("\n==========\n")
    print("GENERATION:")
    print("```")
    print(tokenizer.decode(y.tolist()))
    print("```")

    # print("\n==========\n")
    # print("PERFORMANCE:")
    # tokens_per_second = perf_stats["total_toks_per_sec"]
    # decode_tokens = perf_stats["decode_tokens"]
    # total_seconds = perf_stats["total_seconds"]
    # memory_used_gb = perf_stats["memory_used_gb"]

    # print(
    #     f"Time: {total_seconds:.02f} sec total, {tokens_per_second:.02f} tokens/sec, {decode_tokens} tokens"
    # )
    # print(f"Bandwidth: {model_size * tokens_per_second / 1e9:.02f} GB/s")
    # print(f"Memory used: {memory_used_gb} GB")
    # print("\n==========\n")
    # print("DETAILED PERFORMANCE:")
    # print_stats(perf_stats)

    # print("\n==========\n")
    # print("KV CACHE STATISTICS:")
    # cache_stats = model.get_cache_stats(max_prompt_length, decode_tokens)
    # print_stats(cache_stats)

    print("\n==========\n")
    print("KV CACHE CONTENTS:")

    all_tokens = y.cpu().numpy()
    print_highlighted_tokens(all_tokens, model, 0, 0, tokenizer)  # layer 0, head 0
    print_highlighted_tokens(all_tokens, model, 16, 0, tokenizer)  # layer 16, head 1
    print_highlighted_tokens(all_tokens, model, 31, 0, tokenizer)  # layer 31, head 2


def print_retained_tokens(all_tokens, model, i, j, tokenizer):
    cache = model.layers[i].attention.kv_cache
    pos = cache.pos
    pos = pos.cpu().numpy()
    """Prints the retained tokens in the cache."""
    head_pos = pos[0, j, :]  # take 1st head and all layers for simplicity
    # print(f"Pos of tokens in layer {i} head {j} cache:\n{head_pos}")
    head_tokens = all_tokens[head_pos]
    # print(f"Token ids in layer {i} head {j} cache:\n{head_tokens}")
    head_tokens_decoded = tokenizer.decode(head_tokens.tolist())
    print(f"Retained tokens in layer {i} head {j} cache:\n```\n{head_tokens_decoded}\n```")

def escape_xml_manual(text):
    """
    Escapes special XML characters in a string without using external libraries.
    """
    text = text.replace("&", "&amp;")  # Must be done first to avoid double-escaping
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("'", "&apos;")
    text = text.replace('"', "&quot;")
    return text

def escape_llama_tags(text):
    """
    Escapes Llama turn tags within a given string by replacing them with escaped versions.

    Args:
    text (str): The input string potentially containing Llama turn tags.

    Returns:
    str: The string with Llama turn tags replaced by their escaped versions.
    """
    # replacements = {
    #     "<|begin_of_text|>": "\\<|begin_of_text|>",
    #     "<|end_of_text|>": "\\<|end_of_text|>",
    #     "<|eot_id|>": "\\<|eot_id|>",
    #     "<|start_header_id|>": "\\<|start_header_id|>",
    #     "<|end_header_id|>": "\\<|end_header_id|>"
    # }
    replacements = {
        "<|begin_of_text|>": "[begin of text]",
        "<|end_of_text|>": "[end of text]",
        "<|eot_id|>": "[eot id]",
        "<|start_header_id|>": "[start header id]",
        "<|end_header_id|>": "[end header id]"
    }

    for old_tag, escaped_tag in replacements.items():
        text = text.replace(old_tag, escaped_tag)
    return text

def escape_latex(text):
    """
    Escapes special LaTeX characters in a string.
    """
    latex_special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}', # Often needed for math environments
        '>': r'\textgreater{}', # Often needed for math environments
    }
    
    # Escape backslashes first to avoid double-escaping issues
    text = text.replace('\\', r'\textbackslash{}') 

    for char, replacement in latex_special_chars.items():
        if char != '\\': # Skip backslash as it's handled separately
            text = text.replace(char, replacement)
    return text

def print_highlighted_tokens(all_tokens, model, i, j, tokenizer):
    """Prints the highlighted tokens in the cache."""
    cache = model.layers[i].attention.kv_cache
    pos = cache.pos
    pos = pos.cpu().numpy()
    head_pos = pos[0, j, :]  # take 1st head and all layers for simplicity
    # head_tokens = all_tokens[head_pos]
    # head_tokens_decoded = tokenizer.decode(head_tokens.tolist())
    
    text_buffer = []
    for i, t in enumerate(all_tokens):
        decoded = tokenizer.decode([t])
        text = decoded.replace("\n", "\\n")
        text = escape_llama_tags(text)  # Escape Llama tags
        # text = text.replace("$", "\\$")
        # text = escape_xml_manual(text)  # Escape XML characters manually
        if i in head_pos:
            text = escape_latex(text)  # Escape LaTeX characters
            text_buffer.append('$\\textcolor{green}{\\text{'+ text +'}}$')
        else:
            text_buffer.append(text)
    highlighted_text = ''.join(text_buffer)
    print(f"Highlighted tokens in layer {i} head {j} cache:\n----------------------------\n{highlighted_text}\n--------------------------\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Simple Single Prompt Generation (for development and debugging purposes)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="long_prompt_short_output.txt",
        help="Input prompt. If it ends in .txt, we will load the prompt from the ./prompts dir.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of new tokens."
    )

    parser.add_argument(
        "--cache_config",
        type=str,
        default=None,
        help="Name of YAML file in ./cache_configs.",
    )

    add_generation_arguments(parser)
    add_cache_arguments(parser)

    args = merge_cache_config(parser.parse_args())

    if args.prompt.endswith(".txt"):
        prompt_fn = Path(__file__).resolve().parent / "prompts" / args.prompt
        with open(prompt_fn) as fd:
            args.prompt = fd.read().strip()

    cache_compatibility(args)

    main(
        args.prompt,
        args.max_new_tokens,
        args.checkpoint_path,
        args.compile,
        args.feed_long_prompts,
        args.attn_top_k,
        args.profile,
        args.device,
        cache_kwargs=vars(args),
    )
