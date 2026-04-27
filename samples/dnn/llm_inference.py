# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
LLM inference using the LLM class with multiple tokenizer backends.

Supports three modes via --tokenizer_type flag:
  ort_genai  : ORT-GenAI tokenizer (Llama, Phi, DeepSeek, etc.)
  gpt2       : OpenCV BPE tokenizer (GPT-2)
  qwen       : OpenCV BPE tokenizer (Qwen2.5)

=== ORT-GenAI (Llama, Phi, DeepSeek, etc.) ===

Download and convert a model using ORT-GenAI model builder:

    pip install onnxruntime-genai
    python3 -m onnxruntime_genai.models.builder \
        -m <hf_model_id> -o <output_dir> -p <precision> -e cpu

    where <precision> is fp32, fp16, or int4.

Build OpenCV with:
    -DWITH_ONNXRUNTIME=ON -DWITH_ONNXRUNTIME_GENAI=ON

To run:
    python llm_inference.py --tokenizer_type=ort_genai --model=/path/to/ort_genai_model_dir \
        --prompt="What is OpenCV?" --max_new_tokens=100

=== GPT-2 with OpenCV BPE tokenizer ===

Exporting GPT-2 model to ONNX:

1. Clone fork of Andrej Karpathy's GPT-2 repository:

    git clone -b fix-dynamic-axis-export https://github.com/nklskyoy/build-nanogpt

2. Install the required dependencies:

    pip install -r requirements.txt

3. Export the model to ONNX:

    python export2onnx.py --promt=<Any-promt-you-want>

To run:
    python llm_inference.py --tokenizer_type=gpt2 --model=/path/to/gpt2.onnx \
        --tokenizer=/path/to/gpt2/config.json \
        --prompt=<use-promt-of-the-same-length-used-while-exporting>

=== Qwen2.5 with OpenCV BPE tokenizer ===

Model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Exporting Qwen2.5 model to ONNX:

1. Install the required dependencies:

    pip install optimum[exporters] torch transformers

2. Export the model to ONNX:

    optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct --task causal-lm qwen2.5_instruct_onnx/

To run:
    python llm_inference.py --tokenizer_type=qwen --model=/path/to/qwen2.5.onnx \
        --tokenizer=/path/to/qwen2.5/config.json --prompt="What is OpenCV?" --max_new_tokens=100
'''

import numpy as np
import argparse
import cv2 as cv


def run_ort_genai(model_path, user_prompt, max_new_tokens):
    '''ORT-GenAI path: uses OGA tokenizer + single run() call for full generation.'''

    # 1. Create LLM with ORT-GenAI tokenizer
    llm = cv.dnn.LLM.create(model_path, cv.dnn.TOKENIZER_ORT_GENAI)
    tokenizer = llm.getTokenizer()

    print(f"Model type  : {llm.getModelType()}")
    print(f"Device type : {llm.getDeviceType()}")

    # 2. Apply chat template and encode prompt
    messages = '[{"role": "user", "content": "' + user_prompt + '"}]'
    prompt = llm.applyChatTemplate(messages)
    tokens = tokenizer.encode(prompt)

    # 3. Configure generation parameters
    llm.setSearchOption("max_length", float(len(tokens) + max_new_tokens))
    llm.setSearchOptionBool("do_sample", False)

    # 4. Run inference and decode output
    out = llm.run(tokens)
    print(tokenizer.decode(out.flatten().tolist()))


def run_gpt2(model_path, tokenizer_cfg, user_prompt, max_seq_len):
    '''GPT-2 path: uses OpenCV BPE tokenizer + autoregressive greedy decoding loop.'''

    # 1. Create LLM with OpenCV BPE tokenizer
    llm = cv.dnn.LLM.create(model_path, cv.dnn.TOKENIZER_OPENCV_BPE, tokenizer_cfg)
    tokenizer = llm.getTokenizer()

    # 2. Encode prompt
    tokens = list(tokenizer.encode(user_prompt))

    # 3. Autoregressive greedy decoding loop
    stop_token = 50256  # <|endoftext|>
    remaining = max_seq_len

    while remaining > 0 and tokens[-1] != stop_token:
        logits = llm.run(tokens, 'idx')  # (1, seq_len, vocab_size)
        logits = logits[:, -1, :]

        new_idx = int(np.argmax(logits.reshape(-1)))
        tokens.append(new_idx)
        remaining -= 1

    # 4. Decode and print
    print(tokenizer.decode(tokens))


def run_qwen(model_path, tokenizer_cfg, user_prompt, max_new_tokens):
    '''Qwen2.5 path: uses OpenCV BPE tokenizer + autoregressive greedy decoding loop
    with multiple named inputs (input_ids, attention_mask, position_ids).'''

    # 1. Create LLM with OpenCV BPE tokenizer and ENGINE_NEW
    llm = cv.dnn.LLM.create(model_path, cv.dnn.TOKENIZER_OPENCV_BPE, tokenizer_cfg, cv.dnn.ENGINE_NEW)
    tokenizer = llm.getTokenizer()

    # 2. Encode prompt with ChatML format
    chatml_prompt = '<|im_start|>user\n' + user_prompt + '<|im_end|>\n<|im_start|>assistant\n'
    tokens = tokenizer.encode(chatml_prompt)
    tokens = np.array(tokens, dtype=np.int64).reshape(1, -1)

    # 3. Autoregressive greedy decoding loop
    stop_ids = (151645, 151643)  # <|im_end|>, <|endoftext|>

    for _ in range(max_new_tokens):
        seq_len = tokens.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        logits = llm.run([tokens, attention_mask, position_ids],
                         ['input_ids', 'attention_mask', 'position_ids'])
        logits = logits[:, -1, :]

        new_id = int(np.argmax(logits.reshape(-1)))
        tokens = np.concatenate((tokens, np.array([[new_id]], dtype=np.int64)), axis=1)

        if new_id in stop_ids:
            break

    # 4. Decode and print
    print(tokenizer.decode(tokens[0].tolist()))


def parse_args():
    parser = argparse.ArgumentParser(description='LLM inference using the LLM class with multiple tokenizer backends.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tokenizer_type', type=str, default='ort_genai',
                        choices=['ort_genai', 'gpt2', 'qwen'],
                        help='Tokenizer type: ort_genai, gpt2, or qwen.')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model file or ORT-GenAI model directory.')
    parser.add_argument('--tokenizer', type=str, default='', help='Path to tokenizer config.json (required for gpt2/qwen).')
    parser.add_argument('--prompt', type=str, default='What is OpenCV?', help='User prompt text.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate (ort_genai/qwen).')
    parser.add_argument('--max_seq_len', type=int, default=32, help='Number of tokens to continue (gpt2 only).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.tokenizer_type == 'ort_genai':
        run_ort_genai(args.model, args.prompt, args.max_new_tokens)

    elif args.tokenizer_type == 'gpt2':
        if not args.tokenizer:
            print("Error: --tokenizer is required for gpt2 tokenizer_type.")
            exit(1)
        run_gpt2(args.model, args.tokenizer, args.prompt, args.max_seq_len)

    elif args.tokenizer_type == 'qwen':
        if not args.tokenizer:
            print("Error: --tokenizer is required for qwen tokenizer_type.")
            exit(1)
        run_qwen(args.model, args.tokenizer, args.prompt, args.max_new_tokens)

    else:
        print(f"Error: Unknown tokenizer_type '{args.tokenizer_type}'. Use ort_genai, gpt2, or qwen.")
        exit(1)
