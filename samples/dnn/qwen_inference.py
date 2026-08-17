'''
This is a sample script to run Qwen2.5 inference in OpenCV using ONNX model.
The script loads the Qwen2.5 model and runs inference on a given prompt using
the ChatML format (<|im_start|> / <|im_end|> special tokens).

Model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Exporting Qwen2.5 model to ONNX:

1. Install the required dependencies:

    pip install optimum[exporters] optimum-onnx[onnxruntime] torch transformers

2. Export the model to ONNX:

    Without KV-cache:

        optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct --task causal-lm qwen2.5_instruct_onnx/

    With KV-cache (recommended, faster autoregressive inference):

        optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct --task causal-lm-with-past qwen2.5_instruct_onnx_with_past/


Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    Without KV-cache (causal-lm export):

        python qwen_inference.py --model=<path-to-onnx-model> \
                                 --tokenizer_path=<path-to-qwen2.5-config.json> \
                                 --prompt="What is OpenCV?"

    With KV-cache (causal-lm-with-past export):

        python qwen_inference.py --model=<path-to-onnx-model> \
                                 --tokenizer_path=<path-to-qwen2.5-config.json> \
                                 --prompt="What is OpenCV?" \
                                 --use_kv_cache


Paged KV-cache and reserveKVCache():

    OpenCV keeps K/V in a paged, pre-packed cache only for models whose attention imports as
    a single fused op. The optimum-cli exports above decompose attention into MatMul/Softmax,
    so they fall back to carrying state through present.* -> past_key_values.* and
    reserveKVCache() logs a warning and does nothing.

    To get the paged cache, export with the dynamo exporter at opset 23, which lowers
    scaled_dot_product_attention to a single ai.onnx Attention node (needs onnxscript):

        import torch
        from transformers import AutoModelForCausalLM
        from torch.export import Dim

        m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',
                                                 dtype=torch.float32,
                                                 attn_implementation='sdpa').eval()

        class W(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, input_ids, position_ids):
                return self.m(input_ids=input_ids, position_ids=position_ids,
                              use_cache=False).logits

        T = Dim('T', min=1, max=2048)
        torch.onnx.export(W(m).eval(),
                          (torch.randint(0, 1000, (1, 8)), torch.arange(8).unsqueeze(0)),
                          'qwen25_op23/model.onnx', dynamo=True, opset_version=23,
                          dynamic_shapes=({1: T}, {1: T}),
                          input_names=['input_ids', 'position_ids'],
                          output_names=['logits'])

    Export without past_key_values: the paged cache holds K/V across forwards, so the graph
    only ever sees the current chunk. Pass position_ids explicitly instead.

    With --use_kv_cache the script then calls reserveKVCache(prompt_len + max_new_tokens)
    before prefill, sizing the page pool once so the decode loop allocates nothing.
'''

import numpy as np
import argparse
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run Qwen2.5 inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, required=True, help='Path to Qwen2.5 ONNX model file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to Qwen2.5 tokenizer config.json.')
    parser.add_argument('--prompt', type=str, default='What is OpenCV?', help='User prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=64, help='Maximum number of new tokens to generate.')
    parser.add_argument('--use_kv_cache', action='store_true', default=False, help='Enable KV-cache for faster inference (requires causal-lm-with-past export).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def set_optional_input(net, name, value):
    '''setInput() for a graph input the model may not declare. Returns True if it took.'''
    try:
        net.setInput(value, name)
        return True
    except cv.error:
        return False

def build_chatml_prompt(user_prompt):
    '''Wrap user prompt in Qwen2.5 ChatML format.'''
    return '<|im_start|>user\n' + user_prompt + '<|im_end|>\n<|im_start|>assistant\n'

def qwen_inference(net, prompt, max_new_tokens, tokenizer, use_kv_cache=True):

    print("Inferencing Qwen2.5 model...")

    tokens = list(tokenizer.encode(prompt))
    input_ids = np.array(tokens, dtype=np.int64).reshape(1, -1)

    # Qwen2.5 special token IDs
    im_end_id = 151645   # <|im_end|>
    eos_id    = 151643   # <|endoftext|>
    stop_ids  = (im_end_id, eos_id)

    generated = []

    if use_kv_cache:
        net.enableKVCache()
        prompt_len = input_ids.shape[1]

        # Pre-size the cache so the decode loop allocates no pages. Must precede prefill.
        net.reserveKVCache(prompt_len + max_new_tokens)

        # Prefill: process full prompt once to populate KV-cache
        net.setInput(input_ids, 'input_ids')
        # opset-23 dynamo exports take only input_ids/position_ids; optimum ones also want a mask.
        has_mask = set_optional_input(net, 'attention_mask',
                                      np.ones((1, prompt_len), dtype=np.int64))
        net.setInput(np.arange(prompt_len, dtype=np.int64).reshape(1, -1), 'position_ids')
        logits = net.forward()
        new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
        generated = [new_id]

        # Generate: feed one new token per step; the cache supplies all previous keys/values
        for _ in range(max_new_tokens - 1):
            if new_id in stop_ids:
                break
            cur_len = prompt_len + len(generated)
            net.setInput(np.array([[new_id]], dtype=np.int64), 'input_ids')
            if has_mask:
                net.setInput(np.ones((1, cur_len), dtype=np.int64), 'attention_mask')
            net.setInput(np.array([[cur_len - 1]], dtype=np.int64), 'position_ids')
            logits = net.forward()
            new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
            generated.append(new_id)
    else:
        # Without KV-cache: feed full growing sequence each step
        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            net.setInput(input_ids, 'input_ids')
            set_optional_input(net, 'attention_mask', np.ones((1, seq_len), dtype=np.int64))
            net.setInput(np.arange(seq_len, dtype=np.int64).reshape(1, -1), 'position_ids')
            logits = net.forward()
            new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
            if new_id in stop_ids:
                break
            generated.append(new_id)
            input_ids = np.concatenate([input_ids, [[new_id]]], axis=1)

    return np.array([tokens + generated], dtype=np.int64)

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(args.seed)

    print("Preparing Qwen2.5 model...")
    tokenizer = cv.dnn.Tokenizer.load(args.tokenizer_path)

    net = cv.dnn.readNetFromONNX(args.model, cv.dnn.ENGINE_OPENCV)
    if net.empty():
        raise SystemExit('Failed to load the model - readNetFromONNX() only warns, it does not raise. '
                         'Re-run with OPENCV_LOG_LEVEL=INFO to see which node was rejected and why.')

    chatml_prompt = build_chatml_prompt(args.prompt)
    print(f"Prompt:\n{chatml_prompt}")

    prompt_len = len(tokenizer.encode(chatml_prompt))
    tokens = qwen_inference(net, chatml_prompt, args.max_new_tokens, tokenizer, args.use_kv_cache)
    response = tokenizer.decode(tokens[0][prompt_len:].tolist())
    print(f"Response:\n{response}")
