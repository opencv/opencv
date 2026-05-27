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

        # Prefill: process full prompt once to populate KV-cache
        net.setInput(input_ids, 'input_ids')
        net.setInput(np.ones((1, prompt_len), dtype=np.int64), 'attention_mask')
        net.setInput(np.arange(prompt_len, dtype=np.int64).reshape(1, -1), 'position_ids')
        logits = net.forward()
        new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
        generated = [new_id]

        # Generate: feed one new token per step; OpenCV routes present.* -> past_key_values.*
        for _ in range(max_new_tokens - 1):
            if new_id in stop_ids:
                break
            cur_len = prompt_len + len(generated)
            net.setInput(np.array([[new_id]], dtype=np.int64), 'input_ids')
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
            net.setInput(np.ones((1, seq_len), dtype=np.int64), 'attention_mask')
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

    net = cv.dnn.readNetFromONNX(args.model, cv.dnn.ENGINE_NEW)

    chatml_prompt = build_chatml_prompt(args.prompt)
    print(f"Prompt:\n{chatml_prompt}")

    prompt_len = len(tokenizer.encode(chatml_prompt))
    tokens = qwen_inference(net, chatml_prompt, args.max_new_tokens, tokenizer, args.use_kv_cache)
    response = tokenizer.decode(tokens[0][prompt_len:].tolist())
    print(f"Response:\n{response}")
