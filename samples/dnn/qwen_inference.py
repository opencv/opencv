'''
This is a sample script to run Qwen2.5 inference in OpenCV using ONNX model.
The script loads the Qwen2.5 model and runs inference on a given prompt using
the ChatML format (<|im_start|> / <|im_end|> special tokens).

Model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Exporting Qwen2.5 model to ONNX:

1. Install the required dependencies:

    pip install optimum[exporters] torch transformers

2. Export the model to ONNX:

    optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct --task causal-lm qwen2.5_instruct_onnx/


Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python qwen_inference.py --model=<path-to-onnx-model> \
                             --tokenizer_path=<path-to-qwen2.5-config.json> \
                             --prompt="What is OpenCV?"
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
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def stable_softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def build_chatml_prompt(user_prompt):
    '''Wrap user prompt in Qwen2.5 ChatML format.'''
    return '<|im_start|>user\n' + user_prompt + '<|im_end|>\n<|im_start|>assistant\n'

def qwen_inference(net, prompt, max_new_tokens, tokenizer):

    print("Inferencing Qwen2.5 model...")

    tokens = tokenizer.encode(prompt)
    tokens = np.array(tokens, dtype=np.int64).reshape(1, -1)

    # Qwen2.5 special token IDs
    im_end_id = 151645   # <|im_end|>
    eos_id    = 151643   # <|endoftext|>
    stop_ids  = (im_end_id, eos_id)

    for _ in range(max_new_tokens):
        seq_len = tokens.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        net.setInput(tokens, 'input_ids')
        net.setInput(attention_mask, 'attention_mask')
        net.setInput(position_ids, 'position_ids')
        logits = net.forward()          # (1, seq_len, vocab_size)
        logits = logits[:, -1, :]       # take last token logits

        new_id = int(np.argmax(logits.reshape(-1)))
        tokens = np.concatenate((tokens, np.array([[new_id]], dtype=np.int64)), axis=1)

        if new_id in stop_ids:
            break

    return tokens

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(args.seed)

    print("Preparing Qwen2.5 model...")
    tokenizer = cv.dnn.Tokenizer.load(args.tokenizer_path)

    net = cv.dnn.readNetFromONNX(args.model, cv.dnn.ENGINE_NEW)

    chatml_prompt = build_chatml_prompt(args.prompt)
    print(f"Prompt:\n{chatml_prompt}")

    tokens = qwen_inference(net, chatml_prompt, args.max_new_tokens, tokenizer)
    response = tokenizer.decode(tokens[0].tolist())
    print(f"Response:\n{response}")
