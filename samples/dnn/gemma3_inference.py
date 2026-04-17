'''
This is a sample script to run Gemma3 inference in OpenCV using ONNX model.
The script loads the Gemma3 model and runs inference on a given prompt using
the Gemma3 chat format (<start_of_turn> / <end_of_turn> special tokens).

Model: https://huggingface.co/google/gemma-3-1b-it

Exporting Gemma3 model to ONNX:

1. Install the required dependencies:

    pip install optimum[exporters] torch transformers

2. Export the model to ONNX:

    optimum-cli export onnx --model google/gemma-3-1b-it --task causal-lm gemma3_instruct_onnx/


Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python gemma3_inference.py --model=<path-to-onnx-model> \
                               --tokenizer_path=<path-to-opencv-tokenizer-config.json> \
                               --prompt="What is OpenCV?"

    The tokenizer_path should point to an OpenCV-format config.json (e.g., from
    opencv_extra/testdata/dnn/llm/gemma3/config.json), NOT the HuggingFace tokenizer_config.json.
'''

import numpy as np
import argparse
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run Gemma3 inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, required=True, help='Path to Gemma3 ONNX model file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to Gemma3 tokenizer config.json.')
    parser.add_argument('--prompt', type=str, default='What is OpenCV?', help='User prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=64, help='Maximum number of new tokens to generate.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def build_gemma3_prompt(user_prompt):
    '''Wrap user prompt in Gemma3 chat format.'''
    return '<start_of_turn>user\n' + user_prompt + '<end_of_turn>\n<start_of_turn>model\n'

def gemma3_inference(net, prompt, max_new_tokens, tokenizer):

    print("Inferencing Gemma3 model...")

    tokens = tokenizer.encode(prompt)
    # Prepend BOS token (id=2) as required by Gemma3
    tokens = [2] + list(tokens)
    tokens = np.array(tokens, dtype=np.int64).reshape(1, -1)

    # Gemma3 special token IDs
    eos_id     = 1    # <eos>
    eot_id     = 106  # <end_of_turn>
    stop_ids   = (eos_id, eot_id)

    for _ in range(max_new_tokens):
        seq_len = tokens.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)

        net.setInput(tokens, 'input_ids')
        net.setInput(attention_mask, 'attention_mask')
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

    print("Preparing Gemma3 model...")
    tokenizer = cv.dnn.Tokenizer.load(args.tokenizer_path)

    net = cv.dnn.readNetFromONNX(args.model, cv.dnn.ENGINE_NEW)

    gemma3_prompt = build_gemma3_prompt(args.prompt)
    print(f"Prompt:\n{gemma3_prompt}")

    prompt_len = len(tokenizer.encode(gemma3_prompt)) + 1  # +1 for BOS token
    tokens = gemma3_inference(net, gemma3_prompt, args.max_new_tokens, tokenizer)
    response = tokenizer.decode(tokens[0][prompt_len:].tolist())
    print(f"Response:\n{response}")
