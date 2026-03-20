'''
This is a sample script to run GPT-2 inference in OpenCV using ONNX model.
The script loads the GPT-2 model and runs inference on a given prompt.
Currently script only works with fixed size window, that means
you will have to specify prompt of the same length as when model was exported to ONNX.


Exporting GPT-2 model to ONNX.
To export GPT-2 model to ONNX, you can use the following procedure:

1. Clone fork of Andrej Karpathy's GPT-2 repository:

    git clone -b fix-dynamic-axis-export  https://github.com/nklskyoy/build-nanogpt

2. Install the required dependencies:

    pip install -r requirements.txt

3  Export the model to ONNX:

    python export2onnx.py --promt=<Any-promt-you-want>


Run the script:
1. Install the required dependencies:

    pip install tiktoken==0.7.0 numpy tqdm

2. Run the script:
    python gpt2_inference.py --model=<path-to-onnx-model> --tokenizer_path=<path-to-tokenizer-config> --prompt=<use-promt-of-the-same-length-used-while-exporting>
'''

import numpy as np
import argparse
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run GPT-2 inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, required=True, help='Path to GPT-2 model ONNX model file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to GPT-2 tokenizer config file.')
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,", help="Prompt to start with.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Number of tokens to continue.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()

def stable_softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)



def gpt2_inference(net, prompt, max_length, tokenizer):

    print("Inferencing GPT-2 model...")

    tokens = tokenizer.encode(prompt).reshape(1,-1)

    stop_tokens = (50256, ) ## could be extended to include more stop tokens
    while 0 < max_length and tokens[:, -1] not in stop_tokens:

        net.setInputsNames(['idx'])
        net.setInput(tokens, 'idx')
        logits = net.forward()
        logits = logits[:, -1, :]  # (B, vocab_size)

        # use hard sampling
        new_idx = np.argmax(logits.reshape(-1)).reshape(1,1)

        tokens = np.concatenate((tokens, new_idx), axis=1)

        max_length -= 1
    return tokens



if __name__ == '__main__':

    args = parse_args()
    print("Preparing GPT-2 model...")
    max_length = args.max_seq_len
    prompt = args.prompt
    tokenizer_path = args.tokenizer_path

    net = cv.dnn.readNetFromONNX(args.model, 4)
    tokenizer = cv.dnn.Tokenizer.load(tokenizer_path)

    tokens = gpt2_inference(net, prompt, max_length, tokenizer)
    print(tokenizer.decode(tokens[0]))
