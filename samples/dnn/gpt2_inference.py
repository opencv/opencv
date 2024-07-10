'''
This is a sample script to run GPT-2 inference in OpenCV using ONNX model.
The script loads the GPT-2 model and runs inference on a given prompt.
Currently script only works with fixed size window, that means
you will have to specify prompt of the same length as when model was exported to ONNX.


Exporting GPT-2 model to ONNX.
To export GPT-2 model to ONNX, you can use the following procedure:

1. Clone fork of Andrej Karpathy's GPT-2 repository:

    git clone https://github.com/Abdurrahheem/build-nanogpt/tree/ash/export-gpt2-onnx

2. Install the required dependencies:

    pip install -r requirements.txt

3  Export the model to ONNX:

    python export2onnx.py --promt=<Any-promt-you-want> --batch_size=<batch-size>


Run the script:
1. Install the required dependencies:

    pip install tiktoken==0.7.0 numpy==1.23.0

2. Run the script:

    python gpt2_inference.py --model=<path-to-onnx-model> --max_seq_len=<max-output-lenght> --batch_size=<use-one-used-while-exportinh> --prompt=<use-promt-of-the-same-length-used-while-exporting>
'''



from copy import deepcopy
import numpy as np
import tiktoken
import argparse
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run GPT-2 inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, required=True, help='Path to GPT-2 model ONNX model file.')
    parser.add_argument("--max_seq_len", type=int, default=30, help="Number of tokens to continue.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of batches.")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,", help="Prompt to start with.")
    return parser.parse_args()

def stable_softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def gpt2_inference(net, tokens, max_length, num_return_sequences=5):

    print("Inferencing GPT-2 model...")
    x = np.array(tokens)
    x = np.tile(x, (num_return_sequences, 1))

    output_buffer = deepcopy(x)
    counter = x.shape[1]
    while counter < max_length:

        net.setInput(x)
        logits = net.forward()

        # logits is assumed to be (B, seq_length, vocab_size) and needs to be the last token's logits
        logits = logits[:, -1, :]  # (B, vocab_size)

        # Get the probabilities using softmax
        probs = stable_softmax(logits)

        # Do top-k sampling of 50
        topk_indices = np.argpartition(probs, -50, axis=-1)[:, -50:]
        topk_probs = np.take_along_axis(probs, topk_indices, axis=-1)

        # Normalize top-k probabilities
        topk_probs /= np.sum(topk_probs, axis=-1, keepdims=True)

        # Select a token from the top-k probabilities
        sampled_indices = [np.random.choice(topk_indices[i], p=topk_probs[i]) for i in range(len(topk_probs))]
        sampled_indices = np.array(sampled_indices).reshape(-1, 1)

        # Append to the sequence
        x = np.concatenate((x, sampled_indices), axis=1)
        x = x[:, 1:] ## issue due to fixes size window in opencv

        output_buffer = np.concatenate((output_buffer, sampled_indices), axis=1)
        counter += 1
    print("Inference done!")
    return output_buffer

if __name__ == '__main__':

    np.random.seed(0)

    args = parse_args()
    max_length = args.max_seq_len
    num_return_sequences = args.batch_size
    prompt = args.prompt

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)

    net = cv.dnn.readNet(args.model)
    output_buffer = gpt2_inference(net, tokens, max_length, num_return_sequences)

    for i in range(num_return_sequences):
        tokens = output_buffer[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">>>>", decoded)