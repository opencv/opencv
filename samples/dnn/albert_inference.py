# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
This is a sample script to run ALBERT (albert-large-v2) masked-LM inference
in OpenCV using an ONNX model. The input text must contain a single literal
"[MASK]" token; the script prints the top predictions for that position.

Model: https://huggingface.co/Xenova/albert-large-v2

Downloading the ALBERT model and tokenizer:

1. Install the Hugging Face CLI:

    pip install -U "hf"

2. Download only the files needed (full-precision ONNX model, config.json and
   the SentencePiece tokenizer.json) into a local directory:

    hf download Xenova/albert-large-v2 \
        onnx/model.onnx config.json tokenizer.json tokenizer_config.json \
        --local-dir albert-large-v2

Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python albert_inference.py --model_dir=<path-to-albert-large-v2-dir> \
                                --text="Paris is the [MASK] of France." \
                                --topk=5
'''

import numpy as np
import argparse
import os
import cv2 as cv

CLS_ID = 2
SEP_ID = 3
MASK_ID = 4

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run ALBERT masked-LM inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the ALBERT model directory (config.json and the ONNX model file are located automatically).')
    parser.add_argument('--text', type=str, default='Paris is the [MASK] of France.', help='Input text containing a single [MASK] token.')
    parser.add_argument('--topk', type=int, default=5, help='Number of top predictions to print.')
    return parser.parse_args()

def find_file(model_dir, filename):
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = [d for d in dirs if d != '.git']
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f'{filename} not found under {model_dir}')

def encode(tokenizer, text):
    ids = [CLS_ID]
    parts = text.split('[MASK]')
    for i, part in enumerate(parts):
        if i > 0:
            ids.append(MASK_ID)
        if part:
            ids.extend(int(t) for t in tokenizer.encode(part.lower()))
    ids.append(SEP_ID)
    return ids

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def albert_inference(net, tokenizer, text, topk):

    print("Inferencing ALBERT model...")

    if '[MASK]' not in text:
        raise ValueError('--text must contain a single [MASK] token')

    ids = encode(tokenizer, text)
    if ids.count(MASK_ID) != 1:
        raise ValueError(f'expected exactly one [MASK] token, got {ids.count(MASK_ID)}')
    mask_pos = ids.index(MASK_ID)

    n = len(ids)
    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.ones((1, n), dtype=np.int64)
    token_type_ids = np.zeros((1, n), dtype=np.int64)

    net.setInput(input_ids, 'input_ids')
    net.setInput(attention_mask, 'attention_mask')
    net.setInput(token_type_ids, 'token_type_ids')
    logits = net.forward('logits')

    row = logits[0, mask_pos]
    probs = softmax(row)
    top_ids = np.argsort(row)[::-1][:topk]

    return [(int(t), tokenizer.decode([int(t)]).strip(), float(probs[t])) for t in top_ids]

if __name__ == '__main__':

    args = parse_args()

    print("Preparing ALBERT model...")
    tokenizer = cv.dnn.Tokenizer.load(find_file(args.model_dir, 'config.json'))

    net = cv.dnn.readNetFromONNX(find_file(args.model_dir, 'model.onnx'), cv.dnn.ENGINE_OPENCV)

    print(f"Text: {args.text}")
    predictions = albert_inference(net, tokenizer, args.text, args.topk)
    for rank, (token_id, token, prob) in enumerate(predictions, start=1):
        print(f"  {rank}. {token!r}  id={token_id}  p={prob:.4f}")
