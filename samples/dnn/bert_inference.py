# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
This is a sample script to run BERT (bert-base-uncased) masked-LM inference
in OpenCV using an ONNX model. The input text must contain a single literal
"[MASK]" token; the script prints the top predictions for that position.

Model: https://huggingface.co/google-bert/bert-base-uncased

Downloading the BERT model and tokenizer:

1. Install the Hugging Face CLI:

    pip install -U "hf"

2. Download only the files needed (model.onnx, config.json and the WordPiece
   tokenizer.json/vocab.txt) into a local directory:

    hf download google-bert/bert-base-uncased \
        model.onnx config.json tokenizer.json tokenizer_config.json vocab.txt \
        --local-dir bert-base-uncased

Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python bert_inference.py --model_dir=<path-to-bert-base-uncased-dir> \
                              --text="Paris is the [MASK] of France." \
                              --topk=5
'''

import numpy as np
import argparse
import os
import cv2 as cv

MASK_TOKEN = '[MASK]'

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run BERT masked-LM inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the BERT model directory (config.json and the ONNX model file are located automatically).')
    parser.add_argument('--text', type=str, default='Paris is the [MASK] of France.', help='Input text containing a single [MASK] token.')
    parser.add_argument('--topk', type=int, default=5, help='Number of top predictions to print.')
    return parser.parse_args()

def find_file(model_dir, filename):
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = [d for d in dirs if d != '.git']
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f'{filename} not found under {model_dir}')

def encode_with_mask(tokenizer, text):
    if text.count(MASK_TOKEN) != 1:
        raise ValueError('expected exactly one [MASK] token')
    mask_id = tokenizer.encode(MASK_TOKEN)[1]
    ids = list(tokenizer.encode(text))
    mask_pos = ids.index(mask_id)
    return ids, mask_pos

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def bert_inference(net, tokenizer, text, topk):

    print("Inferencing BERT model...")

    ids, mask_pos = encode_with_mask(tokenizer, text)
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

    print("Preparing BERT model...")
    tokenizer = cv.dnn.Tokenizer.load(find_file(args.model_dir, 'config.json'))

    net = cv.dnn.readNetFromONNX(find_file(args.model_dir, 'model.onnx'), cv.dnn.ENGINE_OPENCV)

    print(f"Text: {args.text}")
    predictions = bert_inference(net, tokenizer, args.text, args.topk)
    for rank, (token_id, token, prob) in enumerate(predictions, start=1):
        print(f"  {rank}. {token!r}  id={token_id}  p={prob:.4f}")
