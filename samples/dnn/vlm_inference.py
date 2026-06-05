# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
This is a sample script to run PaliGemma2 vision-language inference in OpenCV using
ONNX models. Given an image and a text prompt, it generates a text response
(e.g. a caption).

The model is split into three ONNX files:
    - SigLIP vision encoder : image -> 256 image-feature tokens
    - Embedding             : prompt token ids -> text embeddings
    - Gemma2 language model : [image_features | text_embeds] -> logits

Model: https://huggingface.co/google/paligemma2-3b-pt-224
ONNX:  https://huggingface.co/nklskyoy/paligemma2-3b-pt-224-onnx

Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python vlm_inference.py --siglip=<path-to-vision_model.onnx> \
                            --embedding=<path-to-embedding.onnx> \
                            --gemma=<path-to-gemma2_3b.onnx> \
                            --tokenizer_path=<path-to-opencv-tokenizer-config.json> \
                            --input=<path-to-image> \
                            --prompt="cap en\n"

    The tokenizer_path should point to an OpenCV-format config.json, NOT the
    HuggingFace tokenizer_config.json.
'''

import numpy as np
import argparse
import cv2 as cv

EOS_ID = 1

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run PaliGemma2 vision-language inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--siglip', type=str, required=True, help='Path to SigLIP vision encoder ONNX model file.')
    parser.add_argument('--embedding', type=str, required=True, help='Path to embedding ONNX model file.')
    parser.add_argument('--gemma', type=str, required=True, help='Path to Gemma2 language model ONNX model file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer config.json.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--prompt', type=str, default='cap en\n', help='Task prompt (e.g. "cap en\\n" to caption in English).')
    parser.add_argument('--max_new_tokens', type=int, default=64, help='Maximum number of new tokens to generate.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def preprocess_image(image_path):
    '''Resize to 224x224 and normalize to [-1, 1] in CHW order (SigLIP: mean=0.5, std=0.5).'''
    img = cv.imread(image_path)
    if img is None:
        raise IOError("Could not read image: " + image_path)
    img = cv.resize(img, (224, 224))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)[np.newaxis]
    return img

def vlm_inference(siglip_net, embed_net, gemma_net, pixel_values, prompt, max_new_tokens, tokenizer):

    print("Inferencing PaliGemma2 model...")

    tokens = list(tokenizer.encode(prompt))
    input_ids = np.array([tokens], dtype=np.int64)

    # SigLIP vision encoder: image -> image-feature tokens
    siglip_net.setInput(pixel_values, 'pixel_values')
    image_features = siglip_net.forward()        # (1, 256, 2304)

    # Text embedding: token ids -> text embeddings
    embed_net.setInput(input_ids, 'input_ids')
    text_embeds = embed_net.forward()            # (1, text_len, 2304)

    # Combine [image_features | text_embeds]
    inputs_embeds = np.concatenate([image_features, text_embeds], axis=1)

    generated = []

    # Prefill
    gemma_net.setInput(inputs_embeds, 'inputs_embeds')
    logits = gemma_net.forward()
    new_id = int(np.argmax(logits[0, -1, :]))
    generated.append(new_id)

    # Decode (no KV-cache: feed full growing sequence each step)
    for _ in range(max_new_tokens - 1):
        if new_id == EOS_ID:
            break
        embed_net.setInput(np.array([[new_id]], dtype=np.int64), 'input_ids')
        new_embed     = embed_net.forward()
        inputs_embeds = np.concatenate([inputs_embeds, new_embed], axis=1)
        gemma_net.setInput(inputs_embeds, 'inputs_embeds')
        logits        = gemma_net.forward()
        new_id        = int(np.argmax(logits[0, -1, :]))
        generated.append(new_id)

    if generated and generated[-1] == EOS_ID:
        generated.pop()

    return generated

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(args.seed)

    print("Preparing PaliGemma2 model...")
    tokenizer = cv.dnn.Tokenizer.load(args.tokenizer_path)

    siglip_net = cv.dnn.readNetFromONNX(args.siglip, cv.dnn.ENGINE_NEW)
    embed_net  = cv.dnn.readNetFromONNX(args.embedding, cv.dnn.ENGINE_NEW)
    gemma_net  = cv.dnn.readNetFromONNX(args.gemma, cv.dnn.ENGINE_NEW)

    print(f"Prompt:\n{args.prompt}")
    pixel_values = preprocess_image(args.input)

    generated = vlm_inference(siglip_net, embed_net, gemma_net, pixel_values,
                              args.prompt, args.max_new_tokens, tokenizer)
    response = tokenizer.decode(generated)
    print(f"Response:\n{response}")
