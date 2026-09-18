# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
This is a sample script to run PaddleOCR-VL-1.5 vision-language inference in OpenCV
using ONNX models. Given a page image, it recognizes and outputs its text/layout.

Model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
ONNX:  https://huggingface.co/onnx-community/PaddleOCR-VL-1.5-ONNX

Unlike granite_docling_inference.py, this script is verified end to end against the
real published ONNX export above, not just against the model's config:
    - vision encoder (pixel_values + image_grid_thw -> image_embeds): loaded and run
      through OpenCV's engine on the real (quantized) weights; output shape matched
      onnxruntime run on the same real weights exactly (609x1024 for a 600x800 test
      image, i.e. num_patches/merge_size^2 with no fudging).
    - decoder (inputs_embeds + attention_mask, KV-cache via present.*/past_key_values.*):
      loaded and run through OpenCV's engine on the real (full-precision) weights;
      both a prefill forward() and one enableKVCache()-backed decode-step forward()
      produced logits at the expected vocab size (103424) and sequence positions.
      reserveKVCache() logs "has no effect" for this decoder -- expected, since its
      attention is decomposed rather than a fused paged-attention op; the
      present.*/past_key_values.* routing that actually carries the cache is unaffected.
    - the full chain -- real vision encoder, real tokenizer, real embedding, the
      merge, and a real decoder prefill -- was run together once and produced the
      expected (1, prompt_len, 103424) logits with no shape mismatches anywhere
      in between.
    - embedding (input_ids -> embeddings): loaded and run through OpenCV's engine on
      the real weights.
    - the smart_resize + patch-packing preprocessing below was copied from
      PaddlePaddle/PaddleOCR-VL-1.5's own image_processing_paddleocr_vl.py (reading
      the literal source, not a paraphrase of it -- an early summary of that file
      mislabeled temporal tiling and had to be corrected against the raw file), then
      exercised against the real vision encoder above.
    - image_token_id, the special image-wrapper tokens (<|IMAGE_START|>,
      <|IMAGE_PLACEHOLDER|>, <|IMAGE_END|>) and the chat template (User: .../
      Assistant:\\n) were read from this ONNX repo's own tokenizer_config.json and
      chat_template.jinja, not guessed.

What this script does NOT do: this export's decoder graph decomposes attention into
primitive MatMul/Softmax ops rather than the fused com.microsoft::GroupQueryAttention
node -- it does not exercise the GroupQueryAttentionLayer this PR adds. It does confirm
PaddleOCR-VL-1.5 runs in OpenCV's DNN engine today, independent of that layer.

Note on quantized variants: the int4-kquant and int8 (MatMulInteger) decoder variants
in that repo do not currently load in OpenCV -- int4-kquant uses MatMulNBits' asymmetric
(with zero-point) form, which OpenCV's importer only supports in its symmetric 3-input
form, and the int8 variant uses ai.onnx MatMulInteger, which isn't implemented at all.
Neither gap is specific to this PR. Use the full-precision decoder.onnx (or export a
symmetric-quantized variant) until one of those is addressed.

Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python paddleocr_vl_inference.py --vision=<path-to-vision_encoder.onnx> \\
                                     --embedding=<path-to-embedding.onnx> \\
                                     --model=<path-to-decoder.onnx> \\
                                     --tokenizer_path=<path-to-opencv-tokenizer-config.json> \\
                                     --input=<path-to-page-image>

    The tokenizer_path should point to an OpenCV-format config.json, NOT the
    HuggingFace tokenizer_config.json.
'''

import math
import numpy as np
import argparse
import cv2 as cv

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28
MIN_PIXELS = 112896
MAX_PIXELS = 1003520
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5

IMAGE_TOKEN_ID = 100295
EOS_ID = 2

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run PaddleOCR-VL-1.5 vision-language inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vision', type=str, required=True, help='Path to the vision encoder ONNX model file.')
    parser.add_argument('--embedding', type=str, required=True, help='Path to embedding ONNX model file.')
    parser.add_argument('--model', type=str, required=True, help='Path to the decoder ONNX model file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer config.json.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input page image.')
    parser.add_argument('--prompt', type=str, default='OCR:', help='Task instruction.')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    '''Verbatim from PaddleOCR-VL-1.5's image_processing_paddleocr_vl.py.'''
    if height < factor or width < factor:
        if height < width:
            width = round(factor * width / height)
            height = factor
        else:
            height = round(factor * height / width)
            width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio must be <= 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def preprocess_image(image_path):
    '''smart_resize then pack into (1, num_patches, 3, 14, 14) + image_grid_thw (1, 3),
    matching the real vision encoder's input contract. Ported from
    image_processing_paddleocr_vl.py's _preprocess (temporal_patch_size is asserted
    to be 1 there, so no temporal tiling happens for a single image).'''
    img = cv.imread(image_path)
    if img is None:
        raise IOError("Could not read image: " + image_path)
    h, w = img.shape[:2]
    resized_h, resized_w = smart_resize(h, w)

    img = cv.resize(img, (resized_w, resized_h), interpolation=cv.INTER_CUBIC)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGE_MEAN) / IMAGE_STD
    img = img.transpose(2, 0, 1)[np.newaxis]        # (1, 3, H, W)

    grid_t, grid_h, grid_w = 1, resized_h // PATCH_SIZE, resized_w // PATCH_SIZE
    patches = img.reshape(grid_t, 1, 3, grid_h, PATCH_SIZE, grid_w, PATCH_SIZE)
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    pixel_values = patches.reshape(1, grid_t * grid_h * grid_w, 3, PATCH_SIZE, PATCH_SIZE).astype(np.float32)
    image_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, image_grid_thw

def build_prompt(task_prompt, num_image_tokens):
    '''PaddleOCR-VL-1.5's chat template (chat_template.jinja): a single-turn user
    message with the image wrapped in IMAGE_START/END, no assistant content yet.'''
    image_block = '<|IMAGE_START|>' + ('<|IMAGE_PLACEHOLDER|>' * num_image_tokens) + '<|IMAGE_END|>'
    return f'<|begin_of_sentence|>User: {image_block}{task_prompt}\nAssistant:\n'

def set_optional_input(net, name, value):
    '''setInput() for a graph input the model may not declare. Returns True if it took.'''
    try:
        net.setInput(value, name)
        return True
    except cv.error:
        return False

def merge_image_features(input_ids, text_embeds, image_features):
    '''masked_scatter equivalent: replace the embedding at every IMAGE_TOKEN_ID
    position, in order, with the next row of image_features.'''
    merged = text_embeds.copy()
    positions = np.where(input_ids[0] == IMAGE_TOKEN_ID)[0]
    if len(positions) != image_features.shape[0]:
        raise ValueError(f"Prompt has {len(positions)} image-token placeholders but the "
                         f"vision encoder produced {image_features.shape[0]} feature tokens.")
    merged[0, positions, :] = image_features
    return merged

def paddleocr_vl_inference(vision_net, embed_net, decoder_net, pixel_values, image_grid_thw,
                           prompt, max_new_tokens, tokenizer):

    print("Inferencing PaddleOCR-VL-1.5 model...")

    # Vision encoder: patches -> merged image-feature tokens, one per merge_size^2
    # block, already projected to the decoder's hidden size (1024).
    vision_net.setInput(pixel_values, 'pixel_values')
    vision_net.setInput(image_grid_thw, 'image_grid_thw')
    image_features = vision_net.forward()           # (num_merged_tokens, 1024)

    num_image_tokens = image_features.shape[0]
    full_prompt = build_prompt(prompt, num_image_tokens)
    tokens = list(tokenizer.encode(full_prompt))
    input_ids = np.array([tokens], dtype=np.int64)

    embed_net.setInput(input_ids, 'input_ids')
    text_embeds = embed_net.forward()                # (1, prompt_len, 1024)
    inputs_embeds = merge_image_features(input_ids, text_embeds, image_features)

    decoder_net.enableKVCache()
    prompt_len = inputs_embeds.shape[1]
    decoder_net.reserveKVCache(prompt_len + max_new_tokens)

    # Prefill: process the merged image+text embeddings once to populate the KV-cache.
    decoder_net.setInput(inputs_embeds, 'inputs_embeds')
    has_mask = set_optional_input(decoder_net, 'attention_mask',
                                  np.ones((1, prompt_len), dtype=np.int64))
    logits = decoder_net.forward()
    new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
    generated = [new_id]

    # Decode: feed one new token's embedding per step; the cache supplies the rest.
    for _ in range(max_new_tokens - 1):
        if new_id == EOS_ID:
            break
        cur_len = prompt_len + len(generated)
        embed_net.setInput(np.array([[new_id]], dtype=np.int64), 'input_ids')
        new_embed = embed_net.forward()
        decoder_net.setInput(new_embed, 'inputs_embeds')
        if has_mask:
            decoder_net.setInput(np.ones((1, cur_len), dtype=np.int64), 'attention_mask')
        logits = decoder_net.forward()
        new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
        generated.append(new_id)

    if generated and generated[-1] == EOS_ID:
        generated.pop()

    return generated

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(args.seed)

    print("Preparing PaddleOCR-VL-1.5 model...")
    tokenizer = cv.dnn.Tokenizer.load(args.tokenizer_path)

    vision_net  = cv.dnn.readNetFromONNX(args.vision, cv.dnn.ENGINE_OPENCV)
    embed_net   = cv.dnn.readNetFromONNX(args.embedding, cv.dnn.ENGINE_OPENCV)
    decoder_net = cv.dnn.readNetFromONNX(args.model, cv.dnn.ENGINE_OPENCV)

    print(f"Task:\n{args.prompt}")
    pixel_values, image_grid_thw = preprocess_image(args.input)

    generated = paddleocr_vl_inference(vision_net, embed_net, decoder_net, pixel_values,
                                       image_grid_thw, args.prompt, args.max_new_tokens, tokenizer)
    response = tokenizer.decode(generated)
    print(f"Response:\n{response}")
