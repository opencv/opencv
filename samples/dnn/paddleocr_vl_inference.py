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

Model directory layout (matches modules/vlm's engines, so a directory that works
here also works with cv.vlm.create() once that module lands):

    <model_dir>/
      config.json               OpenCV tokenizer config -- NOT HuggingFace's
                                 tokenizer_config.json. Needs model_type/method
                                 (for cv.dnn.Tokenizer.load) plus image_token_id
                                 and eos_token_id (read directly here).
      processor_config.json     image_processor: patch_size, merge_size, min_pixels,
                                 max_pixels.
      tokenizer.json
      onnx/
        vision_encoder.onnx
        embedding.onnx
        decoder.onnx

Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python paddleocr_vl_inference.py --model_dir=<path-to-model-dir> \\
                                     --input=<path-to-page-image>
'''

import math
import os
import numpy as np
import argparse
import cv2 as cv

IMAGE_MEAN = 0.5
IMAGE_STD = 0.5

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run PaddleOCR-VL-1.5 vision-language inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory (see the layout in this script\'s docstring).')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input page image.')
    parser.add_argument('--prompt', type=str, default='OCR:', help='Task instruction.')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def open_json_config_or_throw(path):
    '''Mirrors modules/vlm/src/config_json.cpp's openJsonConfigOrThrow, so the config
    reading here matches what cv.vlm will do with the same directory.'''
    fs = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f'vlm: could not open config file: {path}')
    return fs

def get_int(node, name, fallback):
    child = node.getNode(name)
    return int(child.real()) if not child.empty() else fallback

def get_int_with_text_config_fallback(config, name, fallback):
    '''Mirrors config_json.cpp's getIntWithTextConfigFallback: some HF configs nest
    the generation-relevant ids (image_token_id, eos_token_id) under text_config.'''
    node = config.getNode(name)
    if not node.empty():
        return int(node.real())
    text_config = config.getNode('text_config')
    if not text_config.empty():
        child = text_config.getNode(name)
        if not child.empty():
            return int(child.real())
    return fallback

def smart_resize(height, width, factor, min_pixels, max_pixels):
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

def preprocess_image(image_path, patch_size, merge_size, min_pixels, max_pixels):
    '''smart_resize then pack into (1, num_patches, 3, patch_size, patch_size) +
    image_grid_thw (1, 3), matching the real vision encoder's input contract. Ported
    from image_processing_paddleocr_vl.py's _preprocess (temporal_patch_size is
    asserted to be 1 there, so no temporal tiling happens for a single image).'''
    img = cv.imread(image_path)
    if img is None:
        raise IOError("Could not read image: " + image_path)
    h, w = img.shape[:2]
    factor = patch_size * merge_size
    resized_h, resized_w = smart_resize(h, w, factor, min_pixels, max_pixels)

    img = cv.resize(img, (resized_w, resized_h), interpolation=cv.INTER_CUBIC)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGE_MEAN) / IMAGE_STD
    img = img.transpose(2, 0, 1)[np.newaxis]        # (1, 3, H, W)

    grid_t, grid_h, grid_w = 1, resized_h // patch_size, resized_w // patch_size
    patches = img.reshape(grid_t, 1, 3, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    pixel_values = patches.reshape(1, grid_t * grid_h * grid_w, 3, patch_size, patch_size).astype(np.float32)
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

def merge_image_features(input_ids, text_embeds, image_features, image_token_id):
    '''masked_scatter equivalent: replace the embedding at every image_token_id
    position, in order, with the next row of image_features.'''
    merged = text_embeds.copy()
    positions = np.where(input_ids[0] == image_token_id)[0]
    if len(positions) != image_features.shape[0]:
        raise ValueError(f"Prompt has {len(positions)} image-token placeholders but the "
                         f"vision encoder produced {image_features.shape[0]} feature tokens.")
    merged[0, positions, :] = image_features
    return merged

def paddleocr_vl_inference(vision_net, embed_net, decoder_net, pixel_values, image_grid_thw,
                           prompt, max_new_tokens, tokenizer, image_token_id, eos_id):

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
    inputs_embeds = merge_image_features(input_ids, text_embeds, image_features, image_token_id)

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
        if new_id == eos_id:
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

    if generated and generated[-1] == eos_id:
        generated.pop()

    return generated

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(args.seed)

    print("Preparing PaddleOCR-VL-1.5 model...")
    tokenizer = cv.dnn.Tokenizer.load(os.path.join(args.model_dir, 'config.json'))

    config = open_json_config_or_throw(os.path.join(args.model_dir, 'config.json'))
    processor = open_json_config_or_throw(os.path.join(args.model_dir, 'processor_config.json'))
    image_processor = processor.getNode('image_processor')

    image_token_id = get_int_with_text_config_fallback(config, 'image_token_id', 0)
    eos_id = get_int_with_text_config_fallback(config, 'eos_token_id', 2)
    patch_size = get_int(image_processor, 'patch_size', 14)
    merge_size = get_int(image_processor, 'merge_size', 2)
    min_pixels = get_int(image_processor, 'min_pixels', 28 * 28 * 130)
    max_pixels = get_int(image_processor, 'max_pixels', 28 * 28 * 1280)

    onnx_dir = os.path.join(args.model_dir, 'onnx')
    vision_net  = cv.dnn.readNetFromONNX(os.path.join(onnx_dir, 'vision_encoder.onnx'), cv.dnn.ENGINE_OPENCV)
    embed_net   = cv.dnn.readNetFromONNX(os.path.join(onnx_dir, 'embedding.onnx'), cv.dnn.ENGINE_OPENCV)
    decoder_net = cv.dnn.readNetFromONNX(os.path.join(onnx_dir, 'decoder.onnx'), cv.dnn.ENGINE_OPENCV)

    print(f"Task:\n{args.prompt}")
    pixel_values, image_grid_thw = preprocess_image(args.input, patch_size, merge_size,
                                                     min_pixels, max_pixels)

    generated = paddleocr_vl_inference(vision_net, embed_net, decoder_net, pixel_values,
                                       image_grid_thw, args.prompt, args.max_new_tokens,
                                       tokenizer, image_token_id, eos_id)
    response = tokenizer.decode(generated)
    print(f"Response:\n{response}")
