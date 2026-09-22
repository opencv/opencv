# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
This is a sample script to run IBM Granite-Docling-258M vision-language inference in
OpenCV using ONNX models. Given a page image, it converts it to DocTags markup
(document structure: text, tables, headers, ...).

Model: https://huggingface.co/ibm-granite/granite-docling-258M
ONNX export: https://huggingface.co/onnx-community/granite-docling-258M-ONNX

This is a real export, not one this script's I/O split invented: its
decoder_model_merged.onnx already uses the fused GroupQueryAttention op (30 nodes,
one per layer, do_rotary=1, num_heads=9, kv_num_heads=3, head_dim=64, scale=0.125,
local_window_size=-1, softcap=0.0) and SkipSimplifiedLayerNormalization (60 nodes),
landing on both the layer this PR adds and the decomposition path upstream/5.x
already provides for the other op.

Model directory layout (matches modules/vlm's engines, so a directory that works
here also works with cv.vlm.create() once that module lands):

    <model_dir>/
      config.json               OpenCV tokenizer config -- NOT HuggingFace's
                                 tokenizer_config.json. Needs model_type/method
                                 (for cv.dnn.Tokenizer.load) plus image_token_id
                                 and eos_token_id (read directly here).
      preprocessor_config.json  image_mean, image_std, max_image_size.longest_edge
      processor_config.json     image_seq_len
      tokenizer.json
      onnx/
        vision_encoder.onnx
        embed_tokens.onnx
        decoder_model_merged.onnx

Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python granite_docling_inference.py --model_dir=<path-to-model-dir> \\
                                        --input=<path-to-page-image> \\
                                        --prompt="Convert this page to docling."

'''

import math
import os
import numpy as np
import argparse
import cv2 as cv

FAKE_IMAGE_TOKEN = '<fake_token_around_image>'
GLOBAL_IMAGE_TOKEN = '<global-img>'
IMAGE_TOKEN = '<image>'

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run Granite-Docling-258M vision-language inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory (see the layout in this script\'s docstring).')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input page image.')
    parser.add_argument('--prompt', type=str, default='Convert this page to docling.', help='Task instruction (see the model card for alternatives like "Convert table to OTSL.").')
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

def resize_for_vision_encoder(height, width, max_edge):
    '''Verbatim from Idefics3ImageProcessor.resize_for_vision_encoder: round both sides
    up to a multiple of max_edge while preserving the aspect ratio, so the image divides
    evenly into max_edge x max_edge tiles.'''
    aspect_ratio = width / height
    if width >= height:
        width = math.ceil(width / max_edge) * max_edge
        height = int(width / aspect_ratio)
        height = math.ceil(height / max_edge) * max_edge
    else:
        height = math.ceil(height / max_edge) * max_edge
        width = int(height * aspect_ratio)
        width = math.ceil(width / max_edge) * max_edge
    return height, width

def _normalize(bgr_tile):
    '''BGR uint8 tile -> normalized CHW float32 in [-1, 1] (mean=std=0.5).'''
    rgb = cv.cvtColor(bgr_tile, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - 0.5) / 0.5).transpose(2, 0, 1)

def preprocess_image(image_path, image_size):
    '''Idefics3 image splitting (do_image_splitting=True, this model's configured
    default): round the page up to a multiple of image_size on both sides, cut it into
    image_size x image_size tiles, and append the whole page downscaled to
    image_size x image_size as a final "global" frame. Each frame becomes
    image_seq_len image tokens at the caller, so a 4x4 page yields 17 * 64 = 1088 of
    them instead of the 64 a single global view would give -- that resolution is what
    makes body text legible at all.

    Returns pixel_values (1, n_frames, 3, image_size, image_size), an all-valid
    pixel_attention_mask (1, n_frames, image_size, image_size), and the (rows, cols)
    tile grid the prompt needs.'''
    img = cv.imread(image_path)
    if img is None:
        raise IOError("Could not read image: " + image_path)

    h, w = img.shape[:2]
    resized_h, resized_w = resize_for_vision_encoder(h, w, image_size)
    rows, cols = resized_h // image_size, resized_w // image_size

    resized = cv.resize(img, (resized_w, resized_h), interpolation=cv.INTER_CUBIC)
    frames = [_normalize(resized[r * image_size:(r + 1) * image_size,
                                 c * image_size:(c + 1) * image_size])
              for r in range(rows) for c in range(cols)]
    # The global view goes last, matching Idefics3ImageProcessor.split_images().
    frames.append(_normalize(cv.resize(img, (image_size, image_size),
                                       interpolation=cv.INTER_CUBIC)))

    pixel_values = np.stack(frames)[np.newaxis].astype(np.float32)
    pixel_attention_mask = np.ones((1, len(frames), image_size, image_size), dtype=bool)
    return pixel_values, pixel_attention_mask, (rows, cols)

def build_prompt(task_prompt, grid, image_seq_len):
    '''chat_template.jinja's user/assistant turns, with the single <image> placeholder
    it emits expanded exactly as Idefics3Processor.replace_image_token() does for a
    split image: one <row_R_col_C>-tagged block of image_seq_len image tokens per tile,
    then the global view. The tile order here must match the frame order
    preprocess_image() stacks, since merge_image_features() pairs them positionally.'''
    rows, cols = grid
    image_block = ''
    for r in range(rows):
        for c in range(cols):
            image_block += (FAKE_IMAGE_TOKEN + f'<row_{r + 1}_col_{c + 1}>' +
                            IMAGE_TOKEN * image_seq_len)
        image_block += '\n'
    image_block += ('\n' + FAKE_IMAGE_TOKEN + GLOBAL_IMAGE_TOKEN +
                    IMAGE_TOKEN * image_seq_len + FAKE_IMAGE_TOKEN)

    return ('<|start_of_role|>user<|end_of_role|>' + image_block + task_prompt +
            '<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>')

def set_optional_input(net, name, value):
    '''setInput() for a graph input the model may not declare. Returns True if it took.'''
    try:
        net.setInput(value, name)
        return True
    except cv.error:
        return False

def merge_image_features(input_ids, text_embeds, image_features, image_token_id):
    '''masked_scatter equivalent: replace the embedding at every image_token_id
    position, in order, with the next row of image_features. image_features arrives as
    (n_frames, image_seq_len, hidden) and is flattened frame-major, so tile k's tokens
    land in the k-th <row_R_col_C> block of the prompt.'''
    merged = text_embeds.copy()
    positions = np.where(input_ids[0] == image_token_id)[0]
    features = image_features.reshape(-1, image_features.shape[-1])
    if len(positions) != features.shape[0]:
        raise ValueError(f"Prompt has {len(positions)} image-token placeholders but the "
                         f"vision encoder produced {features.shape[0]} feature tokens.")
    merged[0, positions, :] = features
    return merged

def granite_docling_inference(vision_net, embed_net, text_net, pixel_values,
                              pixel_attention_mask, prompt, max_new_tokens, tokenizer,
                              eos_id):

    print("Inferencing Granite-Docling-258M model...")

    tokens = list(tokenizer.encode(prompt))
    input_ids = np.array([tokens], dtype=np.int64)

    # SigLIP2 encoder + pixel-shuffle connector, run over all tiles in one call:
    # each frame -> image_seq_len feature tokens already projected to the text hidden
    # size, i.e. (n_frames, 64, 576).
    vision_net.setInput(pixel_values, 'pixel_values')
    vision_net.setInput(pixel_attention_mask, 'pixel_attention_mask')
    image_features = vision_net.forward()

    # Text embedding: token ids -> text embeddings, then splice in the image features
    # at every <image> placeholder position.
    embed_net.setInput(input_ids, 'input_ids')
    text_embeds = embed_net.forward()               # (1, prompt_len, 576)
    inputs_embeds = merge_image_features(input_ids, text_embeds, image_features, image_token_id)

    text_net.enableKVCache()
    prompt_len = inputs_embeds.shape[1]

    # Pre-size the cache so the decode loop allocates no pages. Must precede prefill.
    text_net.reserveKVCache(prompt_len + max_new_tokens)

    # Prefill: process the merged image+text embeddings once to populate the KV-cache.
    text_net.setInput(inputs_embeds, 'inputs_embeds')
    has_mask = set_optional_input(text_net, 'attention_mask',
                                  np.ones((1, prompt_len), dtype=np.int64))
    set_optional_input(text_net, 'position_ids',
                       np.arange(prompt_len, dtype=np.int64).reshape(1, -1))
    logits = text_net.forward()
    new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
    generated = [new_id]

    # Decode: feed one new token's embedding per step; the cache supplies the rest.
    for _ in range(max_new_tokens - 1):
        if new_id == eos_id:
            break
        cur_len = prompt_len + len(generated)
        embed_net.setInput(np.array([[new_id]], dtype=np.int64), 'input_ids')
        new_embed = embed_net.forward()
        text_net.setInput(new_embed, 'inputs_embeds')
        if has_mask:
            text_net.setInput(np.ones((1, cur_len), dtype=np.int64), 'attention_mask')
        set_optional_input(text_net, 'position_ids', np.array([[cur_len - 1]], dtype=np.int64))
        logits = text_net.forward()
        new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
        generated.append(new_id)

    if generated and generated[-1] == eos_id:
        generated.pop()

    return generated

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(args.seed)

    print("Preparing Granite-Docling-258M model...")
    tokenizer = cv.dnn.Tokenizer.load(os.path.join(args.model_dir, 'config.json'))

    config = open_json_config_or_throw(os.path.join(args.model_dir, 'config.json'))
    preprocessor = open_json_config_or_throw(os.path.join(args.model_dir, 'preprocessor_config.json'))
    processor = open_json_config_or_throw(os.path.join(args.model_dir, 'processor_config.json'))

    image_token_id = get_int_with_text_config_fallback(config, 'image_token_id', 0)
    eos_id = get_int_with_text_config_fallback(config, 'eos_token_id', 2)
    image_seq_len = get_int(processor, 'image_seq_len', 64)
    image_size = get_int(preprocessor.getNode('max_image_size'), 'longest_edge', 512)

    onnx_dir = os.path.join(args.model_dir, 'onnx')
    vision_net = cv.dnn.readNetFromONNX(os.path.join(onnx_dir, 'vision_encoder.onnx'), cv.dnn.ENGINE_OPENCV)
    embed_net  = cv.dnn.readNetFromONNX(os.path.join(onnx_dir, 'embed_tokens.onnx'), cv.dnn.ENGINE_OPENCV)
    text_net   = cv.dnn.readNetFromONNX(os.path.join(onnx_dir, 'decoder_model_merged.onnx'), cv.dnn.ENGINE_OPENCV)

    print(f"Task:\n{args.prompt}")
    pixel_values, pixel_attention_mask, grid = preprocess_image(args.input, image_size)
    print(f"Image split into {grid[0]}x{grid[1]} tiles + 1 global view "
          f"({pixel_values.shape[1] * image_seq_len} image tokens)")
    prompt = build_prompt(args.prompt, grid, image_seq_len)

    generated = granite_docling_inference(vision_net, embed_net, text_net, pixel_values,
                                          pixel_attention_mask, prompt, args.max_new_tokens,
                                          tokenizer, eos_id)
    response = tokenizer.decode(generated)
    print(f"DocTags:\n{response}")
