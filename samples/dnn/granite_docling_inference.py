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
    (onnx/vision_encoder.onnx, onnx/embed_tokens.onnx, onnx/decoder_model_merged.onnx
    -- pass full-precision or any same-named quantized variant to --vision/--embedding/
    --model). This is a real export, not one this script's I/O split invented: its
    decoder_model_merged.onnx already uses the fused GroupQueryAttention op (30 nodes,
    one per layer, do_rotary=1, num_heads=9, kv_num_heads=3, head_dim=64, scale=0.125,
    local_window_size=-1, softcap=0.0) and SkipSimplifiedLayerNormalization (60 nodes),
    landing on both the layer this PR adds and the decomposition path upstream/5.x
    already provides for the other op.

Run the script:
1. Install the required dependencies:

    pip install numpy

2. Run the script:

    python granite_docling_inference.py --vision=<path-to-vision_model.onnx> \\
                                        --embedding=<path-to-embedding.onnx> \\
                                        --model=<path-to-text_decoder.onnx> \\
                                        --tokenizer_path=<path-to-opencv-tokenizer-config.json> \\
                                        --input=<path-to-page-image> \\
                                        --prompt="Convert this page to docling."

'''

import math
import numpy as np
import argparse
import cv2 as cv

IMAGE_SIZE = 512
IMAGE_TOKEN_ID = 100270
IMAGE_SEQ_LEN = 64
FAKE_IMAGE_TOKEN = '<fake_token_around_image>'
GLOBAL_IMAGE_TOKEN = '<global-img>'
IMAGE_TOKEN = '<image>'
EOS_ID = 100257  # <|end_of_text|>

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run Granite-Docling-258M vision-language inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vision', type=str, required=True, help='Path to the vision encoder + connector ONNX model file.')
    parser.add_argument('--embedding', type=str, required=True, help='Path to embedding ONNX model file.')
    parser.add_argument('--model', type=str, required=True, help='Path to the text decoder ONNX model file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer config.json.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input page image.')
    parser.add_argument('--prompt', type=str, default='Convert this page to docling.', help='Task instruction (see the model card for alternatives like "Convert table to OTSL.").')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def resize_for_vision_encoder(height, width, max_edge=IMAGE_SIZE):
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

def preprocess_image(image_path):
    '''Idefics3 image splitting (do_image_splitting=True, this model's configured
    default): round the page up to a multiple of 512 on both sides, cut it into
    512x512 tiles, and append the whole page downscaled to 512x512 as a final "global"
    frame. Each frame becomes IMAGE_SEQ_LEN image tokens, so a 4x4 page yields
    17 * 64 = 1088 of them instead of the 64 a single global view would give -- that
    resolution is what makes body text legible at all (see VERIFICATION STATUS above).

    Returns pixel_values (1, n_frames, 3, 512, 512), an all-valid pixel_attention_mask
    (1, n_frames, 512, 512), and the (rows, cols) tile grid the prompt needs.'''
    img = cv.imread(image_path)
    if img is None:
        raise IOError("Could not read image: " + image_path)

    h, w = img.shape[:2]
    resized_h, resized_w = resize_for_vision_encoder(h, w)
    rows, cols = resized_h // IMAGE_SIZE, resized_w // IMAGE_SIZE

    resized = cv.resize(img, (resized_w, resized_h), interpolation=cv.INTER_CUBIC)
    frames = [_normalize(resized[r * IMAGE_SIZE:(r + 1) * IMAGE_SIZE,
                                 c * IMAGE_SIZE:(c + 1) * IMAGE_SIZE])
              for r in range(rows) for c in range(cols)]
    # The global view goes last, matching Idefics3ImageProcessor.split_images().
    frames.append(_normalize(cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                                       interpolation=cv.INTER_CUBIC)))

    pixel_values = np.stack(frames)[np.newaxis].astype(np.float32)
    pixel_attention_mask = np.ones((1, len(frames), IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    return pixel_values, pixel_attention_mask, (rows, cols)

def build_prompt(task_prompt, grid):
    '''chat_template.jinja's user/assistant turns, with the single <image> placeholder
    it emits expanded exactly as Idefics3Processor.replace_image_token() does for a
    split image: one <row_R_col_C>-tagged block of IMAGE_SEQ_LEN image tokens per tile,
    then the global view. The tile order here must match the frame order
    preprocess_image() stacks, since merge_image_features() pairs them positionally.'''
    rows, cols = grid
    image_block = ''
    for r in range(rows):
        for c in range(cols):
            image_block += (FAKE_IMAGE_TOKEN + f'<row_{r + 1}_col_{c + 1}>' +
                            IMAGE_TOKEN * IMAGE_SEQ_LEN)
        image_block += '\n'
    image_block += ('\n' + FAKE_IMAGE_TOKEN + GLOBAL_IMAGE_TOKEN +
                    IMAGE_TOKEN * IMAGE_SEQ_LEN + FAKE_IMAGE_TOKEN)

    return ('<|start_of_role|>user<|end_of_role|>' + image_block + task_prompt +
            '<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>')

def set_optional_input(net, name, value):
    '''setInput() for a graph input the model may not declare. Returns True if it took.'''
    try:
        net.setInput(value, name)
        return True
    except cv.error:
        return False

def merge_image_features(input_ids, text_embeds, image_features):
    '''masked_scatter equivalent: replace the embedding at every IMAGE_TOKEN_ID
    position, in order, with the next row of image_features. image_features arrives as
    (n_frames, IMAGE_SEQ_LEN, hidden) and is flattened frame-major, so tile k's tokens
    land in the k-th <row_R_col_C> block of the prompt.'''
    merged = text_embeds.copy()
    positions = np.where(input_ids[0] == IMAGE_TOKEN_ID)[0]
    features = image_features.reshape(-1, image_features.shape[-1])
    if len(positions) != features.shape[0]:
        raise ValueError(f"Prompt has {len(positions)} image-token placeholders but the "
                         f"vision encoder produced {features.shape[0]} feature tokens.")
    merged[0, positions, :] = features
    return merged

def granite_docling_inference(vision_net, embed_net, text_net, pixel_values,
                              pixel_attention_mask, prompt, max_new_tokens, tokenizer):

    print("Inferencing Granite-Docling-258M model...")

    tokens = list(tokenizer.encode(prompt))
    input_ids = np.array([tokens], dtype=np.int64)

    # SigLIP2 encoder + pixel-shuffle connector, run over all tiles in one call:
    # each frame -> IMAGE_SEQ_LEN feature tokens already projected to the text hidden
    # size, i.e. (n_frames, 64, 576).
    vision_net.setInput(pixel_values, 'pixel_values')
    vision_net.setInput(pixel_attention_mask, 'pixel_attention_mask')
    image_features = vision_net.forward()

    # Text embedding: token ids -> text embeddings, then splice in the image features
    # at every <image> placeholder position.
    embed_net.setInput(input_ids, 'input_ids')
    text_embeds = embed_net.forward()               # (1, prompt_len, 576)
    inputs_embeds = merge_image_features(input_ids, text_embeds, image_features)

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
        if new_id == EOS_ID:
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

    if generated and generated[-1] == EOS_ID:
        generated.pop()

    return generated

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(args.seed)

    print("Preparing Granite-Docling-258M model...")
    tokenizer = cv.dnn.Tokenizer.load(args.tokenizer_path)

    vision_net = cv.dnn.readNetFromONNX(args.vision, cv.dnn.ENGINE_OPENCV)
    embed_net  = cv.dnn.readNetFromONNX(args.embedding, cv.dnn.ENGINE_OPENCV)
    text_net   = cv.dnn.readNetFromONNX(args.model, cv.dnn.ENGINE_OPENCV)

    print(f"Task:\n{args.prompt}")
    pixel_values, pixel_attention_mask, grid = preprocess_image(args.input)
    print(f"Image split into {grid[0]}x{grid[1]} tiles + 1 global view "
          f"({pixel_values.shape[1] * IMAGE_SEQ_LEN} image tokens)")
    prompt = build_prompt(args.prompt, grid)

    generated = granite_docling_inference(vision_net, embed_net, text_net, pixel_values,
                                          pixel_attention_mask, prompt, args.max_new_tokens, tokenizer)
    response = tokenizer.decode(generated)
    print(f"DocTags:\n{response}")
