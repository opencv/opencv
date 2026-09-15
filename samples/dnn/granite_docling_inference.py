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

Architecture (Idefics3: SigLIP2 vision encoder + pixel-shuffle connector + Llama-style
text decoder with GroupQueryAttention):
    - Vision  : 512x512 image -> SigLIP2 encoder -> pixel-shuffle connector -> 64
                image-feature tokens already projected to the text hidden size (576).
    - Embedding: prompt token ids -> text embeddings.
    - Text decoder: GroupQueryAttention (9 heads / 3 KV heads, rotary, no sliding
                window/softcap) with KV-cache, same shape this PR's
                GroupQueryAttentionLayer implements.

The three ONNX files (vision, embedding, text decoder) are not split by an official
export anywhere; this script defines that split itself, the same way vlm_inference.py
(PaliGemma2) and qwen_inference.py do for their models. No ONNX export of this model
exists yet, so this script has not been run end to end against real weights.

VERIFICATION STATUS (what is/isn't grounded in something checked):
    - image preprocessing (512x512, mean/std=0.5), the pixel-shuffle formula, the
      masked_scatter image-token merge, image_seq_len=64, image_token_id=100270 and
      the <fake_token_around_image>/<image> wrapper: read directly from
      ibm-granite/granite-docling-258M's own config/processor/tokenizer files and the
      Idefics3 reference implementation in transformers.
    - the text decoder's do_rotary=1 GroupQueryAttention shape: verified in this
      session by exporting a Llama model built from this model's exact published
      text_config through onnxruntime-genai's model_builder and inspecting the
      resulting GroupQueryAttention node's attributes.
    - the outer chat wrapper (<|start_of_role|>user<|end_of_role|>...) follows the
      standard IBM Granite chat format consistent with this model's bos/eos tokens;
      no chat_template.json was available to confirm it token-for-token.
    - single global image (do_image_splitting=False) is used instead of Idefics3's
      default tiling for large pages: a real, supported processor mode, chosen here
      to keep this first sample tractable, not an approximation of a different mode.
    - merge_image_features() (the placeholder-position scatter) was exercised end to
      end through cv.dnn against synthetic vision/embedding ONNX models shaped like
      this script expects (64 image tokens, 576 hidden), confirming it touches exactly
      the <image>-token rows and nothing else. The KV-cache decode loop itself was not
      run here; it follows qwen_inference.py's enableKVCache()/reserveKVCache() usage
      combined with vlm_inference.py's inputs_embeds input, each already exercised
      independently by those samples, but not together in this one.

Exporting the model to ONNX (outline - adjust to your export tooling):
    The vision tower + pixel-shuffle connector, the embedding lookup, and the text
    decoder need separate ONNX graphs, each corresponding to one of --vision,
    --embedding and --model below. The text decoder should be exported with a fused
    GroupQueryAttention node (do_rotary=1) the same way as the opset-23 dynamo export
    documented in qwen_inference.py, so it lands on the layer this PR adds.

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

    The tokenizer_path should point to an OpenCV-format config.json, NOT the
    HuggingFace tokenizer_config.json.
'''

import numpy as np
import argparse
import cv2 as cv

IMAGE_SIZE = 512
IMAGE_TOKEN_ID = 100270
IMAGE_SEQ_LEN = 64
FAKE_IMAGE_TOKEN = '<fake_token_around_image>'
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

def preprocess_image(image_path):
    '''Resize to 512x512 (stretched, matching do_image_splitting=False) and normalize
    to [-1, 1] in CHW order (mean=std=0.5).'''
    img = cv.imread(image_path)
    if img is None:
        raise IOError("Could not read image: " + image_path)
    img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)[np.newaxis]
    return img

def build_prompt(task_prompt):
    '''IBM Granite chat format wrapping a single global image (no tiling) followed by
    the task instruction.'''
    image_block = FAKE_IMAGE_TOKEN + (IMAGE_TOKEN * IMAGE_SEQ_LEN) + FAKE_IMAGE_TOKEN
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
    position, in order, with the next row of image_features.'''
    merged = text_embeds.copy()
    positions = np.where(input_ids[0] == IMAGE_TOKEN_ID)[0]
    if len(positions) != image_features.shape[1]:
        raise ValueError(f"Prompt has {len(positions)} image-token placeholders but the "
                         f"vision encoder produced {image_features.shape[1]} feature tokens.")
    merged[0, positions, :] = image_features[0]
    return merged

def granite_docling_inference(vision_net, embed_net, text_net, pixel_values, prompt,
                              max_new_tokens, tokenizer):

    print("Inferencing Granite-Docling-258M model...")

    tokens = list(tokenizer.encode(prompt))
    input_ids = np.array([tokens], dtype=np.int64)

    # SigLIP2 encoder + pixel-shuffle connector: image -> 64 image-feature tokens,
    # already projected to the text hidden size.
    vision_net.setInput(pixel_values, 'pixel_values')
    image_features = vision_net.forward()          # (1, 64, 576)

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
    text_net.setInput(np.arange(prompt_len, dtype=np.int64).reshape(1, -1), 'position_ids')
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
        text_net.setInput(np.array([[cur_len - 1]], dtype=np.int64), 'position_ids')
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

    prompt = build_prompt(args.prompt)
    print(f"Task:\n{args.prompt}")
    pixel_values = preprocess_image(args.input)

    generated = granite_docling_inference(vision_net, embed_net, text_net, pixel_values,
                                          prompt, args.max_new_tokens, tokenizer)
    response = tokenizer.decode(generated)
    print(f"DocTags:\n{response}")
