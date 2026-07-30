# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
This is a sample script demonstrating cv2.vlm: a single API for running vision-language
OCR / document-understanding inference either locally (PaddleOCR-VL-1.5, Granite-Docling-258M)
or via a hosted cloud API (OpenAI, Anthropic, Gemini, Grok), given a model type, a local ONNX
export directory (or cloud model name), and an input image.

Run the script:

    python vlm_ocr.py --model_type=paddleocr-vl --model_dir=<dir> --input=<path-to-image>
    python vlm_ocr.py --model_type=openai --api_key=<key> --input=<path-to-image>
'''

import argparse
import cv2 as cv

MODEL_TYPES = {
    'paddleocr-vl': cv.vlm.VLM_MODEL_PADDLEOCR_VL,
    'granite-docling': cv.vlm.VLM_MODEL_GRANITE_DOCLING,
    'openai': cv.vlm.VLM_MODEL_OPENAI,
    'anthropic': cv.vlm.VLM_MODEL_ANTHROPIC,
    'gemini': cv.vlm.VLM_MODEL_GEMINI,
    'grok': cv.vlm.VLM_MODEL_GROK,
}

CLOUD_MODEL_TYPES = {'openai', 'anthropic', 'gemini', 'grok'}

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run vision-language OCR / '
                                                  'document-understanding inference in OpenCV',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_type', type=str, required=True, choices=sorted(MODEL_TYPES),
                        help='Which VLM to run.')
    parser.add_argument('--model_dir', type=str, default='',
                        help='Local model types: path to the ONNX export directory. '
                             'Cloud model types: provider model name, e.g. gpt-4o (required, no default).')
    parser.add_argument('--api_key', type=str, default='',
                        help='API key for cloud model types (openai/anthropic/gemini/grok); ignored otherwise.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--prompt', type=str, default='', help="Task prompt (default: the model's built-in prompt).")
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--engine', type=str, default='new', choices=['new', 'ort'],
                        help='Local model types only: dnn engine used to load each ONNX sub-model.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Local model types only: compute device.')
    args = parser.parse_args()

    if not args.model_dir:
        parser.error('--model_dir is required (local: ONNX export directory, cloud: provider model name)')
    if args.model_type in CLOUD_MODEL_TYPES and not args.api_key:
        parser.error('--api_key is required for cloud model types')
    return args

if __name__ == '__main__':

    args = parse_args()

    device = 'cloud' if args.model_type in CLOUD_MODEL_TYPES else args.device

    print(f'Preparing {args.model_type} model...')
    model = cv.vlm.create(MODEL_TYPES[args.model_type], args.model_dir,
                          args.engine, device, args.api_key)

    print(f'Running inference on {args.input}...')
    results = model.inferDocument(args.input, args.prompt, args.max_new_tokens)
    for i, text in enumerate(results):
        print(f'Page {i + 1}:\n{text}')
