#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
Text-to-speech synthesis with Qwen3-TTS: reads a sentence and writes the spoken audio
as a .wav file.

How to use:
    Sample command to run:
        `python qwen3_tts.py --model=<onnx dir> --tokenizer=<tokenizer dir> --text="Hello."`

    Download the cpu_fp32 variant of the model (a directory of ONNX graphs) from
        https://huggingface.co/onnx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice
    and pass that directory with --model. Speech synthesis runs six of its graphs:
        text_embed     : text_ids[1,S] i64            -> [1,S,1024] f32
        codec_embed    : codec_ids[1,S] i64           -> [1,S,1024] f32
        talker_cache   : inputs_embeds[1,T,1024] f32 + position_ids[3,1,T] i64
                         + attention_mask[1,T] i64 + 56 past K/V [1,8,past,128] f32
                         -> logits[1,T,3072] f32, hidden_states[1,T,1024] f32
        code_predictor : talker_hidden[1,1024] f32 + codec_ids[1,16] i64 -> [1,15,2048] f32
        residual_embed : codec_ids[1,16] i64          -> [1,1024] f32
        tok_decoder    : audio_codes[1,25,16] i64     -> waveform[1,1,L] f32

    --tokenizer takes a directory holding config.json and tokenizer.json of any Qwen2-family
    tokenizer; Qwen3-TTS reuses that vocabulary and adds only ids this sample never encodes.
'''

import argparse
import json
import os
import wave

import cv2 as cv
import numpy as np

from common import *

# Audio and graph geometry, all fixed by the export.
SAMPLE_RATE = 24000
HIDDEN = 1024
GROUPS = 16       # codes per frame
DEC_FRAMES = 25   # tok_decoder is exported at a fixed frame count
LAYERS, KV_HEADS, HEAD_DIM = 28, 8, 128

# Control ids, from the model's config.json: the first three are top-level, the rest come
# from its talker_config. Codes in the top 1024 of the vocabulary are control ids.
TTS_BOS, TTS_EOS, TTS_PAD = 151672, 151673, 151671
CODEC_PAD, CODEC_BOS, CODEC_EOS = 2148, 2149, 2150
CODEC_THINK, CODEC_NOTHINK = 2154, 2155
CODEC_THINK_BOS, CODEC_THINK_EOS = 2156, 2157
VOCAB, CONTROL_IDS = 3072, 1024
REPETITION_PENALTY = 1.05
MAX_CONSECUTIVE_REPEAT = 12

# config.json talker_config: spk_id and codec_language_id.
SPEAKER_IDS = {'serena': 3066, 'vivian': 3065, 'uncle_fu': 3010, 'ryan': 3061, 'aiden': 2861,
               'ono_anna': 2873, 'sohee': 2864, 'eric': 2875, 'dylan': 2878}
LANGUAGE_IDS = {'chinese': 2055, 'english': 2050, 'german': 2053, 'italian': 2070,
                'portuguese': 2071, 'spanish': 2054, 'japanese': 2058, 'korean': 2064,
                'french': 2061, 'russian': 2069, 'beijing_dialect': 2074,
                'sichuan_dialect': 2062}

def help():
    print(
        '''
        Use this script for text-to-speech synthesis with Qwen3-TTS using OpenCV.

        To run:
            python qwen3_tts.py --model=<onnx dir> --tokenizer=<tokenizer dir> \\
                --text="OpenCV is an open source computer vision library." --speaker=serena

        Synthesis is greedy, so the same text always produces the same audio.
        '''
    )

def get_args_parser():
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(description='Text-to-speech synthesis with Qwen3-TTS.')
    parser.add_argument('--model', help='Directory with the Qwen3-TTS ONNX graphs.')
    parser.add_argument('--tokenizer', help='Directory with config.json and tokenizer.json.')
    parser.add_argument('--text', default='OpenCV is an open source computer vision library.',
                        help='Text to synthesize.')
    parser.add_argument('--speaker', default='serena', choices=sorted(SPEAKER_IDS),
                        help='Speaker name.')
    parser.add_argument('--language', default='english',
                        choices=sorted(LANGUAGE_IDS) + ['auto'],
                        help='Language tag, or auto to let the model decide.')
    parser.add_argument('--output', default='qwen3_tts.wav', help='Path of the .wav file to write.')
    parser.add_argument('--max_frames', type=int, default=400,
                        help='Stop after this many code frames (about 12 per second).')
    parser.add_argument('--backend', default='default', choices=backends,
                        help='Computation backend.')
    parser.add_argument('--target', default='cpu', choices=targets,
                        help='Target computation device.')
    return parser.parse_args()

def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError('cannot open ' + path)

def embed(net, name, ids):
    net.setInput(np.asarray(ids, np.int64).reshape(1, -1), name)
    return net.forward().reshape(-1, HIDDEN)

def special_token_id(tokenizer_dir, token):
    path = os.path.join(tokenizer_dir, 'tokenizer_config.json')
    require_file(path)
    with open(path, 'rt', encoding='utf-8') as f:
        added = json.load(f).get('added_tokens_decoder', {})
    for token_id, entry in added.items():
        if entry.get('content') == token:
            return int(token_id)
    raise ValueError('token %s is missing from %s' % (token, path))

def write_wav(path, samples, rate):
    with wave.open(path, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype('<i2').tobytes())

def main():
    args = get_args_parser()
    if args.model is None or args.tokenizer is None:
        help()
        exit(1)

    cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_WARNING)

    tokenizer_config = os.path.join(args.tokenizer, 'config.json')
    require_file(tokenizer_config)
    require_file(os.path.join(args.tokenizer, 'tokenizer.json'))
    tokenizer = cv.dnn.Tokenizer.load(tokenizer_config)

    role_ids = [special_token_id(args.tokenizer, '<|im_start|>')] + \
        list(tokenizer.encode('assistant\n'))
    text_ids = list(tokenizer.encode(args.text))
    if not text_ids:
        raise ValueError('the text tokenized to nothing')

    nets = {}
    for name in ('text_embed', 'codec_embed', 'talker_cache',
                 'code_predictor', 'residual_embed', 'tok_decoder'):
        path = os.path.join(args.model, name + '.onnx')
        require_file(path)
        print('loading  : %s' % name)
        nets[name] = cv.dnn.readNetFromONNX(path)
        nets[name].setPreferableBackend(get_backend_id(args.backend))
        nets[name].setPreferableTarget(get_target_id(args.target))

    # Prefill: role embeddings, then the codec tag block, then the text and its closing tags.
    special = embed(nets['text_embed'], 'text_ids', [TTS_BOS, TTS_EOS, TTS_PAD])
    bos_embed, eos_embed, pad_embed = special[0:1], special[1:2], special[2:3]

    if args.language == 'auto':
        tags = [CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS]
    else:
        tags = [CODEC_THINK, CODEC_THINK_BOS, LANGUAGE_IDS[args.language], CODEC_THINK_EOS]
    codec_input = np.concatenate([embed(nets['codec_embed'], 'codec_ids', tags),
                                  embed(nets['codec_embed'], 'codec_ids',
                                        [SPEAKER_IDS[args.speaker]]),
                                  embed(nets['codec_embed'], 'codec_ids',
                                        [CODEC_PAD, CODEC_BOS])])

    pad_block = np.concatenate([np.repeat(pad_embed, len(codec_input) - 2, axis=0), bos_embed])

    text_body = embed(nets['text_embed'], 'text_ids', text_ids)
    text_block = np.concatenate([text_body, eos_embed]) + \
        embed(nets['codec_embed'], 'codec_ids', [CODEC_PAD] * (len(text_body) + 1))
    start_block = embed(nets['codec_embed'], 'codec_ids', [CODEC_BOS]) + pad_embed

    talker_in = np.concatenate([embed(nets['text_embed'], 'text_ids', role_ids),
                                pad_block + codec_input[:-1],
                                text_block, start_block])

    empty_past = np.zeros((1, KV_HEADS, 0, HEAD_DIM), np.float32)
    past_names = ['past_kv' if l == 0 and kv == 0 else 'past_kv_%d_%d' % (l, kv)
                  for l in range(LAYERS) for kv in range(2)]

    print('text     : "%s" (%d tokens)' % (args.text, len(text_ids)))

    codes = []
    seen_first = np.zeros(VOCAB - CONTROL_IDS, bool)
    last_first = -1
    repeat_run = 0
    seen_sub = [np.zeros(VOCAB - CONTROL_IDS, bool) for _ in range(GROUPS)]
    last_sub = [-1] * GROUPS
    repeat_run_sub = [0] * GROUPS
    for _ in range(args.max_frames):
        total = len(talker_in)
        talker = nets['talker_cache']
        talker.setInput(talker_in.reshape(1, total, HIDDEN).astype(np.float32), 'inputs_embeds')
        talker.setInput(np.tile(np.arange(total, dtype=np.int64), (3, 1, 1)), 'position_ids')
        talker.setInput(np.ones((1, total), np.int64), 'attention_mask')
        for name in past_names:
            talker.setInput(empty_past, name)
        logits, hidden = talker.forward(['logits', 'hidden_states'])

        step = logits[0, -1]
        allowed = step[:VOCAB - CONTROL_IDS].astype(np.float64).copy()
        penalized = allowed[seen_first]
        allowed[seen_first] = np.where(penalized < 0, penalized * REPETITION_PENALTY,
                                       penalized / REPETITION_PENALTY)
        first = int(np.argmax(allowed))
        if step[CODEC_EOS] > allowed[first]:
            break
        repeat_run = repeat_run + 1 if first == last_first else 1
        if repeat_run >= MAX_CONSECUTIVE_REPEAT:
            break
        last_first = first
        seen_first[first] = True

        frame = np.zeros(GROUPS, np.int64)
        frame[0] = first
        talker_hidden = hidden[0, -1].reshape(1, HIDDEN).astype(np.float32)

        stuck_sub = False
        for g in range(1, GROUPS):
            nets['code_predictor'].setInput(talker_hidden, 'talker_hidden')
            nets['code_predictor'].setInput(frame.reshape(1, GROUPS), 'codec_ids')
            gv = nets['code_predictor'].forward()[0, g - 1].astype(np.float64).copy()
            penalized = gv[seen_sub[g]]
            gv[seen_sub[g]] = np.where(penalized < 0, penalized * REPETITION_PENALTY,
                                       penalized / REPETITION_PENALTY)
            val = int(np.argmax(gv))
            repeat_run_sub[g] = repeat_run_sub[g] + 1 if val == last_sub[g] else 1
            if repeat_run_sub[g] >= MAX_CONSECUTIVE_REPEAT:
                stuck_sub = True
            last_sub[g] = val
            seen_sub[g][val] = True
            frame[g] = val
        if stuck_sub:
            break
        codes.append(frame.copy())

        nets['residual_embed'].setInput(frame.reshape(1, GROUPS), 'codec_ids')
        step_embed = nets['residual_embed'].forward().reshape(1, HIDDEN) + pad_embed
        talker_in = np.concatenate([talker_in, step_embed])

    print('frames   : %d' % len(codes))
    if not codes:
        raise RuntimeError('the model produced no code frames')
    codes = np.stack(codes)

    # tok_decoder takes a fixed number of frames, so a short tail is padded by repeating its
    # frames and the extra audio is trimmed back off.
    waveform = []
    for start in range(0, len(codes), DEC_FRAMES):
        chunk = codes[start:start + DEC_FRAMES]
        have = len(chunk)
        if have < DEC_FRAMES:
            chunk = chunk[np.arange(DEC_FRAMES) % have]
        nets['tok_decoder'].setInput(chunk.reshape(1, DEC_FRAMES, GROUPS), 'audio_codes')
        audio = nets['tok_decoder'].forward().reshape(-1)
        if have < DEC_FRAMES:
            audio = audio[:round(len(audio) * have / DEC_FRAMES)]
        waveform.append(audio)
    waveform = np.concatenate(waveform)

    write_wav(args.output, waveform, SAMPLE_RATE)
    print('wrote    : %s (%.2f s)' % (args.output, len(waveform) / SAMPLE_RATE))

if __name__ == '__main__':
    main()
