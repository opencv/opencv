#!/usr/bin/env python

'''
Test for Tokenizer Python bindings
'''

from __future__ import print_function

import cv2 as cv
import os
import json

from tests_common import NewOpenCVTests

def _tf(filename=""):
    base = os.environ.get("OPENCV_TEST_DATA_PATH") or os.getcwd()
    path = os.path.join(base, "dnn", "llm", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing test data: {path}. "
            "Set OPENCV_TEST_DATA_PATH to the testdata root contains dnn/llm."
        )
    return path

class TokenizerBindingTest(NewOpenCVTests):
    def test_tokenizer_binding(self):
        try:
            tokenizer = cv.dnn.Tokenizer
            print("Tokenizer binding is available.", tokenizer)
            gpt2_model = _tf("gpt2/config.json")
            tokenizer = cv.dnn.Tokenizer.load(gpt2_model)
            print("Tokenizer loaded from:", gpt2_model)
        except AttributeError:
            self.fail("Tokenizer binding is NOT available.")

    def test_tokenizer_gpt2(self):
        tok = cv.dnn.Tokenizer.load((_tf("gpt2/config.json")))
        ids = tok.encode("hello world")
        print(ids)
        txt = tok.decode(ids)
        self.assertEqual(txt, "hello world")

    def test_tokenizer_gpt4(self):
        tok = cv.dnn.Tokenizer.load(_tf("gpt4/config.json"))
        tokens = tok.encode("hello world")
        self.assertEqual(list(tokens), [15339, 1917])
        sent = tok.decode([15339, 1917])
        self.assertEqual(sent, "hello world")

    def test_tokenizer_bert_encode_pair(self):
        tok = cv.dnn.Tokenizer.load(_tf("bert/config.json"))
        a = list(tok.encode("hello world"))
        b = list(tok.encode("OpenCV is Great"))
        pair = list(tok.encodePair("hello world", "OpenCV is Great"))
        self.assertEqual(pair, a + b[1:])
        self.assertEqual(pair, [101, 7592, 2088, 102, 2330, 2278, 2615, 2003, 2307, 102])

    def test_tokenizer_encode_pair_unsupported(self):
        for cfg in ["gpt2/config.json", "gemma2/config.json", "t5/config.json"]:
            tok = cv.dnn.Tokenizer.load(_tf(cfg))
            with self.assertRaises(cv.error):
                tok.encodePair("hello", "world")

    def test_tokenizer_malformed_utf8(self):
        tok = cv.dnn.Tokenizer.load(_tf("t5/config.json"))
        with self.assertRaises(cv.error):
            tok.encode(b"\xff")
        with self.assertRaises(cv.error):
            tok.encode(b"\xc3")

    def test_tokenizer_explicit_model_type(self):
        cases = [
            ("gpt2/config.json", cv.dnn.DNN_TOKENIZER_BPE),
            ("gemma2/config.json", cv.dnn.DNN_TOKENIZER_SENTENCEPIECE),
            ("gemma3/config.json", cv.dnn.DNN_TOKENIZER_SENTENCEPIECE),
            ("t5/config.json", cv.dnn.DNN_TOKENIZER_UNIGRAM),
            ("bert/config.json", cv.dnn.DNN_TOKENIZER_WORDPIECE),
        ]
        for cfg, model_type in cases:
            auto_tok = cv.dnn.Tokenizer.load(_tf(cfg))
            explicit_tok = cv.dnn.Tokenizer.load(_tf(cfg), model_type)
            self.assertEqual(list(auto_tok.encode("hello world")),
                              list(explicit_tok.encode("hello world")),
                              msg=f"Mismatch for '{cfg}' with modelType={model_type}")

    def test_with_hf_tiktoken(self):
        tok = cv.dnn.Tokenizer.load(_tf("gpt2/config.json"))
        with open(_tf("gpt2/gpt2_hf_tik_testdata.json"), "r", encoding="utf-8") as f:
            golden = json.load(f)

        for s in golden["samples"]:
            text = s["text"]
            expected = s["ids"]
            got = tok.encode(text).tolist()
            self.assertEqual(
                got, expected,
                msg=f"Mismatch for sample '{s['name']}'"
            )
            self.assertEqual(tok.decode(expected), text)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
