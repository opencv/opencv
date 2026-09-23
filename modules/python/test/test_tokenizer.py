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

    def test_tokenizer_bert_encode_chunks(self):
        tok = cv.dnn.Tokenizer.load(_tf("bert/config.json"))
        a = list(tok.encode("hello world"))
        b = list(tok.encode("OpenCV is Great"))
        pair = list(tok.encode(["hello world", "OpenCV is Great"]))
        self.assertEqual(pair, a + b[1:])
        self.assertEqual(pair, [101, 7592, 2088, 102, 2330, 2278, 2615, 2003, 2307, 102])

        # Any number of chunks, and a one-chunk list matches the plain string call.
        c = list(tok.encode("third one"))
        self.assertEqual(list(tok.encode(["hello world", "OpenCV is Great", "third one"])),
                         a + b[1:] + c[1:])
        self.assertEqual(list(tok.encode(["hello world"])), a)

    # A Python str is a sequence of one-character strings, so the string overload has
    # to win over the chunk-list one; getting this wrong encodes text letter by letter.
    def test_tokenizer_encode_str_is_not_a_chunk_list(self):
        tok = cv.dnn.Tokenizer.load(_tf("bert/config.json"))
        self.assertEqual(list(tok.encode("hello world")),
                         list(tok.encode(["hello world"])))

    def test_tokenizer_encode_chunks_unsupported(self):
        # Byte-level BPE models declare no pair template to repeat.
        for cfg in ["gpt2/config.json", "gpt4/config.json"]:
            tok = cv.dnn.Tokenizer.load(_tf(cfg))
            with self.assertRaises(cv.error):
                tok.encode(["hello", "world"])

    def test_tokenizer_encode_chunks_from_pair_template(self):
        # T5 wraps as "A </s> B </s>", Gemma as "<bos> A <bos> B".
        for cfg in ["t5/config.json", "gemma2/config.json"]:
            tok = cv.dnn.Tokenizer.load(_tf(cfg))
            a = list(tok.encode("hello"))
            b = list(tok.encode("world"))
            self.assertEqual(list(tok.encode(["hello", "world"])), a + b)

    def test_tokenizer_malformed_utf8(self):
        # Malformed sequences resolve to U+FFFD rather than raising, so a single
        # bad byte cannot abort a whole prompt.
        tok = cv.dnn.Tokenizer.load(_tf("t5/config.json"))
        self.assertGreater(len(tok.encode(b"\xff")), 0)
        self.assertGreater(len(tok.encode(b"\xc3")), 0)

    def test_tokenizer_albert_sequence_normalizer(self):
        # ALBERT wraps with [CLS]=2 / [SEP]=3 and folds case and accents.
        tok = cv.dnn.Tokenizer.load(_tf("albert/config.json"))
        self.assertEqual(list(tok.encode("Hello world")), [2, 10975, 126, 3])
        self.assertEqual(list(tok.encode("café")), [2, 6241, 3])
        self.assertEqual(list(tok.encode("Hello world")), list(tok.encode("hello world")))

    def test_tokenizer_strip_accents_keeps_non_mark_decompositions(self):
        # Stripping accents must not delete a spacing mark or half a Hangul syllable.
        tok = cv.dnn.Tokenizer.load(_tf("bert/config.json"))
        self.assertEqual(list(tok.encode("हिन्दी")),
                         [101, 1339, 29877, 29863, 29861, 29878, 102])
        for text in ("মৌশল", "தமிழ்", "ଓଡ଼ିଆ", "안녕하세요"):
            self.assertEqual(list(tok.encode(text)), [101, 100, 102], text)
        self.assertEqual(list(tok.encode("café")), [101, 7668, 102])

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
