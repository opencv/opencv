#!/usr/bin/env python

'''
Test for Tokenizer Python bindings
'''

from __future__ import print_function

import cv2 as cv
import sys
import os

from tests_common import NewOpenCVTests

def _tf(filename=""):
    base = (os.environ.get("OPENCV_DNN_TEST_DATA_PATH")
            or os.environ.get("OPENCV_TEST_DATA_PATH")
            or os.getcwd())
    return os.path.join(base, "dnn", "llm", filename)

class TokenizerBindingTest(NewOpenCVTests):
    def test_tokenizer_binding(self):
        try:
            tokenizer = cv.dnn.Tokenizer
            print("Tokenizer binding is available.")
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
        # expects {15339, 1917}
        self.assertEqual(list(tokens), [15339, 1917])
        sent = tok.decode([15339, 1917])
        self.assertEqual(sent, "hello world")

    def test_with_hf_tiktoken(self):
        from tiktoken import encoding_for_model
        from transformers import AutoTokenizer

        tik_tokenizer = encoding_for_model("gpt2")
        hf_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)
        cv_tokenizer = cv.dnn.Tokenizer.load(_tf("gpt2/config.json"))

        sentences = {
            "Spanish": "¡Hola, mundo! ¿Cómo estás?",
            "Chinese": "你好，世界！今天天气不错。",
            "Japanese": "こんにちは世界。今日はいい天気ですね。",
            "Arabic": "مرحبًا بالعالم! كيف الحال؟",
            "Hindi": "नमस्ते दुनिया! आप कैसे हैं?",
            "Russian": "Привет, мир! Как дела?",
            "Korean": "안녕하세요 세계! 오늘은 날씨가 좋아요.",
            "Thai": "สวัสดีโลก วันนี้อากาศดี",
            "Greek": "Γεια σου κόσμε! Τι κάνεις;",
            "Hebrew": "שלום עולם! מה שלומך?",
            "Emojis": "👩🏽‍💻✨🚀 — coding time!",
            "Vietnamese": "Xin chào thế giới! Hôm nay trời đẹp.",
            "Turkish": "Merhaba dünya! Nasılsın?",
        }

        for i, (name, text) in enumerate(sentences.items()):
            cv_ids = cv_tokenizer.encode(text)
            tik_ids = tik_tokenizer.encode(text)
            hf_ids = hf_tokenizer.encode(text)

            print(f"--- {name} ---")
            print("text: ", text)
            print(f"cv token count: {len(cv_ids)} | tiktoken token count {len(tik_ids)} | hf token count {len(hf_ids)}")
            if i == 0: # First iteration output the types returned by encode for all 3 Tokenizer libraries
                print(f"type of cv ids [{type(cv_ids)}] | type of tiktoken ids [{type(tik_ids)}] | type of hf ids [{type(hf_ids)}]")
            self.assertEqual(cv_ids.tolist(), tik_ids, hf_ids)
            
if __name__ == '__main__':
    NewOpenCVTests.bootstrap()