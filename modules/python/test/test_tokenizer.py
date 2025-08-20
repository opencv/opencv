#!/usr/bin/env python

'''
Test for Tokenizer Python bindings
'''

from __future__ import print_function

import cv2 as cv
import sys
import os

from tests_common import NewOpenCVTests

def _tf_gpt2(filename=""):
    base = os.environ.get('OPENCV_DNN_TEST_DATA_PATH') or \
           os.environ.get('OPENCV_TEST_DATA_PATH', os.getcwd())
    return os.path.join(base, "testdata", "dnn", "llm", "gpt2", filename)

def _tf_gpt4(filename=""):
    base = os.environ.get('OPENCV_DNN_TEST_DATA_PATH') or \
           os.environ.get('OPENCV_TEST_DATA_PATH', os.getcwd())
    return os.path.join(base, "testdata", "dnn", "llm", "gpt4", filename)

class TokenizerBindingTest(NewOpenCVTests):
    def test_tokenizer_binding(self):
        try:
            tokenizer = cv.dnn.Tokenizer
            print("Tokenizer binding is available.")
            gpt2_dir = _tf_gpt2("")
            tokenizer = cv.dnn.Tokenizer.load(gpt2_dir)
            print("Tokenizer loaded from:", gpt2_dir)
        except AttributeError:
            self.fail("Tokenizer binding is NOT available.")

    def test_tokenizer_gpt2(self):
        tok = cv.dnn.Tokenizer.load((_tf_gpt2("")))
        ids = tok.encode("hello world")
        print(ids)
        txt = tok.decode(ids)
        self.assertEqual(txt, "hello world")

    def test_encoding_gpt4(self):
        tok = cv.dnn.Tokenizer.load(_tf_gpt4())
        tokens = tok.encode("hello world")
        # expects {15339, 1917}
        self.assertEqual(list(tokens), [15339, 1917])
        sent = tok.decode([15339, 1917])
        self.assertEqual(sent, "hello world")
        
        
if __name__ == '__main__':
    NewOpenCVTests.bootstrap()