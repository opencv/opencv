#!/usr/bin/env python

'''
Test for Tokenizer Python bindings
'''

from __future__ import print_function

import cv2 as cv
import sys

from tests_common import NewOpenCVTests

class TokenizerBindingTest(NewOpenCVTests):
    def test_tokenizer_binding(self):
        try:
            tokenizer = cv.dnn.tokenizer.Tokenizer
            print("Tokenizer binding is available.")
        except AttributeError:
            self.fail("Tokenizer binding is NOT available.")

        gpt4_tokenizer = cv.dnn.tokenizer.Tokenizer.from_pretrained(
            "cl100k_base", "/Users/jorgevelez/Desktop/data/cl100k_base.tiktoken"
        )

        print(gpt4_tokenizer.encode("hello world", True))
        
if __name__ == '__main__':
    NewOpenCVTests.bootstrap()