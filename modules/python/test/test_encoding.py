#!/usr/bin/env python
"""
Test for CL100K_BASE and GPT‑2 BPE encode/decode against the Python references
"""

from __future__ import print_function
import cv2 as cv
import tiktoken 
from transformers import GPT2TokenizerFast

from tests_common import NewOpenCVTests
print("cv2 loaded from:", cv.__file__)

class tokenizer_test(NewOpenCVTests):
    def test_cl100k_case_encode_decode(self):
        samples = [
            "tiktoken is great!",
            "antidisestablishmentarianism",
            "お誕生日おめでとう",
            "   leading and   multiple   whitespace  ",
        ]
        py_enc = tiktoken.get_encoding("cl100k_base")
        # cv_enc = cv.dnn.getEncodingForCl100k_base("cl100k_base")

        for s in samples:
            ref_ids = py_enc.encode(s)
            cpp_ids = cv.dnn_encodeCl100k_base("cl100k_base", s)
            self.assertEqual(cpp_ids, ref_ids, f"CL100K_BASE encode mismatch for '{s}'")


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
