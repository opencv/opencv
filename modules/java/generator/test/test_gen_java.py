#!/usr/bin/env python

import os
import shutil
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR_DIR = os.path.dirname(SCRIPT_DIR)
SRC2_DIR = os.path.normpath(os.path.join(GENERATOR_DIR, "..", "..", "python", "src2"))

sys.path.insert(0, SRC2_DIR)
sys.path.insert(0, GENERATOR_DIR)

import hdr_parser  # noqa: E402
import gen_java  # noqa: E402

gen_java.hdr_parser = hdr_parser


class GenJavaTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_gen_java_")
        self.out_dir = os.path.join(self.tmpdir, "out")
        self.jni_dir = os.path.join(self.tmpdir, "jni")
        self.java_dir = os.path.join(self.tmpdir, "java")
        os.makedirs(self.out_dir)
        os.makedirs(self.jni_dir)
        os.makedirs(self.java_dir)
        self._saved_type_dict = gen_java.type_dict.copy()

    def tearDown(self):
        gen_java.type_dict.clear()
        gen_java.type_dict.update(self._saved_type_dict)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _generate(self, header_code, module="testmod"):
        hdr_path = os.path.join(self.tmpdir, "test_header.hpp")
        with open(hdr_path, "w", encoding="utf-8") as f:
            f.write(header_code)

        gen_java.module = module
        gen_java.Module = module.capitalize()
        generator = gen_java.JavaWrapperGenerator()
        generator.gen([hdr_path], module, self.out_dir, self.jni_dir, self.java_dir, [])

        java_files = {}
        module_java_dir = os.path.join(self.java_dir, module)
        for fn in os.listdir(module_java_dir):
            if fn.endswith(".java"):
                with open(os.path.join(module_java_dir, fn), encoding="utf-8") as f:
                    java_files[fn] = f.read()

        with open(os.path.join(self.jni_dir, module + ".inl.hpp"), encoding="utf-8") as f:
            jni_code = f.read()

        return java_files, jni_code

    def test_namespaced_enum_default_arg(self):
        header = """
        namespace cv { namespace xfeatures2d {
        class CV_EXPORTS_W BEBLID {
        public:
            enum BeblidSize {
                SIZE_512_BITS = 100,
                SIZE_256_BITS = 101
            };
            CV_WRAP static Ptr<BEBLID> create(float scale_factor,
                                              BEBLID::BeblidSize n_bits = BEBLID::SIZE_512_BITS);
        };
        }}
        """
        java_files, jni_code = self._generate(header, module="xfeatures2d")
        java_code = java_files["BEBLID.java"]

        self.assertIn("public static BEBLID create(float scale_factor, int n_bits)", java_code)
        self.assertIn("public static BEBLID create(float scale_factor)", java_code)
        self.assertNotIn("public static BEBLID create()", java_code)
        self.assertIn("(cv::xfeatures2d::BEBLID::BeblidSize)n_bits", jni_code)
        self.assertIn("cv::xfeatures2d::BEBLID::create( (float)scale_factor )", jni_code)
        self.assertNotIn("cv::xfeatures2d::BEBLID::create()", jni_code)

    def test_hidden_default_arg_preserves_required_args(self):
        header = """
        namespace cv {
        class CV_EXPORTS_W Sample {
        public:
            CV_WRAP static void run(float scale_factor,
                                    UnknownEnum mode = DEFAULT_MODE);
            CV_WRAP static void runWithOpt(int required_arg,
                                           int opt_arg = 1,
                                           UnknownEnum mode = DEFAULT_MODE);
            CV_WRAP static void runAllDefault(UnknownEnum mode = DEFAULT_MODE,
                                              int opt_arg = 2);
        };
        }
        """
        java_files, jni_code = self._generate(header, module="core_test")
        java_code = java_files["Sample.java"]

        self.assertIn("public static void run(float scale_factor)", java_code)
        self.assertNotIn("public static void run()", java_code)
        self.assertIn("cv::Sample::run( (float)scale_factor, DEFAULT_MODE )", jni_code)
        self.assertNotIn("cv::Sample::run()", jni_code)

        self.assertIn("public static void runWithOpt(int required_arg, int opt_arg)", java_code)
        self.assertIn("public static void runWithOpt(int required_arg)", java_code)
        self.assertNotIn("public static void runWithOpt()", java_code)

        self.assertIn("public static void runAllDefault(int opt_arg)", java_code)
        self.assertIn("public static void runAllDefault()", java_code)


if __name__ == "__main__":
    unittest.main()
