// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "fuzz_precomp.hpp"

using namespace cv;

////////////////////////////////////////////////////////////////////////////////

static void FuzzSVD(const Mat& image, int code) {
  InitializeOpenCV();
  try {
    const SVD svd = SVD(image, code);
    (void)svd;
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Core, FuzzSVD)
    .WithDomains(/*image:*/Arbitrary2DMat(),
                 /*code:*/fuzztest::InRange<int>(0, SVD::MODIFY_A |
                                                 SVD::NO_UV | SVD::FULL_UV));

////////////////////////////////////////////////////////////////////////////////

static void FuzzStorage(const std::string& data) {
  FileStorage storage;
  try {
    storage.open(data, FileStorage::READ + FileStorage::MEMORY);
  } catch (Exception e) {
  }
}

FUZZ_TEST(Core, FuzzStorage)
    .WithDomains(/*data:*/fuzztest::String());

////////////////////////////////////////////////////////////////////////////////

static void FuzzT(const cv::Mat& mat) {
  try {
    mat.t();
  } catch (Exception e) {
  }
}

FUZZ_TEST(Core, FuzzT)
    .WithDomains(/*mat:*/Arbitrary2DMat());

////////////////////////////////////////////////////////////////////////////////

static void FuzzInv(const cv::Mat& mat) {
  try {
    mat.inv();
  } catch (Exception e) {
  }
}

FUZZ_TEST(Core, FuzzInv)
    .WithDomains(/*mat:*/Arbitrary2DMat());

////////////////////////////////////////////////////////////////////////////////

static void FuzzDiag(const cv::Mat& mat) {
  try {
    mat.diag();
  } catch (Exception e) {
  }
}

FUZZ_TEST(Core, FuzzDiag)
    .WithDomains(/*mat:*/Arbitrary2DMat());

////////////////////////////////////////////////////////////////////////////////

static void FuzzSum(const cv::Mat& mat) {
  try {
    sum(mat);
  } catch (Exception e) {
  }
}

FUZZ_TEST(Core, FuzzSum)
    .WithDomains(/*mat:*/Arbitrary2DMat());

////////////////////////////////////////////////////////////////////////////////

static void FuzzMean(const cv::Mat& mat) {
  try {
    mean(mat);
  } catch (Exception e) {
  }
}

FUZZ_TEST(Core, FuzzMean)
    .WithDomains(/*mat:*/Arbitrary2DMat());

////////////////////////////////////////////////////////////////////////////////

static void FuzzTrace(const cv::Mat& mat) {
  try {
    trace(mat);
  } catch (Exception e) {
  }
}

FUZZ_TEST(Core, FuzzTrace)
    .WithDomains(/*mat:*/Arbitrary2DMat());
