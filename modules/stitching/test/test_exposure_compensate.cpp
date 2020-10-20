/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test {
namespace {

double minPSNR(UMat src1, UMat src2)
{
    std::vector<UMat> src1_channels, src2_channels;
    split(src1, src1_channels);
    split(src2, src2_channels);

    double psnr = cvtest::PSNR(src1_channels[0], src2_channels[0]);
    psnr = std::min(psnr, cvtest::PSNR(src1_channels[1], src2_channels[1]));
    return std::min(psnr, cvtest::PSNR(src1_channels[2], src2_channels[2]));
}

TEST(ExposureCompensate, SimilarityThreshold)
{
  UMat source;
  imread(cvtest::TS::ptr()->get_data_path() + "stitching/s1.jpg").copyTo(source);

  UMat image1 = source.clone();
  UMat image2 = source.clone();

  // Add a big artifact
  image2(Rect(150, 150, 100, 100)).setTo(Scalar(0, 0, 255));

  UMat mask(image1.size(), CV_8U);
  mask.setTo(255);

  detail::BlocksChannelsCompensator compensator;
  compensator.setNrGainsFilteringIterations(0); // makes it more clear

  // Feed the compensator, image 1 and 2 are perfectly
  // identical, except for the red artifact in image 2
  // Apart from that artifact, there is no exposure to compensate
  compensator.setSimilarityThreshold(1);
  compensator.feed(
      {{}, {}},
      {image1, image2},
      {{mask, 255}, {mask, 255}}
  );
  // Verify that the artifact in image 2 did create
  // an artifact in image1 during the exposure compensation
  UMat image1_result = image1.clone();
  compensator.apply(0, {}, image1_result, mask);
  double psnr_no_similarity_mask = minPSNR(image1, image1_result);
  EXPECT_LT(psnr_no_similarity_mask, 45);

  // Add a similarity threshold and verify that
  // the artifact in image1 is gone
  compensator.setSimilarityThreshold(0.1);
  compensator.feed(
      {{}, {}},
      {image1, image2},
      {{mask, 255}, {mask, 255}}
  );
  image1_result = image1.clone();
  compensator.apply(0, {}, image1_result, mask);
  double psnr_similarity_mask = minPSNR(image1, image1_result);
  EXPECT_GT(psnr_similarity_mask, 65);
}

} // namespace
} // namespace opencv_test
