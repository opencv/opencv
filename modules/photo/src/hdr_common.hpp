/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_HDR_COMMON_HPP__
#define __OPENCV_HDR_COMMON_HPP__

#include "precomp.hpp"
#include "opencv2/photo.hpp"

namespace cv
{

void checkImageDimensions(const std::vector<Mat>& images);

void mertensWeightRow(const float* grayUp, const float* grayMid, const float* grayDn,
                      const float* src, float* dst, int cols, int channels,
                      float wcon, float wsat, float wexp, float* sum);
void mertensConvertGray(const uchar* src, size_t srcStep, float* img, size_t imgStep,
                        float* gray, size_t grayStep, int rows, int cols);
void mertensPyrUpSub(const float* src, size_t srcStep, float* dst, size_t dstStep,
                     int srcRows, int srcCols, int dstRows, int dstCols, int channels);
void mertensBlendAcc(const float* img, size_t imgStep, const float* weight, size_t weightStep,
                     float* acc, size_t accStep, int rows, int cols, int channels, bool accumulate);

Mat triangleWeights(int length = LDR_SIZE);

void mapLuminance(Mat src, Mat dst, Mat lum, Mat new_lum, float saturation);

Mat RobertsonWeights(int length = LDR_SIZE);

Mat linearResponse(int channels, int length = LDR_SIZE);
}

#endif
