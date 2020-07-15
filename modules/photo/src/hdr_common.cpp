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

#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "hdr_common.hpp"

namespace cv
{

void checkImageDimensions(const std::vector<Mat>& images)
{
    CV_Assert(!images.empty());
    int width = images[0].cols;
    int height = images[0].rows;
    int type = images[0].type();

    for(size_t i = 0; i < images.size(); i++) {
        CV_Assert(images[i].cols == width && images[i].rows == height);
        CV_Assert(images[i].type() == type);
    }
}

Mat triangleWeights()
{
    // hat function
    Mat w(LDR_SIZE, 1, CV_32F);
    int half = LDR_SIZE / 2;
    for(int i = 0; i < LDR_SIZE; i++) {
        w.at<float>(i) = i < half ? i + 1.0f : LDR_SIZE - i;
    }
    return w;
}

Mat RobertsonWeights()
{
    Mat weight(LDR_SIZE, 1, CV_32FC3);
    float q = (LDR_SIZE - 1) / 4.0f;
    float e4 = exp(4.f);
    float scale = e4/(e4 - 1.f);
    float shift = 1 / (1.f - e4);

    for(int i = 0; i < LDR_SIZE; i++) {
        float value = i / q - 2.0f;
        value = scale*exp(-value * value) + shift;
        weight.at<Vec3f>(i) = Vec3f::all(value);
    }
    return weight;
}

void mapLuminance(Mat src, Mat dst, Mat lum, Mat new_lum, float saturation)
{
    std::vector<Mat> channels(3);
    split(src, channels);
    for(int i = 0; i < 3; i++) {
        channels[i] = channels[i].mul(1.0f / lum);
        pow(channels[i], saturation, channels[i]);
        channels[i] = channels[i].mul(new_lum);
    }
    merge(channels, dst);
}

Mat linearResponse(int channels)
{
    Mat response = Mat(LDR_SIZE, 1, CV_MAKETYPE(CV_32F, channels));
    for(int i = 0; i < LDR_SIZE; i++) {
        response.at<Vec3f>(i) = Vec3f::all(static_cast<float>(i));
    }
    return response;
}

}
