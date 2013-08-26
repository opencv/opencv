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

Mat tringleWeights()
{
    Mat w(256, 1, CV_32F);
    for(int i = 0; i < 256; i++) {
        w.at<float>(i) = i < 128 ? i + 1.0f : 256.0f - i;
    }
    return w;
}

Mat RobertsonWeights()
{
    Mat weight(256, 1, CV_32FC3);
    for(int i = 0; i < 256; i++) {
        float value = exp(-4.0f * pow(i - 127.5f, 2.0f) / pow(127.5f, 2.0f));
        for(int c = 0; c < 3; c++) {
            weight.at<Vec3f>(i)[c] = value;
        }
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

};
