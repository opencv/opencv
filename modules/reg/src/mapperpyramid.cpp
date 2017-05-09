/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// Copyright (C) 2013, Alfonso Sanchez-Beato, all rights reserved.
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
// In no event shall the contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/reg/mapperpyramid.hpp"

using namespace std;

namespace cv {
namespace reg {


////////////////////////////////////////////////////////////////////////////////////////////////////
MapperPyramid::MapperPyramid(const Mapper& baseMapper)
    : numLev_(3), numIterPerScale_(3), baseMapper_(baseMapper)
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void MapperPyramid::calculate(const Mat& img1, const Mat& image2, Ptr<Map>& res) const
{
    Mat img2;

    if(!res.empty()) {
        // We have initial values for the registration: we move img2 to that initial reference
        res->inverseWarp(image2, img2);
    } else {
        res = baseMapper_.getMap();
        img2 = image2;
    }

    cv::Ptr<Map> ident = baseMapper_.getMap();

    // Precalculate pyramid images
    vector<Mat> pyrIm1(numLev_), pyrIm2(numLev_);
    pyrIm1[0] = img1;
    pyrIm2[0] = img2;
    for(size_t im_i = 1; im_i < numLev_; ++im_i) {
        pyrDown(pyrIm1[im_i - 1], pyrIm1[im_i]);
        pyrDown(pyrIm2[im_i - 1], pyrIm2[im_i]);
    }

    Mat currRef, currImg;
    for(size_t lv_i = 0; lv_i < numLev_; ++lv_i) {
        currRef = pyrIm1[numLev_ - 1 - lv_i];
        currImg = pyrIm2[numLev_ - 1 - lv_i];
        // Scale the transformation as we are incresing the resolution in each iteration
        if(lv_i != 0) {
            ident->scale(2.);
        }
        for(size_t it_i = 0; it_i < numIterPerScale_; ++it_i) {
            baseMapper_.calculate(currRef, currImg, ident);
        }
    }

    res->compose(*ident.obj);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Ptr<Map> MapperPyramid::getMap(void) const
{
    return cv::Ptr<Map>(0);
}


}}  // namespace cv::reg
