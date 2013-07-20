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
#include "opencv2/reg/mappergradshift.hpp"
#include "opencv2/reg/mapshift.hpp"

namespace cv {
namespace reg {


////////////////////////////////////////////////////////////////////////////////////////////////////
MapperGradShift::MapperGradShift(void)
{
}


////////////////////////////////////////////////////////////////////////////////////////////////////
MapperGradShift::~MapperGradShift(void)
{
}


////////////////////////////////////////////////////////////////////////////////////////////////////
void MapperGradShift::calculate(
    const cv::Mat& img1, const cv::Mat& image2, cv::Ptr<Map>& res) const
{
    Mat gradx, grady, imgDiff;
    Mat img2;

    CV_DbgAssert(img1.size() == image2.size());

    if(!res.empty()) {
        // We have initial values for the registration: we move img2 to that initial reference
        res->inverseWarp(image2, img2);
    } else {
        img2 = image2;
    }

    // Get gradient in all channels
    gradient(img1, img2, gradx, grady, imgDiff);

    // Calculate parameters using least squares
    Matx<double, 2, 2> A;
    Vec<double, 2> b;
    // For each value in A, all the matrix elements are added and then the channels are also added,
    // so we have two calls to "sum". The result can be found in the first element of the final
    // Scalar object.

    A(0, 0) = sum(sum(gradx.mul(gradx)))[0];
    A(0, 1) = sum(sum(gradx.mul(grady)))[0];
    A(1, 1) = sum(sum(grady.mul(grady)))[0];
    A(1, 0) = A(0, 1);

    b(0) = -sum(sum(imgDiff.mul(gradx)))[0];
    b(1) = -sum(sum(imgDiff.mul(grady)))[0];

    // Calculate shift. We use Cholesky decomposition, as A is symmetric.
    Vec<double, 2> shift = A.inv(DECOMP_CHOLESKY)*b;

    if(res.empty()) {
        res = new MapShift(shift);
    } else {
        MapShift newTr(shift);
        res->compose(newTr);
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Ptr<Map> MapperGradShift::getMap(void) const
{
    return cv::Ptr<Map>(new MapShift());
}


}}  // namespace cv::reg
