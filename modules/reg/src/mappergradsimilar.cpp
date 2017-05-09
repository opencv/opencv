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
#include "opencv2/reg/mappergradsimilar.hpp"
#include "opencv2/reg/mapaffine.hpp"

namespace cv {
namespace reg {


////////////////////////////////////////////////////////////////////////////////////////////////////
MapperGradSimilar::MapperGradSimilar(void)
{
}


////////////////////////////////////////////////////////////////////////////////////////////////////
MapperGradSimilar::~MapperGradSimilar(void)
{
}


////////////////////////////////////////////////////////////////////////////////////////////////////
void MapperGradSimilar::calculate(
    const cv::Mat& img1, const cv::Mat& image2, cv::Ptr<Map>& res) const
{
    Mat gradx, grady, imgDiff;
    Mat img2;

    CV_DbgAssert(img1.size() == image2.size());
    CV_DbgAssert(img1.channels() == image2.channels());
    CV_DbgAssert(img1.channels() == 1 || img1.channels() == 3);

    if(!res.empty()) {
        // We have initial values for the registration: we move img2 to that initial reference
        res->inverseWarp(image2, img2);
    } else {
        img2 = image2;
    }

    // Get gradient in all channels
    gradient(img1, img2, gradx, grady, imgDiff);

    // Matrices with reference frame coordinates
    Mat grid_r, grid_c;
    grid(img1, grid_r, grid_c);

    // Calculate parameters using least squares
    Matx<double, 4, 4> A;
    Vec<double, 4> b;
    // For each value in A, all the matrix elements are added and then the channels are also added,
    // so we have two calls to "sum". The result can be found in the first element of the final
    // Scalar object.
    Mat xIx_p_yIy = grid_c.mul(gradx);
    xIx_p_yIy += grid_r.mul(grady);
    Mat yIx_m_xIy = grid_r.mul(gradx);
    yIx_m_xIy -= grid_c.mul(grady);

    A(0, 0) = sum(sum(sqr(xIx_p_yIy)))[0];
    A(0, 1) = sum(sum(xIx_p_yIy.mul(yIx_m_xIy)))[0];
    A(0, 2) = sum(sum(gradx.mul(xIx_p_yIy)))[0];
    A(0, 3) = sum(sum(grady.mul(xIx_p_yIy)))[0];

    A(1, 1) = sum(sum(sqr(yIx_m_xIy)))[0];
    A(1, 2) = sum(sum(gradx.mul(yIx_m_xIy)))[0];
    A(1, 3) = sum(sum(grady.mul(yIx_m_xIy)))[0];

    A(2, 2) = sum(sum(sqr(gradx)))[0];
    A(2, 3) = sum(sum(gradx.mul(grady)))[0];

    A(3, 3) = sum(sum(sqr(grady)))[0];

    // Lower half values (A is symmetric)
    A(1, 0) = A(0, 1);
    A(2, 0) = A(0, 2);
    A(3, 0) = A(0, 3);

    A(2, 1) = A(1, 2);
    A(3, 1) = A(1, 3);

    A(3, 2) = A(2, 3);

    // Calculation of b
    b(0) = -sum(sum(imgDiff.mul(xIx_p_yIy)))[0];
    b(1) = -sum(sum(imgDiff.mul(yIx_m_xIy)))[0];
    b(2) = -sum(sum(imgDiff.mul(gradx)))[0];
    b(3) = -sum(sum(imgDiff.mul(grady)))[0];

    // Calculate affine transformation. We use Cholesky decomposition, as A is symmetric.
    Vec<double, 4> k = A.inv(DECOMP_CHOLESKY)*b;

    Matx<double, 2, 2> linTr(k(0) + 1., k(1), -k(1), k(0) + 1.);
    Vec<double, 2> shift(k(2), k(3));
    if(res.empty()) {
        res = new MapAffine(linTr, shift);
    } else {
        MapAffine newTr(linTr, shift);
        res->compose(newTr);
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Ptr<Map> MapperGradSimilar::getMap(void) const
{
    return cv::Ptr<Map>(new MapAffine());
}


}}  // namespace cv::reg
