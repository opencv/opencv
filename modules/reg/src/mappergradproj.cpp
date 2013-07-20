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
#include "opencv2/reg/mappergradproj.hpp"
#include "opencv2/reg/mapprojec.hpp"

namespace cv {
namespace reg {


////////////////////////////////////////////////////////////////////////////////////////////////////
MapperGradProj::MapperGradProj(void)
{
}


////////////////////////////////////////////////////////////////////////////////////////////////////
MapperGradProj::~MapperGradProj(void)
{
}


////////////////////////////////////////////////////////////////////////////////////////////////////
void MapperGradProj::calculate(
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
    Matx<double, 8, 8> A;
    Vec<double, 8> b;
    // For each value in A, all the matrix elements are added and then the channels are also added,
    // so we have two calls to "sum". The result can be found in the first element of the final
    // Scalar object.
    Mat xIx = grid_c.mul(gradx);
    Mat xIy = grid_c.mul(grady);
    Mat yIx = grid_r.mul(gradx);
    Mat yIy = grid_r.mul(grady);
    Mat Ix2 = gradx.mul(gradx);
    Mat Iy2 = grady.mul(grady);
    Mat xy = grid_c.mul(grid_r);
    Mat IxIy = gradx.mul(grady);
    Mat x2 = grid_c.mul(grid_c);
    Mat y2 = grid_r.mul(grid_r);
    Mat G = xIx + yIy;
    Mat G2 = sqr(G);
    Mat IxG = gradx.mul(G);
    Mat IyG = grady.mul(G);

    A(0, 0) = sum(sum(x2.mul(Ix2)))[0];
    A(1, 0) = sum(sum(xy.mul(Ix2)))[0];
    A(2, 0) = sum(sum(grid_c.mul(Ix2)))[0];
    A(3, 0) = sum(sum(x2.mul(IxIy)))[0];
    A(4, 0) = sum(sum(xy.mul(IxIy)))[0];
    A(5, 0) = sum(sum(grid_c.mul(IxIy)))[0];
    A(6, 0) = -sum(sum(x2.mul(IxG)))[0];
    A(7, 0) = -sum(sum(xy.mul(IxG)))[0];

    A(1, 1) = sum(sum(y2.mul(Ix2)))[0];
    A(2, 1) = sum(sum(grid_r.mul(Ix2)))[0];
    A(3, 1) = A(4, 0);
    A(4, 1) = sum(sum(y2.mul(IxIy)))[0];
    A(5, 1) = sum(sum(grid_r.mul(IxIy)))[0];
    A(6, 1) = A(7, 0);
    A(7, 1) = -sum(sum(y2.mul(IxG)))[0];

    A(2, 2) = sum(sum(Ix2))[0];
    A(3, 2) = A(5, 0);
    A(4, 2) = A(5, 1);
    A(5, 2) = sum(sum(IxIy))[0];
    A(6, 2) = -sum(sum(grid_c.mul(IxG)))[0];
    A(7, 2) = -sum(sum(grid_r.mul(IxG)))[0];

    A(3, 3) = sum(sum(x2.mul(Iy2)))[0];
    A(4, 3) = sum(sum(xy.mul(Iy2)))[0];
    A(5, 3) = sum(sum(grid_c.mul(Iy2)))[0];
    A(6, 3) = -sum(sum(x2.mul(IyG)))[0];
    A(7, 3) = -sum(sum(xy.mul(IyG)))[0];

    A(4, 4) = sum(sum(y2.mul(Iy2)))[0];
    A(5, 4) = sum(sum(grid_r.mul(Iy2)))[0];
    A(6, 4) = A(7, 3);
    A(7, 4) = -sum(sum(y2.mul(IyG)))[0];

    A(5, 5) = sum(sum(Iy2))[0];
    A(6, 5) = -sum(sum(grid_c.mul(IyG)))[0];
    A(7, 5) = -sum(sum(grid_r.mul(IyG)))[0];

    A(6, 6) = sum(sum(x2.mul(G2)))[0];
    A(7, 6) = sum(sum(xy.mul(G2)))[0];

    A(7, 7) = sum(sum(y2.mul(G2)))[0];

    // Upper half values (A is symmetric)
    A(0, 1) = A(1, 0);
    A(0, 2) = A(2, 0);
    A(0, 3) = A(3, 0);
    A(0, 4) = A(4, 0);
    A(0, 5) = A(5, 0);
    A(0, 6) = A(6, 0);
    A(0, 7) = A(7, 0);
    
    A(1, 2) = A(2, 1);
    A(1, 3) = A(3, 1);
    A(1, 4) = A(4, 1);
    A(1, 5) = A(5, 1);
    A(1, 6) = A(6, 1);
    A(1, 7) = A(7, 1);
    
    A(2, 3) = A(3, 2);
    A(2, 4) = A(4, 2);
    A(2, 5) = A(5, 2);
    A(2, 6) = A(6, 2);
    A(2, 7) = A(7, 2);
    
    A(3, 4) = A(4, 3);
    A(3, 5) = A(5, 3);
    A(3, 6) = A(6, 3);
    A(3, 7) = A(7, 3);
    
    A(4, 5) = A(5, 4);
    A(4, 6) = A(6, 4);
    A(4, 7) = A(7, 4);
    
    A(5, 6) = A(6, 5);
    A(5, 7) = A(7, 5);
    
    A(6, 7) = A(7, 6);

    // Calculation of b
    b(0) = -sum(sum(imgDiff.mul(xIx)))[0];
    b(1) = -sum(sum(imgDiff.mul(yIx)))[0];
    b(2) = -sum(sum(imgDiff.mul(gradx)))[0];
    b(3) = -sum(sum(imgDiff.mul(xIy)))[0];
    b(4) = -sum(sum(imgDiff.mul(yIy)))[0];
    b(5) = -sum(sum(imgDiff.mul(grady)))[0];
    b(6) = sum(sum(imgDiff.mul(grid_c.mul(G))))[0];
    b(7) = sum(sum(imgDiff.mul(grid_r.mul(G))))[0];

    // Calculate affine transformation. We use Cholesky decomposition, as A is symmetric.
    Vec<double, 8> k = A.inv(DECOMP_CHOLESKY)*b;

    Matx<double, 3, 3> H(k(0) + 1., k(1), k(2), k(3), k(4) + 1., k(5), k(6), k(7), 1.);
    if(res.empty()) {
        res = new MapProjec(H);
    } else {
        MapProjec newTr(H);
        res->compose(newTr);
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Ptr<Map> MapperGradProj::getMap(void) const
{
    return cv::Ptr<Map>(new MapProjec());
}


}}  // namespace cv::reg
