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
#include <opencv2/imgproc.hpp>
#include "opencv2/reg/mapper.hpp"

namespace cv {
namespace reg {


////////////////////////////////////////////////////////////////////////////////////////////////////
void Mapper::gradient(const Mat& img1, const Mat& img2, Mat& Ix, Mat& Iy, Mat& It) const
{
    Size sz1 = img2.size();

    Mat xkern = (Mat_<double>(1, 3) << -1., 0., 1.)/2.;
    filter2D(img2, Ix, -1, xkern, Point(-1,-1), 0., BORDER_REPLICATE);

    Mat ykern = (Mat_<double>(3, 1) << -1., 0., 1.)/2.;
    filter2D(img2, Iy, -1, ykern, Point(-1,-1), 0., BORDER_REPLICATE);

    It = Mat::zeros(sz1, img1.type());
    It = img2 - img1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void Mapper::grid(const Mat& img, Mat& grid_r, Mat& grid_c) const
{
    // Matrices with reference frame coordinates
    grid_r.create(img.size(), img.type());
    grid_c.create(img.size(), img.type());
    if(img.channels() == 1) {
        for(int r_i = 0; r_i < img.rows; ++r_i) {
            for(int c_i = 0; c_i < img.cols; ++c_i) {
                grid_r.at<double>(r_i, c_i) = r_i;
                grid_c.at<double>(r_i, c_i) = c_i;
            }
        }
    } else {
        Vec3d ones(1., 1., 1.);
        for(int r_i = 0; r_i < img.rows; ++r_i) {
            for(int c_i = 0; c_i < img.cols; ++c_i) {
                grid_r.at<Vec3d>(r_i, c_i) = r_i*ones;
                grid_c.at<Vec3d>(r_i, c_i) = c_i*ones;
            }
        }
    }
}


}}  // namespace cv::reg
