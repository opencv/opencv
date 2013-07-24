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
#include "opencv2/reg/mapprojec.hpp"


namespace cv {
namespace reg {


////////////////////////////////////////////////////////////////////////////////////////////////////
MapProjec::MapProjec(void)
    : projTr_(Matx<double, 3, 3>::eye())
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////
MapProjec::MapProjec(const Matx<double, 3, 3>& projTr)
    : projTr_(projTr)
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////
MapProjec::~MapProjec(void)
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void MapProjec::inverseWarp(const Mat& img1, Mat& img2) const
{
    // Rows and columns in destination
    Mat dest_r, dest_c;
    dest_r.create(img1.size(), CV_32FC1);
    dest_c.create(img1.size(), CV_32FC1);
    for(int r_i = 0; r_i < img1.rows; ++r_i)
    {
        for(int c_i = 0; c_i < img1.cols; ++c_i)
        {
            double z = c_i*projTr_(2, 0) + r_i*projTr_(2, 1) + projTr_(2, 2);
            dest_c.at<float>(r_i, c_i) =
                float((c_i*projTr_(0, 0) + r_i*projTr_(0, 1) + projTr_(0, 2))/z);
            dest_r.at<float>(r_i, c_i) =
                float((c_i*projTr_(1, 0) + r_i*projTr_(1, 1) + projTr_(1, 2))/z);
        }
    }

    //remap(img1, img2, dest_c, dest_r, INTER_CUBIC, BORDER_REPLICATE);
    // Parts that cannot be interpolated will be as in img1 (BORDER_TRANSPARENT means that
    // remap will not touch them).
    img1.copyTo(img2);
    remap(img1, img2, dest_c, dest_r, INTER_CUBIC, BORDER_TRANSPARENT);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Ptr<Map> MapProjec::inverseMap(void) const
{
    Matx<double, 3, 3> invProjTr = projTr_.inv(DECOMP_LU);
    return Ptr<Map>(new MapProjec(invProjTr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void MapProjec::compose(const Map& map)
{
    // Composition of homographies H and H' is (H o H') = H'*H
    const MapProjec& mapProj = static_cast<const MapProjec&>(map);
    Matx<double, 3, 3> compProjTr = mapProj.getProjTr()*projTr_;
    projTr_ = compProjTr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void MapProjec::scale(double factor)
{
    // Shift is multiplied, projective factors are divided
    projTr_(0, 2) *= factor;
    projTr_(1, 2) *= factor;
    projTr_(2, 0) /= factor;
    projTr_(2, 1) /= factor;
}


}}  // namespace cv::reg
