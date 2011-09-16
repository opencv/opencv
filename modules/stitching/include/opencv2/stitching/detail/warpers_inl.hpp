/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_STITCHING_WARPERS_INL_HPP__
#define __OPENCV_STITCHING_WARPERS_INL_HPP__

#include "opencv2/core/core.hpp"
#include "warpers.hpp" // Make your IDE see declarations

namespace cv {
namespace detail {

template <class P>
Point WarperBase<P>::warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst,
                          int interp_mode, int border_mode)
{
    src_size_ = src.size();
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(dst_tl, dst_br);

    Mat xmap(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);
    Mat ymap(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);

    float x, y;
    for (int v = dst_tl.y; v <= dst_br.y; ++v)
    {
        for (int u = dst_tl.x; u <= dst_br.x; ++u)
        {
            projector_.mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
            xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;
            ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;
        }
    }

    dst.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, src.type());
    remap(src, dst, xmap, ymap, interp_mode, border_mode);

    return dst_tl;
}


template <class P>
Rect WarperBase<P>::warpRoi(const Size &sz, const Mat &K, const Mat &R)
{
    src_size_ = sz;
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(dst_tl, dst_br);

    return Rect(dst_tl, Point(dst_br.x + 1, dst_br.y + 1));
}


template <class P>
void WarperBase<P>::detectResultRoi(Point &dst_tl, Point &dst_br)
{
    float tl_uf = std::numeric_limits<float>::max();
    float tl_vf = std::numeric_limits<float>::max();
    float br_uf = -std::numeric_limits<float>::max();
    float br_vf = -std::numeric_limits<float>::max();

    float u, v;
    for (int y = 0; y < src_size_.height; ++y)
    {
        for (int x = 0; x < src_size_.width; ++x)
        {
            projector_.mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
            tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
            br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


template <class P>
void WarperBase<P>::detectResultRoiByBorder(Point &dst_tl, Point &dst_br)
{
    float tl_uf = std::numeric_limits<float>::max();
    float tl_vf = std::numeric_limits<float>::max();
    float br_uf = -std::numeric_limits<float>::max();
    float br_vf = -std::numeric_limits<float>::max();

    float u, v;
    for (float x = 0; x < src_size_.width; ++x)
    {
        projector_.mapForward(static_cast<float>(x), 0, u, v);
        tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
        br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

        projector_.mapForward(static_cast<float>(x), static_cast<float>(src_size_.height - 1), u, v);
        tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
        br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
    }
    for (int y = 0; y < src_size_.height; ++y)
    {
        projector_.mapForward(0, static_cast<float>(y), u, v);
        tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
        br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

        projector_.mapForward(static_cast<float>(src_size_.width - 1), static_cast<float>(y), u, v);
        tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
        br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


inline
void PlaneProjector::mapForward(float x, float y, float &u, float &v)
{
    float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
    float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
    float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

    u = scale * x_ / z_;
    v = scale * y_ / z_;
}


inline
void PlaneProjector::mapBackward(float u, float v, float &x, float &y)
{
    u /= scale;
    v /= scale;

    float z;
    x = k_rinv[0] * u + k_rinv[1] * v + k_rinv[2];
    y = k_rinv[3] * u + k_rinv[4] * v + k_rinv[5];
    z = k_rinv[6] * u + k_rinv[7] * v + k_rinv[8];

    x /= z;
    y /= z;
}


inline
void SphericalProjector::mapForward(float x, float y, float &u, float &v)
{    
    float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
    float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
    float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

    u = scale * atan2f(x_, z_);
    v = scale * (static_cast<float>(CV_PI) - acosf(y_ / sqrtf(x_ * x_ + y_ * y_ + z_ * z_)));
}


inline
void SphericalProjector::mapBackward(float u, float v, float &x, float &y)
{
    u /= scale;
    v /= scale;

    float sinv = sinf(static_cast<float>(CV_PI) - v);
    float x_ = sinv * sinf(u);
    float y_ = cosf(static_cast<float>(CV_PI) - v);
    float z_ = sinv * cosf(u);

    float z;
    x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
    y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
    z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;

    x /= z;
    y /= z;
}


inline
void CylindricalProjector::mapForward(float x, float y, float &u, float &v)
{
    float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
    float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
    float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

    u = scale * atan2f(x_, z_);
    v = scale * y_ / sqrtf(x_ * x_ + z_ * z_);
}


inline
void CylindricalProjector::mapBackward(float u, float v, float &x, float &y)
{
    u /= scale;
    v /= scale;

    float x_ = sinf(u);
    float y_ = v;
    float z_ = cosf(u);

    float z;
    x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
    y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
    z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;

    x /= z;
    y /= z;
}

} // namespace detail
} // namespace cv

#endif // __OPENCV_STITCHING_WARPERS_INL_HPP__
