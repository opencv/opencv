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

#include "precomp.hpp"

using namespace std;

namespace cv {
namespace detail {

void ProjectorBase::setCameraParams(const Mat &K, const Mat &R)
{
    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);

    Mat_<float> K_(K);
    k[0] = K_(0,0); k[1] = K_(0,1); k[2] = K_(0,2);
    k[3] = K_(1,0); k[4] = K_(1,1); k[5] = K_(1,2);
    k[6] = K_(2,0); k[7] = K_(2,1); k[8] = K_(2,2);

    Mat_<float> Rinv = R.t();
    rinv[0] = Rinv(0,0); rinv[1] = Rinv(0,1); rinv[2] = Rinv(0,2);
    rinv[3] = Rinv(1,0); rinv[4] = Rinv(1,1); rinv[5] = Rinv(1,2);
    rinv[6] = Rinv(2,0); rinv[7] = Rinv(2,1); rinv[8] = Rinv(2,2);

    Mat_<float> R_Kinv = R * K.inv();
    r_kinv[0] = R_Kinv(0,0); r_kinv[1] = R_Kinv(0,1); r_kinv[2] = R_Kinv(0,2); 
    r_kinv[3] = R_Kinv(1,0); r_kinv[4] = R_Kinv(1,1); r_kinv[5] = R_Kinv(1,2); 
    r_kinv[6] = R_Kinv(2,0); r_kinv[7] = R_Kinv(2,1); r_kinv[8] = R_Kinv(2,2); 

    Mat_<float> K_Rinv = K * Rinv;
    k_rinv[0] = K_Rinv(0,0); k_rinv[1] = K_Rinv(0,1); k_rinv[2] = K_Rinv(0,2); 
    k_rinv[3] = K_Rinv(1,0); k_rinv[4] = K_Rinv(1,1); k_rinv[5] = K_Rinv(1,2); 
    k_rinv[6] = K_Rinv(2,0); k_rinv[7] = K_Rinv(2,1); k_rinv[8] = K_Rinv(2,2); 
}


void PlaneWarper::detectResultRoi(Point &dst_tl, Point &dst_br)
{
    float tl_uf = numeric_limits<float>::max();
    float tl_vf = numeric_limits<float>::max();
    float br_uf = -numeric_limits<float>::max();
    float br_vf = -numeric_limits<float>::max();

    float u, v;

    projector_.mapForward(0, 0, u, v);
    tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
    br_uf = max(br_uf, u); br_vf = max(br_vf, v);

    projector_.mapForward(0, static_cast<float>(src_size_.height - 1), u, v);
    tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
    br_uf = max(br_uf, u); br_vf = max(br_vf, v);

    projector_.mapForward(static_cast<float>(src_size_.width - 1), 0, u, v);
    tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
    br_uf = max(br_uf, u); br_vf = max(br_vf, v);

    projector_.mapForward(static_cast<float>(src_size_.width - 1), static_cast<float>(src_size_.height - 1), u, v);
    tl_uf = min(tl_uf, u); tl_vf = min(tl_vf, v);
    br_uf = max(br_uf, u); br_vf = max(br_vf, v);

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}

#ifndef ANDROID
Point PlaneWarperGpu::warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst, int interp_mode, int border_mode)
{
    src_size_ = src.size();
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(dst_tl, dst_br);

    gpu::buildWarpPlaneMaps(src.size(), Rect(dst_tl, Point(dst_br.x+1, dst_br.y+1)),
                            K, R, projector_.scale, d_xmap_, d_ymap_);

    gpu::ensureSizeIsEnough(src.size(), src.type(), d_src_);
    d_src_.upload(src);

    gpu::ensureSizeIsEnough(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, src.type(), d_dst_);

    gpu::remap(d_src_, d_dst_, d_xmap_, d_ymap_, interp_mode, border_mode);

    d_dst_.download(dst);

    return dst_tl;
}
#endif


void SphericalWarper::detectResultRoi(Point &dst_tl, Point &dst_br)
{
    detectResultRoiByBorder(dst_tl, dst_br);

    float tl_uf = static_cast<float>(dst_tl.x);
    float tl_vf = static_cast<float>(dst_tl.y);
    float br_uf = static_cast<float>(dst_br.x);
    float br_vf = static_cast<float>(dst_br.y);

    float x = projector_.rinv[1];
    float y = projector_.rinv[4];
    float z = projector_.rinv[7];
    if (y > 0.f)
    {
        //x = projector_.focal * x / z + src_size_.width * 0.5f;
        //y = projector_.focal * y / z + src_size_.height * 0.5f;
        float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
        float y_ = projector_.k[4] * y / z + projector_.k[5];
        if (x_ > 0.f && x_ < src_size_.width && y_ > 0.f && y_ < src_size_.height)
        {
            tl_uf = min(tl_uf, 0.f); tl_vf = min(tl_vf, static_cast<float>(CV_PI * projector_.scale));
            br_uf = max(br_uf, 0.f); br_vf = max(br_vf, static_cast<float>(CV_PI * projector_.scale));
        }
    }

    x = projector_.rinv[1];
    y = -projector_.rinv[4];
    z = projector_.rinv[7];
    if (y > 0.f)
    {
        //x = projector_.focal * x / z + src_size_.width * 0.5f;
        //y = projector_.focal * y / z + src_size_.height * 0.5f;
        float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
        float y_ = projector_.k[4] * y / z + projector_.k[5];
        if (x_ > 0.f && x_ < src_size_.width && y_ > 0.f && y_ < src_size_.height)
        {
            tl_uf = min(tl_uf, 0.f); tl_vf = min(tl_vf, static_cast<float>(0));
            br_uf = max(br_uf, 0.f); br_vf = max(br_vf, static_cast<float>(0));
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}

#ifndef ANDROID
Point SphericalWarperGpu::warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst,
                               int interp_mode, int border_mode)
{
    src_size_ = src.size();
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(dst_tl, dst_br);

    gpu::buildWarpSphericalMaps(src.size(), Rect(dst_tl, Point(dst_br.x+1, dst_br.y+1)),
                                K, R, projector_.scale, d_xmap_, d_ymap_);

    gpu::ensureSizeIsEnough(src.size(), src.type(), d_src_);
    d_src_.upload(src);

    gpu::ensureSizeIsEnough(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, src.type(), d_dst_);

    gpu::remap(d_src_, d_dst_, d_xmap_, d_ymap_, interp_mode, border_mode);

    d_dst_.download(dst);

    return dst_tl;
}


Point CylindricalWarperGpu::warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst,
                                 int interp_mode, int border_mode)
{
    src_size_ = src.size();
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(dst_tl, dst_br);

    gpu::buildWarpCylindricalMaps(src.size(), Rect(dst_tl, Point(dst_br.x+1, dst_br.y+1)),
                                  K, R, projector_.scale, d_xmap_, d_ymap_);

    gpu::ensureSizeIsEnough(src.size(), src.type(), d_src_);
    d_src_.upload(src);

    gpu::ensureSizeIsEnough(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, src.type(), d_dst_);

    gpu::remap(d_src_, d_dst_, d_xmap_, d_ymap_, interp_mode, border_mode);

    d_dst_.download(dst);

    return dst_tl;
}
#endif

} // namespace detail
} // namespace cv
