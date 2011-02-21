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

#if !defined(HAVE_CUDA)

void cv::gpu::transformPoints(const GpuMat&, const Mat&, const Mat&,
                              GpuMat&) { throw_nogpu(); }

void cv::gpu::projectPoints(const GpuMat&, const Mat&, const Mat&,
                            const Mat&, const Mat&, GpuMat&) { throw_nogpu(); }

#else

namespace cv { namespace gpu { namespace transform_points 
{
    void call(const DevMem2D_<float3> src, const float* rot, const float* transl, DevMem2D_<float3> dst);
}}}

void cv::gpu::transformPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                              GpuMat& dst)
{
    CV_Assert(src.rows == 1 && src.cols > 0 && src.type() == CV_32FC3);
    CV_Assert(rvec.size() == Size(3, 1) && rvec.type() == CV_32F);
    CV_Assert(tvec.size() == Size(3, 1) && tvec.type() == CV_32F);

    // Convert rotation vector into matrix
    Mat rot;
    Rodrigues(rvec, rot);

    dst.create(src.size(), src.type());
    transform_points::call(src, rot.ptr<float>(), tvec.ptr<float>(), dst);
}


namespace cv { namespace gpu { namespace project_points 
{
    void call(const DevMem2D_<float3> src, const float* rot, const float* transl, const float* proj, DevMem2D_<float2> dst);
}}}

void cv::gpu::projectPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                            const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst)
{
    CV_Assert(src.rows == 1 && src.cols > 0 && src.type() == CV_32FC3);
    CV_Assert(rvec.size() == Size(3, 1) && rvec.type() == CV_32F);
    CV_Assert(tvec.size() == Size(3, 1) && tvec.type() == CV_32F);
    CV_Assert(camera_mat.size() == Size(3, 3) && camera_mat.type() == CV_32F);
    CV_Assert(dist_coef.empty()); // Undistortion isn't supported

    // Convert rotation vector into matrix
    Mat rot;
    Rodrigues(rvec, rot);

    dst.create(src.size(), CV_32FC2);
    project_points::call(src, rot.ptr<float>(), tvec.ptr<float>(), camera_mat.ptr<float>(), dst);
}

#endif
