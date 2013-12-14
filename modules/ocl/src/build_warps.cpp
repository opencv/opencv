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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

//////////////////////////////////////////////////////////////////////////////
// buildWarpPlaneMaps

void cv::ocl::buildWarpPlaneMaps(Size /*src_size*/, Rect dst_roi, const Mat &K, const Mat &R, const Mat &T,
                                 float scale, oclMat &xmap, oclMat &ymap)
{
    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);
    CV_Assert((T.size() == Size(3, 1) || T.size() == Size(1, 3)) && T.type() == CV_32F && T.isContinuous());

    Mat K_Rinv = K * R.t();
    CV_Assert(K_Rinv.isContinuous());

    Mat KRT_mat(1, 12, CV_32FC1); // 9 + 3
    KRT_mat(Range::all(), Range(0, 8)) = K_Rinv.reshape(1, 1);
    KRT_mat(Range::all(), Range(9, 11)) = T;

    oclMat KRT_oclMat(KRT_mat);
    // transfer K_Rinv and T into a single cl_mem
    xmap.create(dst_roi.size(), CV_32F);
    ymap.create(dst_roi.size(), CV_32F);

    int tl_u = dst_roi.tl().x;
    int tl_v = dst_roi.tl().y;

    int xmap_step = xmap.step / xmap.elemSize(), xmap_offset = xmap.offset / xmap.elemSize();
    int ymap_step = ymap.step / ymap.elemSize(), ymap_offset = ymap.offset / ymap.elemSize();

    vector< pair<size_t, const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&xmap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&ymap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&KRT_mat.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&tl_u));
    args.push_back( make_pair( sizeof(cl_int), (void *)&tl_v));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_offset));
    args.push_back( make_pair( sizeof(cl_float), (void *)&scale));

    size_t globalThreads[3] = { xmap.cols, xmap.rows, 1 };
#ifdef ANDROID
    size_t localThreads[3]  = {32, 4, 1};
#else
    size_t localThreads[3]  = {32, 8, 1};
#endif
    openCLExecuteKernel(Context::getContext(), &build_warps, "buildWarpPlaneMaps", globalThreads, localThreads, args, -1, -1);
}

//////////////////////////////////////////////////////////////////////////////
// buildWarpCylyndricalMaps

void cv::ocl::buildWarpCylindricalMaps(Size /*src_size*/, Rect dst_roi, const Mat &K, const Mat &R, float scale,
                                       oclMat &xmap, oclMat &ymap)
{
    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);

    Mat K_Rinv = K * R.t();
    CV_Assert(K_Rinv.isContinuous());

    oclMat KR_oclMat(K_Rinv.reshape(1, 1));

    xmap.create(dst_roi.size(), CV_32F);
    ymap.create(dst_roi.size(), CV_32F);

    int tl_u = dst_roi.tl().x;
    int tl_v = dst_roi.tl().y;

    int xmap_step = xmap.step / xmap.elemSize(), xmap_offset = xmap.offset / xmap.elemSize();
    int ymap_step = ymap.step / ymap.elemSize(), ymap_offset = ymap.offset / ymap.elemSize();

    vector< pair<size_t, const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&xmap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&ymap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&KR_oclMat.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&tl_u));
    args.push_back( make_pair( sizeof(cl_int), (void *)&tl_v));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_offset));
    args.push_back( make_pair( sizeof(cl_float), (void *)&scale));

    size_t globalThreads[3] = { xmap.cols, xmap.rows, 1 };
#ifdef ANDROID
    size_t localThreads[3]  = {32, 1, 1};
#else
    size_t localThreads[3]  = {32, 8, 1};
#endif
    openCLExecuteKernel(Context::getContext(), &build_warps, "buildWarpCylindricalMaps", globalThreads, localThreads, args, -1, -1);
}

//////////////////////////////////////////////////////////////////////////////
// buildWarpSphericalMaps

void cv::ocl::buildWarpSphericalMaps(Size /*src_size*/, Rect dst_roi, const Mat &K, const Mat &R, float scale,
                                     oclMat &xmap, oclMat &ymap)
{
    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);

    Mat K_Rinv = K * R.t();
    CV_Assert(K_Rinv.isContinuous());

    oclMat KR_oclMat(K_Rinv.reshape(1, 1));
    // transfer K_Rinv, R_Kinv into a single cl_mem
    xmap.create(dst_roi.size(), CV_32F);
    ymap.create(dst_roi.size(), CV_32F);

    int tl_u = dst_roi.tl().x;
    int tl_v = dst_roi.tl().y;

    int xmap_step = xmap.step / xmap.elemSize(), xmap_offset = xmap.offset / xmap.elemSize();
    int ymap_step = ymap.step / ymap.elemSize(), ymap_offset = ymap.offset / ymap.elemSize();

    vector< pair<size_t, const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&xmap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&ymap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&KR_oclMat.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&tl_u));
    args.push_back( make_pair( sizeof(cl_int), (void *)&tl_v));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_offset));
    args.push_back( make_pair( sizeof(cl_float), (void *)&scale));

    size_t globalThreads[3] = { xmap.cols, xmap.rows, 1 };
#ifdef ANDROID
    size_t localThreads[3]  = {32, 4, 1};
#else
    size_t localThreads[3]  = {32, 8, 1};
#endif
    openCLExecuteKernel(Context::getContext(), &build_warps, "buildWarpSphericalMaps", globalThreads, localThreads, args, -1, -1);
}

//////////////////////////////////////////////////////////////////////////////
// buildWarpAffineMaps

void cv::ocl::buildWarpAffineMaps(const Mat &M, bool inverse, Size dsize, oclMat &xmap, oclMat &ymap)
{
    CV_Assert(M.rows == 2 && M.cols == 3);
    CV_Assert(dsize.area());

    xmap.create(dsize, CV_32FC1);
    ymap.create(dsize, CV_32FC1);

    float coeffs[2 * 3];
    Mat coeffsMat(2, 3, CV_32F, (void *)coeffs);

    if (inverse)
        M.convertTo(coeffsMat, coeffsMat.type());
    else
    {
        cv::Mat iM;
        invertAffineTransform(M, iM);
        iM.convertTo(coeffsMat, coeffsMat.type());
    }

    int xmap_step = xmap.step / xmap.elemSize(), xmap_offset = xmap.offset / xmap.elemSize();
    int ymap_step = ymap.step / ymap.elemSize(), ymap_offset = ymap.offset / ymap.elemSize();

    oclMat coeffsOclMat(coeffsMat.reshape(1, 1));

    vector< pair<size_t, const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&xmap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&ymap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&coeffsOclMat.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_offset));

    size_t globalThreads[3] = { xmap.cols, xmap.rows, 1 };
#ifdef ANDROID
    size_t localThreads[3]  = {32, 4, 1};
#else
    size_t localThreads[3]  = {32, 8, 1};
#endif
    openCLExecuteKernel(Context::getContext(), &build_warps, "buildWarpAffineMaps", globalThreads, localThreads, args, -1, -1);
}

//////////////////////////////////////////////////////////////////////////////
// buildWarpPerspectiveMaps

void cv::ocl::buildWarpPerspectiveMaps(const Mat &M, bool inverse, Size dsize, oclMat &xmap, oclMat &ymap)
{
    CV_Assert(M.rows == 3 && M.cols == 3);
    CV_Assert(dsize.area() > 0);

    xmap.create(dsize, CV_32FC1);
    ymap.create(dsize, CV_32FC1);

    float coeffs[3 * 3];
    Mat coeffsMat(3, 3, CV_32F, (void *)coeffs);

    if (inverse)
        M.convertTo(coeffsMat, coeffsMat.type());
    else
    {
        cv::Mat iM;
        invert(M, iM);
        iM.convertTo(coeffsMat, coeffsMat.type());
    }

    oclMat coeffsOclMat(coeffsMat.reshape(1, 1));

    int xmap_step = xmap.step / xmap.elemSize(), xmap_offset = xmap.offset / xmap.elemSize();
    int ymap_step = ymap.step / ymap.elemSize(), ymap_offset = ymap.offset / ymap.elemSize();

    vector< pair<size_t, const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&xmap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&ymap.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&coeffsOclMat.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap.rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&xmap_offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&ymap_offset));

    size_t globalThreads[3] = { xmap.cols, xmap.rows, 1 };

    openCLExecuteKernel(Context::getContext(), &build_warps, "buildWarpPerspectiveMaps", globalThreads, NULL, args, -1, -1);
}
