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
                                 float scale, oclMat &map_x, oclMat &map_y)
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
    map_x.create(dst_roi.size(), CV_32F);
    map_y.create(dst_roi.size(), CV_32F);

    int tl_u = dst_roi.tl().x;
    int tl_v = dst_roi.tl().y;

    Context *clCxt = Context::getContext();
    String kernelName = "buildWarpPlaneMaps";
    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&map_x.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&map_y.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&KRT_mat.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&tl_u));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&tl_v));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_y.step));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&scale));

    size_t globalThreads[3] = {map_x.cols, map_x.rows, 1};
    size_t localThreads[3]  = {32, 8, 1};
    openCLExecuteKernel(clCxt, &build_warps, kernelName, globalThreads, localThreads, args, -1, -1);
}

//////////////////////////////////////////////////////////////////////////////
// buildWarpCylyndricalMaps

void cv::ocl::buildWarpCylindricalMaps(Size /*src_size*/, Rect dst_roi, const Mat &K, const Mat &R, float scale,
                                       oclMat &map_x, oclMat &map_y)
{
    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);

    Mat K_Rinv = K * R.t();
    CV_Assert(K_Rinv.isContinuous());

    oclMat KR_oclMat(K_Rinv.reshape(1, 1));

    map_x.create(dst_roi.size(), CV_32F);
    map_y.create(dst_roi.size(), CV_32F);

    int tl_u = dst_roi.tl().x;
    int tl_v = dst_roi.tl().y;

    Context *clCxt = Context::getContext();
    String kernelName = "buildWarpCylindricalMaps";
    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&map_x.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&map_y.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&KR_oclMat.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&tl_u));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&tl_v));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_y.step));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&scale));

    size_t globalThreads[3] = {map_x.cols, map_x.rows, 1};
    size_t localThreads[3]  = {32, 8, 1};
    openCLExecuteKernel(clCxt, &build_warps, kernelName, globalThreads, localThreads, args, -1, -1);
}

//////////////////////////////////////////////////////////////////////////////
// buildWarpSphericalMaps
void cv::ocl::buildWarpSphericalMaps(Size /*src_size*/, Rect dst_roi, const Mat &K, const Mat &R, float scale,
                                     oclMat &map_x, oclMat &map_y)
{
    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);

    Mat K_Rinv = K * R.t();
    CV_Assert(K_Rinv.isContinuous());

    oclMat KR_oclMat(K_Rinv.reshape(1, 1));
    // transfer K_Rinv, R_Kinv into a single cl_mem
    map_x.create(dst_roi.size(), CV_32F);
    map_y.create(dst_roi.size(), CV_32F);

    int tl_u = dst_roi.tl().x;
    int tl_v = dst_roi.tl().y;

    Context *clCxt = Context::getContext();
    String kernelName = "buildWarpSphericalMaps";
    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&map_x.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&map_y.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&KR_oclMat.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&tl_u));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&tl_v));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_x.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&map_y.step));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&scale));

    size_t globalThreads[3] = {map_x.cols, map_x.rows, 1};
    size_t localThreads[3]  = {32, 8, 1};
    openCLExecuteKernel(clCxt, &build_warps, kernelName, globalThreads, localThreads, args, -1, -1);
}


void cv::ocl::buildWarpAffineMaps(const Mat &M, bool inverse, Size dsize, oclMat &xmap, oclMat &ymap)
{

    CV_Assert(M.rows == 2 && M.cols == 3);

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

    oclMat coeffsOclMat(coeffsMat.reshape(1, 1));

    Context *clCxt = Context::getContext();
    String kernelName = "buildWarpAffineMaps";
    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&xmap.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&ymap.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&coeffsOclMat.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&xmap.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&xmap.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&xmap.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&ymap.step));

    size_t globalThreads[3] = {xmap.cols, xmap.rows, 1};
    size_t localThreads[3]  = {32, 8, 1};
    openCLExecuteKernel(clCxt, &build_warps, kernelName, globalThreads, localThreads, args, -1, -1);
}

void cv::ocl::buildWarpPerspectiveMaps(const Mat &M, bool inverse, Size dsize, oclMat &xmap, oclMat &ymap)
{

    CV_Assert(M.rows == 3 && M.cols == 3);

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

    Context *clCxt = Context::getContext();
    String kernelName = "buildWarpPerspectiveMaps";
    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&xmap.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&ymap.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&coeffsOclMat.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&xmap.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&xmap.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&xmap.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&ymap.step));

    size_t globalThreads[3] = {xmap.cols, xmap.rows, 1};
    size_t localThreads[3]  = {32, 8, 1};
    openCLExecuteKernel(clCxt, &build_warps, kernelName, globalThreads, localThreads, args, -1, -1);
}
