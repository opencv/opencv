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
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;

#ifdef HAVE_CUDA

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        void buildWarpPlaneMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
                                const float k_rinv[9], const float r_kinv[9], const float t[3], float scale,
                                cudaStream_t stream);

        void buildWarpSphericalMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
                                    const float k_rinv[9], const float r_kinv[9], float scale,
                                    cudaStream_t stream);

        void buildWarpCylindricalMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
                                      const float k_rinv[9], const float r_kinv[9], float scale,
                                      cudaStream_t stream);
    }
}}}

static void buildWarpPlaneMaps(Size src_size, Rect dst_roi, InputArray _K, InputArray _R, InputArray _T,
                               float scale, OutputArray _map_x, OutputArray _map_y, Stream& stream = Stream::Null())
{
    (void) src_size;

    Mat K = _K.getMat();
    Mat R = _R.getMat();
    Mat T = _T.getMat();

    CV_Assert( K.size() == Size(3,3) && K.type() == CV_32FC1 );
    CV_Assert( R.size() == Size(3,3) && R.type() == CV_32FC1 );
    CV_Assert( (T.size() == Size(3,1) || T.size() == Size(1,3)) && T.type() == CV_32FC1 && T.isContinuous() );

    Mat K_Rinv = K * R.t();
    Mat R_Kinv = R * K.inv();
    CV_Assert( K_Rinv.isContinuous() );
    CV_Assert( R_Kinv.isContinuous() );

    _map_x.create(dst_roi.size(), CV_32FC1);
    _map_y.create(dst_roi.size(), CV_32FC1);

    GpuMat map_x = _map_x.getGpuMat();
    GpuMat map_y = _map_y.getGpuMat();

    device::imgproc::buildWarpPlaneMaps(dst_roi.tl().x, dst_roi.tl().y, map_x, map_y, K_Rinv.ptr<float>(), R_Kinv.ptr<float>(),
                       T.ptr<float>(), scale, StreamAccessor::getStream(stream));
}

static void buildWarpSphericalMaps(Size src_size, Rect dst_roi, InputArray _K, InputArray _R, float scale,
                                   OutputArray _map_x, OutputArray _map_y, Stream& stream = Stream::Null())
{
    (void) src_size;

    Mat K = _K.getMat();
    Mat R = _R.getMat();

    CV_Assert( K.size() == Size(3,3) && K.type() == CV_32FC1 );
    CV_Assert( R.size() == Size(3,3) && R.type() == CV_32FC1 );

    Mat K_Rinv = K * R.t();
    Mat R_Kinv = R * K.inv();
    CV_Assert( K_Rinv.isContinuous() );
    CV_Assert( R_Kinv.isContinuous() );

    _map_x.create(dst_roi.size(), CV_32FC1);
    _map_y.create(dst_roi.size(), CV_32FC1);

    GpuMat map_x = _map_x.getGpuMat();
    GpuMat map_y = _map_y.getGpuMat();

    device::imgproc::buildWarpSphericalMaps(dst_roi.tl().x, dst_roi.tl().y, map_x, map_y, K_Rinv.ptr<float>(), R_Kinv.ptr<float>(), scale, StreamAccessor::getStream(stream));
}

static void buildWarpCylindricalMaps(Size src_size, Rect dst_roi, InputArray _K, InputArray _R, float scale,
                                     OutputArray _map_x, OutputArray _map_y, Stream& stream = Stream::Null())
{
    (void) src_size;

    Mat K = _K.getMat();
    Mat R = _R.getMat();

    CV_Assert( K.size() == Size(3,3) && K.type() == CV_32FC1 );
    CV_Assert( R.size() == Size(3,3) && R.type() == CV_32FC1 );

    Mat K_Rinv = K * R.t();
    Mat R_Kinv = R * K.inv();
    CV_Assert( K_Rinv.isContinuous() );
    CV_Assert( R_Kinv.isContinuous() );

    _map_x.create(dst_roi.size(), CV_32FC1);
    _map_y.create(dst_roi.size(), CV_32FC1);

    GpuMat map_x = _map_x.getGpuMat();
    GpuMat map_y = _map_y.getGpuMat();

    device::imgproc::buildWarpCylindricalMaps(dst_roi.tl().x, dst_roi.tl().y, map_x, map_y, K_Rinv.ptr<float>(), R_Kinv.ptr<float>(), scale, StreamAccessor::getStream(stream));
}

#endif

Rect cv::detail::PlaneWarperGpu::buildMaps(Size src_size, InputArray K, InputArray R,
                                           cuda::GpuMat & xmap, cuda::GpuMat & ymap)
{
    return buildMaps(src_size, K, R, Mat::zeros(3, 1, CV_32F), xmap, ymap);
}

Rect cv::detail::PlaneWarperGpu::buildMaps(Size src_size, InputArray K, InputArray R, InputArray T,
                                           cuda::GpuMat & xmap, cuda::GpuMat & ymap)
{
#ifndef HAVE_CUDA
    (void)src_size;
    (void)K;
    (void)R;
    (void)T;
    (void)xmap;
    (void)ymap;
    throw_no_cuda();
    return Rect();
#else
    projector_.setCameraParams(K, R, T);

    Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);

    ::buildWarpPlaneMaps(src_size, Rect(dst_tl, Point(dst_br.x + 1, dst_br.y + 1)),
                         K, R, T, projector_.scale, xmap, ymap);

    return Rect(dst_tl, dst_br);
#endif
}

Point cv::detail::PlaneWarperGpu::warp(const cuda::GpuMat & src, InputArray K, InputArray R,
                                       int interp_mode, int border_mode,
                                       cuda::GpuMat & dst)
{
    return warp(src, K, R, Mat::zeros(3, 1, CV_32F), interp_mode, border_mode, dst);
}


Point cv::detail::PlaneWarperGpu::warp(const cuda::GpuMat & src, InputArray K, InputArray R, InputArray T,
                                       int interp_mode, int border_mode,
                                       cuda::GpuMat & dst)
{
#ifndef HAVE_OPENCV_CUDAWARPING
    (void)src;
    (void)K;
    (void)R;
    (void)T;
    (void)interp_mode;
    (void)border_mode;
    (void)dst;
    throw_no_cuda();
    return Point();
#else
    Rect dst_roi = buildMaps(src.size(), K, R, T, d_xmap_, d_ymap_);
    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    cuda::remap(src, dst, d_xmap_, d_ymap_, interp_mode, border_mode);
    return dst_roi.tl();
#endif
}

Rect cv::detail::SphericalWarperGpu::buildMaps(Size src_size, InputArray K, InputArray R, cuda::GpuMat & xmap, cuda::GpuMat & ymap)
{
#ifndef HAVE_CUDA
    (void)src_size;
    (void)K;
    (void)R;
    (void)xmap;
    (void)ymap;
    throw_no_cuda();
    return Rect();
#else
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);

    ::buildWarpSphericalMaps(src_size, Rect(dst_tl, Point(dst_br.x + 1, dst_br.y + 1)),
                             K, R, projector_.scale, xmap, ymap);

    return Rect(dst_tl, dst_br);
#endif
}

Point cv::detail::SphericalWarperGpu::warp(const cuda::GpuMat & src, InputArray K, InputArray R,
                                           int interp_mode, int border_mode,
                                           cuda::GpuMat & dst)
{
#ifndef HAVE_OPENCV_CUDAWARPING
    (void)src;
    (void)K;
    (void)R;
    (void)interp_mode;
    (void)border_mode;
    (void)dst;
    throw_no_cuda();
    return Point();
#else
    Rect dst_roi = buildMaps(src.size(), K, R, d_xmap_, d_ymap_);
    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    cuda::remap(src, dst, d_xmap_, d_ymap_, interp_mode, border_mode);
    return dst_roi.tl();
#endif
}


Rect cv::detail::CylindricalWarperGpu::buildMaps(Size src_size, InputArray K, InputArray R,
                                                 cuda::GpuMat & xmap, cuda::GpuMat & ymap)
{
#ifndef HAVE_CUDA
    (void)src_size;
    (void)K;
    (void)R;
    (void)xmap;
    (void)ymap;
    throw_no_cuda();
    return Rect();
#else
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);

    ::buildWarpCylindricalMaps(src_size, Rect(dst_tl, Point(dst_br.x + 1, dst_br.y + 1)),
                               K, R, projector_.scale, xmap, ymap);

    return Rect(dst_tl, dst_br);
#endif
}

Point cv::detail::CylindricalWarperGpu::warp(const cuda::GpuMat & src, InputArray K, InputArray R,
                                             int interp_mode, int border_mode,
                                             cuda::GpuMat & dst)
{
#ifndef HAVE_OPENCV_CUDAWARPING
    (void)src;
    (void)K;
    (void)R;
    (void)interp_mode;
    (void)border_mode;
    (void)dst;
    throw_no_cuda();
    return Point();
#else
    Rect dst_roi = buildMaps(src.size(), K, R, d_xmap_, d_ymap_);
    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    cuda::remap(src, dst, d_xmap_, d_ymap_, interp_mode, border_mode);
    return dst_roi.tl();
#endif
}
