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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudev/ptr2d/glob.hpp"

using namespace cv::cudev;

void RGB_to_YV12(const GpuMat& src, GpuMat& dst);

namespace
{
    __device__ __forceinline__ void rgb_to_y(const uchar b, const uchar g, const uchar r, uchar& y)
    {
        y = static_cast<uchar>(((int)(30 * r) + (int)(59 * g) + (int)(11 * b)) / 100);
    }

    __device__ __forceinline__ void rgb_to_yuv(const uchar b, const uchar g, const uchar r, uchar& y, uchar& u, uchar& v)
    {
        rgb_to_y(b, g, r, y);
        u = static_cast<uchar>(((int)(-17 * r) - (int)(33 * g) + (int)(50 * b) + 12800) / 100);
        v = static_cast<uchar>(((int)(50 * r) - (int)(42 * g) - (int)(8 * b) + 12800) / 100);
    }

    __global__ void Gray_to_YV12(const GlobPtrSz<uchar> src, GlobPtr<uchar> dst)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

        if (x + 1 >= src.cols || y + 1 >= src.rows)
            return;

        // get pointers to the data
        const size_t planeSize = src.rows * dst.step;
        GlobPtr<uchar> y_plane = globPtr(dst.data, dst.step);
        GlobPtr<uchar> u_plane = globPtr(y_plane.data + planeSize, dst.step / 2);
        GlobPtr<uchar> v_plane = globPtr(u_plane.data + (planeSize / 4), dst.step / 2);

        uchar pix;
        uchar y_val, u_val, v_val;

        pix = src(y, x);
        rgb_to_y(pix, pix, pix, y_val);
        y_plane(y, x) = y_val;

        pix = src(y, x + 1);
        rgb_to_y(pix, pix, pix, y_val);
        y_plane(y, x + 1) = y_val;

        pix = src(y + 1, x);
        rgb_to_y(pix, pix, pix, y_val);
        y_plane(y + 1, x) = y_val;

        pix = src(y + 1, x + 1);
        rgb_to_yuv(pix, pix, pix, y_val, u_val, v_val);
        y_plane(y + 1, x + 1) = y_val;
        u_plane(y / 2, x / 2) = u_val;
        v_plane(y / 2, x / 2) = v_val;
    }

    template <typename T>
    __global__ void RGB_to_YV12(const GlobPtrSz<T> src, GlobPtr<uchar> dst)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

        if (x + 1 >= src.cols || y + 1 >= src.rows)
            return;

        // get pointers to the data
        const size_t planeSize = src.rows * dst.step;
        GlobPtr<uchar> y_plane = globPtr(dst.data, dst.step);
        GlobPtr<uchar> u_plane = globPtr(y_plane.data + planeSize, dst.step / 2);
        GlobPtr<uchar> v_plane = globPtr(u_plane.data + (planeSize / 4), dst.step / 2);

        T pix;
        uchar y_val, u_val, v_val;

        pix = src(y, x);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y, x) = y_val;

        pix = src(y, x + 1);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y, x + 1) = y_val;

        pix = src(y + 1, x);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y + 1, x) = y_val;

        pix = src(y + 1, x + 1);
        rgb_to_yuv(pix.z, pix.y, pix.x, y_val, u_val, v_val);
        y_plane(y + 1, x + 1) = y_val;
        u_plane(y / 2, x / 2) = u_val;
        v_plane(y / 2, x / 2) = v_val;
    }
}

void RGB_to_YV12(const GpuMat& src, GpuMat& dst)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(src.cols, block.x * 2), divUp(src.rows, block.y * 2));

    switch (src.channels())
    {
    case 1:
        Gray_to_YV12<<<grid, block>>>(globPtr<uchar>(src), globPtr<uchar>(dst));
        break;
    case 3:
        RGB_to_YV12<<<grid, block>>>(globPtr<uchar3>(src), globPtr<uchar>(dst));
        break;
    case 4:
        RGB_to_YV12<<<grid, block>>>(globPtr<uchar4>(src), globPtr<uchar>(dst));
        break;
    }

    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

#endif
