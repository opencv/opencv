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

#ifndef __cvt_color_internal_h__
#define __cvt_color_internal_h__

namespace cv { namespace cuda { namespace device
{
#define OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name) \
    void name(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

#define OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(name)       \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _8u)    \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _16u)   \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _32f)

#define OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(name)    \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _8u)   \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _32f)

#define OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(name)    \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _8u)        \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _32f)       \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _full_8u)   \
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(name ## _full_32f)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_rgba)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr_to_bgr555)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr_to_bgr565)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(rgb_to_bgr555)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(rgb_to_bgr565)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgra_to_bgr555)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgra_to_bgr565)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(rgba_to_bgr555)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(rgba_to_bgr565)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr555_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr565_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr555_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr565_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr555_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr565_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr555_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr565_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(gray_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(gray_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(gray_to_bgr555)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(gray_to_bgr565)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr555_to_gray)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ONE(bgr565_to_gray)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgb_to_gray)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_gray)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgba_to_gray)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_gray)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgb_to_yuv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgba_to_yuv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgb_to_yuv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgba_to_yuv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_yuv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_yuv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_yuv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_yuv4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(yuv_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(yuv_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(yuv4_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(yuv4_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(yuv_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(yuv_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(yuv4_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(yuv4_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgb_to_YCrCb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgba_to_YCrCb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgb_to_YCrCb4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgba_to_YCrCb4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_YCrCb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_YCrCb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_YCrCb4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_YCrCb4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(YCrCb_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(YCrCb_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(YCrCb4_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(YCrCb4_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(YCrCb_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(YCrCb_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(YCrCb4_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(YCrCb4_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgb_to_xyz)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgba_to_xyz)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgb_to_xyz4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(rgba_to_xyz4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_xyz)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_xyz)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgr_to_xyz4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(bgra_to_xyz4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(xyz_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(xyz4_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(xyz_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(xyz4_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(xyz_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(xyz4_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(xyz_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_ALL(xyz4_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(rgb_to_hsv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(rgba_to_hsv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(rgb_to_hsv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(rgba_to_hsv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(bgr_to_hsv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(bgra_to_hsv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(bgr_to_hsv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(bgra_to_hsv4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hsv_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hsv_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hsv4_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hsv4_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hsv_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hsv_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hsv4_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hsv4_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(rgb_to_hls)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(rgba_to_hls)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(rgb_to_hls4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(rgba_to_hls4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(bgr_to_hls)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(bgra_to_hls)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(bgr_to_hls4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(bgra_to_hls4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hls_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hls_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hls4_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hls4_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hls_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hls_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hls4_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL(hls4_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(rgb_to_lab)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(rgba_to_lab)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(rgb_to_lab4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(rgba_to_lab4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(bgr_to_lab)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(bgra_to_lab)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(bgr_to_lab4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(bgra_to_lab4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lrgb_to_lab)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lrgba_to_lab)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lrgb_to_lab4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lrgba_to_lab4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lbgr_to_lab)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lbgra_to_lab)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lbgr_to_lab4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lbgra_to_lab4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab4_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab4_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab4_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab4_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab_to_lrgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab4_to_lrgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab_to_lrgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab4_to_lrgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab_to_lbgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab4_to_lbgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab_to_lbgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lab4_to_lbgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(rgb_to_luv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(rgba_to_luv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(rgb_to_luv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(rgba_to_luv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(bgr_to_luv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(bgra_to_luv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(bgr_to_luv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(bgra_to_luv4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lrgb_to_luv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lrgba_to_luv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lrgb_to_luv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lrgba_to_luv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lbgr_to_luv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lbgra_to_luv)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lbgr_to_luv4)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(lbgra_to_luv4)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv4_to_rgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv4_to_rgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv4_to_bgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv_to_bgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv4_to_bgra)

    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv_to_lrgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv4_to_lrgb)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv_to_lrgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv4_to_lrgba)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv_to_lbgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv4_to_lbgr)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv_to_lbgra)
    OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F(luv4_to_lbgra)

    #undef OPENCV_CUDA_DECLARE_CVTCOLOR_ONE
    #undef OPENCV_CUDA_DECLARE_CVTCOLOR_ALL
    #undef OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F
    #undef OPENCV_CUDA_DECLARE_CVTCOLOR_8U32F_FULL
}}}

#endif
