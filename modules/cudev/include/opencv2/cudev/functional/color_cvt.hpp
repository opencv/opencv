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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once

#ifndef __OPENCV_CUDEV_FUNCTIONAL_COLOR_CVT_HPP__
#define __OPENCV_CUDEV_FUNCTIONAL_COLOR_CVT_HPP__

#include "../common.hpp"
#include "detail/color_cvt.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// Various 3/4-channel to 3/4-channel RGB transformations

#define CV_CUDEV_RGB2RGB_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2RGB<SrcDepth, scn, dcn, bidx> \
    { \
    };

CV_CUDEV_RGB2RGB_INST(BGR_to_RGB, 3, 3, 2)
CV_CUDEV_RGB2RGB_INST(BGR_to_BGRA, 3, 4, 0)
CV_CUDEV_RGB2RGB_INST(BGR_to_RGBA, 3, 4, 2)
CV_CUDEV_RGB2RGB_INST(BGRA_to_BGR, 4, 3, 0)
CV_CUDEV_RGB2RGB_INST(BGRA_to_RGB, 4, 3, 2)
CV_CUDEV_RGB2RGB_INST(BGRA_to_RGBA, 4, 4, 2)

#undef CV_CUDEV_RGB2RGB_INST

// RGB to Grayscale

#define CV_CUDEV_RGB2GRAY_INST(name, scn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2Gray<SrcDepth, scn, bidx> \
    { \
    };

CV_CUDEV_RGB2GRAY_INST(RGB_to_GRAY, 3, 2)
CV_CUDEV_RGB2GRAY_INST(BGR_to_GRAY, 3, 0)
CV_CUDEV_RGB2GRAY_INST(RGBA_to_GRAY, 4, 2)
CV_CUDEV_RGB2GRAY_INST(BGRA_to_GRAY, 4, 0)

#undef CV_CUDEV_RGB2GRAY_INST

// Grayscale to RGB

#define CV_CUDEV_GRAY2RGB_INST(name, dcn) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::Gray2RGB<SrcDepth, dcn> \
    { \
    };

CV_CUDEV_GRAY2RGB_INST(GRAY_to_BGR, 3)
CV_CUDEV_GRAY2RGB_INST(GRAY_to_BGRA, 4)

#undef CV_CUDEV_GRAY2RGB_INST

// RGB to YUV

#define CV_CUDEV_RGB2YUV_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2YUV<SrcDepth, scn, dcn, bidx> \
    { \
    };

CV_CUDEV_RGB2YUV_INST(RGB_to_YUV, 3, 3, 2)
CV_CUDEV_RGB2YUV_INST(RGBA_to_YUV, 4, 3, 2)
CV_CUDEV_RGB2YUV_INST(RGB_to_YUV4, 3, 4, 2)
CV_CUDEV_RGB2YUV_INST(RGBA_to_YUV4, 4, 4, 2)
CV_CUDEV_RGB2YUV_INST(BGR_to_YUV, 3, 3, 0)
CV_CUDEV_RGB2YUV_INST(BGRA_to_YUV, 4, 3, 0)
CV_CUDEV_RGB2YUV_INST(BGR_to_YUV4, 3, 4, 0)
CV_CUDEV_RGB2YUV_INST(BGRA_to_YUV4, 4, 4, 0)

#undef CV_CUDEV_RGB2YUV_INST

// YUV to RGB

#define CV_CUDEV_YUV2RGB_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::YUV2RGB<SrcDepth, scn, dcn, bidx> \
    { \
    };

CV_CUDEV_YUV2RGB_INST(YUV_to_RGB, 3, 3, 2)
CV_CUDEV_YUV2RGB_INST(YUV_to_RGBA, 3, 4, 2)
CV_CUDEV_YUV2RGB_INST(YUV4_to_RGB, 4, 3, 2)
CV_CUDEV_YUV2RGB_INST(YUV4_to_RGBA, 4, 4, 2)
CV_CUDEV_YUV2RGB_INST(YUV_to_BGR, 3, 3, 0)
CV_CUDEV_YUV2RGB_INST(YUV_to_BGRA, 3, 4, 0)
CV_CUDEV_YUV2RGB_INST(YUV4_to_BGR, 4, 3, 0)
CV_CUDEV_YUV2RGB_INST(YUV4_to_BGRA, 4, 4, 0)

#undef CV_CUDEV_YUV2RGB_INST

// RGB to YCrCb

#define CV_CUDEV_RGB2YCrCb_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2YCrCb<SrcDepth, scn, dcn, bidx> \
    { \
    };

CV_CUDEV_RGB2YCrCb_INST(RGB_to_YCrCb, 3, 3, 2)
CV_CUDEV_RGB2YCrCb_INST(RGBA_to_YCrCb, 4, 3, 2)
CV_CUDEV_RGB2YCrCb_INST(RGB_to_YCrCb4, 3, 4, 2)
CV_CUDEV_RGB2YCrCb_INST(RGBA_to_YCrCb4, 4, 4, 2)
CV_CUDEV_RGB2YCrCb_INST(BGR_to_YCrCb, 3, 3, 0)
CV_CUDEV_RGB2YCrCb_INST(BGRA_to_YCrCb, 4, 3, 0)
CV_CUDEV_RGB2YCrCb_INST(BGR_to_YCrCb4, 3, 4, 0)
CV_CUDEV_RGB2YCrCb_INST(BGRA_to_YCrCb4, 4, 4, 0)

#undef CV_CUDEV_RGB2YCrCb_INST

// YCrCb to RGB

#define CV_CUDEV_YCrCb2RGB_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::YCrCb2RGB<SrcDepth, scn, dcn, bidx> \
    { \
    };

CV_CUDEV_YCrCb2RGB_INST(YCrCb_to_RGB, 3, 3, 2)
CV_CUDEV_YCrCb2RGB_INST(YCrCb_to_RGBA, 3, 4, 2)
CV_CUDEV_YCrCb2RGB_INST(YCrCb4_to_RGB, 4, 3, 2)
CV_CUDEV_YCrCb2RGB_INST(YCrCb4_to_RGBA, 4, 4, 2)
CV_CUDEV_YCrCb2RGB_INST(YCrCb_to_BGR, 3, 3, 0)
CV_CUDEV_YCrCb2RGB_INST(YCrCb_to_BGRA, 3, 4, 0)
CV_CUDEV_YCrCb2RGB_INST(YCrCb4_to_BGR, 4, 3, 0)
CV_CUDEV_YCrCb2RGB_INST(YCrCb4_to_BGRA, 4, 4, 0)

#undef CV_CUDEV_YCrCb2RGB_INST

// RGB to XYZ

#define CV_CUDEV_RGB2XYZ_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2XYZ<SrcDepth, scn, dcn, bidx> \
    { \
    };

CV_CUDEV_RGB2XYZ_INST(RGB_to_XYZ, 3, 3, 2)
CV_CUDEV_RGB2XYZ_INST(RGBA_to_XYZ, 4, 3, 2)
CV_CUDEV_RGB2XYZ_INST(RGB_to_XYZ4, 3, 4, 2)
CV_CUDEV_RGB2XYZ_INST(RGBA_to_XYZ4, 4, 4, 2)
CV_CUDEV_RGB2XYZ_INST(BGR_to_XYZ, 3, 3, 0)
CV_CUDEV_RGB2XYZ_INST(BGRA_to_XYZ, 4, 3, 0)
CV_CUDEV_RGB2XYZ_INST(BGR_to_XYZ4, 3, 4, 0)
CV_CUDEV_RGB2XYZ_INST(BGRA_to_XYZ4, 4, 4, 0)

#undef CV_CUDEV_RGB2XYZ_INST

// XYZ to RGB

#define CV_CUDEV_XYZ2RGB_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::XYZ2RGB<SrcDepth, scn, dcn, bidx> \
    { \
    };

CV_CUDEV_XYZ2RGB_INST(XYZ_to_RGB, 3, 3, 2)
CV_CUDEV_XYZ2RGB_INST(XYZ4_to_RGB, 4, 3, 2)
CV_CUDEV_XYZ2RGB_INST(XYZ_to_RGBA, 3, 4, 2)
CV_CUDEV_XYZ2RGB_INST(XYZ4_to_RGBA, 4, 4, 2)
CV_CUDEV_XYZ2RGB_INST(XYZ_to_BGR, 3, 3, 0)
CV_CUDEV_XYZ2RGB_INST(XYZ4_to_BGR, 4, 3, 0)
CV_CUDEV_XYZ2RGB_INST(XYZ_to_BGRA, 3, 4, 0)
CV_CUDEV_XYZ2RGB_INST(XYZ4_to_BGRA, 4, 4, 0)

#undef CV_CUDEV_XYZ2RGB_INST

// RGB to HSV

#define CV_CUDEV_RGB2HSV_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2HSV<SrcDepth, scn, dcn, bidx, 180> \
    { \
    }; \
    template <typename SrcDepth> struct name ## _FULL ## _func : cv::cudev::color_cvt_detail::RGB2HSV<SrcDepth, scn, dcn, bidx, 256> \
    { \
    }; \
    template <> struct name ## _func<float> : cv::cudev::color_cvt_detail::RGB2HSV<float, scn, dcn, bidx, 360> \
    { \
    }; \
    template <> struct name ## _FULL ## _func<float> : cv::cudev::color_cvt_detail::RGB2HSV<float, scn, dcn, bidx, 360> \
    { \
    };

CV_CUDEV_RGB2HSV_INST(RGB_to_HSV, 3, 3, 2)
CV_CUDEV_RGB2HSV_INST(RGBA_to_HSV, 4, 3, 2)
CV_CUDEV_RGB2HSV_INST(RGB_to_HSV4, 3, 4, 2)
CV_CUDEV_RGB2HSV_INST(RGBA_to_HSV4, 4, 4, 2)
CV_CUDEV_RGB2HSV_INST(BGR_to_HSV, 3, 3, 0)
CV_CUDEV_RGB2HSV_INST(BGRA_to_HSV, 4, 3, 0)
CV_CUDEV_RGB2HSV_INST(BGR_to_HSV4, 3, 4, 0)
CV_CUDEV_RGB2HSV_INST(BGRA_to_HSV4, 4, 4, 0)

#undef CV_CUDEV_RGB2HSV_INST

// HSV to RGB

#define CV_CUDEV_HSV2RGB_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::HSV2RGB<SrcDepth, scn, dcn, bidx, 180> \
    { \
    }; \
    template <typename SrcDepth> struct name ## _FULL ## _func : cv::cudev::color_cvt_detail::HSV2RGB<SrcDepth, scn, dcn, bidx, 255> \
    { \
    }; \
    template <> struct name ## _func<float> : cv::cudev::color_cvt_detail::HSV2RGB<float, scn, dcn, bidx, 360> \
    { \
    }; \
    template <> struct name ## _FULL ## _func<float> : cv::cudev::color_cvt_detail::HSV2RGB<float, scn, dcn, bidx, 360> \
    { \
    };

CV_CUDEV_HSV2RGB_INST(HSV_to_RGB, 3, 3, 2)
CV_CUDEV_HSV2RGB_INST(HSV_to_RGBA, 3, 4, 2)
CV_CUDEV_HSV2RGB_INST(HSV4_to_RGB, 4, 3, 2)
CV_CUDEV_HSV2RGB_INST(HSV4_to_RGBA, 4, 4, 2)
CV_CUDEV_HSV2RGB_INST(HSV_to_BGR, 3, 3, 0)
CV_CUDEV_HSV2RGB_INST(HSV_to_BGRA, 3, 4, 0)
CV_CUDEV_HSV2RGB_INST(HSV4_to_BGR, 4, 3, 0)
CV_CUDEV_HSV2RGB_INST(HSV4_to_BGRA, 4, 4, 0)

#undef CV_CUDEV_HSV2RGB_INST

// RGB to HLS

#define CV_CUDEV_RGB2HLS_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2HLS<SrcDepth, scn, dcn, bidx, 180> \
    { \
    }; \
    template <typename SrcDepth> struct name ## _FULL ## _func : cv::cudev::color_cvt_detail::RGB2HLS<SrcDepth, scn, dcn, bidx, 256> \
    { \
    }; \
    template <> struct name ## _func<float> : cv::cudev::color_cvt_detail::RGB2HLS<float, scn, dcn, bidx, 360> \
    { \
    }; \
    template <> struct name ## _FULL ## _func<float> : cv::cudev::color_cvt_detail::RGB2HLS<float, scn, dcn, bidx, 360> \
    { \
    };

CV_CUDEV_RGB2HLS_INST(RGB_to_HLS, 3, 3, 2)
CV_CUDEV_RGB2HLS_INST(RGBA_to_HLS, 4, 3, 2)
CV_CUDEV_RGB2HLS_INST(RGB_to_HLS4, 3, 4, 2)
CV_CUDEV_RGB2HLS_INST(RGBA_to_HLS4, 4, 4, 2)
CV_CUDEV_RGB2HLS_INST(BGR_to_HLS, 3, 3, 0)
CV_CUDEV_RGB2HLS_INST(BGRA_to_HLS, 4, 3, 0)
CV_CUDEV_RGB2HLS_INST(BGR_to_HLS4, 3, 4, 0)
CV_CUDEV_RGB2HLS_INST(BGRA_to_HLS4, 4, 4, 0)

#undef CV_CUDEV_RGB2HLS_INST

// HLS to RGB

#define CV_CUDEV_HLS2RGB_INST(name, scn, dcn, bidx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::HLS2RGB<SrcDepth, scn, dcn, bidx, 180> \
    { \
    }; \
    template <typename SrcDepth> struct name ## _FULL ## _func : cv::cudev::color_cvt_detail::HLS2RGB<SrcDepth, scn, dcn, bidx, 255> \
    { \
    }; \
    template <> struct name ## _func<float> : cv::cudev::color_cvt_detail::HLS2RGB<float, scn, dcn, bidx, 360> \
    { \
    }; \
    template <> struct name ## _FULL ## _func<float> : cv::cudev::color_cvt_detail::HLS2RGB<float, scn, dcn, bidx, 360> \
    { \
    };

CV_CUDEV_HLS2RGB_INST(HLS_to_RGB, 3, 3, 2)
CV_CUDEV_HLS2RGB_INST(HLS_to_RGBA, 3, 4, 2)
CV_CUDEV_HLS2RGB_INST(HLS4_to_RGB, 4, 3, 2)
CV_CUDEV_HLS2RGB_INST(HLS4_to_RGBA, 4, 4, 2)
CV_CUDEV_HLS2RGB_INST(HLS_to_BGR, 3, 3, 0)
CV_CUDEV_HLS2RGB_INST(HLS_to_BGRA, 3, 4, 0)
CV_CUDEV_HLS2RGB_INST(HLS4_to_BGR, 4, 3, 0)
CV_CUDEV_HLS2RGB_INST(HLS4_to_BGRA, 4, 4, 0)

#undef CV_CUDEV_HLS2RGB_INST

// RGB to Lab

#define CV_CUDEV_RGB2Lab_INST(name, scn, dcn, sRGB, blueIdx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2Lab<SrcDepth, scn, dcn, sRGB, blueIdx> \
    { \
    };

CV_CUDEV_RGB2Lab_INST(RGB_to_Lab, 3, 3, true, 2)
CV_CUDEV_RGB2Lab_INST(RGBA_to_Lab, 4, 3, true, 2)
CV_CUDEV_RGB2Lab_INST(RGB_to_Lab4, 3, 4, true, 2)
CV_CUDEV_RGB2Lab_INST(RGBA_to_Lab4, 4, 4, true, 2)
CV_CUDEV_RGB2Lab_INST(BGR_to_Lab, 3, 3, true, 0)
CV_CUDEV_RGB2Lab_INST(BGRA_to_Lab, 4, 3, true, 0)
CV_CUDEV_RGB2Lab_INST(BGR_to_Lab4, 3, 4, true, 0)
CV_CUDEV_RGB2Lab_INST(BGRA_to_Lab4, 4, 4, true, 0)

CV_CUDEV_RGB2Lab_INST(LRGB_to_Lab, 3, 3, false, 2)
CV_CUDEV_RGB2Lab_INST(LRGBA_to_Lab, 4, 3, false, 2)
CV_CUDEV_RGB2Lab_INST(LRGB_to_Lab4, 3, 4, false, 2)
CV_CUDEV_RGB2Lab_INST(LRGBA_to_Lab4, 4, 4, false, 2)
CV_CUDEV_RGB2Lab_INST(LBGR_to_Lab, 3, 3, false, 0)
CV_CUDEV_RGB2Lab_INST(LBGRA_to_Lab, 4, 3, false, 0)
CV_CUDEV_RGB2Lab_INST(LBGR_to_Lab4, 3, 4, false, 0)
CV_CUDEV_RGB2Lab_INST(LBGRA_to_Lab4, 4, 4, false, 0)

#undef CV_CUDEV_RGB2Lab_INST

// Lab to RGB

#define CV_CUDEV_Lab2RGB_INST(name, scn, dcn, sRGB, blueIdx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::Lab2RGB<SrcDepth, scn, dcn, sRGB, blueIdx> \
    { \
    };

CV_CUDEV_Lab2RGB_INST(Lab_to_RGB, 3, 3, true, 2)
CV_CUDEV_Lab2RGB_INST(Lab4_to_RGB, 4, 3, true, 2)
CV_CUDEV_Lab2RGB_INST(Lab_to_RGBA, 3, 4, true, 2)
CV_CUDEV_Lab2RGB_INST(Lab4_to_RGBA, 4, 4, true, 2)
CV_CUDEV_Lab2RGB_INST(Lab_to_BGR, 3, 3, true, 0)
CV_CUDEV_Lab2RGB_INST(Lab4_to_BGR, 4, 3, true, 0)
CV_CUDEV_Lab2RGB_INST(Lab_to_BGRA, 3, 4, true, 0)
CV_CUDEV_Lab2RGB_INST(Lab4_to_BGRA, 4, 4, true, 0)

CV_CUDEV_Lab2RGB_INST(Lab_to_LRGB, 3, 3, false, 2)
CV_CUDEV_Lab2RGB_INST(Lab4_to_LRGB, 4, 3, false, 2)
CV_CUDEV_Lab2RGB_INST(Lab_to_LRGBA, 3, 4, false, 2)
CV_CUDEV_Lab2RGB_INST(Lab4_to_LRGBA, 4, 4, false, 2)
CV_CUDEV_Lab2RGB_INST(Lab_to_LBGR, 3, 3, false, 0)
CV_CUDEV_Lab2RGB_INST(Lab4_to_LBGR, 4, 3, false, 0)
CV_CUDEV_Lab2RGB_INST(Lab_to_LBGRA, 3, 4, false, 0)
CV_CUDEV_Lab2RGB_INST(Lab4_to_LBGRA, 4, 4, false, 0)

#undef CV_CUDEV_Lab2RGB_INST

// RGB to Luv

#define CV_CUDEV_RGB2Luv_INST(name, scn, dcn, sRGB, blueIdx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::RGB2Luv<SrcDepth, scn, dcn, sRGB, blueIdx> \
    { \
    };

CV_CUDEV_RGB2Luv_INST(RGB_to_Luv, 3, 3, true, 2)
CV_CUDEV_RGB2Luv_INST(RGBA_to_Luv, 4, 3, true, 2)
CV_CUDEV_RGB2Luv_INST(RGB_to_Luv4, 3, 4, true, 2)
CV_CUDEV_RGB2Luv_INST(RGBA_to_Luv4, 4, 4, true, 2)
CV_CUDEV_RGB2Luv_INST(BGR_to_Luv, 3, 3, true, 0)
CV_CUDEV_RGB2Luv_INST(BGRA_to_Luv, 4, 3, true, 0)
CV_CUDEV_RGB2Luv_INST(BGR_to_Luv4, 3, 4, true, 0)
CV_CUDEV_RGB2Luv_INST(BGRA_to_Luv4, 4, 4, true, 0)

CV_CUDEV_RGB2Luv_INST(LRGB_to_Luv, 3, 3, false, 2)
CV_CUDEV_RGB2Luv_INST(LRGBA_to_Luv, 4, 3, false, 2)
CV_CUDEV_RGB2Luv_INST(LRGB_to_Luv4, 3, 4, false, 2)
CV_CUDEV_RGB2Luv_INST(LRGBA_to_Luv4, 4, 4, false, 2)
CV_CUDEV_RGB2Luv_INST(LBGR_to_Luv, 3, 3, false, 0)
CV_CUDEV_RGB2Luv_INST(LBGRA_to_Luv, 4, 3, false, 0)
CV_CUDEV_RGB2Luv_INST(LBGR_to_Luv4, 3, 4, false, 0)
CV_CUDEV_RGB2Luv_INST(LBGRA_to_Luv4, 4, 4, false, 0)

#undef CV_CUDEV_RGB2Luv_INST

// Luv to RGB

#define CV_CUDEV_Luv2RGB_INST(name, scn, dcn, sRGB, blueIdx) \
    template <typename SrcDepth> struct name ## _func : cv::cudev::color_cvt_detail::Luv2RGB<SrcDepth, scn, dcn, sRGB, blueIdx> \
    { \
    };

CV_CUDEV_Luv2RGB_INST(Luv_to_RGB, 3, 3, true, 2)
CV_CUDEV_Luv2RGB_INST(Luv4_to_RGB, 4, 3, true, 2)
CV_CUDEV_Luv2RGB_INST(Luv_to_RGBA, 3, 4, true, 2)
CV_CUDEV_Luv2RGB_INST(Luv4_to_RGBA, 4, 4, true, 2)
CV_CUDEV_Luv2RGB_INST(Luv_to_BGR, 3, 3, true, 0)
CV_CUDEV_Luv2RGB_INST(Luv4_to_BGR, 4, 3, true, 0)
CV_CUDEV_Luv2RGB_INST(Luv_to_BGRA, 3, 4, true, 0)
CV_CUDEV_Luv2RGB_INST(Luv4_to_BGRA, 4, 4, true, 0)

CV_CUDEV_Luv2RGB_INST(Luv_to_LRGB, 3, 3, false, 2)
CV_CUDEV_Luv2RGB_INST(Luv4_to_LRGB, 4, 3, false, 2)
CV_CUDEV_Luv2RGB_INST(Luv_to_LRGBA, 3, 4, false, 2)
CV_CUDEV_Luv2RGB_INST(Luv4_to_LRGBA, 4, 4, false, 2)
CV_CUDEV_Luv2RGB_INST(Luv_to_LBGR, 3, 3, false, 0)
CV_CUDEV_Luv2RGB_INST(Luv4_to_LBGR, 4, 3, false, 0)
CV_CUDEV_Luv2RGB_INST(Luv_to_LBGRA, 3, 4, false, 0)
CV_CUDEV_Luv2RGB_INST(Luv4_to_LBGRA, 4, 4, false, 0)

#undef CV_CUDEV_Luv2RGB_INST

// 24/32-bit RGB to 16-bit (565 or 555) RGB

#define CV_CUDEV_RGB2RGB5x5_INST(name, scn, bidx, green_bits) \
    typedef cv::cudev::color_cvt_detail::RGB2RGB5x5<scn, bidx, green_bits> name ## _func;

CV_CUDEV_RGB2RGB5x5_INST(BGR_to_BGR555, 3, 0, 5)
CV_CUDEV_RGB2RGB5x5_INST(BGR_to_BGR565, 3, 0, 6)
CV_CUDEV_RGB2RGB5x5_INST(RGB_to_BGR555, 3, 2, 5)
CV_CUDEV_RGB2RGB5x5_INST(RGB_to_BGR565, 3, 2, 6)
CV_CUDEV_RGB2RGB5x5_INST(BGRA_to_BGR555, 4, 0, 5)
CV_CUDEV_RGB2RGB5x5_INST(BGRA_to_BGR565, 4, 0, 6)
CV_CUDEV_RGB2RGB5x5_INST(RGBA_to_BGR555, 4, 2, 5)
CV_CUDEV_RGB2RGB5x5_INST(RGBA_to_BGR565, 4, 2, 6)

#undef CV_CUDEV_RGB2RGB5x5_INST

// 16-bit (565 or 555) RGB to 24/32-bit RGB

#define CV_CUDEV_RGB5x52RGB_INST(name, dcn, bidx, green_bits) \
    typedef cv::cudev::color_cvt_detail::RGB5x52RGB<dcn, bidx, green_bits> name ## _func;

CV_CUDEV_RGB5x52RGB_INST(BGR555_to_RGB, 3, 2, 5)
CV_CUDEV_RGB5x52RGB_INST(BGR565_to_RGB, 3, 2, 6)
CV_CUDEV_RGB5x52RGB_INST(BGR555_to_BGR, 3, 0, 5)
CV_CUDEV_RGB5x52RGB_INST(BGR565_to_BGR, 3, 0, 6)
CV_CUDEV_RGB5x52RGB_INST(BGR555_to_RGBA, 4, 2, 5)
CV_CUDEV_RGB5x52RGB_INST(BGR565_to_RGBA, 4, 2, 6)
CV_CUDEV_RGB5x52RGB_INST(BGR555_to_BGRA, 4, 0, 5)
CV_CUDEV_RGB5x52RGB_INST(BGR565_to_BGRA, 4, 0, 6)

#undef CV_CUDEV_RGB5x52RGB_INST

// Grayscale to 16-bit (565 or 555) RGB

#define CV_CUDEV_GRAY2RGB5x5_INST(name, green_bits) \
    typedef cv::cudev::color_cvt_detail::Gray2RGB5x5<green_bits> name ## _func;

CV_CUDEV_GRAY2RGB5x5_INST(GRAY_to_BGR555, 5)
CV_CUDEV_GRAY2RGB5x5_INST(GRAY_to_BGR565, 6)

#undef CV_CUDEV_GRAY2RGB5x5_INST

// 16-bit (565 or 555) RGB to Grayscale

#define CV_CUDEV_RGB5x52GRAY_INST(name, green_bits) \
    typedef cv::cudev::color_cvt_detail::RGB5x52Gray<green_bits> name ## _func;

CV_CUDEV_RGB5x52GRAY_INST(BGR555_to_GRAY, 5)
CV_CUDEV_RGB5x52GRAY_INST(BGR565_to_GRAY, 6)

#undef CV_CUDEV_RGB5x52GRAY_INST

//! @}

}}

#endif
