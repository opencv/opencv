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
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////// cvtColor////////////////////////

CV_ENUM(ConversionTypes, COLOR_RGB2GRAY, COLOR_RGB2BGR, COLOR_RGB2YUV, COLOR_YUV2RGB, COLOR_RGB2YCrCb,
        COLOR_YCrCb2RGB, COLOR_RGB2XYZ, COLOR_XYZ2RGB, COLOR_RGB2HSV, COLOR_HSV2RGB, COLOR_RGB2HLS,
        COLOR_HLS2RGB, COLOR_BGR5652BGR, COLOR_BGR2BGR565, COLOR_RGBA2mRGBA, COLOR_mRGBA2RGBA,
        COLOR_RGB2Lab, COLOR_Lab2BGR, COLOR_RGB2Luv, COLOR_Luv2LBGR, COLOR_YUV2RGB_NV12, COLOR_YUV2RGB_IYUV,
        COLOR_YUV2GRAY_420, COLOR_RGB2YUV_IYUV, COLOR_YUV2RGB_YUY2, COLOR_YUV2GRAY_YUY2)

typedef tuple<Size, tuple<ConversionTypes, int, int> > CvtColorParams;
typedef TestBaseWithParam<CvtColorParams> CvtColorFixture;

OCL_PERF_TEST_P(CvtColorFixture, CvtColor, testing::Combine(
                OCL_TEST_SIZES,
                testing::Values(
                    make_tuple(ConversionTypes(COLOR_RGB2GRAY), 3, 1),
                    make_tuple(ConversionTypes(COLOR_RGB2BGR), 3, 3),
                    make_tuple(ConversionTypes(COLOR_RGB2YUV), 3, 3),
                    make_tuple(ConversionTypes(COLOR_YUV2RGB), 3, 3),
                    make_tuple(ConversionTypes(COLOR_RGB2YCrCb), 3, 3),
                    make_tuple(ConversionTypes(COLOR_YCrCb2RGB), 3, 3),
                    make_tuple(ConversionTypes(COLOR_RGB2XYZ), 3, 3),
                    make_tuple(ConversionTypes(COLOR_XYZ2RGB), 3, 3),
                    make_tuple(ConversionTypes(COLOR_RGB2HSV), 3, 3),
                    make_tuple(ConversionTypes(COLOR_HSV2RGB), 3, 3),
                    make_tuple(ConversionTypes(COLOR_RGB2HLS), 3, 3),
                    make_tuple(ConversionTypes(COLOR_HLS2RGB), 3, 3),
                    make_tuple(ConversionTypes(COLOR_BGR5652BGR), 2, 3),
                    make_tuple(ConversionTypes(COLOR_BGR2BGR565), 3, 2),
                    make_tuple(ConversionTypes(COLOR_RGBA2mRGBA), 4, 4),
                    make_tuple(ConversionTypes(COLOR_mRGBA2RGBA), 4, 4),
                    make_tuple(ConversionTypes(COLOR_RGB2Lab), 3, 3),
                    make_tuple(ConversionTypes(COLOR_Lab2BGR), 3, 4),
                    make_tuple(ConversionTypes(COLOR_RGB2Luv), 3, 3),
                    make_tuple(ConversionTypes(COLOR_Luv2LBGR), 3, 4),
                    make_tuple(ConversionTypes(COLOR_YUV2RGB_NV12), 1, 3),
                    make_tuple(ConversionTypes(COLOR_YUV2RGB_IYUV), 1, 3),
                    make_tuple(ConversionTypes(COLOR_YUV2GRAY_420), 1, 1),
                    make_tuple(ConversionTypes(COLOR_RGB2YUV_IYUV), 3, 1),
                    make_tuple(ConversionTypes(COLOR_YUV2RGB_YUY2), 2, 3),
                    make_tuple(ConversionTypes(COLOR_YUV2GRAY_YUY2), 2, 1)
                    )))
{
    CvtColorParams params = GetParam();
    const Size srcSize = get<0>(params);
    const tuple<int, int, int> conversionParams = get<1>(params);
    const int code = get<0>(conversionParams), scn = get<1>(conversionParams),
            dcn = get<2>(conversionParams);

    UMat src(srcSize, CV_8UC(scn)), dst(srcSize, CV_8UC(scn));
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::cvtColor(src, dst, code, dcn);

    SANITY_CHECK(dst, 1);
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
