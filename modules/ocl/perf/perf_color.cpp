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
//     and/or other oclMaterials provided with the distribution.
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
#include "perf_precomp.hpp"

using namespace perf;

///////////// cvtColor////////////////////////

typedef TestBaseWithParam<Size> cvtColorFixture;

PERF_TEST_P(cvtColorFixture, DISABLED_cvtColor,
            ::testing::Values(Size(100, 100), Size(200, 200)))
{
    const Size srcSize = GetParam();
    const std::string impl = getSelectedImpl();

    Mat src(srcSize, CV_8UC4), dst;

    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(src.size(), CV_8UC4);

        ocl::cvtColor(oclSrc, oclDst, CV_RGBA2GRAY, 4);
        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        dst.create(srcSize, CV_8UC4);

        TEST_CYCLE() cv::cvtColor(src, dst, CV_RGBA2GRAY, 4);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}
