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
#include "perf_precomp.hpp"

using namespace perf;

///////////// Canny ////////////////////////

PERF_TEST(CannyFixture, Canny)
{
    Mat img = imread(getDataPath("gpu/stereobm/aloe-L.png"), cv::IMREAD_GRAYSCALE),
            edges(img.size(), CV_8UC1);
    ASSERT_TRUE(!img.empty()) << "can't open aloe-L.png";

    declare.in(img).out(edges);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclImg(img), oclEdges(img.size(), CV_8UC1);

        OCL_TEST_CYCLE() ocl::Canny(oclImg, oclEdges, 50.0, 100.0);
        oclEdges.download(edges);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() Canny(img, edges, 50.0, 100.0);
    }
    else
        OCL_PERF_ELSE

    int value = 0;
    SANITY_CHECK(value);
}
