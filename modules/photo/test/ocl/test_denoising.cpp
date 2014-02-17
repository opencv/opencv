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

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(FastNlMeansDenoisingTestBase, Channels, bool)
{
    int cn, templateWindowSize, searchWindowSize;
    float h;
    bool use_roi;

    TEST_DECLARE_INPUT_PARAMETER(src)
    TEST_DECLARE_OUTPUT_PARAMETER(dst)

    virtual void SetUp()
    {
        cn = GET_PARAM(0);
        use_roi = GET_PARAM(1);

        templateWindowSize = 7;
        searchWindowSize = 21;
        h = 3.0f;
    }

    virtual void generateTestData()
    {
        const int type = CV_8UC(cn);

        Size roiSize = randomSize(10, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 0, 255);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, 0, 255);

        UMAT_UPLOAD_INPUT_PARAMETER(src)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst)
    }
};

typedef FastNlMeansDenoisingTestBase FastNlMeansDenoising;

OCL_TEST_P(FastNlMeansDenoising, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::fastNlMeansDenoising(src_roi, dst_roi, h, templateWindowSize, searchWindowSize));
        OCL_ON(cv::fastNlMeansDenoising(usrc_roi, udst_roi, h, templateWindowSize, searchWindowSize));

//        Mat difference;
//        cv::subtract(dst_roi, udst_roi, difference);
//        print(difference);

        OCL_EXPECT_MATS_NEAR(dst, 1)
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Photo, FastNlMeansDenoising, Combine(Values((Channels)1), Bool()));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
