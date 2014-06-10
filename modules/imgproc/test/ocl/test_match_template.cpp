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

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"
#include "iostream"
#include "fstream"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////////////////////////////////////// matchTemplate //////////////////////////////////////////////////////////

CV_ENUM(MatchTemplType, CV_TM_CCORR, CV_TM_CCORR_NORMED, CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED, CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED)

PARAM_TEST_CASE(MatchTemplate, MatDepth, Channels, MatchTemplType, bool)
{
    int type;
    int depth;
    int method;
    bool use_roi;

    TEST_DECLARE_INPUT_PARAMETER(image);
    TEST_DECLARE_INPUT_PARAMETER(templ);
    TEST_DECLARE_OUTPUT_PARAMETER(result);

    virtual void SetUp()
    {
        type = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
        depth = GET_PARAM(0);
        method = GET_PARAM(2);
        use_roi = GET_PARAM(3);
    }

    virtual void generateTestData()
    {
        Size image_roiSize = randomSize(2, 100);
        Size templ_roiSize = Size(randomInt(1, image_roiSize.width), randomInt(1, image_roiSize.height));
        Size result_roiSize = Size(image_roiSize.width - templ_roiSize.width + 1,
                                   image_roiSize.height - templ_roiSize.height + 1);

        const double upValue = 256;

        Border imageBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(image, image_roi, image_roiSize, imageBorder, type, -upValue, upValue);

        Border templBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(templ, templ_roi, templ_roiSize, templBorder, type, -upValue, upValue);

        Border resultBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(result, result_roi, result_roiSize, resultBorder, CV_32FC1, -upValue, upValue);

        UMAT_UPLOAD_INPUT_PARAMETER(image);
        UMAT_UPLOAD_INPUT_PARAMETER(templ);
        UMAT_UPLOAD_OUTPUT_PARAMETER(result);
    }

    void Near(double threshold = 0.0)
    {
        OCL_EXPECT_MATS_NEAR_RELATIVE(result, threshold);
    }
};

OCL_TEST_P(MatchTemplate, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::matchTemplate(image_roi, templ_roi, result_roi, method));
        OCL_ON(cv::matchTemplate(uimage_roi, utempl_roi, uresult_roi, method));

        Near(1.5e-4);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(ImageProc, MatchTemplate, Combine(
                                Values(CV_8U, CV_32F),
                                Values(1, 2, 3, 4),
                                MatchTemplType::all(),
                                Bool())
                           );
} } // namespace cvtest::ocl

#endif
