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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////////////////////////////////////// matchTemplate //////////////////////////////////////////////////////////

CV_ENUM(MatchTemplType, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)

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

    void generateTestData()
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

    void Near()
    {
        bool isNormed =
        method == TM_CCORR_NORMED ||
        method == TM_SQDIFF_NORMED ||
        method == TM_CCOEFF_NORMED;

        if (isNormed)
            OCL_EXPECT_MATS_NEAR(result, 3e-2);
        else
            OCL_EXPECT_MATS_NEAR_RELATIVE_SPARSE(result, 1.5e-2);
    }
};

OCL_TEST_P(MatchTemplate, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::matchTemplate(image_roi, templ_roi, result_roi, method));
        OCL_ON(cv::matchTemplate(uimage_roi, utempl_roi, uresult_roi, method));

        Near();
    }
}

OCL_INSTANTIATE_TEST_CASE_P(ImageProc, MatchTemplate, Combine(
                                Values(CV_8U, CV_32F),
                                Values(1, 2, 3, 4),
                                MatchTemplType::all(),
                                Bool())
                           );

// Regression test for https://github.com/opencv/opencv/issues/21788: the OpenCL
// TM_CCOEFF_NORMED kernel computes each window's variance-like denominator as a difference of
// two comparable-magnitude sums pulled from CV_32F integral images -- classic
// catastrophic-cancellation territory. On a realistic-sized image the rounding error in that
// subtraction can dwarf a genuinely small-but-nonzero window variance, corrupting the ratio
// enough to spuriously hit the +-1 safety clamp for windows that are not actually degenerate.
// The parameterized MatchTemplate case above never reaches this: it only exercises small
// (<=100x100) images of uniformly random full-range noise, where windows have large variance
// and integral sums never accumulate far enough to lose the precision this needs.
TEST(MatchTemplate, ccoeff_normed_large_low_contrast_image)
{
    if (!cv::ocl::haveOpenCL())
        throw SkipTestException("OpenCL is not available");

    Mat image(1080, 1920, CV_8UC1);
    cv::theRNG().fill(image, RNG::UNIFORM, 178, 183);

    Mat templ = image(Rect(5, 5, 32, 32)).clone();

    bool useOCL = cv::ocl::useOpenCL();
    Mat cpuResult;
    cv::ocl::setUseOpenCL(false);
    cv::matchTemplate(image, templ, cpuResult, TM_CCOEFF_NORMED);

    UMat gpuResultU;
    cv::ocl::setUseOpenCL(true);
    cv::matchTemplate(image.getUMat(ACCESS_READ), templ.getUMat(ACCESS_READ), gpuResultU, TM_CCOEFF_NORMED);
    cv::ocl::setUseOpenCL(useOCL);
    Mat gpuResult = gpuResultU.getMat(ACCESS_READ);

    ASSERT_EQ(cpuResult.size(), gpuResult.size());

    double minCpu = 0, maxCpu = 0, minGpu = 0, maxGpu = 0;
    cv::minMaxLoc(cpuResult, &minCpu, &maxCpu);
    cv::minMaxLoc(gpuResult, &minGpu, &maxGpu);

    EXPECT_NEAR(minCpu, minGpu, 5e-2) << "CPU minVal=" << minCpu << " GPU minVal=" << minGpu;
    EXPECT_NEAR(maxCpu, maxGpu, 5e-2) << "CPU maxVal=" << maxCpu << " GPU maxVal=" << maxGpu;
    EXPECT_LE(cv::norm(cpuResult, gpuResult, NORM_INF), 5e-2);
}

} } // namespace opencv_test::ocl

#endif
