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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Zero Lin, Zero.Lin@amd.com
//    Zhang Ying, zhangying913@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
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
#include "cvconfig.h"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(FilterTestBase, MatType,
                int, // kernel size
                Size, // dx, dy
                int, // border type
                double, // optional parameter
                bool) // roi or not
{
    int type, borderType, ksize;
    Size size;
    double param;
    bool useRoi;

    TEST_DECLARE_INPUT_PARAMETER(src)
    TEST_DECLARE_OUTPUT_PARAMETER(dst)

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        size = GET_PARAM(2);
        borderType = GET_PARAM(3);
        param = GET_PARAM(4);
        useRoi = GET_PARAM(5);
    }

    void random_roi(int minSize = 1)
    {
        if (minSize == 0)
            minSize = ksize;

        Size roiSize = randomSize(minSize, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -60, 70);

        UMAT_UPLOAD_INPUT_PARAMETER(src)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst)
    }

    void Near()
    {
        int depth = CV_MAT_DEPTH(type);
        bool isFP = depth >= CV_32F;

        if (isFP)
            Near(1e-6, true);
        else
            Near(1, false);
    }

    void Near(double threshold, bool relative)
    {
        if (relative)
        {
            EXPECT_MAT_NEAR_RELATIVE(dst, udst, threshold);
            EXPECT_MAT_NEAR_RELATIVE(dst_roi, udst_roi, threshold);
        }
        else
        {
            EXPECT_MAT_NEAR(dst, udst, threshold);
            EXPECT_MAT_NEAR(dst_roi, udst_roi, threshold);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral

typedef FilterTestBase Bilateral;

OCL_TEST_P(Bilateral, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        double sigmacolor = rng.uniform(20, 100);
        double sigmaspace = rng.uniform(10, 40);

        OCL_OFF(cv::bilateralFilter(src_roi, dst_roi, ksize, sigmacolor, sigmaspace, borderType));
        OCL_ON(cv::bilateralFilter(usrc_roi, udst_roi, ksize, sigmacolor, sigmaspace, borderType));

        Near();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define FILTER_BORDER_SET_NO_ISOLATED \
    Values((int)BORDER_CONSTANT, (int)BORDER_REPLICATE, (int)BORDER_REFLECT, (int)BORDER_WRAP, (int)BORDER_REFLECT_101/*, \
            (int)BORDER_CONSTANT|BORDER_ISOLATED, (int)BORDER_REPLICATE|BORDER_ISOLATED, \
            (int)BORDER_REFLECT|BORDER_ISOLATED, (int)BORDER_WRAP|BORDER_ISOLATED, \
            (int)BORDER_REFLECT_101|BORDER_ISOLATED*/) // WRAP and ISOLATED are not supported by cv:: version

#define FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED \
    Values((int)BORDER_CONSTANT, (int)BORDER_REPLICATE, (int)BORDER_REFLECT, /*(int)BORDER_WRAP,*/ (int)BORDER_REFLECT_101/*, \
            (int)BORDER_CONSTANT|BORDER_ISOLATED, (int)BORDER_REPLICATE|BORDER_ISOLATED, \
            (int)BORDER_REFLECT|BORDER_ISOLATED, (int)BORDER_WRAP|BORDER_ISOLATED, \
            (int)BORDER_REFLECT_101|BORDER_ISOLATED*/) // WRAP and ISOLATED are not supported by cv:: version

#define FILTER_DATATYPES Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, \
                                CV_32FC1, CV_32FC3, CV_32FC4, \
                                CV_64FC1, CV_64FC3, CV_64FC4)

OCL_INSTANTIATE_TEST_CASE_P(Filter, Bilateral, Combine(
                            Values((MatType)CV_8UC1),
                            Values(5, 9),
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool()));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
