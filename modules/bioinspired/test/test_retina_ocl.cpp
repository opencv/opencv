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
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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

#include "test_precomp.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/bioinspired.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#if defined(HAVE_OPENCV_OCL)

#include "opencv2/ocl.hpp"
#define RETINA_ITERATIONS 5

static double checkNear(const cv::Mat &m1, const cv::Mat &m2)
{
    return cv::norm(m1, m2, cv::NORM_INF);
}

#define PARAM_TEST_CASE(name, ...) struct name : testing::TestWithParam< std::tr1::tuple< __VA_ARGS__ > >
#define GET_PARAM(k) std::tr1::get< k >(GetParam())

static int oclInit = false;

PARAM_TEST_CASE(Retina_OCL, bool, int, bool, double, double)
{
    bool colorMode;
    int colorSamplingMethod;
    bool useLogSampling;
    double reductionFactor;
    double samplingStrength;

    cv::ocl::DevicesInfo infos;

    virtual void SetUp()
    {
        colorMode           = GET_PARAM(0);
        colorSamplingMethod = GET_PARAM(1);
        useLogSampling      = GET_PARAM(2);
        reductionFactor     = GET_PARAM(3);
        samplingStrength    = GET_PARAM(4);

        if(!oclInit)
        {
            cv::ocl::getOpenCLDevices(infos);
            std::cout << "Device name:" << infos[0]->deviceName << std::endl;
            oclInit = true;
        }
    }
};

TEST_P(Retina_OCL, Accuracy)
{
    using namespace cv;
    Mat input = imread(cvtest::TS::ptr()->get_data_path() + "shared/lena.png", colorMode);
    CV_Assert(!input.empty());
    ocl::oclMat ocl_input(input);

    Ptr<bioinspired::Retina> ocl_retina = bioinspired::createRetina_OCL(
        input.size(),
        colorMode,
        colorSamplingMethod,
        useLogSampling,
        reductionFactor,
        samplingStrength);

    Ptr<bioinspired::Retina> gold_retina = bioinspired::createRetina(
        input.size(),
        colorMode,
        colorSamplingMethod,
        useLogSampling,
        reductionFactor,
        samplingStrength);

    Mat gold_parvo;
    Mat gold_magno;
    ocl::oclMat ocl_parvo;
    ocl::oclMat ocl_magno;

    for(int i = 0; i < RETINA_ITERATIONS; i ++)
    {
        ocl_retina->run(ocl_input);
        gold_retina->run(input);

        gold_retina->getParvo(gold_parvo);
        gold_retina->getMagno(gold_magno);

        ocl_retina->getParvo(ocl_parvo);
        ocl_retina->getMagno(ocl_magno);

        int eps = colorMode ? 2 : 1;

        EXPECT_LE(checkNear(gold_parvo, (Mat)ocl_parvo), eps);
        EXPECT_LE(checkNear(gold_magno, (Mat)ocl_magno), eps);
    }
}

INSTANTIATE_TEST_CASE_P(Contrib, Retina_OCL, testing::Combine(
                            testing::Bool(),
                            testing::Values((int)cv::bioinspired::RETINA_COLOR_BAYER),
                            testing::Values(false/*,true*/),
                            testing::Values(1.0, 0.5),
                            testing::Values(10.0, 5.0)));
#endif
