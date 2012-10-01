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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#include <test_precomp.hpp>

#ifdef HAVE_CUDA

using cv::gpu::GpuMat;

TEST(SoftCascade, readCascade)
{
    std::string xml = cvtest::TS::ptr()->get_data_path() + "../cv/cascadeandhog/icf-template.xml";
    cv::gpu::SoftCascade cascade;
    ASSERT_TRUE(cascade.load(xml));

}

TEST(SoftCascade, detect)
{
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "../cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml";
    cv::gpu::SoftCascade cascade;
    ASSERT_TRUE(cascade.load(xml));

    cv::Mat coloredCpu = cv::imread(cvtest::TS::ptr()->get_data_path()
        + "../cv/cascadeandhog/bahnhof/image_00000000_0.png");
    ASSERT_FALSE(coloredCpu.empty());
    GpuMat colored(coloredCpu), objectBoxes(1, 1000, CV_8UC1), rois;

    // ASSERT_NO_THROW(
    // {
        cascade.detectMultiScale(colored, rois, objectBoxes);
    // });
}

class SCSpecific : public ::testing::TestWithParam<std::tr1::tuple<std::string, int> > {
};

namespace {
std::string itoa(long i)
{
    static char s[65];
    sprintf(s, "%ld", i);
    return std::string(s);
}
}

TEST_P(SCSpecific, detect)
{
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "../cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml";
    cv::gpu::SoftCascade cascade;
    ASSERT_TRUE(cascade.load(xml));

    std::string path = GET_PARAM(0);
    cv::Mat coloredCpu = cv::imread(cvtest::TS::ptr()->get_data_path() + path);

    ASSERT_FALSE(coloredCpu.empty());
    GpuMat colored(coloredCpu), objectBoxes(1, 1000, CV_8UC1), rois;

    int level = GET_PARAM(1);
    cascade.detectMultiScale(colored, rois, objectBoxes, 1, level);

    cv::Mat dt(objectBoxes);
    typedef cv::gpu::SoftCascade::Detection detection_t;

    detection_t* dts = (detection_t*)dt.data;
    cv::Mat result(coloredCpu);


    std::cout << "Total detections " << (dt.cols / sizeof(detection_t)) << std::endl;
    for(int i = 0; i  < (int)(dt.cols / sizeof(detection_t)); ++i)
    {
        detection_t d = dts[i];
        std::cout << "detection: [" << std::setw(4) << d.x << " " << std::setw(4) << d.y
                  << "] [" << std::setw(4) << d.w << " " << std::setw(4) << d.h << "] "
                  << std::setw(12)  << d.confidence << std::endl;

        cv::rectangle(result, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(255, 0, 0, 255), 1);
    }

    std::cout << "Result stored in " << "/home/kellan/gpu_res_1_oct_" + itoa(level) << "_"
    + itoa((dt.cols / sizeof(detection_t))) + ".png" << std::endl;
    cv::imwrite("/home/kellan/gpu_res_1_oct_" + itoa(level) + "_" + itoa((dt.cols / sizeof(detection_t))) + ".png",
        result);
    cv::imshow("res", result);
    cv::waitKey(0);
}

INSTANTIATE_TEST_CASE_P(inLevel, SCSpecific,
    testing::Combine(
        testing::Values(std::string("../cv/cascadeandhog/bahnhof/image_00000000_0.png")),
        testing::Range(0, 47)
        ));

#endif