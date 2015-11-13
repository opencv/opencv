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

#include "perf_precomp.hpp"
#include "opencv2/ts/gpu_perf.hpp"

using namespace std;
using namespace testing;
using namespace perf;

///////////////////////////////////////////////////////////////
// HOG

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, ObjDetect_HOG,
            Values<string>("gpu/hog/road.png",
                           "gpu/caltech/image_00000009_0.png",
                           "gpu/caltech/image_00000032_0.png",
                           "gpu/caltech/image_00000165_0.png",
                           "gpu/caltech/image_00000261_0.png",
                           "gpu/caltech/image_00000469_0.png",
                           "gpu/caltech/image_00000527_0.png",
                           "gpu/caltech/image_00000574_0.png"))
{
    declare.time(300.0);

    const cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_img(img);
        std::vector<cv::Rect> gpu_found_locations;

        cv::gpu::HOGDescriptor d_hog;
        d_hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        TEST_CYCLE() d_hog.detectMultiScale(d_img, gpu_found_locations);

        SANITY_CHECK(gpu_found_locations);
    }
    else
    {
        std::vector<cv::Rect> cpu_found_locations;

        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        TEST_CYCLE() hog.detectMultiScale(img, cpu_found_locations);

        SANITY_CHECK(cpu_found_locations);
    }
}

///////////////////////////////////////////////////////////////
// HaarClassifier

typedef pair<string, string> pair_string;
DEF_PARAM_TEST_1(ImageAndCascade, pair_string);

PERF_TEST_P(ImageAndCascade, ObjDetect_HaarClassifier,
            Values<pair_string>(make_pair("gpu/haarcascade/group_1_640x480_VGA.pgm", "gpu/perf/haarcascade_frontalface_alt.xml")))
{
    const cv::Mat img = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (PERF_RUN_GPU())
    {
        cv::gpu::CascadeClassifier_GPU d_cascade;
        ASSERT_TRUE(d_cascade.load(perf::TestBase::getDataPath(GetParam().second)));

        const cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat objects_buffer;
        int detections_num = 0;

        TEST_CYCLE() detections_num = d_cascade.detectMultiScale(d_img, objects_buffer);

        std::vector<cv::Rect> gpu_rects(detections_num);
        cv::Mat gpu_rects_mat(1, detections_num, cv::DataType<cv::Rect>::type, &gpu_rects[0]);
        objects_buffer.colRange(0, detections_num).download(gpu_rects_mat);
        cv::groupRectangles(gpu_rects, 3, 0.2);
        SANITY_CHECK(gpu_rects);
    }
    else
    {
        cv::CascadeClassifier cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath("gpu/perf/haarcascade_frontalface_alt.xml")));

        std::vector<cv::Rect> cpu_rects;

        TEST_CYCLE() cascade.detectMultiScale(img, cpu_rects);

        SANITY_CHECK(cpu_rects);
    }
}

///////////////////////////////////////////////////////////////
// LBP cascade

PERF_TEST_P(ImageAndCascade, ObjDetect_LBPClassifier,
            Values<pair_string>(make_pair("gpu/haarcascade/group_1_640x480_VGA.pgm", "gpu/lbpcascade/lbpcascade_frontalface.xml")))
{
    const cv::Mat img = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (PERF_RUN_GPU())
    {
        cv::gpu::CascadeClassifier_GPU d_cascade;
        ASSERT_TRUE(d_cascade.load(perf::TestBase::getDataPath(GetParam().second)));

        const cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat objects_buffer;
        int detections_num = 0;

        TEST_CYCLE() detections_num = d_cascade.detectMultiScale(d_img, objects_buffer);

        std::vector<cv::Rect> gpu_rects(detections_num);
        cv::Mat gpu_rects_mat(1, detections_num, cv::DataType<cv::Rect>::type, &gpu_rects[0]);
        objects_buffer.colRange(0, detections_num).download(gpu_rects_mat);
        cv::groupRectangles(gpu_rects, 3, 0.2);
        SANITY_CHECK(gpu_rects);
    }
    else
    {
        cv::CascadeClassifier cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath("gpu/lbpcascade/lbpcascade_frontalface.xml")));

        std::vector<cv::Rect> cpu_rects;

        TEST_CYCLE() cascade.detectMultiScale(img, cpu_rects);

        SANITY_CHECK(cpu_rects);
    }
}
