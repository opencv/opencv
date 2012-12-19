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

#include <string>
#include <fstream>

static std::string itoa(int i)
{
    static char s[65];
    sprintf(s, "%03d", i);
    return std::string(s);
}

#include "test_precomp.hpp"
#include <opencv2/highgui/highgui.hpp>

TEST(SCascade, detect1)
{
    typedef cv::SCascade::Detection Detection;
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/sc_cvpr_2012_to_opencv.xml";
    cv::SCascade cascade;
    cascade.set("rejCriteria", cv::SCascade::DOLLAR);
    // cascade.set("minScale", 0.5);
    // cascade.set("scales", 2);
    cv::FileStorage fs("/home/kellan/soft-cascade-17.12.2012/first-soft-cascade-composide-octave_1.xml", cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    for (int sample = 0; sample < 1000; ++sample)
    {

        // std::cout << itoa(sample) << std::endl;
    std::cout << std::string("/home/kellan/bahnhof-l/image_00000" + itoa(sample) + "_0.png") << std::endl;
    cv::Mat colored = cv::imread(std::string("/home/kellan/bahnhof-l/image_00000" + itoa(sample) + "_0.png"));
    ASSERT_FALSE(colored.empty());


    std::vector<Detection> objects;
    cascade.detect(colored, cv::noArray(), objects);

    for (int i = 0; i < (int)objects.size(); ++i)
        cv::rectangle(colored, objects[i].bb, cv::Scalar(51, 160, 255, 255), 1);

    // cv::Mat res;
    // cv::resize(colored, res, cv::Size(), 4,4);
    cv::imshow("detections", colored);
    cv::waitKey(20);
    // cv::imwrite(std::string("/home/kellan/res/image_00000" + itoa(sample) + ".png"), colored);
    }

    // ASSERT_EQ(1459, (int)objects.size());
}

TEST(SCascade, readCascade)
{
    std::string xml = cvtest::TS::ptr()->get_data_path() + "cascadeandhog/test-simple-cascade.xml";
    cv::SCascade cascade;
    cv::FileStorage fs("/home/kellan/soft-cascade-17.12.2012/first-soft-cascade-composide-octave_1.xml", cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));
}

// TEST(SCascade, detect)
// {
//     typedef cv::SCascade::Detection Detection;
//     std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/sc_cvpr_2012_to_opencv.xml";
//     cv::SCascade cascade;
//     // cascade.set("maxScale", 0.5);
//     // cascade.set("minScale", 0.5);
//     // cascade.set("scales", 2);

//     cv::FileStorage fs("/home/kellan/soft-cascade-17.12.2012/first-soft-cascade-composide-octave_1.xml", cv::FileStorage::READ);
//     ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

//     // 454
//     cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/bahnhof/image_00000000_0.png");//"/home/kellan/datasets/INRIA/training_set/pos/octave_-1/sample_1.png");//cvtest::TS::ptr()->get_data_path() + "cascadeandhog/bahnhof/image_00000000_0.png");
//     ASSERT_FALSE(colored.empty());

//     std::vector<Detection> objects;

//     cascade.detect(colored, cv::noArray(), objects);

//     for (int i = 0; i < objects.size(); ++i)
//         cv::rectangle(colored, objects[i].bb, cv::Scalar::all(255), 1);

//     cv::Mat res;
//     cv::resize(colored, res, cv::Size(), 4,4);
//     cv::imshow("detections", colored);
//     cv::waitKey(0);

//     // ASSERT_EQ(1459, (int)objects.size());
// }

// TEST(SCascade, detectSeparate)
// {
//     typedef cv::SCascade::Detection Detection;
//     std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/sc_cvpr_2012_to_opencv.xml";
//     cv::SCascade cascade;
//     cv::FileStorage fs(xml, cv::FileStorage::READ);
//     ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

//     cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/bahnhof/image_00000000_0.png");
//     ASSERT_FALSE(colored.empty());

//     cv::Mat rects, confs;

//     cascade.detect(colored, cv::noArray(), rects, confs);
//     ASSERT_EQ(1459, confs.cols);
// }

// TEST(SCascade, detectRoi)
// {
//     typedef cv::SCascade::Detection Detection;
//     std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/sc_cvpr_2012_to_opencv.xml";
//     cv::SCascade cascade;
//     cv::FileStorage fs(xml, cv::FileStorage::READ);
//     ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

//     cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/bahnhof/image_00000000_0.png");
//     ASSERT_FALSE(colored.empty());

//     std::vector<Detection> objects;
//     std::vector<cv::Rect> rois;
//     rois.push_back(cv::Rect(0, 0, 640, 480));

//     cascade.detect(colored, rois, objects);
//     ASSERT_EQ(1459, (int)objects.size());
// }

// TEST(SCascade, detectNoRoi)
// {
//     typedef cv::SCascade::Detection Detection;
//     std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/sc_cvpr_2012_to_opencv.xml";
//     cv::SCascade cascade;
//     cv::FileStorage fs(xml, cv::FileStorage::READ);
//     ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

//     cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/bahnhof/image_00000000_0.png");
//     ASSERT_FALSE(colored.empty());

//     std::vector<Detection> objects;
//     std::vector<cv::Rect> rois;

//     cascade.detect(colored, rois, objects);

//     ASSERT_EQ(0, (int)objects.size());
// }