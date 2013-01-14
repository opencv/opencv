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

#include "test_precomp.hpp"

// show detection results on input image with cv::imshow
#define SHOW_DETECTIONS

#if defined SHOW_DETECTIONS
# define SHOW(res)           \
    cv::imshow(#res, res);\
    cv::waitKey(30);
#else
# define SHOW(res)
#endif

TEST(SCascade, readCascade)
{
    std::string xml = "/home/kellan/cs-caltech-13.01.2013/caltech-octave_-1.xml";//cvtest::TS::ptr()->get_data_path() + "softcascade/soft-cascade-17.12.2012.xml";
    cv::SCascade cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));
}

// TEST(SCascade, detect)
// {
//     typedef cv::SCascade::Detection Detection;
//     std::string xml =  cvtest::TS::ptr()->get_data_path() + "softcascade/soft-cascade-17.12.2012.xml";// "/home/kellan/cs-caltech-13.01.2013/caltech-octave_-1.xml";//cvtest::TS::ptr()->get_data_path() + "softcascade/soft-cascade-17.12.2012.xml";
//     cv::SCascade cascade;
//     cv::FileStorage fs(xml, cv::FileStorage::READ);
//     ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

//     cv::VideoCapture capture("/home/kellan/datasets/caltech/set00/V009.seq/I0%4d.jpg");
//     ASSERT_TRUE(capture.isOpened());

//     for (;;)
//     {

//         cv::Mat frame;
//         if (!capture.read(frame))
//         {
//             std::cout << "Nothing to read. " << std::endl << std::flush;
//             return;
//         }

//         cv::Mat colored;
//         frame.copyTo(colored);
//         ASSERT_FALSE(colored.empty());

//         std::vector<Detection> objects;
//         cascade.detect(colored, cv::noArray(), objects);

//         for (int i = 0; i  < (int)objects.size(); ++i)
//         {
//             Detection d = objects[i];
//             cv::rectangle(colored, cv::Rect(d.bb.x, d.bb.y, d.bb.width, d.bb.height), cv::Scalar(255, 0, 0, 255), 1);
//         }

//         SHOW(colored)

//         // ASSERT_EQ(823, (int)objects.size());
//     }
//     // cv::Mat colored = //cv::imread(cvtest::TS::ptr()->get_data_path() + "softcascade/bahnhof/image_00000000_0.png");
// }

TEST(SCascade, detectSeparate)
{
    typedef cv::SCascade::Detection Detection;
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "softcascade/soft-cascade-17.12.2012.xml";
    cv::SCascade cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "softcascade/bahnhof/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    cv::Mat rects, confs;

    cascade.detect(colored, cv::noArray(), rects, confs);
    ASSERT_EQ(823, confs.cols);
}

TEST(SCascade, detectRoi)
{
    typedef cv::SCascade::Detection Detection;
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "softcascade/soft-cascade-17.12.2012.xml";
    cv::SCascade cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "softcascade/bahnhof/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    std::vector<Detection> objects;
    std::vector<cv::Rect> rois;
    rois.push_back(cv::Rect(0, 0, 640, 480));

    cascade.detect(colored, rois, objects);
    ASSERT_EQ(823, (int)objects.size());
}

TEST(SCascade, detectNoRoi)
{
    typedef cv::SCascade::Detection Detection;
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "softcascade/soft-cascade-17.12.2012.xml";
    cv::SCascade cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "softcascade/bahnhof/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    std::vector<Detection> objects;
    std::vector<cv::Rect> rois;

    cascade.detect(colored, rois, objects);

    ASSERT_EQ(0, (int)objects.size());
}