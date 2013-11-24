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
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
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
//     and / or other materials provided with the distribution.
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

using namespace cv::softcascade;

TEST(SoftCascadeDetector, readCascade)
{
    std::string xml = cvtest::TS::ptr()->get_data_path() + "cascadeandhog/cascades/inria_caltech-17.01.2013.xml";
     Detector cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));
}

TEST(SoftCascadeDetector, detect)
{
    std::string xml =  cvtest::TS::ptr()->get_data_path()+ "cascadeandhog/cascades/inria_caltech-17.01.2013.xml";
    Detector cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path()  + "cascadeandhog/images/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    std::vector<Detection> objects;
    cascade.detect(colored, cv::noArray(), objects);

    ASSERT_EQ(719, (int)objects.size());
}

TEST(SoftCascadeDetector, detectSeparate)
{
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/cascades/inria_caltech-17.01.2013.xml";
    Detector cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/images/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    cv::Mat rects, confs;

    cascade.detect(colored, cv::noArray(), rects, confs);
    ASSERT_EQ(719, confs.cols);
}

TEST(SoftCascadeDetector, detectRoi)
{
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/cascades/inria_caltech-17.01.2013.xml";
    Detector cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/images/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    std::vector<Detection> objects;
    std::vector<cv::Rect> rois;
    rois.push_back(cv::Rect(0, 0, 640, 480));

    cascade.detect(colored, rois, objects);
    ASSERT_EQ(719, (int)objects.size());
}

TEST(SoftCascadeDetector, detectNoRoi)
{
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/cascades/inria_caltech-17.01.2013.xml";
    Detector cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/images/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    std::vector<Detection> objects;
    std::vector<cv::Rect> rois;

    cascade.detect(colored, rois, objects);

    ASSERT_EQ(719, (int)objects.size());
}

TEST(SoftCascadeDetector, detectEmptyRoi)
{
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/cascades/inria_caltech-17.01.2013.xml";
    Detector cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/images/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    std::vector<Detection> objects;
    cascade.detect(colored, cv::Mat::zeros(colored.size(), CV_8UC1), objects);

    ASSERT_EQ(0, (int)objects.size());
}
