/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                          License Agreement
//               For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * The name of the copyright holders may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
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
//M*/

#include "test_precomp.hpp"

TEST(SoftCascade, readCascade)
{
    std::string xml = cvtest::TS::ptr()->get_data_path() + "cascadeandhog/icf-template.xml";
    cv::SoftCascade cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.read(fs));

}

TEST(SoftCascade, detect)
{
    typedef cv::SoftCascade::Detection detection_t;
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "cascadeandhog/sc_cvpr_2012_to_opencv.xml";
    cv::SoftCascade cascade;
    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(cascade.read(fs));

    cv::Mat colored = cv::imread(cvtest::TS::ptr()->get_data_path() + "cascadeandhog/bahnhof/image_00000000_0.png");
    ASSERT_FALSE(colored.empty());

    std::vector<detection_t> objects;
    std::vector<cv::Rect> rois;
    rois.push_back(cv::Rect(0, 0, 640, 480));

    cascade.detectMultiScale(colored, rois, objects);


    cv::Mat out = colored.clone();
    int level = 0, total = 0;
    int levelWidth = objects[0].rect.width;

    for(int i = 0 ; i < (int)objects.size(); ++i)
    {
        if (objects[i].rect.width != levelWidth)
        {
            std::cout << "Level: " << level << " total " << total << std::endl;
            cv::imshow("out", out);
            cv::waitKey(0);
            out = colored.clone();
            levelWidth = objects[i].rect.width;
            total = 0;
            level++;
        }
        cv::rectangle(out, objects[i].rect, cv::Scalar(255, 0, 0, 255), 1);
        std::cout << "detection: " << objects[i].rect.x
                                   << " " << objects[i].rect.y
                                   << " " << objects[i].rect.width
                                   << " " << objects[i].rect.height << std::endl;
        total++;
    }
    std::cout << "detected: " << (int)objects.size() << std::endl;
    ASSERT_EQ((int)objects.size(), 1469);
}