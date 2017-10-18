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
//                        (3-clause BSD License)
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <algorithm>

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Test_Darknet, read_tiny_yolo_voc)
{
    Net net = readNetFromDarknet(_tf("tiny-yolo-voc.cfg"));
    ASSERT_FALSE(net.empty());
}

TEST(Test_Darknet, read_yolo_voc)
{
    Net net = readNetFromDarknet(_tf("yolo-voc.cfg"));
    ASSERT_FALSE(net.empty());
}

TEST(Reproducibility_TinyYoloVoc, Accuracy)
{
    Net net;
    {
        const string cfg = findDataFile("dnn/tiny-yolo-voc.cfg", false);
        const string model = findDataFile("dnn/tiny-yolo-voc.weights", false);
        net = readNetFromDarknet(cfg, model);
        ASSERT_FALSE(net.empty());
    }

    // dog416.png is dog.jpg that resized to 416x416 in the lossless PNG format
    Mat sample = imread(_tf("dog416.png"));
    ASSERT_TRUE(!sample.empty());

    Size inputSize(416, 416);

    if (sample.size() != inputSize)
        resize(sample, sample, inputSize);

    net.setInput(blobFromImage(sample, 1 / 255.F), "data");
    Mat out = net.forward("detection_out");

    Mat detection;
    const float confidenceThreshold = 0.24;

    for (int i = 0; i < out.rows; i++) {
        const int probability_index = 5;
        const int probability_size = out.cols - probability_index;
        float *prob_array_ptr = &out.at<float>(i, probability_index);
        size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = out.at<float>(i, (int)objectClass + probability_index);

        if (confidence > confidenceThreshold)
            detection.push_back(out.row(i));
    }

    // obtained by: ./darknet detector test ./cfg/voc.data  ./cfg/tiny-yolo-voc.cfg ./tiny-yolo-voc.weights -thresh 0.24 ./dog416.png
    // There are 2 objects (6-car, 11-dog) with 25 values for each:
    // { relative_center_x, relative_center_y, relative_width, relative_height, unused_t0, probability_for_each_class[20] }
    float ref_array[] = {
        0.736762F, 0.239551F, 0.315440F, 0.160779F, 0.761977F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,
        0.000000F, 0.000000F, 0.761967F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,
        0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,

        0.287486F, 0.653731F, 0.315579F, 0.534527F, 0.782737F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,
        0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.780595F,
        0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F
    };

    const int number_of_objects = 2;
    Mat ref(number_of_objects, sizeof(ref_array) / (number_of_objects * sizeof(float)), CV_32FC1, &ref_array);

    normAssert(ref, detection);
}

TEST(Reproducibility_YoloVoc, Accuracy)
{
    Net net;
    {
        const string cfg = findDataFile("dnn/yolo-voc.cfg", false);
        const string model = findDataFile("dnn/yolo-voc.weights", false);
        net = readNetFromDarknet(cfg, model);
        ASSERT_FALSE(net.empty());
    }

    // dog416.png is dog.jpg that resized to 416x416 in the lossless PNG format
    Mat sample = imread(_tf("dog416.png"));
    ASSERT_TRUE(!sample.empty());

    Size inputSize(416, 416);

    if (sample.size() != inputSize)
        resize(sample, sample, inputSize);

    net.setInput(blobFromImage(sample, 1 / 255.F), "data");
    Mat out = net.forward("detection_out");

    Mat detection;
    const float confidenceThreshold = 0.24;

    for (int i = 0; i < out.rows; i++) {
        const int probability_index = 5;
        const int probability_size = out.cols - probability_index;
        float *prob_array_ptr = &out.at<float>(i, probability_index);
        size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = out.at<float>(i, (int)objectClass + probability_index);

        if (confidence > confidenceThreshold)
            detection.push_back(out.row(i));
    }

    // obtained by: ./darknet detector test ./cfg/voc.data  ./cfg/yolo-voc.cfg ./yolo-voc.weights -thresh 0.24 ./dog416.png
    // There are 3 objects (6-car, 1-bicycle, 11-dog) with 25 values for each:
    // { relative_center_x, relative_center_y, relative_width, relative_height, unused_t0, probability_for_each_class[20] }
    float ref_array[] = {
        0.740161F, 0.214100F, 0.325575F, 0.173418F, 0.750769F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,
        0.000000F, 0.000000F, 0.750469F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,
        0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,

        0.501618F, 0.504757F, 0.461713F, 0.481310F, 0.783550F, 0.000000F, 0.780879F, 0.000000F, 0.000000F,
        0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,
        0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,

        0.279968F, 0.638651F, 0.282737F, 0.600284F, 0.901864F, 0.000000F, 0.000000F, 0.000000F, 0.000000F,
        0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.901615F,
        0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F
    };

    const int number_of_objects = 3;
    Mat ref(number_of_objects, sizeof(ref_array) / (number_of_objects * sizeof(float)), CV_32FC1, &ref_array);

    normAssert(ref, detection);
}

}
