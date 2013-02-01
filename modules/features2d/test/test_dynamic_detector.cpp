/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

using namespace cv;

const string FEATURES2D_DIR = "features2d";
const string IMAGE_FILENAME = "tsukuba.png";

class CV_DynamicAdaptedFeatureDetectorTest : public cvtest::BaseTest
{
protected:
    virtual void run(int);
};

void CV_DynamicAdaptedFeatureDetectorTest::run(int)
{
    string imgFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;

    Mat image = imread(imgFilename);
    if(image.empty())
    {
        ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str());
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    const int min_features = 0;
    const int max_features = 5;
    const int max_iters = 1;

    {
        bool save_adjusted_parameters = false;
        Ptr<AdjusterAdapter> adapter = new FastAdjuster();
        Ptr<DynamicAdaptedFeatureDetector> detector = new DynamicAdaptedFeatureDetector(adapter, min_features, max_features, max_iters, save_adjusted_parameters);
        {
            vector<KeyPoint> keypoints_1, keypoints_2;
            detector->detect(image, keypoints_1);
            detector->detect(image, keypoints_2);

            if (keypoints_1.size() != keypoints_2.size())
            {
                ts->printf(cvtest::TS::LOG, "Non-const behaviour of DynamicAdaptedFeatureDetector when save_adjusted_parameters=false\n");
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            }
        }

        detector->setBool("save_adjusted_parameters", true);
        {
            vector<KeyPoint> keypoints_1, keypoints_2;
            detector->detect(image, keypoints_1);
            detector->detect(image, keypoints_2);

            if (keypoints_1.size() == keypoints_2.size())
            {
                ts->printf(cvtest::TS::LOG, "const behaviour of DynamicAdaptedFeatureDetector when save_adjusted_parameters=true\n");
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            }
            if (keypoints_1.size() < keypoints_2.size())
            {
                ts->printf(cvtest::TS::LOG, "Paramters of DynamicAdaptedFeatureDetector are adjusted in a wrong direction\n");
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            }
        }
    }

    {
        bool save_adjusted_parameters = false;
        Ptr<AdjusterAdapter> adapter = new FastAdjuster();
        const Ptr<const DynamicAdaptedFeatureDetector> detector = new DynamicAdaptedFeatureDetector(adapter, min_features, max_features, max_iters, save_adjusted_parameters);
        vector<KeyPoint> keypoints_1, keypoints_2;
        detector->detect(image, keypoints_1);
        detector->detect(image, keypoints_2);

        if (keypoints_1.size() != keypoints_2.size())
        {
            ts->printf(cvtest::TS::LOG, "Non-const behaviour of DynamicAdaptedFeatureDetector when save_adjusted_parameters=false\n");
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }

    {
        bool save_adjusted_parameters = true;
        Ptr<AdjusterAdapter> adapter = new FastAdjuster();
        const Ptr<const DynamicAdaptedFeatureDetector> detector = new DynamicAdaptedFeatureDetector(adapter, min_features, max_features, max_iters, save_adjusted_parameters);

        bool isExceptionThrown = false;
        try
        {
            vector<KeyPoint> keypoints_1;
            detector->detect(image, keypoints_1);
        }
        catch(const cv::Exception &ex)
        {
            isExceptionThrown = true;
        }

        if (!isExceptionThrown)
        {
            ts->printf(cvtest::TS::LOG, "Exception wasn't thrown when calling a const DynamicAdaptedFeatureDetector with save_adjusted_parameters=true\n");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ARG_CHECK);
        }
    }
}

TEST(Features2d_DynamicAdaptedFeatureDetector, validation) { CV_DynamicAdaptedFeatureDetectorTest test; test.safe_run(); }
