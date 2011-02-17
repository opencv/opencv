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
#include <string>

using namespace cv;
using namespace cv::gpu;
using namespace std;

const string FEATURES2D_DIR = "features2d";
const string IMAGE_FILENAME = "aloe.png";
const string VALID_FILE_NAME = "surf.xml.gz";

class CV_GPU_SURFTest : public cvtest::BaseTest
{
public:
    CV_GPU_SURFTest()
    {
    }

protected:
    bool isSimilarKeypoints(const KeyPoint& p1, const KeyPoint& p2);
    void compareKeypointSets(const vector<KeyPoint>& validKeypoints, const vector<KeyPoint>& calcKeypoints,
                             const Mat& validDescriptors, const Mat& calcDescriptors);

    void emptyDataTest(SURF_GPU& fdetector);
    void regressionTest(SURF_GPU& fdetector);

    virtual void run(int);
};

void CV_GPU_SURFTest::emptyDataTest(SURF_GPU& fdetector)
{
    GpuMat image;
    vector<KeyPoint> keypoints;
    vector<float> descriptors;
    try
    {
        fdetector(image, GpuMat(), keypoints, descriptors);
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "detect() on empty image must not generate exception (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    if( !keypoints.empty() )
    {
        ts->printf( cvtest::TS::LOG, "detect() on empty image must return empty keypoints vector (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        return;
    }

    if( !descriptors.empty() )
    {
        ts->printf( cvtest::TS::LOG, "detect() on empty image must return empty descriptors vector (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        return;
    }
}

bool CV_GPU_SURFTest::isSimilarKeypoints(const KeyPoint& p1, const KeyPoint& p2)
{
    const float maxPtDif = 1.f;
    const float maxSizeDif = 1.f;
    const float maxAngleDif = 2.f;
    const float maxResponseDif = 0.1f;

    float dist = (float)norm( p1.pt - p2.pt );
    return (dist < maxPtDif &&
            fabs(p1.size - p2.size) < maxSizeDif &&
            abs(p1.angle - p2.angle) < maxAngleDif &&
            abs(p1.response - p2.response) < maxResponseDif &&
            p1.octave == p2.octave &&
            p1.class_id == p2.class_id );
}

void CV_GPU_SURFTest::compareKeypointSets(const vector<KeyPoint>& validKeypoints, const vector<KeyPoint>& calcKeypoints, 
                                          const Mat& validDescriptors, const Mat& calcDescriptors)
{
    if (validKeypoints.size() != calcKeypoints.size())
    {
        ts->printf(cvtest::TS::LOG, "Keypoints sizes doesn't equal (validCount = %d, calcCount = %d).\n",
                   validKeypoints.size(), calcKeypoints.size());
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return;
    }
    if (validDescriptors.size() != calcDescriptors.size())
    {
        ts->printf(cvtest::TS::LOG, "Descriptors sizes doesn't equal.\n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return;
    }
    for (size_t v = 0; v < validKeypoints.size(); v++)
    {
        int nearestIdx = -1;
        float minDist = std::numeric_limits<float>::max();

        for (size_t c = 0; c < calcKeypoints.size(); c++)
        {
            float curDist = (float)norm(calcKeypoints[c].pt - validKeypoints[v].pt);
            if (curDist < minDist)
            {
                minDist = curDist;
                nearestIdx = c;
            }
        }

        assert(minDist >= 0);
        if (!isSimilarKeypoints(validKeypoints[v], calcKeypoints[nearestIdx]))
        {
            ts->printf(cvtest::TS::LOG, "Bad keypoints accuracy.\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }

        if (norm(validDescriptors.row(v), calcDescriptors.row(nearestIdx), NORM_L2) > 1.5f)
        {
            ts->printf(cvtest::TS::LOG, "Bad descriptors accuracy.\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }
    }
}

void CV_GPU_SURFTest::regressionTest(SURF_GPU& fdetector)
{
    string imgFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;
    string resFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + VALID_FILE_NAME;

    // Read the test image.
    GpuMat image(imread(imgFilename, 0));
    if (image.empty())
    {
        ts->printf( cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str() );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    FileStorage fs(resFilename, FileStorage::READ);

    // Compute keypoints.
    GpuMat mask(image.size(), CV_8UC1, Scalar::all(1));
    mask(Range(0, image.rows / 2), Range(0, image.cols / 2)).setTo(Scalar::all(0));
    vector<KeyPoint> calcKeypoints;
    GpuMat calcDespcriptors;
    fdetector(image, mask, calcKeypoints, calcDespcriptors);

    if (fs.isOpened()) // Compare computed and valid keypoints.
    {
        // Read validation keypoints set.
        vector<KeyPoint> validKeypoints;
        Mat validDespcriptors;
        read(fs["keypoints"], validKeypoints);
        read(fs["descriptors"], validDespcriptors);
        if (validKeypoints.empty() || validDespcriptors.empty())
        {
            ts->printf(cvtest::TS::LOG, "Validation file can not be read.\n");
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        compareKeypointSets(validKeypoints, calcKeypoints, validDespcriptors, calcDespcriptors);
    }
    else // Write detector parameters and computed keypoints as validation data.
    {
        fs.open(resFilename, FileStorage::WRITE);
        if (!fs.isOpened())
        {
            ts->printf(cvtest::TS::LOG, "File %s can not be opened to write.\n", resFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }
        else
        {
            write(fs, "keypoints", calcKeypoints);
            write(fs, "descriptors", (Mat)calcDespcriptors);
        }
    }
}

void CV_GPU_SURFTest::run( int /*start_from*/ )
{
    SURF_GPU fdetector;

    emptyDataTest(fdetector);
    regressionTest(fdetector);
}

TEST(SURF, empty_data_and_regression) { CV_GPU_SURFTest test; test.safe_run(); }
