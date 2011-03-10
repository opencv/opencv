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

class CV_GPU_SURFTest : public cvtest::BaseTest
{
public:
    CV_GPU_SURFTest()
    {
    }

protected:
    bool isSimilarKeypoints(const KeyPoint& p1, const KeyPoint& p2);
    int getValidCount(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& matches);
    void compareKeypointSets(const vector<KeyPoint>& validKeypoints, const vector<KeyPoint>& calcKeypoints,
                             const Mat& validDescriptors, const Mat& calcDescriptors);

    void emptyDataTest();
    void accuracyTest();

    virtual void run(int);
};

void CV_GPU_SURFTest::emptyDataTest()
{
    SURF_GPU fdetector;

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

int CV_GPU_SURFTest::getValidCount(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                     const vector<DMatch>& matches)
{
    int count = 0;

    for (size_t i = 0; i < matches.size(); ++i)
    {
        const DMatch& m = matches[i];

        const KeyPoint& kp1 = keypoints1[m.queryIdx];
        const KeyPoint& kp2 = keypoints2[m.trainIdx];

        if (isSimilarKeypoints(kp1, kp2))
            ++count;
    }

    return count;
}

void CV_GPU_SURFTest::compareKeypointSets(const vector<KeyPoint>& validKeypoints, const vector<KeyPoint>& calcKeypoints, 
                                          const Mat& validDescriptors, const Mat& calcDescriptors)
{
    BruteForceMatcher< L2<float> > matcher;
    vector<DMatch> matches;

    matcher.match(validDescriptors, calcDescriptors, matches);

    int validCount = getValidCount(validKeypoints, calcKeypoints, matches);
    float validRatio = (float)validCount / matches.size();

    if (validRatio < 0.5f)
    {
        ts->printf(cvtest::TS::LOG, "Bad accuracy - %f.\n", validRatio);
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
        return;
    }
}

void CV_GPU_SURFTest::accuracyTest()
{
    string imgFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;

    // Read the test image.
    Mat image = imread(imgFilename, 0);
    if (image.empty())
    {
        ts->printf( cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str() );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }
    
    Mat mask(image.size(), CV_8UC1, Scalar::all(1));
    mask(Range(0, image.rows / 2), Range(0, image.cols / 2)).setTo(Scalar::all(0));

    // Compute keypoints.
    vector<KeyPoint> calcKeypoints;
    GpuMat calcDescriptors;
    SURF_GPU fdetector; fdetector.extended = false;
    fdetector(GpuMat(image), GpuMat(mask), calcKeypoints, calcDescriptors);

    // Calc validation keypoints set.
    vector<KeyPoint> validKeypoints;
    vector<float> validDescriptors;
    SURF fdetector_gold; fdetector_gold.extended = false;
    fdetector_gold(image, mask, validKeypoints, validDescriptors);

    compareKeypointSets(validKeypoints, calcKeypoints, 
        Mat(validKeypoints.size(), fdetector_gold.descriptorSize(), CV_32F, &validDescriptors[0]), calcDescriptors);
}

void CV_GPU_SURFTest::run( int /*start_from*/ )
{
    emptyDataTest();
    accuracyTest();
}

TEST(SURF, empty_data_and_accuracy) { CV_GPU_SURFTest test; test.safe_run(); }
