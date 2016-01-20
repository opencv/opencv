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
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace cv::ml;
using cv::ml::SVMSGD;
using cv::ml::TrainData;



class CV_SVMSGDTrainTest : public cvtest::BaseTest
{
public:
    CV_SVMSGDTrainTest(Mat _weights, float _shift);
private:
    virtual void run( int start_from );
    float decisionFunction(Mat sample, Mat weights, float shift);

    cv::Ptr<TrainData> data;
    cv::Mat testSamples;
    cv::Mat testResponses;
    static const int TEST_VALUE_LIMIT = 50;
};

CV_SVMSGDTrainTest::CV_SVMSGDTrainTest(Mat weights, float shift)
{
    int datasize = 100000;
    int varCount = weights.cols;
    cv::Mat samples = cv::Mat::zeros( datasize, varCount, CV_32FC1 );
    cv::Mat responses = cv::Mat::zeros( datasize, 1, CV_32FC1 );
    cv::RNG rng(0);

    float lowerLimit = -TEST_VALUE_LIMIT;
    float upperLimit = TEST_VALUE_LIMIT;


    rng.fill(samples, RNG::UNIFORM, lowerLimit, upperLimit);
    for (int sampleIndex = 0; sampleIndex < datasize; sampleIndex++)
    {
        responses.at<float>( sampleIndex ) = decisionFunction(samples.row(sampleIndex), weights, shift) > 0 ? 1 : -1;
    }

    data = TrainData::create( samples, cv::ml::ROW_SAMPLE, responses );

    int testSamplesCount = 100000;

    testSamples.create(testSamplesCount, varCount, CV_32FC1);
    rng.fill(testSamples, RNG::UNIFORM, lowerLimit, upperLimit);
    testResponses.create(testSamplesCount, 1, CV_32FC1);

    for (int i = 0 ; i < testSamplesCount; i++)
    {
        testResponses.at<float>(i) = decisionFunction(testSamples.row(i), weights, shift) > 0 ? 1 : -1;
    }
}

void CV_SVMSGDTrainTest::run( int /*start_from*/ )
{
    cv::Ptr<SVMSGD> svmsgd = SVMSGD::create();

    svmsgd->setOptimalParameters(SVMSGD::ASGD);

    svmsgd->train( data );

    Mat responses;

    svmsgd->predict(testSamples, responses);

    int errCount = 0;
    int testSamplesCount = testSamples.rows;

    for (int i = 0; i < testSamplesCount; i++)
    {
        if (responses.at<float>(i) * testResponses.at<float>(i) < 0 )
            errCount++;
    }

    float err = (float)errCount / testSamplesCount;
    std::cout << "err " << err << std::endl;

    if ( err > 0.01 )
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
    }
}

float CV_SVMSGDTrainTest::decisionFunction(Mat sample, Mat weights, float shift)
{
    return sample.dot(weights) + shift;
}

TEST(ML_SVMSGD, train0)
{
    int varCount = 2;

    Mat weights;
    weights.create(1, varCount, CV_32FC1);
    weights.at<float>(0) = 1;
    weights.at<float>(1) = 0;

    float shift = 5;

    CV_SVMSGDTrainTest test(weights, shift);
    test.safe_run();
}

TEST(ML_SVMSGD, train1)
{
    int varCount = 5;

    Mat weights;
    weights.create(1, varCount, CV_32FC1);

    float lowerLimit = -1;
    float upperLimit = 1;
    cv::RNG rng(0);
    rng.fill(weights, RNG::UNIFORM, lowerLimit, upperLimit);

    float shift = rng.uniform(-5.f, 5.f);

    CV_SVMSGDTrainTest test(weights, shift);
    test.safe_run();
}

TEST(ML_SVMSGD, train2)
{
    int varCount = 100;

    Mat weights;
    weights.create(1, varCount, CV_32FC1);

    float lowerLimit = -1;
    float upperLimit = 1;
    cv::RNG rng(0);
    rng.fill(weights, RNG::UNIFORM, lowerLimit, upperLimit);

    float shift = rng.uniform(-1000.f, 1000.f);

    CV_SVMSGDTrainTest test(weights, shift);
    test.safe_run();
}
