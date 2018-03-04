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

namespace opencv_test { namespace {

using cv::ml::SVMSGD;
using cv::ml::TrainData;

class CV_SVMSGDTrainTest : public cvtest::BaseTest
{
public:
    enum TrainDataType
    {
        UNIFORM_SAME_SCALE,
        UNIFORM_DIFFERENT_SCALES
    };

    CV_SVMSGDTrainTest(const Mat &_weights, float shift, TrainDataType type, double precision = 0.01);
private:
    virtual void run( int start_from );
    static float decisionFunction(const Mat &sample, const Mat &weights, float shift);
    void makeData(int samplesCount, const Mat &weights, float shift, RNG &rng, Mat &samples, Mat & responses);
    void generateSameBorders(int featureCount);
    void generateDifferentBorders(int featureCount);

    TrainDataType type;
    double precision;
    std::vector<std::pair<float,float> > borders;
    cv::Ptr<TrainData> data;
    cv::Mat testSamples;
    cv::Mat testResponses;
    static const int TEST_VALUE_LIMIT = 500;
};

void CV_SVMSGDTrainTest::generateSameBorders(int featureCount)
{
    float lowerLimit = -TEST_VALUE_LIMIT;
    float upperLimit = TEST_VALUE_LIMIT;

    for (int featureIndex = 0; featureIndex < featureCount; featureIndex++)
    {
        borders.push_back(std::pair<float,float>(lowerLimit, upperLimit));
    }
}

void CV_SVMSGDTrainTest::generateDifferentBorders(int featureCount)
{
    float lowerLimit = -TEST_VALUE_LIMIT;
    float upperLimit = TEST_VALUE_LIMIT;
    cv::RNG rng(0);

    for (int featureIndex = 0; featureIndex < featureCount; featureIndex++)
    {
        int crit = rng.uniform(0, 2);

        if (crit > 0)
        {
            borders.push_back(std::pair<float,float>(lowerLimit, upperLimit));
        }
        else
        {
            borders.push_back(std::pair<float,float>(lowerLimit/1000, upperLimit/1000));
        }
    }
}

float CV_SVMSGDTrainTest::decisionFunction(const Mat &sample, const Mat &weights, float shift)
{
    return static_cast<float>(sample.dot(weights)) + shift;
}

void CV_SVMSGDTrainTest::makeData(int samplesCount, const Mat &weights, float shift, RNG &rng, Mat &samples, Mat & responses)
{
    int featureCount = weights.cols;

    samples.create(samplesCount, featureCount, CV_32FC1);
    for (int featureIndex = 0; featureIndex < featureCount; featureIndex++)
    {
        rng.fill(samples.col(featureIndex), RNG::UNIFORM, borders[featureIndex].first, borders[featureIndex].second);
    }

    responses.create(samplesCount, 1, CV_32FC1);

    for (int i = 0 ; i < samplesCount; i++)
    {
        responses.at<float>(i) = decisionFunction(samples.row(i), weights, shift) > 0 ? 1.f : -1.f;
    }

}

CV_SVMSGDTrainTest::CV_SVMSGDTrainTest(const Mat &weights, float shift, TrainDataType _type, double _precision)
{
    type = _type;
    precision = _precision;

    int featureCount = weights.cols;

    switch(type)
    {
    case UNIFORM_SAME_SCALE:
        generateSameBorders(featureCount);
        break;
    case UNIFORM_DIFFERENT_SCALES:
        generateDifferentBorders(featureCount);
        break;
    default:
        CV_Error(CV_StsBadArg, "Unknown train data type");
    }

    RNG rng(0);

    Mat trainSamples;
    Mat trainResponses;
    int trainSamplesCount = 10000;
    makeData(trainSamplesCount, weights, shift, rng, trainSamples, trainResponses);
    data = TrainData::create(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);

    int testSamplesCount = 100000;
    makeData(testSamplesCount, weights, shift, rng, testSamples, testResponses);
}

void CV_SVMSGDTrainTest::run( int /*start_from*/ )
{
    cv::Ptr<SVMSGD> svmsgd = SVMSGD::create();

    svmsgd->train(data);

    Mat responses;

    svmsgd->predict(testSamples, responses);

    int errCount = 0;
    int testSamplesCount = testSamples.rows;

    CV_Assert((responses.type() == CV_32FC1) && (testResponses.type() == CV_32FC1));
    for (int i = 0; i < testSamplesCount; i++)
    {
        if (responses.at<float>(i) * testResponses.at<float>(i) < 0)
            errCount++;
    }

    float err = (float)errCount / testSamplesCount;

    if ( err > precision )
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
    }
}

void makeWeightsAndShift(int featureCount, Mat &weights, float &shift)
{
    weights.create(1, featureCount, CV_32FC1);
    cv::RNG rng(0);
    double lowerLimit = -1;
    double upperLimit = 1;

    rng.fill(weights, RNG::UNIFORM, lowerLimit, upperLimit);
    shift = static_cast<float>(rng.uniform(-featureCount, featureCount));
}


TEST(ML_SVMSGD, trainSameScale2)
{
    int featureCount = 2;

    Mat weights;

    float shift = 0;
    makeWeightsAndShift(featureCount, weights, shift);

    CV_SVMSGDTrainTest test(weights, shift, CV_SVMSGDTrainTest::UNIFORM_SAME_SCALE);
    test.safe_run();
}

TEST(ML_SVMSGD, trainSameScale5)
{
    int featureCount = 5;

    Mat weights;

    float shift = 0;
    makeWeightsAndShift(featureCount, weights, shift);

    CV_SVMSGDTrainTest test(weights, shift, CV_SVMSGDTrainTest::UNIFORM_SAME_SCALE);
    test.safe_run();
}

TEST(ML_SVMSGD, trainSameScale100)
{
    int featureCount = 100;

    Mat weights;

    float shift = 0;
    makeWeightsAndShift(featureCount, weights, shift);

    CV_SVMSGDTrainTest test(weights, shift, CV_SVMSGDTrainTest::UNIFORM_SAME_SCALE, 0.02);
    test.safe_run();
}

TEST(ML_SVMSGD, trainDifferentScales2)
{
    int featureCount = 2;

    Mat weights;

    float shift = 0;
    makeWeightsAndShift(featureCount, weights, shift);

    CV_SVMSGDTrainTest test(weights, shift, CV_SVMSGDTrainTest::UNIFORM_DIFFERENT_SCALES, 0.01);
    test.safe_run();
}

TEST(ML_SVMSGD, trainDifferentScales5)
{
    int featureCount = 5;

    Mat weights;

    float shift = 0;
    makeWeightsAndShift(featureCount, weights, shift);

    CV_SVMSGDTrainTest test(weights, shift, CV_SVMSGDTrainTest::UNIFORM_DIFFERENT_SCALES, 0.01);
    test.safe_run();
}

TEST(ML_SVMSGD, trainDifferentScales100)
{
    int featureCount = 100;

    Mat weights;

    float shift = 0;
    makeWeightsAndShift(featureCount, weights, shift);

    CV_SVMSGDTrainTest test(weights, shift, CV_SVMSGDTrainTest::UNIFORM_DIFFERENT_SCALES, 0.01);
    test.safe_run();
}

TEST(ML_SVMSGD, twoPoints)
{
    Mat samples(2, 2, CV_32FC1);
    samples.at<float>(0,0) = 0;
    samples.at<float>(0,1) = 0;
    samples.at<float>(1,0) = 1000;
    samples.at<float>(1,1) = 1;

    Mat responses(2, 1, CV_32FC1);
    responses.at<float>(0) = -1;
    responses.at<float>(1) = 1;

    cv::Ptr<TrainData> trainData = TrainData::create(samples, cv::ml::ROW_SAMPLE, responses);

    Mat realWeights(1, 2, CV_32FC1);
    realWeights.at<float>(0) = 1000;
    realWeights.at<float>(1) = 1;

    float realShift = -500000.5;

    float normRealWeights = static_cast<float>(cv::norm(realWeights)); // TODO cvtest
    realWeights /= normRealWeights;
    realShift /= normRealWeights;

    cv::Ptr<SVMSGD> svmsgd = SVMSGD::create();
    svmsgd->setOptimalParameters();
    svmsgd->train( trainData );

    Mat foundWeights = svmsgd->getWeights();
    float foundShift = svmsgd->getShift();

    float normFoundWeights = static_cast<float>(cv::norm(foundWeights)); // TODO cvtest
    foundWeights /= normFoundWeights;
    foundShift /= normFoundWeights;
    EXPECT_LE(cv::norm(Mat(foundWeights - realWeights)), 0.001); // TODO cvtest
    EXPECT_LE(std::abs((foundShift - realShift) / realShift), 0.05);
}

}} // namespace
