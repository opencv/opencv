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
using namespace std;
using cv::ml::SVM;
using cv::ml::TrainData;

//--------------------------------------------------------------------------------------------
class CV_SVMTrainAutoTest : public cvtest::BaseTest {
public:
    CV_SVMTrainAutoTest() {}
protected:
    virtual void run( int start_from );
};

void CV_SVMTrainAutoTest::run( int /*start_from*/ )
{
    int datasize = 100;
    cv::Mat samples = cv::Mat::zeros( datasize, 2, CV_32FC1 );
    cv::Mat responses = cv::Mat::zeros( datasize, 1, CV_32S );

    RNG rng(0);
    for (int i = 0; i < datasize; ++i)
    {
        int response = rng.uniform(0, 2);  // Random from {0, 1}.
        samples.at<float>( i, 0 ) = rng.uniform(0.f, 0.5f) + response * 0.5f;
        samples.at<float>( i, 1 ) = rng.uniform(0.f, 0.5f) + response * 0.5f;
        responses.at<int>( i, 0 ) = response;
    }

    cv::Ptr<TrainData> data = TrainData::create( samples, cv::ml::ROW_SAMPLE, responses );
    cv::Ptr<SVM> svm = SVM::create();
    svm->trainAuto( data, 10 );  // 2-fold cross validation.

    float test_data0[2] = {0.25f, 0.25f};
    cv::Mat test_point0 = cv::Mat( 1, 2, CV_32FC1, test_data0 );
    float result0 = svm->predict( test_point0 );
    float test_data1[2] = {0.75f, 0.75f};
    cv::Mat test_point1 = cv::Mat( 1, 2, CV_32FC1, test_data1 );
    float result1 = svm->predict( test_point1 );

    if ( fabs( result0 - 0 ) > 0.001 || fabs( result1 - 1 ) > 0.001 )
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
    }
}

TEST(ML_SVM, trainauto) { CV_SVMTrainAutoTest test; test.safe_run(); }
