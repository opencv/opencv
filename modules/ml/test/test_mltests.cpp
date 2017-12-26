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

CV_AMLTest::CV_AMLTest( const char* _modelName ) : CV_MLBaseTest( _modelName )
{
    validationFN = "avalidation.xml";
}

int CV_AMLTest::run_test_case( int testCaseIdx )
{
    CV_TRACE_FUNCTION();
    int code = cvtest::TS::OK;
    code = prepare_test_case( testCaseIdx );

    if (code == cvtest::TS::OK)
    {
        //#define GET_STAT
#ifdef GET_STAT
        const char* data_name = ((CvFileNode*)cvGetSeqElem( dataSetNames, testCaseIdx ))->data.str.ptr;
        printf("%s, %s      ", name, data_name);
        const int icount = 100;
        float res[icount];
        for (int k = 0; k < icount; k++)
        {
#endif
            data->shuffleTrainTest();
            code = train( testCaseIdx );
#ifdef GET_STAT
            float case_result = get_error();

            res[k] = case_result;
        }
        float mean = 0, sigma = 0;
        for (int k = 0; k < icount; k++)
        {
            mean += res[k];
        }
        mean = mean /icount;
        for (int k = 0; k < icount; k++)
        {
            sigma += (res[k] - mean)*(res[k] - mean);
        }
        sigma = sqrt(sigma/icount);
        printf("%f, %f\n", mean, sigma);
#endif
    }
    return code;
}

int CV_AMLTest::validate_test_results( int testCaseIdx )
{
    CV_TRACE_FUNCTION();
    int iters;
    float mean, sigma;
    // read validation params
    FileNode resultNode =
        validationFS.getFirstTopLevelNode()["validation"][modelName][dataSetNames[testCaseIdx]]["result"];
    resultNode["iter_count"] >> iters;
    if ( iters > 0)
    {
        resultNode["mean"] >> mean;
        resultNode["sigma"] >> sigma;
        model->save(format("/Users/vp/tmp/dtree/testcase_%02d.cur.yml", testCaseIdx));
        float curErr = get_test_error( testCaseIdx );
        const int coeff = 4;
        ts->printf( cvtest::TS::LOG, "Test case = %d; test error = %f; mean error = %f (diff=%f), %d*sigma = %f\n",
                                testCaseIdx, curErr, mean, abs( curErr - mean), coeff, coeff*sigma );
        if ( abs( curErr - mean) > coeff*sigma )
        {
            ts->printf( cvtest::TS::LOG, "abs(%f - %f) > %f - OUT OF RANGE!\n", curErr, mean, coeff*sigma, coeff );
            return cvtest::TS::FAIL_BAD_ACCURACY;
        }
        else
            ts->printf( cvtest::TS::LOG, ".\n" );

    }
    else
    {
        ts->printf( cvtest::TS::LOG, "validation info is not suitable" );
        return cvtest::TS::FAIL_INVALID_TEST_DATA;
    }
    return cvtest::TS::OK;
}

TEST(ML_DTree, regression) { CV_AMLTest test( CV_DTREE ); test.safe_run(); }
TEST(ML_Boost, regression) { CV_AMLTest test( CV_BOOST ); test.safe_run(); }
TEST(ML_RTrees, regression) { CV_AMLTest test( CV_RTREES ); test.safe_run(); }
TEST(DISABLED_ML_ERTrees, regression) { CV_AMLTest test( CV_ERTREES ); test.safe_run(); }

TEST(ML_NBAYES, regression_5911)
{
    int N=12;
    Ptr<ml::NormalBayesClassifier> nb = cv::ml::NormalBayesClassifier::create();

    // data:
    Mat_<float> X(N,4);
    X << 1,2,3,4,  1,2,3,4,   1,2,3,4,    1,2,3,4,
         5,5,5,5,  5,5,5,5,   5,5,5,5,    5,5,5,5,
         4,3,2,1,  4,3,2,1,   4,3,2,1,    4,3,2,1;

    // labels:
    Mat_<int> Y(N,1);
    Y << 0,0,0,0, 1,1,1,1, 2,2,2,2;
    nb->train(X, ml::ROW_SAMPLE, Y);

    // single prediction:
    Mat R1,P1;
    for (int i=0; i<N; i++)
    {
        Mat r,p;
        nb->predictProb(X.row(i), r, p);
        R1.push_back(r);
        P1.push_back(p);
    }

    // bulk prediction (continuous memory):
    Mat R2,P2;
    nb->predictProb(X, R2, P2);

    EXPECT_EQ(sum(R1 == R2)[0], 255 * R2.total());
    EXPECT_EQ(sum(P1 == P2)[0], 255 * P2.total());

    // bulk prediction, with non-continuous memory storage
    Mat R3_(N, 1+1, CV_32S),
        P3_(N, 3+1, CV_32F);
    nb->predictProb(X, R3_.col(0), P3_.colRange(0,3));
    Mat R3 = R3_.col(0).clone(),
        P3 = P3_.colRange(0,3).clone();

    EXPECT_EQ(sum(R1 == R3)[0], 255 * R3.total());
    EXPECT_EQ(sum(P1 == P3)[0], 255 * P3.total());
}

TEST(ML_RTrees, getVotes)
{
    int n = 12;
    int count, i;
    int label_size = 3;
    int predicted_class = 0;
    int max_votes = -1;
    int val;
    // RTrees for classification
    Ptr<ml::RTrees> rt = cv::ml::RTrees::create();

    //data
    Mat data(n, 4, CV_32F);
    randu(data, 0, 10);

    //labels
    Mat labels = (Mat_<int>(n,1) << 0,0,0,0, 1,1,1,1, 2,2,2,2);

    rt->train(data, ml::ROW_SAMPLE, labels);

    //run function
    Mat test(1, 4, CV_32F);
    Mat result;
    randu(test, 0, 10);
    rt->getVotes(test, result, 0);

    //count vote amount and find highest vote
    count = 0;
    const int* result_row = result.ptr<int>(1);
    for( i = 0; i < label_size; i++ )
    {
        val = result_row[i];
        //predicted_class = max_votes < val? i;
        if( max_votes < val )
        {
            max_votes = val;
            predicted_class = i;
        }
        count += val;
    }

    EXPECT_EQ(count, (int)rt->getRoots().size());
    EXPECT_EQ(result.at<float>(0, predicted_class), rt->predict(test));
}

/* End of file. */
