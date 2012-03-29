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

using namespace std;
using namespace cv;

void defaultDistribs( vector<Mat>& means, vector<Mat>& covs )
{
    float mp0[] = {0.0f, 0.0f}, cp0[] = {0.67f, 0.0f, 0.0f, 0.67f};
    float mp1[] = {5.0f, 0.0f}, cp1[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float mp2[] = {1.0f, 5.0f}, cp2[] = {1.0f, 0.0f, 0.0f, 1.0f};
    Mat m0( 1, 2, CV_32FC1, mp0 ), c0( 2, 2, CV_32FC1, cp0 );
    Mat m1( 1, 2, CV_32FC1, mp1 ), c1( 2, 2, CV_32FC1, cp1 );
    Mat m2( 1, 2, CV_32FC1, mp2 ), c2( 2, 2, CV_32FC1, cp2 );
    means.resize(3), covs.resize(3);
    m0.copyTo(means[0]), c0.copyTo(covs[0]);
    m1.copyTo(means[1]), c1.copyTo(covs[1]);
    m2.copyTo(means[2]), c2.copyTo(covs[2]);
}

// generate points sets by normal distributions
void generateData( Mat& data, Mat& labels, const vector<int>& sizes, const vector<Mat>& means, const vector<Mat>& covs, int labelType )
{
    vector<int>::const_iterator sit = sizes.begin();
    int total = 0;
    for( ; sit != sizes.end(); ++sit )
        total += *sit;
    assert( means.size() == sizes.size() && covs.size() == sizes.size() );
    assert( !data.empty() && data.rows == total );
    assert( data.type() == CV_32FC1 );
    
    labels.create( data.rows, 1, labelType );

    randn( data, Scalar::all(0.0), Scalar::all(1.0) );
    vector<Mat>::const_iterator mit = means.begin(), cit = covs.begin();
    int bi, ei = 0;
    sit = sizes.begin();
    for( int p = 0, l = 0; sit != sizes.end(); ++sit, ++mit, ++cit, l++ )
    {
        bi = ei;
        ei = bi + *sit;
        assert( mit->rows == 1 && mit->cols == data.cols );
        assert( cit->rows == data.cols && cit->cols == data.cols );
        for( int i = bi; i < ei; i++, p++ )
        {
            Mat r(1, data.cols, CV_32FC1, data.ptr<float>(i));
            r =  r * (*cit) + *mit; 
            if( labelType == CV_32FC1 )
                labels.at<float>(p, 0) = (float)l;
            else if( labelType == CV_32SC1 )
                labels.at<int>(p, 0) = l;
            else
                CV_DbgAssert(0);
        }
    }
}

int maxIdx( const vector<int>& count )
{
    int idx = -1;
    int maxVal = -1;
    vector<int>::const_iterator it = count.begin();
    for( int i = 0; it != count.end(); ++it, i++ )
    {
        if( *it > maxVal)
        {
            maxVal = *it;
            idx = i;
        }
    }
    assert( idx >= 0);
    return idx;
}

bool getLabelsMap( const Mat& labels, const vector<int>& sizes, vector<int>& labelsMap )
{
    int total = 0, setCount = (int)sizes.size();
    vector<int>::const_iterator sit = sizes.begin();
    for( ; sit != sizes.end(); ++sit )
        total += *sit;
    assert( !labels.empty() );
    assert( labels.rows == total && labels.cols == 1 );
    assert( labels.type() == CV_32SC1 || labels.type() == CV_32FC1 );

    bool isFlt = labels.type() == CV_32FC1;
    labelsMap.resize(setCount);
    vector<int>::iterator lmit = labelsMap.begin();
    vector<bool> buzy(setCount, false);
    int bi, ei = 0;
    for( sit = sizes.begin(); sit != sizes.end(); ++sit, ++lmit )
    {
        vector<int> count( setCount, 0 );
        bi = ei;
        ei = bi + *sit;
        if( isFlt )
        {
            for( int i = bi; i < ei; i++ )
                count[(int)labels.at<float>(i, 0)]++;
        }
        else
        {
            for( int i = bi; i < ei; i++ )
                count[labels.at<int>(i, 0)]++;
        }
  
        *lmit = maxIdx( count );
        if( buzy[*lmit] )
            return false;
        buzy[*lmit] = true;
    }
    return true;    
}

float calcErr( const Mat& labels, const Mat& origLabels, const vector<int>& sizes, bool labelsEquivalent = true )
{
    int err = 0;
    assert( !labels.empty() && !origLabels.empty() );
    assert( labels.cols == 1 && origLabels.cols == 1 );
    assert( labels.rows == origLabels.rows );
    assert( labels.type() == origLabels.type() );
    assert( labels.type() == CV_32SC1 || labels.type() == CV_32FC1 );

    vector<int> labelsMap;
    bool isFlt = labels.type() == CV_32FC1;
    if( !labelsEquivalent )
    {
        getLabelsMap( labels, sizes, labelsMap );
        for( int i = 0; i < labels.rows; i++ )
            if( isFlt )
                err += labels.at<float>(i, 0) != labelsMap[(int)origLabels.at<float>(i, 0)];
            else
                err += labels.at<int>(i, 0) != labelsMap[origLabels.at<int>(i, 0)];
    }
    else
    {
        for( int i = 0; i < labels.rows; i++ )
            if( isFlt )
                err += labels.at<float>(i, 0) != origLabels.at<float>(i, 0);
            else
                err += labels.at<int>(i, 0) != origLabels.at<int>(i, 0);
    }
    return (float)err / (float)labels.rows;
}

//--------------------------------------------------------------------------------------------
class CV_KMeansTest : public cvtest::BaseTest {
public:
    CV_KMeansTest() {}
protected:
    virtual void run( int start_from );
};

void CV_KMeansTest::run( int /*start_from*/ )
{
    const int iters = 100;
    int sizesArr[] = { 5000, 7000, 8000 };
    int pointsCount = sizesArr[0]+ sizesArr[1] + sizesArr[2];
    
    Mat data( pointsCount, 2, CV_32FC1 ), labels;
    vector<int> sizes( sizesArr, sizesArr + sizeof(sizesArr) / sizeof(sizesArr[0]) );
    vector<Mat> means, covs;
    defaultDistribs( means, covs );
    generateData( data, labels, sizes, means, covs, CV_32SC1 );
    
    int code = cvtest::TS::OK;
    float err;
    Mat bestLabels;
    // 1. flag==KMEANS_PP_CENTERS
    kmeans( data, 3, bestLabels, TermCriteria( TermCriteria::COUNT, iters, 0.0), 0, KMEANS_PP_CENTERS, noArray() );
    err = calcErr( bestLabels, labels, sizes, false );
    if( err > 0.01f )
    {
        ts->printf( cvtest::TS::LOG, "Bad accuracy (%f) if flag==KMEANS_PP_CENTERS.\n", err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    // 2. flag==KMEANS_RANDOM_CENTERS
    kmeans( data, 3, bestLabels, TermCriteria( TermCriteria::COUNT, iters, 0.0), 0, KMEANS_RANDOM_CENTERS, noArray() );
    err = calcErr( bestLabels, labels, sizes, false );
    if( err > 0.01f )
    {
        ts->printf( cvtest::TS::LOG, "Bad accuracy (%f) if flag==KMEANS_PP_CENTERS.\n", err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    // 3. flag==KMEANS_USE_INITIAL_LABELS
    labels.copyTo( bestLabels );
    RNG rng;
    for( int i = 0; i < 0.5f * pointsCount; i++ )
        bestLabels.at<int>( rng.next() % pointsCount, 0 ) = rng.next() % 3;
    kmeans( data, 3, bestLabels, TermCriteria( TermCriteria::COUNT, iters, 0.0), 0, KMEANS_USE_INITIAL_LABELS, noArray() );
    err = calcErr( bestLabels, labels, sizes, false );
    if( err > 0.01f )
    {
        ts->printf( cvtest::TS::LOG, "Bad accuracy (%f) if flag==KMEANS_PP_CENTERS.\n", err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    ts->set_failed_test_info( code );
}

//--------------------------------------------------------------------------------------------
class CV_KNearestTest : public cvtest::BaseTest {
public:
    CV_KNearestTest() {}
protected:
    virtual void run( int start_from );
};

void CV_KNearestTest::run( int /*start_from*/ )
{
    int sizesArr[] = { 500, 700, 800 };
    int pointsCount = sizesArr[0]+ sizesArr[1] + sizesArr[2];

    // train data
    Mat trainData( pointsCount, 2, CV_32FC1 ), trainLabels;
    vector<int> sizes( sizesArr, sizesArr + sizeof(sizesArr) / sizeof(sizesArr[0]) );
    vector<Mat> means, covs;
    defaultDistribs( means, covs );
    generateData( trainData, trainLabels, sizes, means, covs, CV_32FC1 );

    // test data
    Mat testData( pointsCount, 2, CV_32FC1 ), testLabels, bestLabels;
    generateData( testData, testLabels, sizes, means, covs, CV_32FC1 );

    int code = cvtest::TS::OK;
    KNearest knearest;
    knearest.train( trainData, trainLabels );
    knearest.find_nearest( testData, 4, &bestLabels );
    float err = calcErr( bestLabels, testLabels, sizes, true );
    if( err > 0.01f )
    {
        ts->printf( cvtest::TS::LOG, "Bad accuracy (%f) on test data.\n", err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }
    ts->set_failed_test_info( code );
}

//--------------------------------------------------------------------------------------------
class CV_EMTest : public cvtest::BaseTest {
public:
    CV_EMTest() {}
protected:
    virtual void run( int start_from );
};

void CV_EMTest::run( int /*start_from*/ )
{
    int sizesArr[] = { 5000, 7000, 8000 };
    int pointsCount = sizesArr[0]+ sizesArr[1] + sizesArr[2];

    // train data
    Mat trainData( pointsCount, 2, CV_32FC1 ), trainLabels;
    vector<int> sizes( sizesArr, sizesArr + sizeof(sizesArr) / sizeof(sizesArr[0]) );
    vector<Mat> means, covs;
    defaultDistribs( means, covs );
    generateData( trainData, trainLabels, sizes, means, covs, CV_32SC1 );

    // test data
    Mat testData( pointsCount, 2, CV_32FC1 ), testLabels, bestLabels;
    generateData( testData, testLabels, sizes, means, covs, CV_32SC1 );

    int code = cvtest::TS::OK;
    float err;
    ExpectationMaximization em;
    CvEMParams params;
    params.nclusters = 3;
    em.train( trainData, Mat(), params, &bestLabels );

    // check train error
    err = calcErr( bestLabels, trainLabels, sizes, false );
    if( err > 0.002f )
    {
        ts->printf( cvtest::TS::LOG, "Bad accuracy (%f) on train data.\n", err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    // check test error
    bestLabels.create( testData.rows, 1, CV_32SC1 );
    for( int i = 0; i < testData.rows; i++ )
    {
        Mat sample( 1, testData.cols, CV_32FC1, testData.ptr<float>(i));
        bestLabels.at<int>(i,0) = (int)em.predict( sample, 0 );
    }
    err = calcErr( bestLabels, testLabels, sizes, false );
    if( err > 0.005f )
    {
        ts->printf( cvtest::TS::LOG, "Bad accuracy (%f) on test data.\n", err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }
    
    ts->set_failed_test_info( code );
}

class CV_EMTest_Smoke : public cvtest::BaseTest {
public:
    CV_EMTest_Smoke() {}
protected:
    virtual void run( int /*start_from*/ )
    {
        int code = cvtest::TS::OK;
        CvEM em;

        Mat samples = Mat(3,2,CV_32F);
        samples.at<float>(0,0) = 1;
        samples.at<float>(1,0) = 2;
        samples.at<float>(2,0) = 3;

        CvEMParams params;
        params.nclusters = 2;

        Mat labels;

        em.train(samples, Mat(), params, &labels);

        Mat firstResult(samples.rows, 1, CV_32FC1);
        for( int i = 0; i < samples.rows; i++)
            firstResult.at<float>(i) = em.predict( samples.row(i) );

        // Write out
        string filename = tempfile() + ".xml";
        {
            FileStorage fs = FileStorage(filename, FileStorage::WRITE);

            try
            {
                em.write(fs.fs, "EM");
            }
            catch(...)
            {
                ts->printf( cvtest::TS::LOG, "Crash in write method.\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_EXCEPTION );
            }
        }

        em.clear();

        // Read in
        {
            FileStorage fs = FileStorage(filename, FileStorage::READ);
            FileNode fileNode = fs["EM"];

            try
            {
                em.read(const_cast<CvFileStorage*>(fileNode.fs), const_cast<CvFileNode*>(fileNode.node));
            }
            catch(...)
            {
                ts->printf( cvtest::TS::LOG, "Crash in read method.\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_EXCEPTION );
            }
        }

        remove( filename.c_str() );

        int errCaseCount = 0;
        for( int i = 0; i < samples.rows; i++)
            errCaseCount = std::abs(em.predict(samples.row(i)) - firstResult.at<float>(i)) < FLT_EPSILON ? 0 : 1;

        if( errCaseCount > 0 )
        {
            ts->printf( cvtest::TS::LOG, "Different prediction results before writeing and after reading (errCaseCount=%d).\n", errCaseCount );
            code = cvtest::TS::FAIL_BAD_ACCURACY;
        }

        ts->set_failed_test_info( code );
    }
};

TEST(ML_KMeans, accuracy) { CV_KMeansTest test; test.safe_run(); }
TEST(ML_KNearest, accuracy) { CV_KNearestTest test; test.safe_run(); }
TEST(ML_EM, accuracy) { CV_EMTest test; test.safe_run(); }
TEST(ML_EM, smoke) { CV_EMTest_Smoke test; test.safe_run(); }
