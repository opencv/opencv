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
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include <algorithm>
#include <vector>
#include <iostream>

using namespace cv;
using namespace cv::flann;

//--------------------------------------------------------------------------------
class NearestNeighborTest : public cvtest::BaseTest
{
public:
    NearestNeighborTest() {}
protected:
    static const int minValue = 0;
    static const int maxValue = 1;
    static const int dims = 30;
    static const int featuresCount = 2000;
    static const int K = 1; // * should also test 2nd nn etc.?


    virtual void run( int start_from );
    virtual void createModel( const Mat& data ) = 0;
    virtual int findNeighbors( Mat& points, Mat& neighbors ) = 0;
    virtual int checkGetPoins( const Mat& data );
    virtual int checkFindBoxed();
    virtual int checkFind( const Mat& data );
    virtual void releaseModel() = 0;
};

int NearestNeighborTest::checkGetPoins( const Mat& )
{
   return cvtest::TS::OK;
}

int NearestNeighborTest::checkFindBoxed()
{
    return cvtest::TS::OK;
}

int NearestNeighborTest::checkFind( const Mat& data )
{
    int code = cvtest::TS::OK;
    int pointsCount = 1000;
    float noise = 0.2f;

    RNG rng;
    Mat points( pointsCount, dims, CV_32FC1 );
    Mat results( pointsCount, K, CV_32SC1 );

    std::vector<int> fmap( pointsCount );
    for( int pi = 0; pi < pointsCount; pi++ )
    {
        int fi = rng.next() % featuresCount;
        fmap[pi] = fi;
        for( int d = 0; d < dims; d++ )
            points.at<float>(pi, d) = data.at<float>(fi, d) + rng.uniform(0.0f, 1.0f) * noise;
    }

    code = findNeighbors( points, results );

    if( code == cvtest::TS::OK )
    {
        int correctMatches = 0;
        for( int pi = 0; pi < pointsCount; pi++ )
        {
            if( fmap[pi] == results.at<int>(pi, 0) )
                correctMatches++;
        }

        double correctPerc = correctMatches / (double)pointsCount;
        if (correctPerc < .75)
        {
            ts->printf( cvtest::TS::LOG, "correct_perc = %d\n", correctPerc );
            code = cvtest::TS::FAIL_BAD_ACCURACY;
        }
    }

    return code;
}

void NearestNeighborTest::run( int /*start_from*/ ) {
    int code = cvtest::TS::OK, tempCode;
    Mat desc( featuresCount, dims, CV_32FC1 );
    randu( desc, Scalar(minValue), Scalar(maxValue) );

    createModel( desc );
    
    tempCode = checkGetPoins( desc );
    if( tempCode != cvtest::TS::OK )
    {
        ts->printf( cvtest::TS::LOG, "bad accuracy of GetPoints \n" );
        code = tempCode;
    }

    tempCode = checkFindBoxed();
    if( tempCode != cvtest::TS::OK )
    {
        ts->printf( cvtest::TS::LOG, "bad accuracy of FindBoxed \n" );
        code = tempCode;
    }

    tempCode = checkFind( desc );
    if( tempCode != cvtest::TS::OK )
    {
        ts->printf( cvtest::TS::LOG, "bad accuracy of Find \n" );
        code = tempCode;
    }
    
    releaseModel();
    
    ts->set_failed_test_info( code );
}

//--------------------------------------------------------------------------------
class CV_LSHTest : public NearestNeighborTest
{
public:
    CV_LSHTest() {}
protected:
    virtual void createModel( const Mat& data );
    virtual int findNeighbors( Mat& points, Mat& neighbors );
    virtual void releaseModel();
    struct CvLSH* lsh;
    CvMat desc;
};

void CV_LSHTest::createModel( const Mat& data )
{
    desc = data;
    lsh = cvCreateMemoryLSH( data.cols, data.rows, 70, 20, CV_32FC1 );
    cvLSHAdd( lsh, &desc );
}

int CV_LSHTest::findNeighbors( Mat& points, Mat& neighbors )
{
    const int emax = 20;
    Mat dist( points.rows, neighbors.cols, CV_64FC1);
    CvMat _dist = dist, _points = points, _neighbors = neighbors;
    cvLSHQuery( lsh, &_points, &_neighbors, &_dist, neighbors.cols, emax );
    return cvtest::TS::OK;
}

void CV_LSHTest::releaseModel()
{
    cvReleaseLSH( &lsh );
}

//--------------------------------------------------------------------------------
class CV_FeatureTreeTest_C : public NearestNeighborTest
{
public:
    CV_FeatureTreeTest_C() {}
protected:
    virtual int findNeighbors( Mat& points, Mat& neighbors );
    virtual void releaseModel();
    CvFeatureTree* tr;
    CvMat desc;
};

int CV_FeatureTreeTest_C::findNeighbors( Mat& points, Mat& neighbors )
{
    const int emax = 20;
    Mat dist( points.rows, neighbors.cols, CV_64FC1);
    CvMat _dist = dist, _points = points, _neighbors = neighbors;
    cvFindFeatures( tr, &_points, &_neighbors, &_dist, neighbors.cols, emax );
    return cvtest::TS::OK;
}

void CV_FeatureTreeTest_C::releaseModel()
{
    cvReleaseFeatureTree( tr );
}

//--------------------------------------
class CV_SpillTreeTest_C : public CV_FeatureTreeTest_C
{
public:
    CV_SpillTreeTest_C() {}
protected:
    virtual void createModel( const Mat& data );
};

void CV_SpillTreeTest_C::createModel( const Mat& data )
{
    desc = data;
    tr = cvCreateSpillTree( &desc );
}

//--------------------------------------
class CV_KDTreeTest_C : public CV_FeatureTreeTest_C
{
public:
    CV_KDTreeTest_C() {}
protected:
    virtual void createModel( const Mat& data );
    virtual int checkFindBoxed();
};

void CV_KDTreeTest_C::createModel( const Mat& data )
{
    desc = data;
    tr = cvCreateKDTree( &desc );
}

int CV_KDTreeTest_C::checkFindBoxed()
{
    Mat min(1, dims, CV_32FC1 ), max(1, dims, CV_32FC1 ), indices( 1, 1, CV_32SC1 );
    float l = minValue, r = maxValue;
    min.setTo(Scalar(l)), max.setTo(Scalar(r));
    CvMat _min = min, _max = max, _indices = indices;
    // TODO check indices
    if( cvFindFeaturesBoxed( tr, &_min, &_max, &_indices ) != featuresCount )
        return cvtest::TS::FAIL_BAD_ACCURACY;
    return cvtest::TS::OK;
}


TEST(Features2d_LSH, regression) { CV_LSHTest test; test.safe_run(); }
TEST(Features2d_SpillTree, regression) { CV_SpillTreeTest_C test; test.safe_run(); }
TEST(Features2d_KDTree_C, regression) { CV_KDTreeTest_C test; test.safe_run(); }
