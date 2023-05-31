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
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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

namespace opencv_test { namespace {

#ifdef HAVE_OPENCV_FLANN
using namespace cv::flann;
#endif

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
    virtual int checkGetPoints( const Mat& data );
    virtual int checkFindBoxed();
    virtual int checkFind( const Mat& data );
    virtual void releaseModel() = 0;
};

int NearestNeighborTest::checkGetPoints( const Mat& )
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
        EXPECT_GE(correctPerc, .75) << "correctMatches=" << correctMatches << " pointsCount=" << pointsCount;
    }

    return code;
}

void NearestNeighborTest::run( int /*start_from*/ ) {
    int code = cvtest::TS::OK, tempCode;
    Mat desc( featuresCount, dims, CV_32FC1 );
    ts->get_rng().fill( desc, RNG::UNIFORM, minValue, maxValue );

    createModel( desc.clone() );  // .clone() is used to simulate dangling pointers problem: https://github.com/opencv/opencv/issues/17553

    tempCode = checkGetPoints( desc );
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

    if (::testing::Test::HasFailure()) code = cvtest::TS::FAIL_BAD_ACCURACY;
    ts->set_failed_test_info( code );
}

//--------------------------------------------------------------------------------
#ifdef HAVE_OPENCV_FLANN

class CV_FlannTest : public NearestNeighborTest
{
public:
    CV_FlannTest() : NearestNeighborTest(), index(NULL) { }
protected:
    void createIndex( const Mat& data, const IndexParams& params );
    int knnSearch( Mat& points, Mat& neighbors );
    int radiusSearch( Mat& points, Mat& neighbors );
    virtual void releaseModel();
    Index* index;
};

void CV_FlannTest::createIndex( const Mat& data, const IndexParams& params )
{
    // release previously allocated index
    releaseModel();

    index = new Index( data, params );
}

int CV_FlannTest::knnSearch( Mat& points, Mat& neighbors )
{
    Mat dist( points.rows, neighbors.cols, CV_32FC1);
    int knn = 1, j;

    // 1st way
    index->knnSearch( points, neighbors, dist, knn, SearchParams() );

    // 2nd way
    Mat neighbors1( neighbors.size(), CV_32SC1 );
    for( int i = 0; i < points.rows; i++ )
    {
        float* fltPtr = points.ptr<float>(i);
        vector<float> query( fltPtr, fltPtr + points.cols );
        vector<int> indices( neighbors1.cols, 0 );
        vector<float> dists( dist.cols, 0 );
        index->knnSearch( query, indices, dists, knn, SearchParams() );
        vector<int>::const_iterator it = indices.begin();
        for( j = 0; it != indices.end(); ++it, j++ )
            neighbors1.at<int>(i,j) = *it;
    }

    // compare results
    EXPECT_LE(cvtest::norm(neighbors, neighbors1, NORM_L1), 0);

    return ::testing::Test::HasFailure() ? cvtest::TS::FAIL_BAD_ACCURACY : cvtest::TS::OK;
}

int CV_FlannTest::radiusSearch( Mat& points, Mat& neighbors )
{
    Mat dist( 1, neighbors.cols, CV_32FC1);
    Mat neighbors1( neighbors.size(), CV_32SC1 );
    float radius = 10.0f;
    int j;

    // radiusSearch can only search one feature at a time for range search
    for( int i = 0; i < points.rows; i++ )
    {
        // 1st way
        Mat p( 1, points.cols, CV_32FC1, points.ptr<float>(i) ),
            n( 1, neighbors.cols, CV_32SC1, neighbors.ptr<int>(i) );
        index->radiusSearch( p, n, dist, radius, neighbors.cols, SearchParams() );

        // 2nd way
        float* fltPtr = points.ptr<float>(i);
        vector<float> query( fltPtr, fltPtr + points.cols );
        vector<int> indices( neighbors1.cols, 0 );
        vector<float> dists( dist.cols, 0 );
        index->radiusSearch( query, indices, dists, radius, neighbors.cols, SearchParams() );
        vector<int>::const_iterator it = indices.begin();
        for( j = 0; it != indices.end(); ++it, j++ )
            neighbors1.at<int>(i,j) = *it;
    }

    // compare results
    EXPECT_LE(cvtest::norm(neighbors, neighbors1, NORM_L1), 0);

    return ::testing::Test::HasFailure() ? cvtest::TS::FAIL_BAD_ACCURACY : cvtest::TS::OK;
}

void CV_FlannTest::releaseModel()
{
    if (index)
    {
        delete index;
        index = NULL;
    }
}

//---------------------------------------
class CV_FlannLinearIndexTest : public CV_FlannTest
{
public:
    CV_FlannLinearIndexTest() {}
protected:
    virtual void createModel( const Mat& data ) { createIndex( data, LinearIndexParams() ); }
    virtual int findNeighbors( Mat& points, Mat& neighbors ) { return knnSearch( points, neighbors ); }
};

//---------------------------------------
class CV_FlannKMeansIndexTest : public CV_FlannTest
{
public:
    CV_FlannKMeansIndexTest() {}
protected:
    virtual void createModel( const Mat& data ) { createIndex( data, KMeansIndexParams() ); }
    virtual int findNeighbors( Mat& points, Mat& neighbors ) { return radiusSearch( points, neighbors ); }
};

//---------------------------------------
class CV_FlannKDTreeIndexTest : public CV_FlannTest
{
public:
    CV_FlannKDTreeIndexTest() {}
protected:
    virtual void createModel( const Mat& data ) { createIndex( data, KDTreeIndexParams() ); }
    virtual int findNeighbors( Mat& points, Mat& neighbors ) { return radiusSearch( points, neighbors ); }
};

//----------------------------------------
class CV_FlannCompositeIndexTest : public CV_FlannTest
{
public:
    CV_FlannCompositeIndexTest() {}
protected:
    virtual void createModel( const Mat& data ) { createIndex( data, CompositeIndexParams() ); }
    virtual int findNeighbors( Mat& points, Mat& neighbors ) { return knnSearch( points, neighbors ); }
};

//----------------------------------------
class CV_FlannAutotunedIndexTest : public CV_FlannTest
{
public:
    CV_FlannAutotunedIndexTest() {}
protected:
    virtual void createModel( const Mat& data ) { createIndex( data, AutotunedIndexParams() ); }
    virtual int findNeighbors( Mat& points, Mat& neighbors ) { return knnSearch( points, neighbors ); }
};
//----------------------------------------
class CV_FlannSavedIndexTest : public CV_FlannTest
{
public:
    CV_FlannSavedIndexTest() {}
protected:
    virtual void createModel( const Mat& data );
    virtual int findNeighbors( Mat& points, Mat& neighbors ) { return knnSearch( points, neighbors ); }
};

void CV_FlannSavedIndexTest::createModel(const cv::Mat &data)
{
    switch ( cvtest::randInt(ts->get_rng()) % 2 )
    {
        //case 0: createIndex( data, LinearIndexParams() ); break; // nothing to save for linear search
        case 0: createIndex( data, KMeansIndexParams() ); break;
        case 1: createIndex( data, KDTreeIndexParams() ); break;
        //case 2: createIndex( data, CompositeIndexParams() ); break; // nothing to save for linear search
        //case 2: createIndex( data, AutotunedIndexParams() ); break; // possible linear index !
        default: CV_Assert(0);
    }
    string filename = tempfile();
    index->save( filename );

    createIndex( data, SavedIndexParams(filename.c_str()));
    remove( filename.c_str() );
}

TEST(Features2d_FLANN_Linear, regression) { CV_FlannLinearIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_KMeans, regression) { CV_FlannKMeansIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_KDTree, regression) { CV_FlannKDTreeIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_Composite, regression) { CV_FlannCompositeIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_Auto, regression) { CV_FlannAutotunedIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_Saved, regression) { CV_FlannSavedIndexTest test; test.safe_run(); }

#endif

}} // namespace
