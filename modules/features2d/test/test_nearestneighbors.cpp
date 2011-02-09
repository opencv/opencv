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

//--------------------------------------------------------------------------------
class CV_KDTreeTest_CPP : public NearestNeighborTest
{
public:
    CV_KDTreeTest_CPP() {}
protected:
    virtual void createModel( const Mat& data );
    virtual int checkGetPoins( const Mat& data );
    virtual int findNeighbors( Mat& points, Mat& neighbors );
    virtual int checkFindBoxed();
    virtual void releaseModel();
    KDTree* tr;
};


void CV_KDTreeTest_CPP::createModel( const Mat& data )
{
    tr = new KDTree( data, false );
}

int CV_KDTreeTest_CPP::checkGetPoins( const Mat& data )
{
    Mat res1( data.size(), data.type() ),
        res2( data.size(), data.type() ),
        res3( data.size(), data.type() );
    Mat idxs( 1, data.rows, CV_32SC1 );
    for( int pi = 0; pi < data.rows; pi++ )
    {
        idxs.at<int>(0, pi) = pi;
        // 1st way
        const float* point = tr->getPoint(pi);
        for( int di = 0; di < data.cols; di++ )
            res1.at<float>(pi, di) = point[di];
    }
    // 2nd way
    tr->getPoints( idxs.ptr<int>(0), data.rows, res2 );

    // 3d way
    tr->getPoints( idxs, res3 );

    if( norm( res1, data, NORM_L1) != 0 ||
        norm( res2, data, NORM_L1) != 0 ||
        norm( res3, data, NORM_L1) != 0)
        return cvtest::TS::FAIL_BAD_ACCURACY;
    return cvtest::TS::OK;
}

int CV_KDTreeTest_CPP::checkFindBoxed()
{
    vector<float> min( dims, minValue), max(dims, maxValue);
    vector<int> indices;
    tr->findOrthoRange( &min[0], &max[0], &indices );
    // TODO check indices
    if( (int)indices.size() != featuresCount)
        return cvtest::TS::FAIL_BAD_ACCURACY;
    return cvtest::TS::OK;
}

int CV_KDTreeTest_CPP::findNeighbors( Mat& points, Mat& neighbors )
{
    const int emax = 20;
    Mat neighbors2( neighbors.size(), CV_32SC1 );
    int j;
    vector<float> min(points.cols, minValue);
    vector<float> max(points.cols, maxValue);
    for( int pi = 0; pi < points.rows; pi++ )
    {
        // 1st way
        tr->findNearest( points.ptr<float>(pi), neighbors.cols, emax, neighbors.ptr<int>(pi) );

        // 2nd way
        vector<int> neighborsIdx2( neighbors2.cols, 0 );
        tr->findNearest( points.ptr<float>(pi), neighbors2.cols, emax, &neighborsIdx2 );
        vector<int>::const_iterator it2 = neighborsIdx2.begin();
        for( j = 0; it2 != neighborsIdx2.end(); ++it2, j++ )
            neighbors2.at<int>(pi,j) = *it2;
    }

    // compare results
    if( norm( neighbors, neighbors2, NORM_L1 ) != 0 )
        return cvtest::TS::FAIL_BAD_ACCURACY;

    return cvtest::TS::OK;
}

void CV_KDTreeTest_CPP::releaseModel()
{
    delete tr;
}

//--------------------------------------------------------------------------------
class CV_FlannTest : public NearestNeighborTest
{
public:
    CV_FlannTest() {}
protected:
    void createIndex( const Mat& data, const IndexParams& params );
    int knnSearch( Mat& points, Mat& neighbors );
    int radiusSearch( Mat& points, Mat& neighbors );
    virtual void releaseModel();
    Index* index;
};

void CV_FlannTest::createIndex( const Mat& data, const IndexParams& params )
{
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
    if( norm( neighbors, neighbors1, NORM_L1 ) != 0 )
        return cvtest::TS::FAIL_BAD_ACCURACY;

    return cvtest::TS::OK;
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
        index->radiusSearch( p, n, dist, radius, SearchParams() );

        // 2nd way
        float* fltPtr = points.ptr<float>(i);
        vector<float> query( fltPtr, fltPtr + points.cols );
        vector<int> indices( neighbors1.cols, 0 );
        vector<float> dists( dist.cols, 0 );
        index->radiusSearch( query, indices, dists, radius, SearchParams() );
        vector<int>::const_iterator it = indices.begin();
        for( j = 0; it != indices.end(); ++it, j++ )
            neighbors1.at<int>(i,j) = *it;
    }
    // compare results
    if( norm( neighbors, neighbors1, NORM_L1 ) != 0 )
        return cvtest::TS::FAIL_BAD_ACCURACY;

    return cvtest::TS::OK;
}

void CV_FlannTest::releaseModel()
{
    delete index;
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
        default: assert(0);
    }
    char filename[50];
    tmpnam( filename );
    if(filename[0] == '\\') filename[0] = '_';
    index->save( filename );
    
    createIndex( data, SavedIndexParams(filename));
    remove( filename );
}

TEST(Features2d_LSH, regression) { CV_LSHTest test; test.safe_run(); }
TEST(Features2d_SpillTree, regression) { CV_SpillTreeTest_C test; test.safe_run(); }
TEST(Features2d_KDTree_C, regression) { CV_KDTreeTest_C test; test.safe_run(); }
TEST(Features2d_KDTree_CPP, regression) { CV_KDTreeTest_CPP test; test.safe_run(); }
TEST(Features2d_FLANN_Linear, regression) { CV_FlannLinearIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_KMeans, regression) { CV_FlannKMeansIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_KDTree, regression) { CV_FlannKDTreeIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_Composite, regression) { CV_FlannCompositeIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_Auto, regression) { CV_FlannAutotunedIndexTest test; test.safe_run(); }
TEST(Features2d_FLANN_Saved, regression) { CV_FlannSavedIndexTest test; test.safe_run(); }
