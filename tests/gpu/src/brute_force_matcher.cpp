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

#include "gputest.hpp"
#include <algorithm>
#include <iterator>

using namespace cv;
using namespace cv::gpu;
using namespace std;

class CV_GpuBruteForceMatcherTest : public CvTest
{
public:
    CV_GpuBruteForceMatcherTest() :
        CvTest( "GPU-BruteForceMatcher", "BruteForceMatcher" )
    {
    }

protected:
    virtual void run(int);
    
    void emptyDataTest();
    void dataTest(int dim);
    
    void generateData(GpuMat& query, GpuMat& train, int dim);

    void matchTest(const GpuMat& query, const GpuMat& train);
    void knnMatchTest(const GpuMat& query, const GpuMat& train);
    void radiusMatchTest(const GpuMat& query, const GpuMat& train);

private:
    BruteForceMatcher_GPU< L2<float> > dmatcher;

    static const int queryDescCount = 300; // must be even number because we split train data in some cases in two
    static const int countFactor = 4; // do not change it
};

void CV_GpuBruteForceMatcherTest::emptyDataTest()
{
    GpuMat queryDescriptors, trainDescriptors, mask;
    vector<GpuMat> trainDescriptorCollection, masks;
    vector<DMatch> matches;
    vector< vector<DMatch> > vmatches;

    try
    {
        dmatcher.match(queryDescriptors, trainDescriptors, matches, mask);
    }
    catch(...)
    {
        ts->printf( CvTS::LOG, "match() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher.knnMatch(queryDescriptors, trainDescriptors, vmatches, 2, mask);
    }
    catch(...)
    {
        ts->printf( CvTS::LOG, "knnMatch() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher.radiusMatch(queryDescriptors, trainDescriptors, vmatches, 10.f, mask);
    }
    catch(...)
    {
        ts->printf( CvTS::LOG, "radiusMatch() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher.add(trainDescriptorCollection);
    }
    catch(...)
    {
        ts->printf( CvTS::LOG, "add() on empty descriptors must not generate exception.\n" );
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher.match(queryDescriptors, matches, masks);
    }
    catch(...)
    {
        ts->printf( CvTS::LOG, "match() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher.knnMatch(queryDescriptors, vmatches, 2, masks);
    }
    catch(...)
    {
        ts->printf( CvTS::LOG, "knnMatch() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher.radiusMatch( queryDescriptors, vmatches, 10.f, masks );
    }
    catch(...)
    {
        ts->printf( CvTS::LOG, "radiusMatch() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    }

}

void CV_GpuBruteForceMatcherTest::generateData( GpuMat& queryGPU, GpuMat& trainGPU, int dim )
{
    Mat query, train;
    RNG rng(*ts->get_rng());

    // Generate query descriptors randomly.
    // Descriptor vector elements are integer values.
    Mat buf( queryDescCount, dim, CV_32SC1 );
    rng.fill( buf, RNG::UNIFORM, Scalar::all(0), Scalar(3) );
    buf.convertTo( query, CV_32FC1 );

    // Generate train decriptors as follows:
    // copy each query descriptor to train set countFactor times
    // and perturb some one element of the copied descriptors in
    // in ascending order. General boundaries of the perturbation
    // are (0.f, 1.f).
    train.create( query.rows*countFactor, query.cols, CV_32FC1 );
    float step = 1.f / countFactor;
    for( int qIdx = 0; qIdx < query.rows; qIdx++ )
    {
        Mat queryDescriptor = query.row(qIdx);
        for( int c = 0; c < countFactor; c++ )
        {
            int tIdx = qIdx * countFactor + c;
            Mat trainDescriptor = train.row(tIdx);
            queryDescriptor.copyTo( trainDescriptor );
            int elem = rng(dim);
            float diff = rng.uniform( step*c, step*(c+1) );
            trainDescriptor.at<float>(0, elem) += diff;
        }
    }

    queryGPU.upload(query);
    trainGPU.upload(train);
}

void CV_GpuBruteForceMatcherTest::matchTest( const GpuMat& query, const GpuMat& train )
{
    dmatcher.clear();

    // test const version of match()
    {
        vector<DMatch> matches;
        dmatcher.match( query, train, matches );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(CvTS::LOG, "Incorrect matches count while test match() function (1).\n");
            ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                DMatch match = matches[i];
                if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0) )
                    badCount++;
            }
            if (badCount > 0)
            {
                ts->printf( CvTS::LOG, "%f - too large bad matches part while test match() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test version of match() with add()
    {
        vector<DMatch> matches;
        // make add() twice to test such case
        dmatcher.add( vector<GpuMat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher.add( vector<GpuMat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<GpuMat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = GpuMat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher.match( query, matches, masks );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(CvTS::LOG, "Incorrect matches count while test match() function (2).\n");
            ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                DMatch match = matches[i];
                int shift = dmatcher.isMaskSupported() ? 1 : 0;
                {
                    if( i < queryDescCount/2 )
                    {
                        if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + shift) || (match.imgIdx != 0) )
                            badCount++;
                    }
                    else
                    {
                        if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + shift) || (match.imgIdx != 1) )
                            badCount++;
                    }
                }
            }
            if (badCount > 0)
            {
                ts->printf( CvTS::LOG, "%f - too large bad matches part while test match() function (2).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
            }
        }
    }
}

void CV_GpuBruteForceMatcherTest::knnMatchTest( const GpuMat& query, const GpuMat& train )
{
    dmatcher.clear();

    // test const version of knnMatch()
    {
        const int knn = 3;

        vector< vector<DMatch> > matches;
        dmatcher.knnMatch( query, train, matches, knn );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(CvTS::LOG, "Incorrect matches count while test knnMatch() function (1).\n");
            ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != knn )
                    badCount++;
                else
                {
                    int localBadCount = 0;
                    for( int k = 0; k < knn; k++ )
                    {
                        DMatch match = matches[i][k];
                        if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor+k) || (match.imgIdx != 0) )
                            localBadCount++;
                    }
                    badCount += localBadCount > 0 ? 1 : 0;
                }
            }
            if (badCount > 0)
            {
                ts->printf( CvTS::LOG, "%f - too large bad matches part while test knnMatch() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test version of knnMatch() with add()
    {
        const int knn = 2;
        vector<vector<DMatch> > matches;
        // make add() twice to test such case
        dmatcher.add( vector<GpuMat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher.add( vector<GpuMat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<GpuMat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = GpuMat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher.knnMatch( query, matches, knn, masks );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(CvTS::LOG, "Incorrect matches count while test knnMatch() function (2).\n");
            ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            int shift = dmatcher.isMaskSupported() ? 1 : 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != knn )
                    badCount++;
                else
                {
                    int localBadCount = 0;
                    for( int k = 0; k < knn; k++ )
                    {
                        DMatch match = matches[i][k];
                        {
                            if( i < queryDescCount/2 )
                            {
                                if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + k + shift) ||
                                    (match.imgIdx != 0) )
                                    localBadCount++;
                            }
                            else
                            {
                                if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + k + shift) ||
                                    (match.imgIdx != 1) )
                                    localBadCount++;
                            }
                        }
                    }
                    badCount += localBadCount > 0 ? 1 : 0;
                }
            }
            if (badCount > 0)
            {
                ts->printf( CvTS::LOG, "%f - too large bad matches part while test knnMatch() function (2).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
            }
        }
    }
}

void CV_GpuBruteForceMatcherTest::radiusMatchTest( const GpuMat& query, const GpuMat& train )
{
    bool atomics_ok = TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS);
    if (!atomics_ok)
    {
        ts->printf(CvTS::CONSOLE, "\nCode and device atomics support is required for radiusMatch (CC >= 1.1)");
        ts->set_failed_test_info(CvTS::FAIL_GENERIC);
        return;
    }

    dmatcher.clear();
    // test const version of match()
    {
        const float radius = 1.f/countFactor;
        vector< vector<DMatch> > matches;
        dmatcher.radiusMatch( query, train, matches, radius );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(CvTS::LOG, "Incorrect matches count while test radiusMatch() function (1).\n");
            ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != 1 )
                    badCount++;
                else
                {
                    DMatch match = matches[i][0];
                    if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0) )
                        badCount++;
                }
            }
            if (badCount > 0)
            {
                ts->printf( CvTS::LOG, "%f - too large bad matches part while test radiusMatch() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test version of match() with add()
    {
        int n = 3;
        const float radius = 1.f/countFactor * n;
        vector< vector<DMatch> > matches;
        // make add() twice to test such case
        dmatcher.add( vector<GpuMat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher.add( vector<GpuMat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<GpuMat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = GpuMat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher.radiusMatch( query, matches, radius, masks );

        int curRes = CvTS::OK;
        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(CvTS::LOG, "Incorrect matches count while test radiusMatch() function (1).\n");
            ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
        }

        int badCount = 0;
        int shift = dmatcher.isMaskSupported() ? 1 : 0;
        int needMatchCount = dmatcher.isMaskSupported() ? n-1 : n;
        for( size_t i = 0; i < matches.size(); i++ )
        {
            if( (int)matches[i].size() != needMatchCount )
                badCount++;
            else
            {
                int localBadCount = 0;
                for( int k = 0; k < needMatchCount; k++ )
                {
                    DMatch match = matches[i][k];
                    {
                        if( i < queryDescCount/2 )
                        {
                            if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + k + shift) ||
                                (match.imgIdx != 0) )
                                localBadCount++;
                        }
                        else
                        {
                            if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + k + shift) ||
                                (match.imgIdx != 1) )
                                localBadCount++;
                        }
                    }
                }
                badCount += localBadCount > 0 ? 1 : 0;
            }
        }

        if (badCount > 0)
        {
            curRes = CvTS::FAIL_INVALID_OUTPUT;
            ts->printf( CvTS::LOG, "%f - too large bad matches part while test radiusMatch() function (2).\n",
                        (float)badCount/(float)queryDescCount );
            ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
        }
    }
}

void CV_GpuBruteForceMatcherTest::dataTest(int dim)
{
    GpuMat query, train;
    generateData(query, train, dim);

    matchTest(query, train);
    knnMatchTest(query, train);
    radiusMatchTest(query, train);

    dmatcher.clear();
}

void CV_GpuBruteForceMatcherTest::run(int)
{
    try
    {
        emptyDataTest();

        dataTest(50);
        dataTest(64);
        dataTest(100);
        dataTest(128);
        dataTest(200);
        dataTest(256);
        dataTest(300);
    }
    catch(cv::Exception& e)
    {
        if (!check_and_treat_gpu_exception(e, ts))
            throw; 
        return;
    }
}

CV_GpuBruteForceMatcherTest CV_GpuBruteForceMatcher_test;
