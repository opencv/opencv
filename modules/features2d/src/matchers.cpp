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

#include "precomp.hpp"

#ifdef HAVE_EIGEN2
#include <Eigen/Array>
#endif

using namespace std;

namespace cv
{

Mat windowedMatchingMask( const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                          float maxDeltaX, float maxDeltaY )
{
    if( keypoints1.empty() || keypoints2.empty() )
        return Mat();

    Mat mask( keypoints1.size(), keypoints2.size(), CV_8UC1 );
    for( size_t i = 0; i < keypoints1.size(); i++ )
    {
        for( size_t j = 0; j < keypoints2.size(); j++ )
        {
            Point2f diff = keypoints2[j].pt - keypoints1[i].pt;
            mask.at<uchar>(i, j) = std::abs(diff.x) < maxDeltaX && std::abs(diff.y) < maxDeltaY;
        }
    }
    return mask;
}

/****************************************************************************************\
*                                      DescriptorMatcher                                 *
\****************************************************************************************/
void DescriptorMatcher::DescriptorCollection::set( const vector<Mat>& descCollection )
{
    clear();

    size_t imageCount = descCollection.size();
    CV_Assert( imageCount > 0 );

    startIdxs.resize( imageCount );

    int dim = -1;
    int type = -1;
    startIdxs[0] = 0;
    for( size_t i = 1; i < imageCount; i++ )
    {
        int s = 0;
        if( !descCollection[i-1].empty() )
        {
            dim = descCollection[i-1].cols;
            type = descCollection[i-1].type();
            s = descCollection[i-1].rows;
        }
        startIdxs[i] = startIdxs[i-1] + s;
    }
    if( imageCount == 1 )
    {
        if( descCollection[0].empty() ) return;

        dim = descCollection[0].cols;
        type = descCollection[0].type();
    }
    assert( dim > 0 );

    int count = startIdxs[imageCount-1] + descCollection[imageCount-1].rows;

    if( count > 0 )
    {
        dmatrix.create( count, dim, type );
        for( size_t i = 0; i < imageCount; i++ )
        {
            if( !descCollection[i].empty() )
            {
                CV_Assert( descCollection[i].cols == dim && descCollection[i].type() == type );
                Mat m = dmatrix.rowRange( startIdxs[i], startIdxs[i] + descCollection[i].rows );
                descCollection[i].copyTo(m);
            }
        }
    }
}

void DescriptorMatcher::DescriptorCollection::clear()
{
    startIdxs.clear();
    dmatrix.release();
}

const Mat DescriptorMatcher::DescriptorCollection::getDescriptor( int imgIdx, int localDescIdx ) const
{
    CV_Assert( imgIdx < (int)startIdxs.size() );
    int globalIdx = startIdxs[imgIdx] + localDescIdx;
    CV_Assert( globalIdx < (int)size() );

    return getDescriptor( globalIdx );
}

const Mat DescriptorMatcher::DescriptorCollection::getDescriptor( int globalDescIdx ) const
{
    CV_Assert( globalDescIdx < size() );
    return dmatrix.row( globalDescIdx );
}

void DescriptorMatcher::DescriptorCollection::getLocalIdx( int globalDescIdx, int& imgIdx, int& localDescIdx ) const
{
    imgIdx = -1;
    CV_Assert( globalDescIdx < size() );
    for( size_t i = 1; i < startIdxs.size(); i++ )
    {
        if( globalDescIdx < startIdxs[i] )
        {
            imgIdx = i - 1;
            break;
        }
    }
    imgIdx = imgIdx == -1 ? startIdxs.size() -1 : imgIdx;
    localDescIdx = globalDescIdx - startIdxs[imgIdx];
}

/*
 * DescriptorMatcher
 */
void convertMatches( const vector<vector<DMatch> >& knnMatches, vector<DMatch>& matches )
{
    matches.clear();
    matches.reserve( knnMatches.size() );
    for( size_t i = 0; i < knnMatches.size(); i++ )
    {
        CV_Assert( knnMatches[i].size() <= 1 );
        if( !knnMatches[i].empty() )
            matches.push_back( knnMatches[i][0] );
    }
}

void DescriptorMatcher::add( const vector<Mat>& descCollection )
{
    trainDescCollection.insert( trainDescCollection.end(), descCollection.begin(), descCollection.end() );
}

void DescriptorMatcher::clear()
{
    trainDescCollection.clear();
}

void DescriptorMatcher::match( const Mat& queryDescs, const Mat& trainDescs, vector<DMatch>& matches, const Mat& mask ) const
{
    Ptr<DescriptorMatcher> tempMatcher = cloneWithoutData();
    tempMatcher->add( vector<Mat>(1, trainDescs) );
    tempMatcher->match( queryDescs, matches, vector<Mat>(1, mask) );
}

void DescriptorMatcher::knnMatch( const Mat& queryDescs, const Mat& trainDescs, vector<vector<DMatch> >& matches, int knn,
                                  const Mat& mask, bool compactResult ) const
{
    Ptr<DescriptorMatcher> tempMatcher = cloneWithoutData();
    tempMatcher->add( vector<Mat>(1, trainDescs) );
    tempMatcher->knnMatch( queryDescs, matches, knn, vector<Mat>(1, mask), compactResult );
}

void DescriptorMatcher::radiusMatch( const Mat& queryDescs, const Mat& trainDescs, vector<vector<DMatch> >& matches, float maxDistance,
                                     const Mat& mask, bool compactResult ) const
{
    Ptr<DescriptorMatcher> tempMatcher = cloneWithoutData();
    tempMatcher->add( vector<Mat>(1, trainDescs) );
    tempMatcher->radiusMatch( queryDescs, matches, maxDistance, vector<Mat>(1, mask), compactResult );
}

void DescriptorMatcher::match( const Mat& queryDescs, vector<DMatch>& matches, const vector<Mat>& masks )
{
    vector<vector<DMatch> > knnMatches;
    knnMatch( queryDescs, knnMatches, 1, masks, true /*compactResult*/ );
    convertMatches( knnMatches, matches );
}

void DescriptorMatcher::knnMatch( const Mat& queryDescs, vector<vector<DMatch> >& matches, int knn,
                                  const vector<Mat>& masks, bool compactResult )
{
    train();
    knnMatchImpl( queryDescs, matches, knn, masks, compactResult );
}

void DescriptorMatcher::radiusMatch( const Mat& queryDescs, vector<vector<DMatch> >& matches, float maxDistance,
                                     const vector<Mat>& masks, bool compactResult )
{
    train();
    radiusMatchImpl( queryDescs, matches, maxDistance, masks, compactResult );
}

template<>
void BruteForceMatcher<L2<float> >::knnMatchImpl( const Mat& queryDescs, vector<vector<DMatch> >& matches, int knn,
                                                const vector<Mat>& masks, bool compactResult )
{
#ifndef HAVE_EIGEN2
    bfKnnMatchImpl<L2<float> >( *this, queryDescs, matches, knn, masks, compactResult );
#else
    CV_Assert( queryDescs.type() == CV_32FC1 ||  queryDescs.empty() );
    CV_Assert( masks.empty() || masks.size() == trainDescCollection.size() );

    matches.reserve(queryDescs.rows);
    size_t imgCount = trainDescCollection.size();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> e_query_t;
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > e_trainCollection(trainDescCollection.size());
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > e_trainNorms2(trainDescCollection.size());
    cv2eigen( queryDescs.t(), e_query_t);
    for( size_t i = 0; i < trainDescCollection.size(); i++ )
    {
        cv2eigen( trainDescCollection[i], e_trainCollection[i] );
        e_trainNorms2[i] = e_trainCollection[i].rowwise().squaredNorm() / 2;
    }

    vector<Eigen::Matrix<float, Eigen::Dynamic, 1> > e_allDists( imgCount ); // distances between one query descriptor and all train descriptors

    for( int qIdx = 0; qIdx < queryDescs.rows; qIdx++ )
    {
        if( maskedOut( masks, qIdx ) )
        {
            if( !compactResult ) // push empty vector
                matches.push_back( vector<DMatch>() );
        }
        else
        {
            float queryNorm2 = e_query_t.col(qIdx).squaredNorm();
            // 1. compute distances between i-th query descriptor and all train descriptors
            for( size_t iIdx = 0; iIdx < imgCount; iIdx++ )
            {
                CV_Assert( masks.empty() || masks[iIdx].empty() ||
                           ( masks[iIdx].rows == queryDescs.rows && masks[iIdx].cols == trainDescCollection[iIdx].rows &&
                             masks[iIdx].type() == CV_8UC1 ) );
                CV_Assert( trainDescCollection[iIdx].type() == CV_32FC1 ||  trainDescCollection[iIdx].empty() );
                CV_Assert( queryDescs.cols == trainDescCollection[iIdx].cols );

                e_allDists[iIdx] = e_trainCollection[iIdx] *e_query_t.col(qIdx);
                e_allDists[iIdx] -= e_trainNorms2[iIdx];

                if( !masks.empty() && !masks[iIdx].empty() )
                {
                    const uchar* maskPtr = (uchar*)masks[iIdx].ptr(qIdx);
                    for( int c = 0; c < masks[iIdx].cols; c++ )
                    {
                        if( maskPtr[c] == 0 )
                            e_allDists[iIdx](c) = std::numeric_limits<float>::min();
                    }
                }
            }

            // 2. choose knn nearest matches for query[i]
            matches.push_back( vector<DMatch>() );
            vector<vector<DMatch> >::reverse_iterator curMatches = matches.rbegin();
            for( int k = 0; k < knn; k++ )
            {
                float totalMaxCoeff = std::numeric_limits<float>::min();
                int bestTrainIdx = -1, bestImgIdx = -1;
                for( size_t iIdx = 0; iIdx < imgCount; iIdx++ )
                {
                    int loc;
                    float curMaxCoeff = e_allDists[iIdx].maxCoeff( &loc );
                    if( curMaxCoeff > totalMaxCoeff )
                    {
                        totalMaxCoeff = curMaxCoeff;
                        bestTrainIdx = loc;
                        bestImgIdx = iIdx;
                    }
                }
                if( bestTrainIdx == -1 )
                    break;

                e_allDists[bestImgIdx](bestTrainIdx) = std::numeric_limits<float>::min();
                curMatches->push_back( DMatch(qIdx, bestTrainIdx, bestImgIdx, sqrt((-2)*totalMaxCoeff + queryNorm2)) );
            }
            std::sort( curMatches->begin(), curMatches->end() );
        }
    }
#endif
}

template<>
void BruteForceMatcher<L2<float> >::radiusMatchImpl( const Mat& queryDescs, vector<vector<DMatch> >& matches, float maxDistance,
                                                     const vector<Mat>& masks, bool compactResult )
{
#ifndef HAVE_EIGEN2
    bfRadiusMatchImpl<L2<float> >( *this, queryDescs, matches, maxDistance, masks, compactResult );
#else
    CV_Assert( queryDescs.type() == CV_32FC1 ||  queryDescs.empty() );
    CV_Assert( masks.empty() || masks.size() == trainDescCollection.size() );

    matches.reserve(queryDescs.rows);
    size_t imgCount = trainDescCollection.size();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> e_query_t;
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > e_trainCollection(trainDescCollection.size());
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > e_trainNorms2(trainDescCollection.size());
    cv2eigen( queryDescs.t(), e_query_t);
    for( size_t i = 0; i < trainDescCollection.size(); i++ )
    {
        cv2eigen( trainDescCollection[i], e_trainCollection[i] );
        e_trainNorms2[i] = e_trainCollection[i].rowwise().squaredNorm() / 2;
    }

    vector<Eigen::Matrix<float, Eigen::Dynamic, 1> > e_allDists( imgCount ); // distances between one query descriptor and all train descriptors

    for( int qIdx = 0; qIdx < queryDescs.rows; qIdx++ )
    {
        if( maskedOut( masks, qIdx ) )
        {
            if( !compactResult ) // push empty vector
                matches.push_back( vector<DMatch>() );
        }
        else
        {
            float queryNorm2 = e_query_t.col(qIdx).squaredNorm();
            // 1. compute distances between i-th query descriptor and all train descriptors
            for( size_t iIdx = 0; iIdx < imgCount; iIdx++ )
            {
                CV_Assert( masks.empty() || masks[iIdx].empty() ||
                           ( masks[iIdx].rows == queryDescs.rows && masks[iIdx].cols == trainDescCollection[iIdx].rows &&
                             masks[iIdx].type() == CV_8UC1 ) );
                CV_Assert( trainDescCollection[iIdx].type() == CV_32FC1 ||  trainDescCollection[iIdx].empty() );
                CV_Assert( queryDescs.cols == trainDescCollection[iIdx].cols );

                e_allDists[iIdx] = e_trainCollection[iIdx] *e_query_t.col(qIdx);
                e_allDists[iIdx] -= e_trainNorms2[iIdx];
            }

            matches.push_back( vector<DMatch>() );
            vector<vector<DMatch> >::reverse_iterator curMatches = matches.rbegin();
            for( size_t iIdx = 0; iIdx < imgCount; iIdx++ )
            {
                assert( e_allDists[iIdx].rows() == trainDescCollection[iIdx].rows );
                for( int tIdx = 0; tIdx < e_allDists[iIdx].rows(); tIdx++ )
                {
                    if( masks.empty() || possibleMatch(masks[iIdx], qIdx, tIdx) )
                    {
                        float d =  sqrt((-2)*e_allDists[iIdx](tIdx) + queryNorm2);
                        if( d < maxDistance )
                            curMatches->push_back( DMatch( qIdx, tIdx, iIdx, d ) );
                    }
                }
            }
            std::sort( curMatches->begin(), curMatches->end() );
        }
    }
#endif
}

/*
 * Flann based matcher
 */
FlannBasedMatcher::FlannBasedMatcher( const Ptr<flann::IndexParams>& _indexParams, const Ptr<flann::SearchParams>& _searchParams )
    : indexParams(_indexParams), searchParams(_searchParams), addedDescCount(0)
{
    CV_Assert( !_indexParams.empty() );
    CV_Assert( !_searchParams.empty() );
}

void FlannBasedMatcher::add( const vector<Mat>& descCollection )
{
    DescriptorMatcher::add( descCollection );
    for( size_t i = 0; i < descCollection.size(); i++ )
    {
        addedDescCount += descCollection[i].rows;
    }
}

void FlannBasedMatcher::clear()
{
    DescriptorMatcher::clear();

    mergedDescriptors.clear();
    flannIndex.release();

    addedDescCount = 0;
}

void FlannBasedMatcher::train()
{
    if( flannIndex.empty() || mergedDescriptors.size() < addedDescCount )
    {
        mergedDescriptors.set( trainDescCollection );
        flannIndex = new flann::Index( mergedDescriptors.getDescriptors(), *indexParams );
    }
}

void FlannBasedMatcher::convertToDMatches( const DescriptorCollection& collection, const Mat& indices, const Mat& dists,
                                           vector<vector<DMatch> >& matches )
{
    matches.resize( indices.rows );
    for( int i = 0; i < indices.rows; i++ )
    {
        for( int j = 0; j < indices.cols; j++ )
        {
            int idx = indices.at<int>(i, j);
            if( idx >= 0 )
            {
                int imgIdx, trainIdx;
                collection.getLocalIdx( idx, imgIdx, trainIdx );
                matches[i].push_back( DMatch( i, trainIdx, imgIdx, std::sqrt(dists.at<float>(i,j))) );
            }
        }
    }
}

void FlannBasedMatcher::knnMatchImpl( const Mat& queryDescs, vector<vector<DMatch> >& matches, int knn,
                                      const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    Mat indices( queryDescs.rows, knn, CV_32SC1 );
    Mat dists( queryDescs.rows, knn, CV_32FC1);
    flannIndex->knnSearch( queryDescs, indices, dists, knn, *searchParams );

    convertToDMatches( mergedDescriptors, indices, dists, matches );
}

void FlannBasedMatcher::radiusMatchImpl( const Mat& queryDescs, vector<vector<DMatch> >& matches, float maxDistance,
                                         const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    const int count = mergedDescriptors.size(); // TODO do count as param?
    Mat indices( queryDescs.rows, count, CV_32SC1, Scalar::all(-1) );
    Mat dists( queryDescs.rows, count, CV_32FC1, Scalar::all(-1) );
    for( int qIdx = 0; qIdx < queryDescs.rows; qIdx++ )
    {
        Mat queryDescsRow = queryDescs.row(qIdx);
        Mat indicesRow = indices.row(qIdx);
        Mat distsRow = dists.row(qIdx);
        flannIndex->radiusSearch( queryDescsRow, indicesRow, distsRow, maxDistance*maxDistance, *searchParams );
    }

    convertToDMatches( mergedDescriptors, indices, dists, matches );
}

/*
 * Factory function for DescriptorMatcher creating
 */
Ptr<DescriptorMatcher> createDescriptorMatcher( const string& descriptorMatcherType )
{
    DescriptorMatcher* dm = 0;
    if( !descriptorMatcherType.compare( "BruteForce" ) )
    {
        dm = new BruteForceMatcher<L2<float> >();
    }
    else if( !descriptorMatcherType.compare( "BruteForce-L1" ) )
    {
        dm = new BruteForceMatcher<L1<float> >();
    }
    else if ( !descriptorMatcherType.compare( "FlannBased" ) )
    {
        dm = new FlannBasedMatcher();
    }
    else
    {
        //CV_Error( CV_StsBadArg, "unsupported descriptor matcher type");
    }

    return dm;
}

/****************************************************************************************\
*                                GenericDescriptorMatcher                                *
\****************************************************************************************/
/*
 * KeyPointCollection
 */
void GenericDescriptorMatcher::KeyPointCollection::add( const vector<Mat>& _images,
                                                        const vector<vector<KeyPoint> >& _points )
{
    CV_Assert( !_images.empty() );
    CV_Assert( _images.size() == _points.size() );

    images.insert( images.end(), _images.begin(), _images.end() );
    points.insert( points.end(), _points.begin(), _points.end() );
    for( size_t i = 0; i < _points.size(); i++ )
        size += _points[i].size();

    size_t prevSize = startIndices.size(), addSize = _images.size();
    startIndices.resize( prevSize + addSize );

    if( prevSize == 0 )
        startIndices[prevSize] = 0; //first
    else
        startIndices[prevSize] = startIndices[prevSize-1] + points[prevSize-1].size();

    for( size_t i = prevSize + 1; i < prevSize + addSize; i++ )
    {
        startIndices[i] = startIndices[i - 1] + points[i - 1].size();
    }
}

void GenericDescriptorMatcher::KeyPointCollection::clear()
{
    points.clear();
}

const KeyPoint& GenericDescriptorMatcher::KeyPointCollection::getKeyPoint( int imgIdx, int localPointIdx ) const
{
    CV_Assert( imgIdx < (int)images.size() );
    CV_Assert( localPointIdx < (int)points[imgIdx].size() );
    return points[imgIdx][localPointIdx];
}

const KeyPoint& GenericDescriptorMatcher::KeyPointCollection::getKeyPoint( int globalPointIdx ) const
{
    int imgIdx, localPointIdx;
    getLocalIdx( globalPointIdx, imgIdx, localPointIdx );
    return points[imgIdx][localPointIdx];
}

void GenericDescriptorMatcher::KeyPointCollection::getLocalIdx( int globalPointIdx, int& imgIdx, int& localPointIdx ) const
{
    imgIdx = -1;
    CV_Assert( globalPointIdx < (int)pointCount() );
    for( size_t i = 1; i < startIndices.size(); i++ )
    {
        if( globalPointIdx < startIndices[i] )
        {
            imgIdx = i - 1;
            break;
        }
    }
    imgIdx = imgIdx == -1 ? startIndices.size() -1 : imgIdx;
    localPointIdx = globalPointIdx - startIndices[imgIdx];
}

/*
 * GenericDescriptorMatcher
 */
void GenericDescriptorMatcher::add( const vector<Mat>& imgCollection,
                                    vector<vector<KeyPoint> >& pointCollection )
{
    trainPointCollection.add( imgCollection, pointCollection );
}

void GenericDescriptorMatcher::clear()
{
    trainPointCollection.clear();
}

void GenericDescriptorMatcher::classify( const Mat& queryImage, vector<KeyPoint>& queryPoints,
                                         const Mat& trainImage, vector<KeyPoint>& trainPoints ) const
{
    vector<DMatch> matches;
    match( queryImage, queryPoints, trainImage, trainPoints, matches );

    // remap keypoint indices to descriptors
    for( size_t i = 0; i < matches.size(); i++ )
        queryPoints[matches[i].queryIdx].class_id = trainPoints[matches[i].trainIdx].class_id;
}

void GenericDescriptorMatcher::classify( const Mat& queryImage, vector<KeyPoint>& queryPoints )
{
    vector<DMatch> matches;
    match( queryImage, queryPoints, matches );

    // remap keypoint indices to descriptors
    for( size_t i = 0; i < matches.size(); i++ )
        queryPoints[matches[i].queryIdx].class_id = trainPointCollection.getKeyPoint( matches[i].trainIdx, matches[i].trainIdx ).class_id;
}

void GenericDescriptorMatcher::match( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                      const Mat& trainImg, vector<KeyPoint>& trainPoints,
                                      vector<DMatch>& matches, const Mat& mask ) const
{
    Ptr<GenericDescriptorMatcher> tempMatcher = createEmptyMatcherCopy();
    vector<vector<KeyPoint> > vecTrainPoints(1, trainPoints);
    tempMatcher->add( vector<Mat>(1, trainImg), vecTrainPoints );
    tempMatcher->match( queryImg, queryPoints, matches, vector<Mat>(1, mask) );
    vecTrainPoints[0].swap( trainPoints );
}

void GenericDescriptorMatcher::knnMatch( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                         const Mat& trainImg, vector<KeyPoint>& trainPoints,
                                         vector<vector<DMatch> >& matches, int knn, const Mat& mask, bool compactResult ) const
{
    Ptr<GenericDescriptorMatcher> tempMatcher = createEmptyMatcherCopy();
    vector<vector<KeyPoint> > vecTrainPoints(1, trainPoints);
    tempMatcher->add( vector<Mat>(1, trainImg), vecTrainPoints );
    tempMatcher->knnMatch( queryImg, queryPoints, matches, knn, vector<Mat>(1, mask), compactResult );
    vecTrainPoints[0].swap( trainPoints );
}

void GenericDescriptorMatcher::radiusMatch( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                            const Mat& trainImg, vector<KeyPoint>& trainPoints,
                                            vector<vector<DMatch> >& matches, float maxDistance,
                                            const Mat& mask, bool compactResult ) const
{
    Ptr<GenericDescriptorMatcher> tempMatcher = createEmptyMatcherCopy();
    vector<vector<KeyPoint> > vecTrainPoints(1, trainPoints);
    tempMatcher->add( vector<Mat>(1, trainImg), vecTrainPoints );
    tempMatcher->radiusMatch( queryImg, queryPoints, matches, maxDistance, vector<Mat>(1, mask), compactResult );
    vecTrainPoints[0].swap( trainPoints );
}

void GenericDescriptorMatcher::match( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                      vector<DMatch>& matches, const vector<Mat>& masks )
{
    vector<vector<DMatch> > knnMatches;
    knnMatch( queryImg, queryPoints, knnMatches, 1, masks, false );
    convertMatches( knnMatches, matches );
}

void GenericDescriptorMatcher::knnMatch( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                         vector<vector<DMatch> >& matches, int knn,
                                         const vector<Mat>& masks, bool compactResult )
{
    train();
    knnMatchImpl( queryImg, queryPoints, matches, knn, masks, compactResult );
}

void GenericDescriptorMatcher::radiusMatch( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                            vector<vector<DMatch> >& matches, float maxDistance,
                                            const vector<Mat>& masks, bool compactResult )
{
    train();
    radiusMatchImpl( queryImg, queryPoints, matches, maxDistance, masks, compactResult );
}

/****************************************************************************************\
*                                OneWayDescriptorMatcher                                  *
\****************************************************************************************/
OneWayDescriptorMatcher::OneWayDescriptorMatcher( const Params& _params)
{
    initialize(_params);
}

OneWayDescriptorMatcher::~OneWayDescriptorMatcher()
{}

void OneWayDescriptorMatcher::initialize( const Params& _params, const Ptr<OneWayDescriptorBase>& _base )
{
    clear();

    if( _base.empty() )
        base = _base;

    params = _params;
}

void OneWayDescriptorMatcher::clear()
{
    GenericDescriptorMatcher::clear();

    prevTrainCount = 0;
    base->clear();
}

void OneWayDescriptorMatcher::train()
{
    if( base.empty() || prevTrainCount < (int)trainPointCollection.pointCount() )
    {
        base = new OneWayDescriptorObject( params.patchSize, params.poseCount, params.pcaFilename,
                                           params.trainPath, params.trainImagesList, params.minScale, params.maxScale, params.stepScale );

        base->Allocate( trainPointCollection.pointCount() );
        prevTrainCount = trainPointCollection.pointCount();

        const vector<vector<KeyPoint> >& points = trainPointCollection.getKeypoints();
        int count = 0;
        for( size_t i = 0; i < points.size(); i++ )
        {
            IplImage _image = trainPointCollection.getImage(i);
            for( size_t j = 0; j < points[i].size(); j++ )
                base->InitializeDescriptor( count++, &_image, points[i][j], "" );
        }

#if defined(_KDTREE)
        base->ConvertDescriptorsArrayToTree();
#endif
    }
}

void OneWayDescriptorMatcher::knnMatchImpl( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                            vector<vector<DMatch> >& matches, int knn,
                                            const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    train();

    CV_Assert( knn == 1 ); // knn > 1 unsupported because of bug in OneWayDescriptorBase for this case

    matches.resize( queryPoints.size() );
    IplImage _qimage = queryImg;
    for( size_t i = 0; i < queryPoints.size(); i++ )
    {
        int descIdx = -1, poseIdx = -1;
        float distance;
        base->FindDescriptor( &_qimage, queryPoints[i].pt, descIdx, poseIdx, distance );
        matches[i].push_back( DMatch(i, descIdx, distance) );
    }
}

void OneWayDescriptorMatcher::radiusMatchImpl( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                               vector<vector<DMatch> >& matches, float maxDistance,
                                               const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    train();

    matches.resize( queryPoints.size() );
    IplImage _qimage = queryImg;
    for( size_t i = 0; i < queryPoints.size(); i++ )
    {
        int descIdx = -1, poseIdx = -1;
        float distance;
        base->FindDescriptor( &_qimage, queryPoints[i].pt, descIdx, poseIdx, distance );
        if( distance < maxDistance )
            matches[i].push_back( DMatch(i, descIdx, distance) );
    }
}

void OneWayDescriptorMatcher::read( const FileNode &fn )
{
    base = new OneWayDescriptorObject( params.patchSize, params.poseCount, string (), string (), string (),
                                       params.minScale, params.maxScale, params.stepScale );
    base->Read (fn);
}

void OneWayDescriptorMatcher::write( FileStorage& fs ) const
{
    base->Write (fs);
}

/****************************************************************************************\
*                                  FernDescriptorMatcher                                 *
\****************************************************************************************/
FernDescriptorMatcher::Params::Params( int _nclasses, int _patchSize, int _signatureSize,
                                     int _nstructs, int _structSize, int _nviews, int _compressionMethod,
                                     const PatchGenerator& _patchGenerator ) :
    nclasses(_nclasses), patchSize(_patchSize), signatureSize(_signatureSize),
    nstructs(_nstructs), structSize(_structSize), nviews(_nviews),
    compressionMethod(_compressionMethod), patchGenerator(_patchGenerator)
{}

FernDescriptorMatcher::Params::Params( const string& _filename )
{
    filename = _filename;
}

FernDescriptorMatcher::FernDescriptorMatcher( const Params& _params )
{
    prevTrainCount = 0;
    params = _params;
    if( !params.filename.empty() )
    {
        classifier = new FernClassifier;
        FileStorage fs(params.filename, FileStorage::READ);
        if( fs.isOpened() )
            classifier->read( fs.getFirstTopLevelNode() );
    }
}

FernDescriptorMatcher::~FernDescriptorMatcher()
{}

void FernDescriptorMatcher::clear()
{
    GenericDescriptorMatcher::clear();

    classifier.release();
    prevTrainCount = 0;
}

void FernDescriptorMatcher::train()
{
    if( classifier.empty() || prevTrainCount < (int)trainPointCollection.pointCount() )
    {
        assert( params.filename.empty() );

        vector<vector<Point2f> > points( trainPointCollection.imageCount() );
        for( size_t imgIdx = 0; imgIdx < trainPointCollection.imageCount(); imgIdx++ )
            KeyPoint::convert( trainPointCollection.getKeypoints(imgIdx), points[imgIdx] );

        classifier = new FernClassifier( points, trainPointCollection.getImages(), vector<vector<int> >(), 0, // each points is a class
                                         params.patchSize, params.signatureSize, params.nstructs, params.structSize,
                                         params.nviews, params.compressionMethod, params.patchGenerator );
    }
}

void FernDescriptorMatcher::calcBestProbAndMatchIdx( const Mat& image, const Point2f& pt,
                                                     float& bestProb, int& bestMatchIdx, vector<float>& signature )
{
    (*classifier)( image, pt, signature);

    bestProb = -FLT_MAX;
    bestMatchIdx = -1;
    for( int ci = 0; ci < classifier->getClassCount(); ci++ )
    {
        if( signature[ci] > bestProb )
        {
            bestProb = signature[ci];
            bestMatchIdx = ci;
        }
    }
}

void FernDescriptorMatcher::knnMatchImpl( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                          vector<vector<DMatch> >& matches, int knn,
                                          const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    train();

    matches.resize( queryPoints.size() );
    vector<float> signature( (size_t)classifier->getClassCount() );

    for( size_t queryIdx = 0; queryIdx < queryPoints.size(); queryIdx++ )
    {
        (*classifier)( queryImg, queryPoints[queryIdx].pt, signature);

        for( int k = 0; k < knn; k++ )
        {
            DMatch bestMatch;
            size_t ci = 0;
            for( ; ci < signature.size(); ci++ )
            {
                if( -signature[ci] < bestMatch.distance )
                {
                    int imgIdx = -1, trainIdx = -1;
                    trainPointCollection.getLocalIdx( ci , imgIdx, trainIdx );
                    bestMatch = DMatch( queryIdx, trainIdx, imgIdx, -signature[ci] );
                }
            }

            if( bestMatch.trainIdx == -1 )
                break;
            signature[ci] = std::numeric_limits<float>::min();
            matches[queryIdx].push_back( bestMatch );
        }
    }
}

void FernDescriptorMatcher::radiusMatchImpl( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                             vector<vector<DMatch> >& matches, float maxDistance,
                                             const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    train();
    matches.resize( queryPoints.size() );
    vector<float> signature( (size_t)classifier->getClassCount() );

    for( size_t i = 0; i < queryPoints.size(); i++ )
    {
        (*classifier)( queryImg, queryPoints[i].pt, signature);

        for( int ci = 0; ci < classifier->getClassCount(); ci++ )
        {
            if( -signature[ci] < maxDistance )
            {
                int imgIdx = -1, trainIdx = -1;
                trainPointCollection.getLocalIdx( ci , imgIdx, trainIdx );
                matches[i].push_back( DMatch( i, trainIdx, imgIdx, -signature[ci] ) );
            }
        }
    }
}

void FernDescriptorMatcher::read( const FileNode &fn )
{
    params.nclasses = fn["nclasses"];
    params.patchSize = fn["patchSize"];
    params.signatureSize = fn["signatureSize"];
    params.nstructs = fn["nstructs"];
    params.structSize = fn["structSize"];
    params.nviews = fn["nviews"];
    params.compressionMethod = fn["compressionMethod"];

    //classifier->read(fn);
}

void FernDescriptorMatcher::write( FileStorage& fs ) const
{
    fs << "nclasses" << params.nclasses;
    fs << "patchSize" << params.patchSize;
    fs << "signatureSize" << params.signatureSize;
    fs << "nstructs" << params.nstructs;
    fs << "structSize" << params.structSize;
    fs << "nviews" << params.nviews;
    fs << "compressionMethod" << params.compressionMethod;

//    classifier->write(fs);
}

/****************************************************************************************\
*                                  VectorDescriptorMatcher                                 *
\****************************************************************************************/
void VectorDescriptorMatcher::add( const vector<Mat>& imgCollection,
                                   vector<vector<KeyPoint> >& pointCollection )
{
    vector<Mat> descCollection;
    extractor->compute( imgCollection, pointCollection, descCollection );

    matcher->add( descCollection );

    trainPointCollection.add( imgCollection, pointCollection );
}

void VectorDescriptorMatcher::clear()
{
    //extractor->clear();
    matcher->clear();
    GenericDescriptorMatcher::clear();
}

void VectorDescriptorMatcher::train()
{
    matcher->train();
}

void VectorDescriptorMatcher::knnMatchImpl( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                            vector<vector<DMatch> >& matches, int knn,
                                            const vector<Mat>& masks, bool compactResult )
{
    Mat queryDescs;
    extractor->compute( queryImg, queryPoints, queryDescs );
    matcher->knnMatch( queryDescs, matches, knn, masks, compactResult );
}

void VectorDescriptorMatcher::radiusMatchImpl( const Mat& queryImg, vector<KeyPoint>& queryPoints,
                                               vector<vector<DMatch> >& matches, float maxDistance,
                                               const vector<Mat>& masks, bool compactResult )
{
    Mat queryDescs;
    extractor->compute( queryImg, queryPoints, queryDescs );
    matcher->radiusMatch( queryDescs, matches, maxDistance, masks, compactResult );
}

void VectorDescriptorMatcher::read( const FileNode& fn )
{
    GenericDescriptorMatcher::read(fn);
    extractor->read (fn);
}

void VectorDescriptorMatcher::write (FileStorage& fs) const
{
    GenericDescriptorMatcher::write(fs);
    extractor->write (fs);
}

/*
 * Factory function for GenericDescriptorMatch creating
 */
Ptr<GenericDescriptorMatcher> createGenericDescriptorMatcher( const string& genericDescritptorMatcherType, const string &paramsFilename )
{
    Ptr<GenericDescriptorMatcher> descriptorMatcher;
    if( ! genericDescritptorMatcherType.compare("ONEWAY") )
    {
        descriptorMatcher = new OneWayDescriptorMatcher();
    }
    else if( ! genericDescritptorMatcherType.compare("FERN") )
    {
        descriptorMatcher = new FernDescriptorMatcher();
    }
    else if( ! genericDescritptorMatcherType.compare ("CALONDER") )
    {
        //descriptorMatch = new CalonderDescriptorMatch ();
    }

    if( !paramsFilename.empty() && descriptorMatcher != 0 )
    {
        FileStorage fs = FileStorage( paramsFilename, FileStorage::READ );
        if( fs.isOpened() )
        {
            descriptorMatcher->read( fs.root() );
            fs.release();
        }
    }
    return descriptorMatcher;
}

}
