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

namespace cv
{

Mat windowedMatchingMask( const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                          float maxDeltaX, float maxDeltaY )
{
    if( keypoints1.empty() || keypoints2.empty() )
        return Mat();

    int n1 = (int)keypoints1.size(), n2 = (int)keypoints2.size();
    Mat mask( n1, n2, CV_8UC1 );
    for( int i = 0; i < n1; i++ )
    {
        for( int j = 0; j < n2; j++ )
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
DescriptorMatcher::DescriptorCollection::DescriptorCollection()
{}

DescriptorMatcher::DescriptorCollection::DescriptorCollection( const DescriptorCollection& collection )
{
    mergedDescriptors = collection.mergedDescriptors.clone();
    copy( collection.startIdxs.begin(), collection.startIdxs.begin(), startIdxs.begin() );
}

DescriptorMatcher::DescriptorCollection::~DescriptorCollection()
{}

void DescriptorMatcher::DescriptorCollection::set( const vector<Mat>& descriptors )
{
    clear();

    size_t imageCount = descriptors.size();
    CV_Assert( imageCount > 0 );

    startIdxs.resize( imageCount );

    int dim = -1;
    int type = -1;
    startIdxs[0] = 0;
    for( size_t i = 1; i < imageCount; i++ )
    {
        int s = 0;
        if( !descriptors[i-1].empty() )
        {
            dim = descriptors[i-1].cols;
            type = descriptors[i-1].type();
            s = descriptors[i-1].rows;
        }
        startIdxs[i] = startIdxs[i-1] + s;
    }
    if( imageCount == 1 )
    {
        if( descriptors[0].empty() ) return;

        dim = descriptors[0].cols;
        type = descriptors[0].type();
    }
    assert( dim > 0 );

    int count = startIdxs[imageCount-1] + descriptors[imageCount-1].rows;

    if( count > 0 )
    {
        mergedDescriptors.create( count, dim, type );
        for( size_t i = 0; i < imageCount; i++ )
        {
            if( !descriptors[i].empty() )
            {
                CV_Assert( descriptors[i].cols == dim && descriptors[i].type() == type );
                Mat m = mergedDescriptors.rowRange( startIdxs[i], startIdxs[i] + descriptors[i].rows );
                descriptors[i].copyTo(m);
            }
        }
    }
}

void DescriptorMatcher::DescriptorCollection::clear()
{
    startIdxs.clear();
    mergedDescriptors.release();
}

const Mat DescriptorMatcher::DescriptorCollection::getDescriptor( int imgIdx, int localDescIdx ) const
{
    CV_Assert( imgIdx < (int)startIdxs.size() );
    int globalIdx = startIdxs[imgIdx] + localDescIdx;
    CV_Assert( globalIdx < (int)size() );

    return getDescriptor( globalIdx );
}

const Mat& DescriptorMatcher::DescriptorCollection::getDescriptors() const
{
    return mergedDescriptors;
}

const Mat DescriptorMatcher::DescriptorCollection::getDescriptor( int globalDescIdx ) const
{
    CV_Assert( globalDescIdx < size() );
    return mergedDescriptors.row( globalDescIdx );
}

void DescriptorMatcher::DescriptorCollection::getLocalIdx( int globalDescIdx, int& imgIdx, int& localDescIdx ) const
{
    CV_Assert( (globalDescIdx>=0) && (globalDescIdx < size()) );
    std::vector<int>::const_iterator img_it = std::upper_bound(startIdxs.begin(), startIdxs.end(), globalDescIdx);
    --img_it;
    imgIdx = img_it - startIdxs.begin();
    localDescIdx = globalDescIdx - (*img_it);
}

int DescriptorMatcher::DescriptorCollection::size() const
{
    return mergedDescriptors.rows;
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

DescriptorMatcher::~DescriptorMatcher()
{}

void DescriptorMatcher::add( const vector<Mat>& descriptors )
{
    trainDescCollection.insert( trainDescCollection.end(), descriptors.begin(), descriptors.end() );
}

const vector<Mat>& DescriptorMatcher::getTrainDescriptors() const
{
    return trainDescCollection;
}

void DescriptorMatcher::clear()
{
    trainDescCollection.clear();
}

bool DescriptorMatcher::empty() const
{
    return trainDescCollection.empty();
}

void DescriptorMatcher::train()
{}

void DescriptorMatcher::match( const Mat& queryDescriptors, const Mat& trainDescriptors, vector<DMatch>& matches, const Mat& mask ) const
{
    Ptr<DescriptorMatcher> tempMatcher = clone(true);
    tempMatcher->add( vector<Mat>(1, trainDescriptors) );
    tempMatcher->match( queryDescriptors, matches, vector<Mat>(1, mask) );
}

void DescriptorMatcher::knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors, vector<vector<DMatch> >& matches, int knn,
                                  const Mat& mask, bool compactResult ) const
{
    Ptr<DescriptorMatcher> tempMatcher = clone(true);
    tempMatcher->add( vector<Mat>(1, trainDescriptors) );
    tempMatcher->knnMatch( queryDescriptors, matches, knn, vector<Mat>(1, mask), compactResult );
}

void DescriptorMatcher::radiusMatch( const Mat& queryDescriptors, const Mat& trainDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                                     const Mat& mask, bool compactResult ) const
{
    Ptr<DescriptorMatcher> tempMatcher = clone(true);
    tempMatcher->add( vector<Mat>(1, trainDescriptors) );
    tempMatcher->radiusMatch( queryDescriptors, matches, maxDistance, vector<Mat>(1, mask), compactResult );
}

void DescriptorMatcher::match( const Mat& queryDescriptors, vector<DMatch>& matches, const vector<Mat>& masks )
{
    vector<vector<DMatch> > knnMatches;
    knnMatch( queryDescriptors, knnMatches, 1, masks, true /*compactResult*/ );
    convertMatches( knnMatches, matches );
}

void DescriptorMatcher::checkMasks( const vector<Mat>& masks, int queryDescriptorsCount ) const
{
	if( isMaskSupported() && !masks.empty() )
	{
		// Check masks
		size_t imageCount = trainDescCollection.size();
		CV_Assert( masks.size() == imageCount );
		for( size_t i = 0; i < imageCount; i++ )
		{
			if( !masks[i].empty() && !trainDescCollection[i].empty() )
			{
				CV_Assert( masks[i].rows == queryDescriptorsCount && 
					       masks[i].cols == trainDescCollection[i].rows &&
						   masks[i].type() == CV_8UC1 );
			}
		}
	}
}

void DescriptorMatcher::knnMatch( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int knn,
                                  const vector<Mat>& masks, bool compactResult )
{
	matches.empty();
	if( empty() || queryDescriptors.empty() )
		return;

	CV_Assert( knn > 0 );
	
	checkMasks( masks, queryDescriptors.rows );

    train();
    knnMatchImpl( queryDescriptors, matches, knn, masks, compactResult );
}

void DescriptorMatcher::radiusMatch( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                                     const vector<Mat>& masks, bool compactResult )
{
	matches.empty();
	if( empty() || queryDescriptors.empty() )
		return;

	CV_Assert( maxDistance > std::numeric_limits<float>::epsilon() );
	
	checkMasks( masks, queryDescriptors.rows );

    train();
    radiusMatchImpl( queryDescriptors, matches, maxDistance, masks, compactResult );
}

void DescriptorMatcher::read( const FileNode& )
{}

void DescriptorMatcher::write( FileStorage& ) const
{}

bool DescriptorMatcher::isPossibleMatch( const Mat& mask, int queryIdx, int trainIdx )
{
    return mask.empty() || mask.at<uchar>(queryIdx, trainIdx);
}

bool DescriptorMatcher::isMaskedOut( const vector<Mat>& masks, int queryIdx )
{
    size_t outCount = 0;
    for( size_t i = 0; i < masks.size(); i++ )
    {
        if( !masks[i].empty() && (countNonZero(masks[i].row(queryIdx)) == 0) )
            outCount++;
    }

    return !masks.empty() && outCount == masks.size() ;
}

/*
 * Factory function for DescriptorMatcher creating
 */
Ptr<DescriptorMatcher> DescriptorMatcher::create( const string& descriptorMatcherType )
{
    DescriptorMatcher* dm = 0;
    if( !descriptorMatcherType.compare( "FlannBased" ) )
    {
        dm = new FlannBasedMatcher();
    }
    else if( !descriptorMatcherType.compare( "BruteForce" ) ) // L2
    {
        dm = new BruteForceMatcher<L2<float> >();
    }
    else if( !descriptorMatcherType.compare( "BruteForce-L1" ) )
    {
        dm = new BruteForceMatcher<L1<float> >();
    }
    else if( !descriptorMatcherType.compare("BruteForce-Hamming") )
    {
        dm = new BruteForceMatcher<Hamming>();
    }
    else if( !descriptorMatcherType.compare( "BruteForce-HammingLUT") )
    {
        dm = new BruteForceMatcher<HammingLUT>();
    }

    return dm;
}

/*
 * BruteForce L2 specialization
 */
template<>
void BruteForceMatcher<L2<float> >::knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int knn,
                                              const vector<Mat>& masks, bool compactResult )
{
#ifndef HAVE_EIGEN2
    commonKnnMatchImpl( *this, queryDescriptors, matches, knn, masks, compactResult );
#else
    CV_Assert( queryDescriptors.type() == CV_32FC1 ||  queryDescriptors.empty() );
    CV_Assert( masks.empty() || masks.size() == trainDescCollection.size() );

    matches.reserve(queryDescriptors.rows);
    size_t imgCount = trainDescCollection.size();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> e_query_t;
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > e_trainCollection(trainDescCollection.size());
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > e_trainNorms2(trainDescCollection.size());
    cv2eigen( queryDescriptors.t(), e_query_t);
    for( size_t i = 0; i < trainDescCollection.size(); i++ )
    {
        cv2eigen( trainDescCollection[i], e_trainCollection[i] );
        e_trainNorms2[i] = e_trainCollection[i].rowwise().squaredNorm() / 2;
    }

    vector<Eigen::Matrix<float, Eigen::Dynamic, 1> > e_allDists( imgCount ); // distances between one query descriptor and all train descriptors

    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        if( isMaskedOut( masks, qIdx ) )
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
                           ( masks[iIdx].rows == queryDescriptors.rows && masks[iIdx].cols == trainDescCollection[iIdx].rows &&
                             masks[iIdx].type() == CV_8UC1 ) );
                CV_Assert( trainDescCollection[iIdx].type() == CV_32FC1 ||  trainDescCollection[iIdx].empty() );
                CV_Assert( queryDescriptors.cols == trainDescCollection[iIdx].cols );

                e_allDists[iIdx] = e_trainCollection[iIdx] *e_query_t.col(qIdx);
                e_allDists[iIdx] -= e_trainNorms2[iIdx];

                if( !masks.empty() && !masks[iIdx].empty() )
                {
                    const uchar* maskPtr = (uchar*)masks[iIdx].ptr(qIdx);
                    for( int c = 0; c < masks[iIdx].cols; c++ )
                    {
                        if( maskPtr[c] == 0 )
                            e_allDists[iIdx](c) = -std::numeric_limits<float>::max();
                    }
                }
            }

            // 2. choose knn nearest matches for query[i]
            matches.push_back( vector<DMatch>() );
            vector<vector<DMatch> >::reverse_iterator curMatches = matches.rbegin();
            for( int k = 0; k < knn; k++ )
            {
                float totalMaxCoeff = -std::numeric_limits<float>::max();
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

                e_allDists[bestImgIdx](bestTrainIdx) = -std::numeric_limits<float>::max();
                curMatches->push_back( DMatch(qIdx, bestTrainIdx, bestImgIdx, sqrt((-2)*totalMaxCoeff + queryNorm2)) );
            }
            std::sort( curMatches->begin(), curMatches->end() );
        }
    }
#endif
}

template<>
void BruteForceMatcher<L2<float> >::radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                                                     const vector<Mat>& masks, bool compactResult )
{
#ifndef HAVE_EIGEN2
    commonRadiusMatchImpl( *this, queryDescriptors, matches, maxDistance, masks, compactResult );
#else
    CV_Assert( queryDescriptors.type() == CV_32FC1 ||  queryDescriptors.empty() );
    CV_Assert( masks.empty() || masks.size() == trainDescCollection.size() );

    matches.reserve(queryDescriptors.rows);
    size_t imgCount = trainDescCollection.size();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> e_query_t;
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > e_trainCollection(trainDescCollection.size());
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > e_trainNorms2(trainDescCollection.size());
    cv2eigen( queryDescriptors.t(), e_query_t);
    for( size_t i = 0; i < trainDescCollection.size(); i++ )
    {
        cv2eigen( trainDescCollection[i], e_trainCollection[i] );
        e_trainNorms2[i] = e_trainCollection[i].rowwise().squaredNorm() / 2;
    }

    vector<Eigen::Matrix<float, Eigen::Dynamic, 1> > e_allDists( imgCount ); // distances between one query descriptor and all train descriptors

    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        if( isMaskedOut( masks, qIdx ) )
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
                           ( masks[iIdx].rows == queryDescriptors.rows && masks[iIdx].cols == trainDescCollection[iIdx].rows &&
                             masks[iIdx].type() == CV_8UC1 ) );
                CV_Assert( trainDescCollection[iIdx].type() == CV_32FC1 ||  trainDescCollection[iIdx].empty() );
                CV_Assert( queryDescriptors.cols == trainDescCollection[iIdx].cols );

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
                    if( masks.empty() || isPossibleMatch(masks[iIdx], qIdx, tIdx) )
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

void FlannBasedMatcher::add( const vector<Mat>& descriptors )
{
    DescriptorMatcher::add( descriptors );
    for( size_t i = 0; i < descriptors.size(); i++ )
    {
        addedDescCount += descriptors[i].rows;
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

bool FlannBasedMatcher::isMaskSupported() const
{
    return false;
}

Ptr<DescriptorMatcher> FlannBasedMatcher::clone( bool emptyTrainData ) const
{
    FlannBasedMatcher* matcher = new FlannBasedMatcher(indexParams, searchParams);
    if( !emptyTrainData )
    {
        CV_Error( CV_StsNotImplemented, "deep clone functionality is not implemented, because "
                  "Flann::Index has not copy constructor or clone method ");
        //matcher->flannIndex;
        matcher->addedDescCount = addedDescCount;
        matcher->mergedDescriptors = DescriptorCollection( mergedDescriptors );
        std::transform( trainDescCollection.begin(), trainDescCollection.end(),
                        matcher->trainDescCollection.begin(), clone_op );
    }
    return matcher;
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

void FlannBasedMatcher::knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int knn,
                                      const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    Mat indices( queryDescriptors.rows, knn, CV_32SC1 );
    Mat dists( queryDescriptors.rows, knn, CV_32FC1);
    flannIndex->knnSearch( queryDescriptors, indices, dists, knn, *searchParams );

    convertToDMatches( mergedDescriptors, indices, dists, matches );
}

void FlannBasedMatcher::radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                                         const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    const int count = mergedDescriptors.size(); // TODO do count as param?
    Mat indices( queryDescriptors.rows, count, CV_32SC1, Scalar::all(-1) );
    Mat dists( queryDescriptors.rows, count, CV_32FC1, Scalar::all(-1) );
    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        Mat queryDescriptorsRow = queryDescriptors.row(qIdx);
        Mat indicesRow = indices.row(qIdx);
        Mat distsRow = dists.row(qIdx);
        flannIndex->radiusSearch( queryDescriptorsRow, indicesRow, distsRow, maxDistance*maxDistance, *searchParams );
    }

    convertToDMatches( mergedDescriptors, indices, dists, matches );
}

/****************************************************************************************\
*                                GenericDescriptorMatcher                                *
\****************************************************************************************/
/*
 * KeyPointCollection
 */
GenericDescriptorMatcher::KeyPointCollection::KeyPointCollection() : pointCount(0)
{}

GenericDescriptorMatcher::KeyPointCollection::KeyPointCollection( const KeyPointCollection& collection )
{
    pointCount = collection.pointCount;

    std::transform( collection.images.begin(), collection.images.end(), images.begin(), clone_op );

    keypoints.resize( collection.keypoints.size() );
    for( size_t i = 0; i < keypoints.size(); i++ )
        copy( collection.keypoints[i].begin(), collection.keypoints[i].end(), keypoints[i].begin() );

    copy( collection.startIndices.begin(), collection.startIndices.end(), startIndices.begin() );
}

void GenericDescriptorMatcher::KeyPointCollection::add( const vector<Mat>& _images,
                                                        const vector<vector<KeyPoint> >& _points )
{
    CV_Assert( !_images.empty() );
    CV_Assert( _images.size() == _points.size() );

    images.insert( images.end(), _images.begin(), _images.end() );
    keypoints.insert( keypoints.end(), _points.begin(), _points.end() );
    for( size_t i = 0; i < _points.size(); i++ )
        pointCount += (int)_points[i].size();

    size_t prevSize = startIndices.size(), addSize = _images.size();
    startIndices.resize( prevSize + addSize );

    if( prevSize == 0 )
        startIndices[prevSize] = 0; //first
    else
        startIndices[prevSize] = (int)(startIndices[prevSize-1] + keypoints[prevSize-1].size());

    for( size_t i = prevSize + 1; i < prevSize + addSize; i++ )
    {
        startIndices[i] = (int)(startIndices[i - 1] + keypoints[i - 1].size());
    }
}

void GenericDescriptorMatcher::KeyPointCollection::clear()
{
    keypoints.clear();
}

size_t GenericDescriptorMatcher::KeyPointCollection::keypointCount() const
{
    return pointCount;
}

size_t GenericDescriptorMatcher::KeyPointCollection::imageCount() const
{
    return images.size();
}

const vector<vector<KeyPoint> >& GenericDescriptorMatcher::KeyPointCollection::getKeypoints() const
{
    return keypoints;
}

const vector<KeyPoint>& GenericDescriptorMatcher::KeyPointCollection::getKeypoints( int imgIdx ) const
{
    CV_Assert( imgIdx < (int)imageCount() );
    return keypoints[imgIdx];
}

const KeyPoint& GenericDescriptorMatcher::KeyPointCollection::getKeyPoint( int imgIdx, int localPointIdx ) const
{
    CV_Assert( imgIdx < (int)images.size() );
    CV_Assert( localPointIdx < (int)keypoints[imgIdx].size() );
    return keypoints[imgIdx][localPointIdx];
}

const KeyPoint& GenericDescriptorMatcher::KeyPointCollection::getKeyPoint( int globalPointIdx ) const
{
    int imgIdx, localPointIdx;
    getLocalIdx( globalPointIdx, imgIdx, localPointIdx );
    return keypoints[imgIdx][localPointIdx];
}

void GenericDescriptorMatcher::KeyPointCollection::getLocalIdx( int globalPointIdx, int& imgIdx, int& localPointIdx ) const
{
    imgIdx = -1;
    CV_Assert( globalPointIdx < (int)keypointCount() );
    for( size_t i = 1; i < startIndices.size(); i++ )
    {
        if( globalPointIdx < startIndices[i] )
        {
            imgIdx = (int)(i - 1);
            break;
        }
    }
    imgIdx = imgIdx == -1 ? (int)(startIndices.size() - 1) : imgIdx;
    localPointIdx = globalPointIdx - startIndices[imgIdx];
}

const vector<Mat>& GenericDescriptorMatcher::KeyPointCollection::getImages() const
{
    return images;
}

const Mat& GenericDescriptorMatcher::KeyPointCollection::getImage( int imgIdx ) const
{
    CV_Assert( imgIdx < (int)imageCount() );
    return images[imgIdx];
}

/*
 * GenericDescriptorMatcher
 */
GenericDescriptorMatcher::GenericDescriptorMatcher()
{}

GenericDescriptorMatcher::~GenericDescriptorMatcher()
{}

void GenericDescriptorMatcher::add( const vector<Mat>& imgCollection,
                                    vector<vector<KeyPoint> >& pointCollection )
{
    trainPointCollection.add( imgCollection, pointCollection );
}

const vector<Mat>& GenericDescriptorMatcher::getTrainImages() const
{
    return trainPointCollection.getImages();
}

const vector<vector<KeyPoint> >& GenericDescriptorMatcher::getTrainKeypoints() const
{
    return trainPointCollection.getKeypoints();
}

void GenericDescriptorMatcher::clear()
{
    trainPointCollection.clear();
}

void GenericDescriptorMatcher::train()
{}

void GenericDescriptorMatcher::classify( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                         const Mat& trainImage, vector<KeyPoint>& trainKeypoints ) const
{
    vector<DMatch> matches;
    match( queryImage, queryKeypoints, trainImage, trainKeypoints, matches );

    // remap keypoint indices to descriptors
    for( size_t i = 0; i < matches.size(); i++ )
        queryKeypoints[matches[i].queryIdx].class_id = trainKeypoints[matches[i].trainIdx].class_id;
}

void GenericDescriptorMatcher::classify( const Mat& queryImage, vector<KeyPoint>& queryKeypoints )
{
    vector<DMatch> matches;
    match( queryImage, queryKeypoints, matches );

    // remap keypoint indices to descriptors
    for( size_t i = 0; i < matches.size(); i++ )
        queryKeypoints[matches[i].queryIdx].class_id = trainPointCollection.getKeyPoint( matches[i].trainIdx, matches[i].trainIdx ).class_id;
}

void GenericDescriptorMatcher::match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                      const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                                      vector<DMatch>& matches, const Mat& mask ) const
{
    Ptr<GenericDescriptorMatcher> tempMatcher = clone( true );
    vector<vector<KeyPoint> > vecTrainPoints(1, trainKeypoints);
    tempMatcher->add( vector<Mat>(1, trainImage), vecTrainPoints );
    tempMatcher->match( queryImage, queryKeypoints, matches, vector<Mat>(1, mask) );
    vecTrainPoints[0].swap( trainKeypoints );
}

void GenericDescriptorMatcher::knnMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                         const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                                         vector<vector<DMatch> >& matches, int knn, const Mat& mask, bool compactResult ) const
{
    Ptr<GenericDescriptorMatcher> tempMatcher = clone( true );
    vector<vector<KeyPoint> > vecTrainPoints(1, trainKeypoints);
    tempMatcher->add( vector<Mat>(1, trainImage), vecTrainPoints );
    tempMatcher->knnMatch( queryImage, queryKeypoints, matches, knn, vector<Mat>(1, mask), compactResult );
    vecTrainPoints[0].swap( trainKeypoints );
}

void GenericDescriptorMatcher::radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                            const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                                            vector<vector<DMatch> >& matches, float maxDistance,
                                            const Mat& mask, bool compactResult ) const
{
    Ptr<GenericDescriptorMatcher> tempMatcher = clone( true );
    vector<vector<KeyPoint> > vecTrainPoints(1, trainKeypoints);
    tempMatcher->add( vector<Mat>(1, trainImage), vecTrainPoints );
    tempMatcher->radiusMatch( queryImage, queryKeypoints, matches, maxDistance, vector<Mat>(1, mask), compactResult );
    vecTrainPoints[0].swap( trainKeypoints );
}

void GenericDescriptorMatcher::match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                      vector<DMatch>& matches, const vector<Mat>& masks )
{
    vector<vector<DMatch> > knnMatches;
    knnMatch( queryImage, queryKeypoints, knnMatches, 1, masks, false );
    convertMatches( knnMatches, matches );
}

void GenericDescriptorMatcher::knnMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                         vector<vector<DMatch> >& matches, int knn,
                                         const vector<Mat>& masks, bool compactResult )
{
    train();
    knnMatchImpl( queryImage, queryKeypoints, matches, knn, masks, compactResult );
}

void GenericDescriptorMatcher::radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                            vector<vector<DMatch> >& matches, float maxDistance,
                                            const vector<Mat>& masks, bool compactResult )
{
    train();
    radiusMatchImpl( queryImage, queryKeypoints, matches, maxDistance, masks, compactResult );
}

void GenericDescriptorMatcher::read( const FileNode& )
{}

void GenericDescriptorMatcher::write( FileStorage& ) const
{}

bool GenericDescriptorMatcher::empty() const
{
    return true;
}

/*
 * Factory function for GenericDescriptorMatch creating
 */
Ptr<GenericDescriptorMatcher> GenericDescriptorMatcher::create( const string& genericDescritptorMatcherType,
                                                                const string &paramsFilename )
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

    if( !paramsFilename.empty() && !descriptorMatcher.empty() )
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

/****************************************************************************************\
*                                OneWayDescriptorMatcher                                  *
\****************************************************************************************/

OneWayDescriptorMatcher::Params::Params( int _poseCount, Size _patchSize, string _pcaFilename,
                                         string _trainPath, string _trainImagesList,
                                         float _minScale, float _maxScale, float _stepScale ) :
                poseCount(_poseCount), patchSize(_patchSize), pcaFilename(_pcaFilename),
                trainPath(_trainPath), trainImagesList(_trainImagesList),
                minScale(_minScale), maxScale(_maxScale), stepScale(_stepScale)
{}


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
    if( base.empty() || prevTrainCount < (int)trainPointCollection.keypointCount() )
    {
        base = new OneWayDescriptorObject( params.patchSize, params.poseCount, params.pcaFilename,
                                           params.trainPath, params.trainImagesList, params.minScale, params.maxScale, params.stepScale );

        base->Allocate( (int)trainPointCollection.keypointCount() );
        prevTrainCount = (int)trainPointCollection.keypointCount();

        const vector<vector<KeyPoint> >& points = trainPointCollection.getKeypoints();
        int count = 0;
        for( size_t i = 0; i < points.size(); i++ )
        {
            IplImage _image = trainPointCollection.getImage((int)i);
            for( size_t j = 0; j < points[i].size(); j++ )
                base->InitializeDescriptor( count++, &_image, points[i][j], "" );
        }

#if defined(_KDTREE)
        base->ConvertDescriptorsArrayToTree();
#endif
    }
}

bool OneWayDescriptorMatcher::isMaskSupported()
{
    return false;
}

void OneWayDescriptorMatcher::knnMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                            vector<vector<DMatch> >& matches, int knn,
                                            const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    train();

    CV_Assert( knn == 1 ); // knn > 1 unsupported because of bug in OneWayDescriptorBase for this case

    matches.resize( queryKeypoints.size() );
    IplImage _qimage = queryImage;
    for( size_t i = 0; i < queryKeypoints.size(); i++ )
    {
        int descIdx = -1, poseIdx = -1;
        float distance;
        base->FindDescriptor( &_qimage, queryKeypoints[i].pt, descIdx, poseIdx, distance );
        matches[i].push_back( DMatch((int)i, descIdx, distance) );
    }
}

void OneWayDescriptorMatcher::radiusMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                               vector<vector<DMatch> >& matches, float maxDistance,
                                               const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    train();

    matches.resize( queryKeypoints.size() );
    IplImage _qimage = queryImage;
    for( size_t i = 0; i < queryKeypoints.size(); i++ )
    {
        int descIdx = -1, poseIdx = -1;
        float distance;
        base->FindDescriptor( &_qimage, queryKeypoints[i].pt, descIdx, poseIdx, distance );
        if( distance < maxDistance )
            matches[i].push_back( DMatch((int)i, descIdx, distance) );
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

bool OneWayDescriptorMatcher::empty() const
{
    return base.empty() || base->empty();
}

Ptr<GenericDescriptorMatcher> OneWayDescriptorMatcher::clone( bool emptyTrainData ) const
{
    OneWayDescriptorMatcher* matcher = new OneWayDescriptorMatcher( params );

    if( !emptyTrainData )
    {
        CV_Error( CV_StsNotImplemented, "deep clone functionality is not implemented, because "
              "OneWayDescriptorBase has not copy constructor or clone method ");

        //matcher->base;
        matcher->params = params;
        matcher->prevTrainCount = prevTrainCount;
        matcher->trainPointCollection = trainPointCollection;
    }
    return matcher;
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
    if( classifier.empty() || prevTrainCount < (int)trainPointCollection.keypointCount() )
    {
        assert( params.filename.empty() );

        vector<vector<Point2f> > points( trainPointCollection.imageCount() );
        for( size_t imgIdx = 0; imgIdx < trainPointCollection.imageCount(); imgIdx++ )
            KeyPoint::convert( trainPointCollection.getKeypoints((int)imgIdx), points[imgIdx] );

        classifier = new FernClassifier( points, trainPointCollection.getImages(), vector<vector<int> >(), 0, // each points is a class
                                         params.patchSize, params.signatureSize, params.nstructs, params.structSize,
                                         params.nviews, params.compressionMethod, params.patchGenerator );
    }
}

bool FernDescriptorMatcher::isMaskSupported()
{
    return false;
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

void FernDescriptorMatcher::knnMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                          vector<vector<DMatch> >& matches, int knn,
                                          const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    train();

    matches.resize( queryKeypoints.size() );
    vector<float> signature( (size_t)classifier->getClassCount() );

    for( size_t queryIdx = 0; queryIdx < queryKeypoints.size(); queryIdx++ )
    {
        (*classifier)( queryImage, queryKeypoints[queryIdx].pt, signature);

        for( int k = 0; k < knn; k++ )
        {
            DMatch bestMatch;
            size_t best_ci = 0;
            for( size_t ci = 0; ci < signature.size(); ci++ )
            {
                if( -signature[ci] < bestMatch.distance )
                {
                    int imgIdx = -1, trainIdx = -1;
                    trainPointCollection.getLocalIdx( (int)ci , imgIdx, trainIdx );
                    bestMatch = DMatch( (int)queryIdx, trainIdx, imgIdx, -signature[ci] );
                    best_ci = ci;
                }
            }

            if( bestMatch.trainIdx == -1 )
                break;
            signature[best_ci] = -std::numeric_limits<float>::max();
            matches[queryIdx].push_back( bestMatch );
        }
    }
}

void FernDescriptorMatcher::radiusMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                             vector<vector<DMatch> >& matches, float maxDistance,
                                             const vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    train();
    matches.resize( queryKeypoints.size() );
    vector<float> signature( (size_t)classifier->getClassCount() );

    for( size_t i = 0; i < queryKeypoints.size(); i++ )
    {
        (*classifier)( queryImage, queryKeypoints[i].pt, signature);

        for( int ci = 0; ci < classifier->getClassCount(); ci++ )
        {
            if( -signature[ci] < maxDistance )
            {
                int imgIdx = -1, trainIdx = -1;
                trainPointCollection.getLocalIdx( ci , imgIdx, trainIdx );
                matches[i].push_back( DMatch( (int)i, trainIdx, imgIdx, -signature[ci] ) );
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

bool FernDescriptorMatcher::empty() const
{
    return classifier.empty() || classifier->empty();
}

Ptr<GenericDescriptorMatcher> FernDescriptorMatcher::clone( bool emptyTrainData ) const
{
    FernDescriptorMatcher* matcher = new FernDescriptorMatcher( params );
    if( !emptyTrainData )
    {
        CV_Error( CV_StsNotImplemented, "deep clone dunctionality is not implemented, because "
              "FernClassifier has not copy constructor or clone method ");

        //matcher->classifier;
        matcher->params = params;
        matcher->prevTrainCount = prevTrainCount;
        matcher->trainPointCollection = trainPointCollection;
    }
    return matcher;
}

/****************************************************************************************\
*                                  VectorDescriptorMatcher                               *
\****************************************************************************************/
VectorDescriptorMatcher::VectorDescriptorMatcher( const Ptr<DescriptorExtractor>& _extractor,
                                                  const Ptr<DescriptorMatcher>& _matcher )
                                : extractor( _extractor ), matcher( _matcher )
{
    CV_Assert( !extractor.empty() && !matcher.empty() );
}

VectorDescriptorMatcher::~VectorDescriptorMatcher()
{}

void VectorDescriptorMatcher::add( const vector<Mat>& imgCollection,
                                   vector<vector<KeyPoint> >& pointCollection )
{
    vector<Mat> descriptors;
    extractor->compute( imgCollection, pointCollection, descriptors );

    matcher->add( descriptors );

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

bool VectorDescriptorMatcher::isMaskSupported()
{
    return matcher->isMaskSupported();
}

void VectorDescriptorMatcher::knnMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                            vector<vector<DMatch> >& matches, int knn,
                                            const vector<Mat>& masks, bool compactResult )
{
    Mat queryDescriptors;
    extractor->compute( queryImage, queryKeypoints, queryDescriptors );
    matcher->knnMatch( queryDescriptors, matches, knn, masks, compactResult );
}

void VectorDescriptorMatcher::radiusMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                               vector<vector<DMatch> >& matches, float maxDistance,
                                               const vector<Mat>& masks, bool compactResult )
{
    Mat queryDescriptors;
    extractor->compute( queryImage, queryKeypoints, queryDescriptors );
    matcher->radiusMatch( queryDescriptors, matches, maxDistance, masks, compactResult );
}

void VectorDescriptorMatcher::read( const FileNode& fn )
{
    GenericDescriptorMatcher::read(fn);
    extractor->read(fn);
}

void VectorDescriptorMatcher::write (FileStorage& fs) const
{
    GenericDescriptorMatcher::write(fs);
    extractor->write (fs);
}

bool VectorDescriptorMatcher::empty() const
{
    return extractor.empty() || extractor->empty() ||
           matcher.empty() || matcher->empty();
}

Ptr<GenericDescriptorMatcher> VectorDescriptorMatcher::clone( bool emptyTrainData ) const
{
    // TODO clone extractor
    return new VectorDescriptorMatcher( extractor, matcher->clone(emptyTrainData) );
}

}
