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

#include "opencv2/core/internal.hpp"
#if defined(HAVE_EIGEN) && EIGEN_WORLD_VERSION == 2
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
    std::copy( collection.startIdxs.begin(), collection.startIdxs.begin(), startIdxs.begin() );
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
    imgIdx = (int)(img_it - startIdxs.begin());
    localDescIdx = globalDescIdx - (*img_it);
}

int DescriptorMatcher::DescriptorCollection::size() const
{
    return mergedDescriptors.rows;
}

/*
 * DescriptorMatcher
 */
static void convertMatches( const vector<vector<DMatch> >& knnMatches, vector<DMatch>& matches )
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
    matches.clear();
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
    matches.clear();
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


///////////////////////////////////////////////////////////////////////////////////////////////////////

BFMatcher::BFMatcher( int _normType, bool _crossCheck )
{
    normType = _normType;
    crossCheck = _crossCheck;
}

Ptr<DescriptorMatcher> BFMatcher::clone( bool emptyTrainData ) const
{
    BFMatcher* matcher = new BFMatcher(normType, crossCheck);
    if( !emptyTrainData )
    {
        matcher->trainDescCollection.resize(trainDescCollection.size());
        std::transform( trainDescCollection.begin(), trainDescCollection.end(),
                        matcher->trainDescCollection.begin(), clone_op );
    }
    return matcher;
}


void BFMatcher::knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int knn,
                              const vector<Mat>& masks, bool compactResult )
{
    const int IMGIDX_SHIFT = 18;
    const int IMGIDX_ONE = (1 << IMGIDX_SHIFT);

    if( queryDescriptors.empty() || trainDescCollection.empty() )
    {
        matches.clear();
        return;
    }
    CV_Assert( queryDescriptors.type() == trainDescCollection[0].type() );

    matches.reserve(queryDescriptors.rows);

    Mat dist, nidx;

    int iIdx, imgCount = (int)trainDescCollection.size(), update = 0;
    int dtype = normType == NORM_HAMMING || normType == NORM_HAMMING2 ||
        (normType == NORM_L1 && queryDescriptors.type() == CV_8U) ? CV_32S : CV_32F;

    CV_Assert( (int64)imgCount*IMGIDX_ONE < INT_MAX );

    for( iIdx = 0; iIdx < imgCount; iIdx++ )
    {
        CV_Assert( trainDescCollection[iIdx].rows < IMGIDX_ONE );
        batchDistance(queryDescriptors, trainDescCollection[iIdx], dist, dtype, nidx,
                      normType, knn, masks.empty() ? Mat() : masks[iIdx], update, crossCheck);
        update += IMGIDX_ONE;
    }

    if( dtype == CV_32S )
    {
        Mat temp;
        dist.convertTo(temp, CV_32F);
        dist = temp;
    }

    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        const float* distptr = dist.ptr<float>(qIdx);
        const int* nidxptr = nidx.ptr<int>(qIdx);

        matches.push_back( vector<DMatch>() );
        vector<DMatch>& mq = matches.back();
        mq.reserve(knn);

        for( int k = 0; k < nidx.cols; k++ )
        {
            if( nidxptr[k] < 0 )
                break;
            mq.push_back( DMatch(qIdx, nidxptr[k] & (IMGIDX_ONE - 1),
                          nidxptr[k] >> IMGIDX_SHIFT, distptr[k]) );
        }

        if( mq.empty() && compactResult )
            matches.pop_back();
    }
}


void BFMatcher::radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches,
                                 float maxDistance, const vector<Mat>& masks, bool compactResult )
{
    if( queryDescriptors.empty() || trainDescCollection.empty() )
    {
        matches.clear();
        return;
    }
    CV_Assert( queryDescriptors.type() == trainDescCollection[0].type() );

    matches.resize(queryDescriptors.rows);
    Mat dist, distf;

    int iIdx, imgCount = (int)trainDescCollection.size();
    int dtype = normType == NORM_HAMMING ||
        (normType == NORM_L1 && queryDescriptors.type() == CV_8U) ? CV_32S : CV_32F;

    for( iIdx = 0; iIdx < imgCount; iIdx++ )
    {
        batchDistance(queryDescriptors, trainDescCollection[iIdx], dist, dtype, noArray(),
                      normType, 0, masks.empty() ? Mat() : masks[iIdx], 0, false);
        if( dtype == CV_32S )
            dist.convertTo(distf, CV_32F);
        else
            distf = dist;

        for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
        {
            const float* distptr = distf.ptr<float>(qIdx);

            vector<DMatch>& mq = matches[qIdx];
            for( int k = 0; k < distf.cols; k++ )
            {
                if( distptr[k] <= maxDistance )
                    mq.push_back( DMatch(qIdx, k, iIdx, distptr[k]) );
            }
        }
    }

    int qIdx0 = 0;
    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        if( matches[qIdx].empty() && compactResult )
            continue;

        if( qIdx0 < qIdx )
            std::swap(matches[qIdx], matches[qIdx0]);

        std::sort( matches[qIdx0].begin(), matches[qIdx0].end() );
        qIdx0++;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

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
        dm = new BFMatcher(NORM_L2);
    }
    else if( !descriptorMatcherType.compare( "BruteForce-SL2" ) ) // Squared L2
    {
        dm = new BFMatcher(NORM_L2SQR);
    }
    else if( !descriptorMatcherType.compare( "BruteForce-L1" ) )
    {
        dm = new BFMatcher(NORM_L1);
    }
    else if( !descriptorMatcherType.compare("BruteForce-Hamming") ||
             !descriptorMatcherType.compare("BruteForce-HammingLUT") )
    {
        dm = new BFMatcher(NORM_HAMMING);
    }
    else if( !descriptorMatcherType.compare("BruteForce-Hamming(2)") )
    {
        dm = new BFMatcher(NORM_HAMMING2);
    }
    else
        CV_Error( CV_StsBadArg, "Unknown matcher name" );

    return dm;
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

void FlannBasedMatcher::read( const FileNode& fn)
{
     if (indexParams == 0)
         indexParams = new flann::IndexParams();

     FileNode ip = fn["indexParams"];
     CV_Assert(ip.type() == FileNode::SEQ);

     for(int i = 0; i < (int)ip.size(); ++i)
     {
        CV_Assert(ip[i].type() == FileNode::MAP);
        std::string _name =  (std::string)ip[i]["name"];
        int type =  (int)ip[i]["type"];

        switch(type)
        {
        case CV_8U:
        case CV_8S:
        case CV_16U:
        case CV_16S:
        case CV_32S:
            indexParams->setInt(_name, (int) ip[i]["value"]);
            break;
        case CV_32F:
            indexParams->setFloat(_name, (float) ip[i]["value"]);
            break;
        case CV_64F:
            indexParams->setDouble(_name, (double) ip[i]["value"]);
            break;
        case CV_USRTYPE1:
            indexParams->setString(_name, (std::string) ip[i]["value"]);
            break;
        case CV_MAKETYPE(CV_USRTYPE1,2):
            indexParams->setBool(_name, (int) ip[i]["value"] != 0);
            break;
        case CV_MAKETYPE(CV_USRTYPE1,3):
            indexParams->setAlgorithm((int) ip[i]["value"]);
            break;
        };
     }

     if (searchParams == 0)
         searchParams = new flann::SearchParams();

     FileNode sp = fn["searchParams"];
     CV_Assert(sp.type() == FileNode::SEQ);

     for(int i = 0; i < (int)sp.size(); ++i)
     {
        CV_Assert(sp[i].type() == FileNode::MAP);
        std::string _name =  (std::string)sp[i]["name"];
        int type =  (int)sp[i]["type"];

        switch(type)
        {
        case CV_8U:
        case CV_8S:
        case CV_16U:
        case CV_16S:
        case CV_32S:
            searchParams->setInt(_name, (int) sp[i]["value"]);
            break;
        case CV_32F:
            searchParams->setFloat(_name, (float) ip[i]["value"]);
            break;
        case CV_64F:
            searchParams->setDouble(_name, (double) ip[i]["value"]);
            break;
        case CV_USRTYPE1:
            searchParams->setString(_name, (std::string) ip[i]["value"]);
            break;
        case CV_MAKETYPE(CV_USRTYPE1,2):
            searchParams->setBool(_name, (int) ip[i]["value"] != 0);
            break;
        case CV_MAKETYPE(CV_USRTYPE1,3):
            searchParams->setAlgorithm((int) ip[i]["value"]);
            break;
        };
     }

    flannIndex.release();
}

void FlannBasedMatcher::write( FileStorage& fs) const
{
     fs << "indexParams" << "[";

     if (indexParams != 0)
     {
         std::vector<std::string> names;
         std::vector<int> types;
         std::vector<std::string> strValues;
         std::vector<double> numValues;

         indexParams->getAll(names, types, strValues, numValues);

         for(size_t i = 0; i < names.size(); ++i)
         {
             fs << "{" << "name" << names[i] << "type" << types[i] << "value";
             switch(types[i])
             {
             case CV_8U:
                 fs << (uchar)numValues[i];
                 break;
             case CV_8S:
                 fs << (char)numValues[i];
                 break;
             case CV_16U:
                 fs << (ushort)numValues[i];
                 break;
             case CV_16S:
                 fs << (short)numValues[i];
                 break;
             case CV_32S:
             case CV_MAKETYPE(CV_USRTYPE1,2):
             case CV_MAKETYPE(CV_USRTYPE1,3):
                 fs << (int)numValues[i];
                 break;
             case CV_32F:
                 fs << (float)numValues[i];
                 break;
             case CV_64F:
                 fs << (double)numValues[i];
                 break;
             case CV_USRTYPE1:
                 fs << strValues[i];
                 break;
             default:
                 fs << (double)numValues[i];
                 fs << "typename" << strValues[i];
                 break;
             }
             fs << "}";
         }
     }

     fs << "]" << "searchParams" << "[";

     if (searchParams != 0)
     {
         std::vector<std::string> names;
         std::vector<int> types;
         std::vector<std::string> strValues;
         std::vector<double> numValues;

         searchParams->getAll(names, types, strValues, numValues);

         for(size_t i = 0; i < names.size(); ++i)
         {
             fs << "{" << "name" << names[i] << "type" << types[i] << "value";
             switch(types[i])
             {
             case CV_8U:
                 fs << (uchar)numValues[i];
                 break;
             case CV_8S:
                 fs << (char)numValues[i];
                 break;
             case CV_16U:
                 fs << (ushort)numValues[i];
                 break;
             case CV_16S:
                 fs << (short)numValues[i];
                 break;
             case CV_32S:
             case CV_MAKETYPE(CV_USRTYPE1,2):
             case CV_MAKETYPE(CV_USRTYPE1,3):
                 fs << (int)numValues[i];
                 break;
             case CV_32F:
                 fs << (float)numValues[i];
                 break;
             case CV_64F:
                 fs << (double)numValues[i];
                 break;
             case CV_USRTYPE1:
                 fs << strValues[i];
                 break;
             default:
                 fs << (double)numValues[i];
                 fs << "typename" << strValues[i];
                 break;
             }
             fs << "}";
         }
     }
     fs << "]";
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
                float dist = 0;
                if (dists.type() == CV_32S)
                    dist = static_cast<float>( dists.at<int>(i,j) );
                else
                    dist = std::sqrt(dists.at<float>(i,j));
                matches[i].push_back( DMatch( i, trainIdx, imgIdx, dist ) );
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
        flannIndex->radiusSearch( queryDescriptorsRow, indicesRow, distsRow, maxDistance*maxDistance, count, *searchParams );
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
        std::copy( collection.keypoints[i].begin(), collection.keypoints[i].end(), keypoints[i].begin() );

    std::copy( collection.startIndices.begin(), collection.startIndices.end(), startIndices.begin() );
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
    pointCount = 0;

    images.clear();
    keypoints.clear();
    startIndices.clear();
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

void GenericDescriptorMatcher::add( const vector<Mat>& images,
                                    vector<vector<KeyPoint> >& keypoints )
{
    CV_Assert( !images.empty() );
    CV_Assert( images.size() == keypoints.size() );

    for( size_t i = 0; i < images.size(); i++ )
    {
        CV_Assert( !images[i].empty() );
        KeyPointsFilter::runByImageBorder( keypoints[i], images[i].size(), 0 );
        KeyPointsFilter::runByKeypointSize( keypoints[i], std::numeric_limits<float>::epsilon() );
    }

    trainPointCollection.add( images, keypoints );
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
    matches.clear();

    if( queryImage.empty() || queryKeypoints.empty() )
        return;

    KeyPointsFilter::runByImageBorder( queryKeypoints, queryImage.size(), 0 );
    KeyPointsFilter::runByKeypointSize( queryKeypoints, std::numeric_limits<float>::epsilon() );

    train();
    knnMatchImpl( queryImage, queryKeypoints, matches, knn, masks, compactResult );
}

void GenericDescriptorMatcher::radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                            vector<vector<DMatch> >& matches, float maxDistance,
                                            const vector<Mat>& masks, bool compactResult )
{
    matches.clear();

    if( queryImage.empty() || queryKeypoints.empty() )
        return;

    KeyPointsFilter::runByImageBorder( queryKeypoints, queryImage.size(), 0 );
    KeyPointsFilter::runByKeypointSize( queryKeypoints, std::numeric_limits<float>::epsilon() );

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
    Ptr<GenericDescriptorMatcher> descriptorMatcher =
        Algorithm::create<GenericDescriptorMatcher>("DescriptorMatcher." + genericDescritptorMatcherType);

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
