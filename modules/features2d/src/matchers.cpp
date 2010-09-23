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
void DescriptorMatcher::add( const Mat& descriptors )
{
    if( m_train.empty() )
    {
        m_train = descriptors;
    }
    else
    {
        // merge train and descriptors
        Mat m( m_train.rows + descriptors.rows, m_train.cols, CV_32F );
        Mat m1 = m.rowRange( 0, m_train.rows );
        m_train.copyTo( m1 );
        Mat m2 = m.rowRange( m_train.rows + 1, m.rows );
        descriptors.copyTo( m2 );
        m_train = m;
    }
}

void DescriptorMatcher::match( const Mat& query, vector<int>& matches ) const
{
    matchImpl( query, m_train, matches, Mat() );
}

void DescriptorMatcher::match( const Mat& query, const Mat& mask,
                               vector<int>& matches ) const
{
    matchImpl( query, m_train, matches, mask );
}

void DescriptorMatcher::match( const Mat& query, vector<DMatch>& matches ) const
{
    matchImpl( query, m_train, matches, Mat() );
}

void DescriptorMatcher::match( const Mat& query, const Mat& mask,
                               vector<DMatch>& matches ) const
{
    matchImpl( query, m_train, matches, mask );
}

void DescriptorMatcher::match( const Mat& query, const Mat& train, vector<DMatch>& matches, const Mat& mask ) const
{
    matchImpl( query, train, matches, mask );
}

void DescriptorMatcher::match( const Mat& query, vector<vector<DMatch> >& matches, float threshold ) const
{
    matchImpl( query, m_train, matches, threshold, Mat() );
}

void DescriptorMatcher::match( const Mat& query, const Mat& mask,
                               vector<vector<DMatch> >& matches, float threshold ) const
{
    matchImpl( query, m_train, matches, threshold, mask );
}

void DescriptorMatcher::clear()
{
    m_train.release();
}

/*
 * BruteForceMatcher L2 specialization
 */
template<>
void BruteForceMatcher<L2<float> >::matchImpl( const Mat& query, const Mat& train, vector<DMatch>& matches, const Mat& mask ) const
{
    assert( mask.empty() || (mask.rows == query.rows && mask.cols == train.rows) );
    assert( query.cols == train.cols ||  query.empty() ||  train.empty() );

    matches.clear();
    matches.reserve( query.rows );
#if (!defined HAVE_EIGEN2)
    Mat norms;
    cv::reduce( train.mul( train ), norms, 1, 0);
    norms = norms.t();
    Mat desc_2t = train.t();
    for( int i=0;i<query.rows;i++ )
    {
        Mat distances = (-2)*query.row(i)*desc_2t;
        distances += norms;
        DMatch match;
        match.indexTrain = -1;
        double minVal;
        Point minLoc;
        if( mask.empty() )
        {
            minMaxLoc ( distances, &minVal, 0, &minLoc );
        }
        else
        {
            minMaxLoc ( distances, &minVal, 0, &minLoc, 0, mask.row( i ) );
        }
        match.indexTrain = minLoc.x;

        if( match.indexTrain != -1 )
        {
            match.indexQuery = i;
            double queryNorm = norm( query.row(i) );
            match.distance = (float)sqrt( minVal + queryNorm*queryNorm );
            matches.push_back( match );
        }
    }

#else
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> desc1t;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> desc2;
    cv2eigen( query.t(), desc1t);
    cv2eigen( train, desc2 );

    Eigen::Matrix<float, Eigen::Dynamic, 1> norms = desc2.rowwise().squaredNorm() / 2;

    if( mask.empty() )
    {
        for( int i=0;i<query.rows;i++ )
        {
            Eigen::Matrix<float, Eigen::Dynamic, 1> distances = desc2*desc1t.col(i);
            distances -= norms;
            DMatch match;
            match.indexQuery = i;
            match.distance = sqrt( (-2)*distances.maxCoeff( &match.indexTrain ) + desc1t.col(i).squaredNorm() );
            matches.push_back( match );
        }
    }
    else
    {
        for( int i=0;i<query.rows;i++ )
        {
            Eigen::Matrix<float, Eigen::Dynamic, 1> distances = desc2*desc1t.col(i);
            distances -= norms;

            float maxCoeff = -std::numeric_limits<float>::max();
            DMatch match;
            match.indexTrain = -1;
            for( int j=0;j<train.rows;j++ )
            {
                if( possibleMatch( mask, i, j ) && distances( j, 0 ) > maxCoeff )
                {
                    maxCoeff = distances( j, 0 );
                    match.indexTrain = j;
                }
            }

            if( match.indexTrain != -1 )
            {
                match.indexQuery = i;
                match.distance = sqrt( (-2)*maxCoeff + desc1t.col(i).squaredNorm() );
                matches.push_back( match );
            }
        }
    }
#endif
}

/****************************************************************************************\
*                Factory function for descriptor matcher creating                        *
\****************************************************************************************/
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

    return dm;
}

/****************************************************************************************\
*                                GenericDescriptorMatch                                  *
\****************************************************************************************/
/*
 * KeyPointCollection
 */
void KeyPointCollection::add( const Mat& _image, const vector<KeyPoint>& _points )
{
    // update m_start_indices
    if( startIndices.empty() )
        startIndices.push_back(0);
    else
        startIndices.push_back((int)(*startIndices.rbegin() + points.rbegin()->size()));

    // add image and keypoints
    images.push_back(_image);
    points.push_back(_points);
}

KeyPoint KeyPointCollection::getKeyPoint( int index ) const
{
    size_t i = 0;
    for(; i < startIndices.size() && startIndices[i] <= index; i++);
    i--;
    assert(i < startIndices.size() && (size_t)index - startIndices[i] < points[i].size());

    return points[i][index - startIndices[i]];
}

size_t KeyPointCollection::calcKeypointCount() const
{
    if( startIndices.empty() )
        return 0;
    return *startIndices.rbegin() + points.rbegin()->size();
}

void KeyPointCollection::clear()
{
    images.clear();
    points.clear();
    startIndices.clear();
}

/*
 * GenericDescriptorMatch
 */

void GenericDescriptorMatch::match( const Mat&, vector<KeyPoint>&, vector<DMatch>& )
{
}

void GenericDescriptorMatch::match( const Mat&, vector<KeyPoint>&, vector<vector<DMatch> >&, float )
{
}

void GenericDescriptorMatch::add( KeyPointCollection& collection )
{
    for( size_t i = 0; i < collection.images.size(); i++ )
        add( collection.images[i], collection.points[i] );
}

void GenericDescriptorMatch::classify( const Mat& image, vector<cv::KeyPoint>& points )
{
    vector<int> keypointIndices;
    match( image, points, keypointIndices );

    // remap keypoint indices to descriptors
    for( size_t i = 0; i < keypointIndices.size(); i++ )
        points[i].class_id = collection.getKeyPoint(keypointIndices[i]).class_id;
};

void GenericDescriptorMatch::clear()
{
    collection.clear();
}

/****************************************************************************************\
*                                OneWayDescriptorMatch                                  *
\****************************************************************************************/
OneWayDescriptorMatch::OneWayDescriptorMatch()
{}

OneWayDescriptorMatch::OneWayDescriptorMatch( const Params& _params)
{
    initialize(_params);
}

OneWayDescriptorMatch::~OneWayDescriptorMatch()
{}

void OneWayDescriptorMatch::initialize( const Params& _params, OneWayDescriptorBase *_base)
{
    base.release();
    if (_base != 0)
    {
        base = _base;
    }
    params = _params;
}

void OneWayDescriptorMatch::add( const Mat& image, vector<KeyPoint>& keypoints )
{
    if( base.empty() )
        base = new OneWayDescriptorObject( params.patchSize, params.poseCount, params.pcaFilename,
                                           params.trainPath, params.trainImagesList, params.minScale, params.maxScale, params.stepScale);

    size_t trainFeatureCount = keypoints.size();

    base->Allocate( (int)trainFeatureCount );

    IplImage _image = image;
    for( size_t i = 0; i < keypoints.size(); i++ )
        base->InitializeDescriptor( (int)i, &_image, keypoints[i], "" );

    collection.add( Mat(), keypoints );

#if defined(_KDTREE)
    base->ConvertDescriptorsArrayToTree();
#endif
}

void OneWayDescriptorMatch::add( KeyPointCollection& keypoints )
{
    if( base.empty() )
        base = new OneWayDescriptorObject( params.patchSize, params.poseCount, params.pcaFilename,
                                           params.trainPath, params.trainImagesList, params.minScale, params.maxScale, params.stepScale);

    size_t trainFeatureCount = keypoints.calcKeypointCount();

    base->Allocate( (int)trainFeatureCount );

    int count = 0;
    for( size_t i = 0; i < keypoints.points.size(); i++ )
    {
        for( size_t j = 0; j < keypoints.points[i].size(); j++ )
        {
            IplImage img = keypoints.images[i];
            base->InitializeDescriptor( count++, &img, keypoints.points[i][j], "" );
        }

        collection.add( Mat(), keypoints.points[i] );
    }

#if defined(_KDTREE)
    base->ConvertDescriptorsArrayToTree();
#endif
}

void OneWayDescriptorMatch::match( const Mat& image, vector<KeyPoint>& points, vector<int>& indices)
{
    vector<DMatch> matchings( points.size() );
    indices.resize(points.size());

    match( image, points, matchings );

    for( size_t i = 0; i < points.size(); i++ )
        indices[i] = matchings[i].indexTrain;
}

void OneWayDescriptorMatch::match( const Mat& image, vector<KeyPoint>& points, vector<DMatch>& matches )
{
    matches.resize( points.size() );
    IplImage _image = image;
    for( size_t i = 0; i < points.size(); i++ )
    {
        int poseIdx = -1;

        DMatch match;
        match.indexQuery = (int)i;
        match.indexTrain = -1;
        base->FindDescriptor( &_image, points[i].pt, match.indexTrain, poseIdx, match.distance );
        matches[i] = match;
    }
}

void OneWayDescriptorMatch::match( const Mat& image, vector<KeyPoint>& points, vector<vector<DMatch> >& matches, float /*threshold*/ )
{
    matches.clear();
    matches.resize( points.size() );

    vector<DMatch> dmatches;
    match( image, points, dmatches );
    for( size_t i=0;i<matches.size();i++ )
    {
        matches[i].push_back( dmatches[i] );
    }

    /*
    printf("Start matching %d points\n", points.size());
    //std::cout << "Start matching " << points.size() << "points\n";
    assert(collection.images.size() == 1);
    int n = collection.points[0].size();

    printf("n = %d\n", n);
    for( size_t i = 0; i < points.size(); i++ )
    {
        //printf("Matching %d\n", i);
        //int poseIdx = -1;

        DMatch match;
        match.indexQuery = i;
        match.indexTrain = -1;


        CvPoint pt = points[i].pt;
        CvRect roi = cvRect(cvRound(pt.x - 24/4),
                            cvRound(pt.y - 24/4),
                            24/2, 24/2);
        cvSetImageROI(&_image, roi);

        std::vector<int> desc_idxs;
        std::vector<int> pose_idxs;
        std::vector<float> distances;
        std::vector<float> _scales;


        base->FindDescriptor(&_image, n, desc_idxs, pose_idxs, distances, _scales);
        cvResetImageROI(&_image);

        for( int j=0;j<n;j++ )
        {
            match.indexTrain = desc_idxs[j];
            match.distance = distances[j];
            matches[i].push_back( match );
        }

        //sort( matches[i].begin(), matches[i].end(), compareIndexTrain );
        //for( int j=0;j<n;j++ )
        //{
            //printf( "%d %f;  ",matches[i][j].indexTrain, matches[i][j].distance);
        //}
        //printf("\n\n\n");



        //base->FindDescriptor( &_image, 100, points[i].pt, match.indexTrain, poseIdx, match.distance );
        //matches[i].push_back( match );
    }
    */
}


void OneWayDescriptorMatch::read( const FileNode &fn )
{
    base = new OneWayDescriptorObject( params.patchSize, params.poseCount, string (), string (), string (),
                                       params.minScale, params.maxScale, params.stepScale );
    base->Read (fn);
}


void OneWayDescriptorMatch::write( FileStorage& fs ) const
{
    base->Write (fs);
}

void OneWayDescriptorMatch::classify( const Mat& image, vector<KeyPoint>& points )
{
    IplImage _image = image;
    for( size_t i = 0; i < points.size(); i++ )
    {
        int descIdx = -1;
        int poseIdx = -1;
        float distance;
        base->FindDescriptor(&_image, points[i].pt, descIdx, poseIdx, distance);
        points[i].class_id = collection.getKeyPoint(descIdx).class_id;
    }
}

void OneWayDescriptorMatch::clear ()
{
    GenericDescriptorMatch::clear();
    base->clear ();
}

/****************************************************************************************\
*                                  FernDescriptorMatch                                   *
\****************************************************************************************/
FernDescriptorMatch::Params::Params( int _nclasses, int _patchSize, int _signatureSize,
                                     int _nstructs, int _structSize, int _nviews, int _compressionMethod,
                                     const PatchGenerator& _patchGenerator ) :
    nclasses(_nclasses), patchSize(_patchSize), signatureSize(_signatureSize),
    nstructs(_nstructs), structSize(_structSize), nviews(_nviews),
    compressionMethod(_compressionMethod), patchGenerator(_patchGenerator)
{}

FernDescriptorMatch::Params::Params( const string& _filename )
{
    filename = _filename;
}

FernDescriptorMatch::FernDescriptorMatch()
{}

FernDescriptorMatch::FernDescriptorMatch( const Params& _params )
{
    params = _params;
}

FernDescriptorMatch::~FernDescriptorMatch()
{}

void FernDescriptorMatch::initialize( const Params& _params )
{
    classifier.release();
    params = _params;
    if( !params.filename.empty() )
    {
        classifier = new FernClassifier;
        FileStorage fs(params.filename, FileStorage::READ);
        if( fs.isOpened() )
            classifier->read( fs.getFirstTopLevelNode() );
    }
}

void FernDescriptorMatch::add( const Mat& image, vector<KeyPoint>& keypoints )
{
    if( params.filename.empty() )
        collection.add( image, keypoints );
}

void FernDescriptorMatch::trainFernClassifier()
{
    if( classifier.empty() )
    {
        assert( params.filename.empty() );

        vector<vector<Point2f> > points;
        for( size_t imgIdx = 0; imgIdx < collection.images.size(); imgIdx++ )
            KeyPoint::convert( collection.points[imgIdx], points[imgIdx] );

        classifier = new FernClassifier( points, collection.images, vector<vector<int> >(), 0, // each points is a class
                                         params.patchSize, params.signatureSize, params.nstructs, params.structSize,
                                         params.nviews, params.compressionMethod, params.patchGenerator );
    }
}

void FernDescriptorMatch::calcBestProbAndMatchIdx( const Mat& image, const Point2f& pt,
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

void FernDescriptorMatch::match( const Mat& image, vector<KeyPoint>& keypoints, vector<int>& indices )
{
    trainFernClassifier();

    indices.resize( keypoints.size() );
    vector<float> signature( (size_t)classifier->getClassCount() );

    for( size_t pi = 0; pi < keypoints.size(); pi++ )
    {
        //calcBestProbAndMatchIdx( image, keypoints[pi].pt, bestProb, indices[pi], signature );
        //TODO: use octave and image pyramid
        indices[pi] = (*classifier)(image, keypoints[pi].pt, signature);
    }
}

void FernDescriptorMatch::match( const Mat& image, vector<KeyPoint>& keypoints, vector<DMatch>& matches )
{
    trainFernClassifier();

    matches.resize( keypoints.size() );
    vector<float> signature( (size_t)classifier->getClassCount() );

    for( int pi = 0; pi < (int)keypoints.size(); pi++ )
    {
        matches[pi].indexQuery = pi;
        calcBestProbAndMatchIdx( image, keypoints[pi].pt, matches[pi].distance, matches[pi].indexTrain, signature );
        //matching[pi].distance is log of probability so we need to transform it
        matches[pi].distance = -matches[pi].distance;
    }
}

void FernDescriptorMatch::match( const Mat& image, vector<KeyPoint>& keypoints, vector<vector<DMatch> >& matches, float threshold )
{
    trainFernClassifier();

    matches.resize( keypoints.size() );
    vector<float> signature( (size_t)classifier->getClassCount() );

    for( int pi = 0; pi < (int)keypoints.size(); pi++ )
    {
        (*classifier)( image, keypoints[pi].pt, signature);

        DMatch match;
        match.indexQuery = pi;

        for( int ci = 0; ci < classifier->getClassCount(); ci++ )
        {
            if( -signature[ci] < threshold )
            {
                match.distance = -signature[ci];
                match.indexTrain = ci;
                matches[pi].push_back( match );
            }
        }
    }
}

void FernDescriptorMatch::classify( const Mat& image, vector<KeyPoint>& keypoints )
{
    trainFernClassifier();

    vector<float> signature( (size_t)classifier->getClassCount() );
    for( size_t pi = 0; pi < keypoints.size(); pi++ )
    {
        float bestProb = 0;
        int bestMatchIdx = -1;
        calcBestProbAndMatchIdx( image, keypoints[pi].pt, bestProb, bestMatchIdx, signature );
        keypoints[pi].class_id = collection.getKeyPoint(bestMatchIdx).class_id;
    }
}

void FernDescriptorMatch::read( const FileNode &fn )
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

void FernDescriptorMatch::write( FileStorage& fs ) const
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

void FernDescriptorMatch::clear ()
{
    GenericDescriptorMatch::clear();
    classifier.release();
}

/****************************************************************************************\
*                                  VectorDescriptorMatch                                 *
\****************************************************************************************/
void VectorDescriptorMatch::add( const Mat& image, vector<KeyPoint>& keypoints )
{
    Mat descriptors;
    extractor->compute( image, keypoints, descriptors );
    matcher->add( descriptors );

    collection.add( Mat(), keypoints );
};

void VectorDescriptorMatch::match( const Mat& image, vector<KeyPoint>& points, vector<int>& keypointIndices )
{
    Mat descriptors;
    extractor->compute( image, points, descriptors );

    matcher->match( descriptors, keypointIndices );
};

void VectorDescriptorMatch::match( const Mat& image, vector<KeyPoint>& points, vector<DMatch>& matches )
{
    Mat descriptors;
    extractor->compute( image, points, descriptors );

    matcher->match( descriptors, matches );
}

void VectorDescriptorMatch::match( const Mat& image, vector<KeyPoint>& points,
                                   vector<vector<DMatch> >& matches, float threshold )
{
    Mat descriptors;
    extractor->compute( image, points, descriptors );

    matcher->match( descriptors, matches, threshold );
}

void VectorDescriptorMatch::clear()
{
    GenericDescriptorMatch::clear();
    matcher->clear();
}

void VectorDescriptorMatch::read( const FileNode& fn )
{
    GenericDescriptorMatch::read(fn);
    extractor->read (fn);
}

void VectorDescriptorMatch::write (FileStorage& fs) const
{
    GenericDescriptorMatch::write(fs);
    extractor->write (fs);
}

/****************************************************************************************\
*                Factory function for GenericDescriptorMatch creating                    *
\****************************************************************************************/
Ptr<GenericDescriptorMatch> createGenericDescriptorMatch( const string& genericDescritptorMatchType,
                                                          const string &paramsFilename )
{
    GenericDescriptorMatch *descriptorMatch = 0;
    if( ! genericDescritptorMatchType.compare("ONEWAY") )
    {
        descriptorMatch = new OneWayDescriptorMatch();
    }
    else if( ! genericDescritptorMatchType.compare("FERN") )
    {
        descriptorMatch = new FernDescriptorMatch();
    }
    else if( ! genericDescritptorMatchType.compare ("CALONDER") )
    {
        //descriptorMatch = new CalonderDescriptorMatch ();
    }

    if( !paramsFilename.empty() && descriptorMatch != 0 )
    {
        FileStorage fs = FileStorage( paramsFilename, FileStorage::READ );
        if( fs.isOpened() )
        {
            descriptorMatch->read( fs.root() );
            fs.release();
        }
    }

    return descriptorMatch;
}

}
