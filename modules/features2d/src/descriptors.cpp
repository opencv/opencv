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

using namespace std;
using namespace cv;

/****************************************************************************************\
*                                 DescriptorExtractor                                    *
\****************************************************************************************/
/*
 *   DescriptorExtractor
 */
struct RoiPredicate
{
    RoiPredicate(float _minX, float _minY, float _maxX, float _maxY)
        : minX(_minX), minY(_minY), maxX(_maxX), maxY(_maxY)
    {}

    bool operator()( const KeyPoint& keyPt) const
    {
        Point2f pt = keyPt.pt;
        return (pt.x < minX) || (pt.x >= maxX) || (pt.y < minY) || (pt.y >= maxY);
    }

    float minX, minY, maxX, maxY;
};

void DescriptorExtractor::removeBorderKeypoints( vector<KeyPoint>& keypoints,
                                                 Size imageSize, int borderPixels )
{
    keypoints.erase( remove_if(keypoints.begin(), keypoints.end(),
                               RoiPredicate((float)borderPixels, (float)borderPixels,
                                            (float)(imageSize.width - borderPixels),
                                            (float)(imageSize.height - borderPixels))),
                     keypoints.end());
}

/****************************************************************************************\
*                                SiftDescriptorExtractor                                  *
\****************************************************************************************/
SiftDescriptorExtractor::SiftDescriptorExtractor( double magnification, bool isNormalize, bool recalculateAngles,
                                                  int nOctaves, int nOctaveLayers, int firstOctave, int angleMode )
    : sift( magnification, isNormalize, recalculateAngles, nOctaves, nOctaveLayers, firstOctave, angleMode )
{}

void SiftDescriptorExtractor::compute( const Mat& image,
                                       vector<KeyPoint>& keypoints,
                                       Mat& descriptors) const
{
    bool useProvidedKeypoints = true;
    sift(image, Mat(), keypoints, descriptors, useProvidedKeypoints);
}

/****************************************************************************************\
*                                SurfDescriptorExtractor                                  *
\****************************************************************************************/
SurfDescriptorExtractor::SurfDescriptorExtractor( int nOctaves,
                                                  int nOctaveLayers, bool extended )
    : surf( 0.0, nOctaves, nOctaveLayers, extended )
{}

void SurfDescriptorExtractor::compute( const Mat& image,
                                       vector<KeyPoint>& keypoints,
                                       Mat& descriptors) const
{
    // Compute descriptors for given keypoints
    vector<float> _descriptors;
    Mat mask;
    bool useProvidedKeypoints = true;
    surf(image, mask, keypoints, _descriptors, useProvidedKeypoints);

    descriptors.create(keypoints.size(), surf.descriptorSize(), CV_32FC1);
    assert( (int)_descriptors.size() == descriptors.rows * descriptors.cols );
    std::copy(_descriptors.begin(), _descriptors.end(), descriptors.begin<float>());
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
        startIndices.push_back(*startIndices.rbegin() + points.rbegin()->size());

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

void OneWayDescriptorMatch::initialize( const Params& _params)
{
    base.release();
    params = _params;
}

void OneWayDescriptorMatch::add( const Mat& image, vector<KeyPoint>& keypoints )
{
    if( base.empty() )
        base = new OneWayDescriptorObject( params.patchSize, params.poseCount, params.pcaFilename,
                                           params.trainPath, params.trainImagesList, params.minScale, params.maxScale, params.stepScale);

    size_t trainFeatureCount = keypoints.size();

    base->Allocate( trainFeatureCount );

    IplImage _image = image;
    for( size_t i = 0; i < keypoints.size(); i++ )
        base->InitializeDescriptor( i, &_image, keypoints[i], "" );

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

    base->Allocate( trainFeatureCount );

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
    indices.resize(points.size());
    IplImage _image = image;
    for( size_t i = 0; i < points.size(); i++ )
    {
        int descIdx = -1;
        int poseIdx = -1;
        float distance;
        base->FindDescriptor( &_image, points[i].pt, descIdx, poseIdx, distance );
        indices[i] = descIdx;
    }
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
*                                CalonderDescriptorMatch                                 *
\****************************************************************************************/
CalonderDescriptorMatch::Params::Params( const RNG& _rng, const PatchGenerator& _patchGen,
                                         int _numTrees, int _depth, int _views,
                                         size_t _reducedNumDim,
                                         int _numQuantBits,
                                         bool _printStatus,
                                         int _patchSize ) :
        rng(_rng), patchGen(_patchGen), numTrees(_numTrees), depth(_depth), views(_views),
        patchSize(_patchSize), reducedNumDim(_reducedNumDim), numQuantBits(_numQuantBits), printStatus(_printStatus)
{}

CalonderDescriptorMatch::Params::Params( const string& _filename )
{
    filename = _filename;
}

CalonderDescriptorMatch::CalonderDescriptorMatch()
{}

CalonderDescriptorMatch::CalonderDescriptorMatch( const Params& _params )
{
    initialize(_params);
}

CalonderDescriptorMatch::~CalonderDescriptorMatch()
{}

void CalonderDescriptorMatch::initialize( const Params& _params )
{
    classifier.release();
    params = _params;
    if( !params.filename.empty() )
    {
        classifier = new RTreeClassifier;
        classifier->read( params.filename.c_str() );
    }
}

void CalonderDescriptorMatch::add( const Mat& image, vector<KeyPoint>& keypoints )
{
    if( params.filename.empty() )
        collection.add( image, keypoints );
}

Mat CalonderDescriptorMatch::extractPatch( const Mat& image, const Point& pt, int patchSize ) const
{
    const int offset = patchSize / 2;
    return image( Rect(pt.x - offset, pt.y - offset, patchSize, patchSize) );
}

void CalonderDescriptorMatch::calcBestProbAndMatchIdx( const Mat& image, const Point& pt,
                                                       float& bestProb, int& bestMatchIdx, float* signature )
{
    IplImage roi = extractPatch( image, pt, params.patchSize );
    classifier->getSignature( &roi, signature );

    bestProb = 0;
    bestMatchIdx = -1;
    for( size_t ci = 0; ci < (size_t)classifier->classes(); ci++ )
    {
        if( signature[ci] > bestProb )
        {
            bestProb = signature[ci];
            bestMatchIdx = ci;
        }
    }
}

void CalonderDescriptorMatch::trainRTreeClassifier()
{
    if( classifier.empty() )
    {
        assert( params.filename.empty() );
        classifier = new RTreeClassifier;

        vector<BaseKeypoint> baseKeyPoints;
        vector<IplImage> iplImages( collection.images.size() );
        for( size_t imageIdx = 0; imageIdx < collection.images.size(); imageIdx++ )
        {
            iplImages[imageIdx] = collection.images[imageIdx];
            for( size_t pointIdx = 0; pointIdx < collection.points[imageIdx].size(); pointIdx++ )
            {
                BaseKeypoint bkp;
                KeyPoint kp = collection.points[imageIdx][pointIdx];
                bkp.x = cvRound(kp.pt.x);
                bkp.y = cvRound(kp.pt.y);
                bkp.image = &iplImages[imageIdx];
                baseKeyPoints.push_back(bkp);
            }
        }
        classifier->train( baseKeyPoints, params.rng, params.patchGen, params.numTrees,
                           params.depth, params.views, params.reducedNumDim, params.numQuantBits,
                           params.printStatus );
    }
}

void CalonderDescriptorMatch::match( const Mat& image, vector<KeyPoint>& keypoints, vector<int>& indices )
{
    trainRTreeClassifier();

    float bestProb = 0;
    AutoBuffer<float> signature( classifier->classes() );
    indices.resize( keypoints.size() );

    for( size_t pi = 0; pi < keypoints.size(); pi++ )
        calcBestProbAndMatchIdx( image, keypoints[pi].pt, bestProb, indices[pi], signature );
}

void CalonderDescriptorMatch::classify( const Mat& image, vector<KeyPoint>& keypoints )
{
    trainRTreeClassifier();

    AutoBuffer<float> signature( classifier->classes() );
    for( size_t pi = 0; pi < keypoints.size(); pi++ )
    {
        float bestProb = 0;
        int bestMatchIdx = -1;
        calcBestProbAndMatchIdx( image, keypoints[pi].pt, bestProb, bestMatchIdx, signature );
        keypoints[pi].class_id = collection.getKeyPoint(bestMatchIdx).class_id;
    }
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

        vector<Point2f> points;
        vector<Ptr<Mat> > refimgs( collection.images.size() );
        vector<int> labels;
        for( size_t imageIdx = 0; imageIdx < collection.images.size(); imageIdx++ )
        {
            refimgs[imageIdx] = &collection.images[imageIdx];
            for( size_t pointIdx = 0; pointIdx < collection.points[imageIdx].size(); pointIdx++ )
            {
                points.push_back(collection.points[imageIdx][pointIdx].pt);
                labels.push_back(imageIdx);
            }
        }

        classifier = new FernClassifier( points, refimgs, labels, params.nclasses, params.patchSize,
                                         params.signatureSize, params.nstructs, params.structSize, params.nviews,
                                         params.compressionMethod, params.patchGenerator );
    }
}

void FernDescriptorMatch::calcBestProbAndMatchIdx( const Mat& image, const Point2f& pt,
                                                   float& bestProb, int& bestMatchIdx, vector<float>& signature )
{
    (*classifier)( image, pt, signature);

    bestProb = 0;
    bestMatchIdx = -1;
    for( size_t ci = 0; ci < (size_t)classifier->getClassCount(); ci++ )
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

    float bestProb = 0;
    indices.resize( keypoints.size() );
    vector<float> signature( (size_t)classifier->getClassCount() );

    for( size_t pi = 0; pi < keypoints.size(); pi++ )
        calcBestProbAndMatchIdx( image, keypoints[pi].pt, bestProb, indices[pi], signature );
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
