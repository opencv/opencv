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

//#define _KDTREE

using namespace std;
namespace cv
{

void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                  const Mat& img2,const vector<KeyPoint>& keypoints2,
                  const vector<int>& matches, Mat& outImg,
                  const Scalar& matchColor, const Scalar& singlePointColor,
                  const vector<char>& matchesMask, int flags )
{
    Size size( img1.cols + img2.cols, MAX(img1.rows, img2.rows) );
    if( flags & DrawMatchesFlags::DRAW_OVER_OUTIMG )
    {
        if( size.width > outImg.cols || size.height > outImg.rows )
            CV_Error( CV_StsBadSize, "outImg has size less than need to draw img1 and img2 together" );
    }
    else
    {
        outImg.create( size, CV_MAKETYPE(img1.depth(), 3) );
        Mat outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
        cvtColor( img1, outImg1, CV_GRAY2RGB );
        Mat outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
        cvtColor( img2, outImg2, CV_GRAY2RGB );
    }

    RNG rng;
    // draw keypoints
    if( !(flags & DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS) )
    {
        bool isRandSinglePointColor = singlePointColor == Scalar::all(-1);
        for( vector<KeyPoint>::const_iterator it = keypoints1.begin(); it < keypoints1.end(); ++it )
        {
            circle( outImg, it->pt, 3, isRandSinglePointColor ?
                    Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)) : singlePointColor );
        }
        for( vector<KeyPoint>::const_iterator it = keypoints2.begin(); it < keypoints2.end(); ++it )
        {
            Point p = it->pt;
            circle( outImg, Point2f(p.x+img1.cols, p.y), 3, isRandSinglePointColor ?
                    Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)) : singlePointColor );
        }
     }

    // draw matches
    bool isRandMatchColor = matchColor == Scalar::all(-1);
    if( matches.size() != keypoints1.size() )
        CV_Error( CV_StsBadSize, "matches must have the same size as keypoints1" );
    if( !matchesMask.empty() && matchesMask.size() != keypoints1.size() )
        CV_Error( CV_StsBadSize, "mask must have the same size as keypoints1" );
    vector<int>::const_iterator mit = matches.begin();
    for( int i1 = 0; mit != matches.end(); ++mit, i1++ )
    {
        if( (matchesMask.empty() || matchesMask[i1] ) && *mit >= 0 )
        {
            Point2f pt1 = keypoints1[i1].pt,
                    pt2 = keypoints2[*mit].pt,
                    dpt2 = Point2f( std::min(pt2.x+img1.cols, float(outImg.cols-1)), pt2.y );
            Scalar randColor( rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256) );
            circle( outImg, pt1, 3, isRandMatchColor ? randColor : matchColor );
            circle( outImg, dpt2, 3, isRandMatchColor ? randColor : matchColor );
            line( outImg, pt1, dpt2, isRandMatchColor ? randColor : matchColor );
        }
    }
}

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

void SiftDescriptorExtractor::read (const FileNode &fn)
{
    double magnification = fn["magnification"];
    bool isNormalize = (int)fn["isNormalize"] != 0;
    bool recalculateAngles = (int)fn["recalculateAngles"] != 0;
    int nOctaves = fn["nOctaves"];
    int nOctaveLayers = fn["nOctaveLayers"];
    int firstOctave = fn["firstOctave"];
    int angleMode = fn["angleMode"];

    sift = SIFT( magnification, isNormalize, recalculateAngles, nOctaves, nOctaveLayers, firstOctave, angleMode );
}

void SiftDescriptorExtractor::write (FileStorage &fs) const
{
//    fs << "algorithm" << getAlgorithmName ();

    SIFT::CommonParams commParams = sift.getCommonParams ();
    SIFT::DescriptorParams descriptorParams = sift.getDescriptorParams ();
    fs << "magnification" << descriptorParams.magnification;
    fs << "isNormalize" << descriptorParams.isNormalize;
    fs << "recalculateAngles" << descriptorParams.recalculateAngles;
    fs << "nOctaves" << commParams.nOctaves;
    fs << "nOctaveLayers" << commParams.nOctaveLayers;
    fs << "firstOctave" << commParams.firstOctave;
    fs << "angleMode" << commParams.angleMode;
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

void SurfDescriptorExtractor::read( const FileNode &fn )
{
    int nOctaves = fn["nOctaves"];
    int nOctaveLayers = fn["nOctaveLayers"];
    bool extended = (int)fn["extended"] != 0;

    surf = SURF( 0.0, nOctaves, nOctaveLayers, extended );
}

void SurfDescriptorExtractor::write( FileStorage &fs ) const
{
//    fs << "algorithm" << getAlgorithmName ();

    fs << "nOctaves" << surf.nOctaves;
    fs << "nOctaveLayers" << surf.nOctaveLayers;
    fs << "extended" << surf.extended;
}

DescriptorExtractor* createDescriptorExtractor( const string& descriptorExtractorType )
{
    DescriptorExtractor* de = 0;
    if( !descriptorExtractorType.compare( "SIFT" ) )
    {
        de = new SiftDescriptorExtractor/*( double magnification=SIFT::DescriptorParams::GET_DEFAULT_MAGNIFICATION(),
                             bool isNormalize=true, bool recalculateAngles=true,
                             int nOctaves=SIFT::CommonParams::DEFAULT_NOCTAVES,
                             int nOctaveLayers=SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS,
                             int firstOctave=SIFT::CommonParams::DEFAULT_FIRST_OCTAVE,
                             int angleMode=SIFT::CommonParams::FIRST_ANGLE )*/;
    }
    else if( !descriptorExtractorType.compare( "SURF" ) )
    {
        de = new SurfDescriptorExtractor/*( int nOctaves=4, int nOctaveLayers=2, bool extended=false )*/;
    }
    else
    {
        //CV_Error( CV_StsBadArg, "unsupported descriptor extractor type");
    }
    return de;
}

DescriptorMatcher* createDescriptorMatcher( const string& descriptorMatcherType )
{
    DescriptorMatcher* dm = 0;
    if( !descriptorMatcherType.compare( "BruteForce" ) )
    {
        dm = new BruteForceMatcher<L2<float> >();
    }
    else if ( !descriptorMatcherType.compare( "BruteForce-L1" ) )
    {
        dm = new BruteForceMatcher<L1<float> >();
    }
    else
    {
        //CV_Error( CV_StsBadArg, "unsupported descriptor matcher type");
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

GenericDescriptorMatch* createGenericDescriptorMatch( const string& genericDescritptorMatchType, const string &paramsFilename )
{
    GenericDescriptorMatch *descriptorMatch = 0;
    if( ! genericDescritptorMatchType.compare ("ONEWAY") )
    {
        descriptorMatch = new OneWayDescriptorMatch ();
    }
    else if( ! genericDescritptorMatchType.compare ("FERN") )
    {
        FernDescriptorMatch::Params params;
        params.signatureSize = numeric_limits<int>::max();
        descriptorMatch = new FernDescriptorMatch (params);
    }
    else if( ! genericDescritptorMatchType.compare ("CALONDER") )
    {
        descriptorMatch = new CalonderDescriptorMatch ();
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
        match.indexQuery = i;
        match.indexTrain = -1;
        base->FindDescriptor( &_image, points[i].pt, match.indexTrain, poseIdx, match.distance );
        matches[i] = match;
    }
}

void OneWayDescriptorMatch::match( const Mat& image, vector<KeyPoint>& points, vector<vector<DMatch> >& matches, float threshold )
{
    matches.clear();
    matches.resize( points.size() );
    IplImage _image = image;


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

void CalonderDescriptorMatch::clear ()
{
    GenericDescriptorMatch::clear();
    classifier.release();
}

void CalonderDescriptorMatch::read( const FileNode &fn )
{
    params.numTrees = fn["numTrees"];
    params.depth = fn["depth"];
    params.views = fn["views"];
    params.patchSize = fn["patchSize"];
    params.reducedNumDim = (int) fn["reducedNumDim"];
    params.numQuantBits = fn["numQuantBits"];
    params.printStatus = (int) fn["printStatus"];
}

void CalonderDescriptorMatch::write( FileStorage& fs ) const
{
    fs << "numTrees" << params.numTrees;
    fs << "depth" << params.depth;
    fs << "views" << params.views;
    fs << "patchSize" << params.patchSize;
    fs << "reducedNumDim" << (int) params.reducedNumDim;
    fs << "numQuantBits" << params.numQuantBits;
    fs << "printStatus" << params.printStatus;
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
        vector<Ptr<Mat> > refimgs;
        vector<int> labels;
        for( size_t imageIdx = 0; imageIdx < collection.images.size(); imageIdx++ )
        {
            for( size_t pointIdx = 0; pointIdx < collection.points[imageIdx].size(); pointIdx++ )
            {
                refimgs.push_back(new Mat (collection.images[imageIdx]));
                points.push_back(collection.points[imageIdx][pointIdx].pt);
                labels.push_back(pointIdx);
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

    bestProb = -FLT_MAX;
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

    for( size_t pi = 0; pi < keypoints.size(); pi++ )
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

    for( size_t pi = 0; pi < keypoints.size(); pi++ )
    {
        (*classifier)( image, keypoints[pi].pt, signature);

        DMatch match;
        match.indexQuery = pi;

        for( size_t ci = 0; ci < (size_t)classifier->getClassCount(); ci++ )
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

}
