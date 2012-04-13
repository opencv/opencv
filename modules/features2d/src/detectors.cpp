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

namespace cv
{
    
/*
 *  FeatureDetector
 */

FeatureDetector::~FeatureDetector()
{}

void FeatureDetector::detect( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    keypoints.clear();

    if( image.empty() )
        return;

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()) );

    detectImpl( image, keypoints, mask );
}

void FeatureDetector::detect(const vector<Mat>& imageCollection, vector<vector<KeyPoint> >& pointCollection, const vector<Mat>& masks ) const
{
    pointCollection.resize( imageCollection.size() );
    for( size_t i = 0; i < imageCollection.size(); i++ )
        detect( imageCollection[i], pointCollection[i], masks.empty() ? Mat() : masks[i] );
}

/*void FeatureDetector::read( const FileNode& )
{}

void FeatureDetector::write( FileStorage& ) const
{}*/

bool FeatureDetector::empty() const
{
    return false;
}

void FeatureDetector::removeInvalidPoints( const Mat& mask, vector<KeyPoint>& keypoints )
{
    KeyPointsFilter::runByPixelsMask( keypoints, mask );
}

Ptr<FeatureDetector> FeatureDetector::create( const string& detectorType )
{
    if( detectorType.find("Grid") == 0 )
    {
        return new GridAdaptedFeatureDetector(FeatureDetector::create(
                                detectorType.substr(strlen("Grid"))));
    }
    
    if( detectorType.find("Pyramid") == 0 )
    {
        return new PyramidAdaptedFeatureDetector(FeatureDetector::create(
                                detectorType.substr(strlen("Pyramid"))));
    }
    
    if( detectorType.find("Dynamic") == 0 )
    {
        return new DynamicAdaptedFeatureDetector(AdjusterAdapter::create(
                                detectorType.substr(strlen("Dynamic"))));
    }
    
    if( detectorType.compare( "HARRIS" ) == 0 )
    {
        Ptr<FeatureDetector> fd = FeatureDetector::create("GFTT");
        fd->set("useHarrisDetector", true);
        return fd;
    }

    return Algorithm::create<FeatureDetector>("Feature2D." + detectorType);
}


GFTTDetector::GFTTDetector( int _nfeatures, double _qualityLevel,
                            double _minDistance, int _blockSize,
                            bool _useHarrisDetector, double _k )
    : nfeatures(_nfeatures), qualityLevel(_qualityLevel), minDistance(_minDistance),
    blockSize(_blockSize), useHarrisDetector(_useHarrisDetector), k(_k)
{
}

void GFTTDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    vector<Point2f> corners;
    goodFeaturesToTrack( grayImage, corners, nfeatures, qualityLevel, minDistance, mask,
                         blockSize, useHarrisDetector, k );
    keypoints.resize(corners.size());
    vector<Point2f>::const_iterator corner_it = corners.begin();
    vector<KeyPoint>::iterator keypoint_it = keypoints.begin();
    for( ; corner_it != corners.end(); ++corner_it, ++keypoint_it )
    {
        *keypoint_it = KeyPoint( *corner_it, (float)blockSize );
    }
}

static Algorithm* createGFTT() { return new GFTTDetector; }
static Algorithm* createHarris()
{
    GFTTDetector* d = new GFTTDetector;
    d->set("useHarris", true);
    return d;
}

static AlgorithmInfo gftt_info("Feature2D.GFTT", createGFTT);
static AlgorithmInfo harris_info("Feature2D.HARRIS", createHarris);
    
AlgorithmInfo* GFTTDetector::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        GFTTDetector obj;
        gftt_info.addParam(obj, "nfeatures", obj.nfeatures);
        gftt_info.addParam(obj, "qualityLevel", obj.qualityLevel);
        gftt_info.addParam(obj, "minDistance", obj.minDistance);
        gftt_info.addParam(obj, "useHarrisDetector", obj.useHarrisDetector);
        gftt_info.addParam(obj, "k", obj.k);
        
        harris_info.addParam(obj, "nfeatures", obj.nfeatures);
        harris_info.addParam(obj, "qualityLevel", obj.qualityLevel);
        harris_info.addParam(obj, "minDistance", obj.minDistance);
        harris_info.addParam(obj, "useHarrisDetector", obj.useHarrisDetector);
        harris_info.addParam(obj, "k", obj.k);
        
        initialized = true;
    }
    return &gftt_info;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 *  DenseFeatureDetector
 */
DenseFeatureDetector::DenseFeatureDetector( float _initFeatureScale, int _featureScaleLevels, 
									  float _featureScaleMul, int _initXyStep, 
									  int _initImgBound, bool _varyXyStepWithScale, 
									  bool _varyImgBoundWithScale ) :
	initFeatureScale(_initFeatureScale), featureScaleLevels(_featureScaleLevels),
	featureScaleMul(_featureScaleMul), initXyStep(_initXyStep), initImgBound(_initImgBound),
	varyXyStepWithScale(_varyXyStepWithScale), varyImgBoundWithScale(_varyImgBoundWithScale)
{}


void DenseFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    float curScale = static_cast<float>(initFeatureScale);
    int curStep = initXyStep;
    int curBound = initImgBound;
    for( int curLevel = 0; curLevel < featureScaleLevels; curLevel++ )
    {
        for( int x = curBound; x < image.cols - curBound; x += curStep )
        {
            for( int y = curBound; y < image.rows - curBound; y += curStep )
            {
                keypoints.push_back( KeyPoint(static_cast<float>(x), static_cast<float>(y), curScale) );
            }
        }

        curScale = static_cast<float>(curScale * featureScaleMul);
        if( varyXyStepWithScale ) curStep = static_cast<int>( curStep * featureScaleMul + 0.5f );
        if( varyImgBoundWithScale ) curBound = static_cast<int>( curBound * featureScaleMul + 0.5f );
    }

    KeyPointsFilter::runByPixelsMask( keypoints, mask );
}
    

static Algorithm* createDense() { return new DenseFeatureDetector; }
static AlgorithmInfo dense_info("Feature2D.Dense", createDense);
    
AlgorithmInfo* DenseFeatureDetector::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        DenseFeatureDetector obj;
        dense_info.addParam(obj, "initFeatureScale", obj.initFeatureScale);
        dense_info.addParam(obj, "featureScaleLevels", obj.featureScaleLevels);
        dense_info.addParam(obj, "featureScaleMul", obj.featureScaleMul);
        dense_info.addParam(obj, "initXyStep", obj.initXyStep);
        dense_info.addParam(obj, "initImgBound", obj.initImgBound);
        dense_info.addParam(obj, "varyXyStepWithScale", obj.varyXyStepWithScale);
        dense_info.addParam(obj, "varyImgBoundWithScale", obj.varyImgBoundWithScale);
        
        initialized = true;
    }
    return &dense_info;
}    

/*
 *  GridAdaptedFeatureDetector
 */
GridAdaptedFeatureDetector::GridAdaptedFeatureDetector( const Ptr<FeatureDetector>& _detector,
                                                        int _maxTotalKeypoints, int _gridRows, int _gridCols )
    : detector(_detector), maxTotalKeypoints(_maxTotalKeypoints), gridRows(_gridRows), gridCols(_gridCols)
{}

bool GridAdaptedFeatureDetector::empty() const
{
    return detector.empty() || (FeatureDetector*)detector->empty();
}

struct ResponseComparator
{
    bool operator() (const KeyPoint& a, const KeyPoint& b)
    {
        return std::abs(a.response) > std::abs(b.response);
    }
};

void keepStrongest( int N, vector<KeyPoint>& keypoints )
{
    if( (int)keypoints.size() > N )
    {
        vector<KeyPoint>::iterator nth = keypoints.begin() + N;
        std::nth_element( keypoints.begin(), nth, keypoints.end(), ResponseComparator() );
        keypoints.erase( nth, keypoints.end() );
    }
}

void GridAdaptedFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    keypoints.reserve(maxTotalKeypoints);

    int maxPerCell = maxTotalKeypoints / (gridRows * gridCols);
    for( int i = 0; i < gridRows; ++i )
    {
        Range row_range((i*image.rows)/gridRows, ((i+1)*image.rows)/gridRows);
        for( int j = 0; j < gridCols; ++j )
        {
            Range col_range((j*image.cols)/gridCols, ((j+1)*image.cols)/gridCols);
            Mat sub_image = image(row_range, col_range);
            Mat sub_mask;
            if( !mask.empty() )
                sub_mask = mask(row_range, col_range);

            vector<KeyPoint> sub_keypoints;
            detector->detect( sub_image, sub_keypoints, sub_mask );
            keepStrongest( maxPerCell, sub_keypoints );
            std::vector<cv::KeyPoint>::iterator it = sub_keypoints.begin(),
                                                end = sub_keypoints.end();
            for( ; it != end; ++it )
            {
                it->pt.x += col_range.start;
                it->pt.y += row_range.start;
            }

            keypoints.insert( keypoints.end(), sub_keypoints.begin(), sub_keypoints.end() );
        }
    }
}

/*
 *  PyramidAdaptedFeatureDetector
 */
PyramidAdaptedFeatureDetector::PyramidAdaptedFeatureDetector( const Ptr<FeatureDetector>& _detector, int _maxLevel )
    : detector(_detector), maxLevel(_maxLevel)
{}

bool PyramidAdaptedFeatureDetector::empty() const
{
    return detector.empty() || (FeatureDetector*)detector->empty();
}

void PyramidAdaptedFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    Mat src = image;
    Mat src_mask = mask;

    Mat dilated_mask;
    if( !mask.empty() )
    {
        dilate( mask, dilated_mask, Mat() );
        Mat mask255( mask.size(), CV_8UC1, Scalar(0) );
        mask255.setTo( Scalar(255), dilated_mask != 0 );
        dilated_mask = mask255;
    }

    for( int l = 0, multiplier = 1; l <= maxLevel; ++l, multiplier *= 2 )
    {
        // Detect on current level of the pyramid
        vector<KeyPoint> new_pts;
        detector->detect( src, new_pts, src_mask );
        vector<KeyPoint>::iterator it = new_pts.begin(),
                                   end = new_pts.end();
        for( ; it != end; ++it)
        {
            it->pt.x *= multiplier;
            it->pt.y *= multiplier;
            it->size *= multiplier;
            it->octave = l;
        }
        keypoints.insert( keypoints.end(), new_pts.begin(), new_pts.end() );

        // Downsample
        if( l < maxLevel )
        {
            Mat dst;
            pyrDown( src, dst );
            src = dst;

            if( !mask.empty() )
                resize( dilated_mask, src_mask, src.size(), 0, 0, CV_INTER_AREA );
        }
    }

    if( !mask.empty() )
        KeyPointsFilter::runByPixelsMask( keypoints, mask );
}
    
    
/////////////////////// AlgorithmInfo for various detector & descriptors ////////////////////////////

/* NOTE!!!
   All the AlgorithmInfo-related stuff should be in the same file as initModule_features2d().
   Otherwise, linker may throw away some seemingly unused stuff.
*/
    
static Algorithm* createBRIEF() { return new BriefDescriptorExtractor; }
static AlgorithmInfo& brief_info()
{
    static AlgorithmInfo brief_info_var("Feature2D.BRIEF", createBRIEF);
    return brief_info_var;
}

static AlgorithmInfo& brief_info_auto = brief_info();

AlgorithmInfo* BriefDescriptorExtractor::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        BriefDescriptorExtractor brief;
        brief_info().addParam(brief, "bytes", brief.bytes_);
        
        initialized = true;
    }
    return &brief_info();
}
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
static Algorithm* createFAST() { return new FastFeatureDetector; }
static AlgorithmInfo& fast_info()
{
    static AlgorithmInfo fast_info_var("Feature2D.FAST", createFAST);
    return fast_info_var;
}

static AlgorithmInfo& fast_info_auto = fast_info();

AlgorithmInfo* FastFeatureDetector::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        FastFeatureDetector obj;
        fast_info().addParam(obj, "threshold", obj.threshold);
        fast_info().addParam(obj, "nonmaxSuppression", obj.nonmaxSuppression);
        
        initialized = true;
    }
    return &fast_info();
}
    

///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
static Algorithm* createStarDetector() { return new StarDetector; }
static AlgorithmInfo& star_info()
{
    static AlgorithmInfo star_info_var("Feature2D.STAR", createStarDetector);
    return star_info_var;
}

static AlgorithmInfo& star_info_auto = star_info();

AlgorithmInfo* StarDetector::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        StarDetector obj;
        star_info().addParam(obj, "maxSize", obj.maxSize);
        star_info().addParam(obj, "responseThreshold", obj.responseThreshold);
        star_info().addParam(obj, "lineThresholdProjected", obj.lineThresholdProjected);
        star_info().addParam(obj, "lineThresholdBinarized", obj.lineThresholdBinarized);
        star_info().addParam(obj, "suppressNonmaxSize", obj.suppressNonmaxSize);
        
        initialized = true;
    }
    return &star_info();
}    
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
static Algorithm* createMSER() { return new MSER; }
static AlgorithmInfo& mser_info()
{
    static AlgorithmInfo mser_info_var("Feature2D.MSER", createMSER);
    return mser_info_var;
}

static AlgorithmInfo& mser_info_auto = mser_info();

AlgorithmInfo* MSER::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        MSER obj;
        mser_info().addParam(obj, "delta", obj.delta);
        mser_info().addParam(obj, "minArea", obj.minArea);
        mser_info().addParam(obj, "maxArea", obj.maxArea);
        mser_info().addParam(obj, "maxVariation", obj.maxVariation);
        mser_info().addParam(obj, "minDiversity", obj.minDiversity);
        mser_info().addParam(obj, "maxEvolution", obj.maxEvolution);
        mser_info().addParam(obj, "areaThreshold", obj.areaThreshold);
        mser_info().addParam(obj, "minMargin", obj.minMargin);
        mser_info().addParam(obj, "edgeBlurSize", obj.edgeBlurSize);
        
        initialized = true;
    }
    return &mser_info();
}
    
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////

static Algorithm* createORB() { return new ORB; }
static AlgorithmInfo& orb_info()
{
    static AlgorithmInfo orb_info_var("Feature2D.ORB", createORB);
    return orb_info_var;
}

static AlgorithmInfo& orb_info_auto = orb_info();

AlgorithmInfo* ORB::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        ORB obj;
        orb_info().addParam(obj, "nFeatures", obj.nfeatures);
        orb_info().addParam(obj, "scaleFactor", obj.scaleFactor);
        orb_info().addParam(obj, "nLevels", obj.nlevels);
        orb_info().addParam(obj, "firstLevel", obj.firstLevel);
        orb_info().addParam(obj, "edgeThreshold", obj.edgeThreshold);
        orb_info().addParam(obj, "patchSize", obj.patchSize);
        orb_info().addParam(obj, "WTA_K", obj.WTA_K);
        orb_info().addParam(obj, "scoreType", obj.scoreType);
        
        initialized = true;
    }
    return &orb_info();
}
    
bool initModule_features2d(void)
{
    Ptr<Algorithm> brief = createBRIEF(), orb = createORB(),
        star = createStarDetector(), fastd = createFAST(), mser = createMSER();
    return brief->info() != 0 && orb->info() != 0 && star->info() != 0 &&
        fastd->info() != 0 && mser->info() != 0;
}
    
}
