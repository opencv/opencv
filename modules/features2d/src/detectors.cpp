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
class MaskPredicate
{
public:
    MaskPredicate( const Mat& _mask ) : mask(_mask) {}
    bool operator() (const KeyPoint& key_pt) const
    {
      return mask.at<uchar>( (int)(key_pt.pt.y + 0.5f), (int)(key_pt.pt.x + 0.5f) ) == 0;
    }
private:
	const Mat mask;
};

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

void FeatureDetector::removeInvalidPoints( const Mat& mask, vector<KeyPoint>& keypoints )
{
    if( mask.empty() )
        return;

    keypoints.erase(remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
};

void FeatureDetector::read( const FileNode& )
{}

void FeatureDetector::write( FileStorage& ) const
{}

bool FeatureDetector::empty() const
{
    return false;
}

Ptr<FeatureDetector> FeatureDetector::create( const string& detectorType )
{
    FeatureDetector* fd = 0;
    size_t pos = 0;

    if( !detectorType.compare( "FAST" ) )
    {
        fd = new FastFeatureDetector();
    }
    else if( !detectorType.compare( "STAR" ) )
    {
        fd = new StarFeatureDetector();
    }
    else if( !detectorType.compare( "SIFT" ) )
    {
        fd = new SiftFeatureDetector();
    }
    else if( !detectorType.compare( "SURF" ) )
    {
        fd = new SurfFeatureDetector();
    }
    else if( !detectorType.compare( "MSER" ) )
    {
        fd = new MserFeatureDetector();
    }
    else if( !detectorType.compare( "GFTT" ) )
    {
        fd = new GoodFeaturesToTrackDetector();
    }
    else if( !detectorType.compare( "HARRIS" ) )
    {
        GoodFeaturesToTrackDetector::Params params;
        params.useHarrisDetector = true;
        fd = new GoodFeaturesToTrackDetector(params);
    }
    else if( (pos=detectorType.find("Grid")) == 0 )
    {
        pos += string("Grid").size();
        fd = new GridAdaptedFeatureDetector( FeatureDetector::create(detectorType.substr(pos)) );
    }
    else if( (pos=detectorType.find("Pyramid")) == 0 )
    {
        pos += string("Pyramid").size();
        fd = new PyramidAdaptedFeatureDetector( FeatureDetector::create(detectorType.substr(pos)) );
    }
    else if( (pos=detectorType.find("Dynamic")) == 0 )
    {
        pos += string("Dynamic").size();
        fd = new DynamicAdaptedFeatureDetector( AdjusterAdapter::create(detectorType.substr(pos)) );
    }

    return fd;
}

/*
 *   FastFeatureDetector
 */
FastFeatureDetector::FastFeatureDetector( int _threshold, bool _nonmaxSuppression )
  : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression)
{}

void FastFeatureDetector::read (const FileNode& fn)
{
    threshold = fn["threshold"];
    nonmaxSuppression = (int)fn["nonmaxSuppression"] ? true : false;
}

void FastFeatureDetector::write (FileStorage& fs) const
{
    fs << "threshold" << threshold;
    fs << "nonmaxSuppression" << nonmaxSuppression;
}

void FastFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );
    FAST( grayImage, keypoints, threshold, nonmaxSuppression );
    removeInvalidPoints( mask, keypoints );
}

/*
 *  GoodFeaturesToTrackDetector
 */
GoodFeaturesToTrackDetector::Params::Params( int _maxCorners, double _qualityLevel, double _minDistance,
                                             int _blockSize, bool _useHarrisDetector, double _k ) :
    maxCorners(_maxCorners), qualityLevel(_qualityLevel), minDistance(_minDistance),
    blockSize(_blockSize), useHarrisDetector(_useHarrisDetector), k(_k)
{}

void GoodFeaturesToTrackDetector::Params::read (const FileNode& fn)
{
    maxCorners = fn["maxCorners"];
    qualityLevel = fn["qualityLevel"];
    minDistance = fn["minDistance"];
    blockSize = fn["blockSize"];
    useHarrisDetector = (int)fn["useHarrisDetector"] != 0;
    k = fn["k"];
}

void GoodFeaturesToTrackDetector::Params::write (FileStorage& fs) const
{
    fs << "maxCorners" << maxCorners;
    fs << "qualityLevel" << qualityLevel;
    fs << "minDistance" << minDistance;
    fs << "blockSize" << blockSize;
    fs << "useHarrisDetector" << useHarrisDetector;
    fs << "k" << k;
}

GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector( const Params& _params ) : params(_params)
{}

GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector( int maxCorners, double qualityLevel,
                                                          double minDistance, int blockSize,
                                                          bool useHarrisDetector, double k )
{
    params = Params( maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k );
}

void GoodFeaturesToTrackDetector::read (const FileNode& fn)
{
    params.read(fn);
}

void GoodFeaturesToTrackDetector::write (FileStorage& fs) const
{
    params.write(fs);
}

void GoodFeaturesToTrackDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    vector<Point2f> corners;
    goodFeaturesToTrack( grayImage, corners, params.maxCorners, params.qualityLevel, params.minDistance, mask,
                         params.blockSize, params.useHarrisDetector, params.k );
    keypoints.resize(corners.size());
    vector<Point2f>::const_iterator corner_it = corners.begin();
    vector<KeyPoint>::iterator keypoint_it = keypoints.begin();
    for( ; corner_it != corners.end(); ++corner_it, ++keypoint_it )
    {
        *keypoint_it = KeyPoint( *corner_it, (float)params.blockSize );
    }
}

/*
 *  MserFeatureDetector
 */
MserFeatureDetector::MserFeatureDetector( int delta, int minArea, int maxArea,
                                          double maxVariation, double minDiversity,
                                          int maxEvolution, double areaThreshold,
                                          double minMargin, int edgeBlurSize )
  : mser( delta, minArea, maxArea, maxVariation, minDiversity,
          maxEvolution, areaThreshold, minMargin, edgeBlurSize )
{}

MserFeatureDetector::MserFeatureDetector( CvMSERParams params )
  : mser( params.delta, params.minArea, params.maxArea, params.maxVariation, params.minDiversity,
          params.maxEvolution, params.areaThreshold, params.minMargin, params.edgeBlurSize )
{}

void MserFeatureDetector::read (const FileNode& fn)
{
    int delta = fn["delta"];
    int minArea = fn["minArea"];
    int maxArea = fn["maxArea"];
    float maxVariation = fn["maxVariation"];
    float minDiversity = fn["minDiversity"];
    int maxEvolution = fn["maxEvolution"];
    double areaThreshold = fn["areaThreshold"];
    double minMargin = fn["minMargin"];
    int edgeBlurSize = fn["edgeBlurSize"];

    mser = MSER( delta, minArea, maxArea, maxVariation, minDiversity,
              maxEvolution, areaThreshold, minMargin, edgeBlurSize );
}

void MserFeatureDetector::write (FileStorage& fs) const
{
    //fs << "algorithm" << getAlgorithmName ();

    fs << "delta" << mser.delta;
    fs << "minArea" << mser.minArea;
    fs << "maxArea" << mser.maxArea;
    fs << "maxVariation" << mser.maxVariation;
    fs << "minDiversity" << mser.minDiversity;
    fs << "maxEvolution" << mser.maxEvolution;
    fs << "areaThreshold" << mser.areaThreshold;
    fs << "minMargin" << mser.minMargin;
    fs << "edgeBlurSize" << mser.edgeBlurSize;
}


void MserFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    vector<vector<Point> > msers;

    mser(image, msers, mask);

    vector<vector<Point> >::const_iterator contour_it = msers.begin();
    for( ; contour_it != msers.end(); ++contour_it )
    {
        // TODO check transformation from MSER region to KeyPoint
        RotatedRect rect = fitEllipse(Mat(*contour_it));
        float diam = sqrt(rect.size.height*rect.size.width);

        if( diam > std::numeric_limits<float>::epsilon() )
            keypoints.push_back( KeyPoint( rect.center, diam, rect.angle) );
    }
}

/*
 *  StarFeatureDetector
 */

StarFeatureDetector::StarFeatureDetector( const CvStarDetectorParams& params )
    : star( params.maxSize, params.responseThreshold, params.lineThresholdProjected,
            params.lineThresholdBinarized, params.suppressNonmaxSize)
{}

StarFeatureDetector::StarFeatureDetector(int maxSize, int responseThreshold,
                                         int lineThresholdProjected,
                                         int lineThresholdBinarized,
                                         int suppressNonmaxSize)
  : star( maxSize, responseThreshold, lineThresholdProjected,
          lineThresholdBinarized, suppressNonmaxSize)
{}

void StarFeatureDetector::read (const FileNode& fn)
{
    int maxSize = fn["maxSize"];
    int responseThreshold = fn["responseThreshold"];
    int lineThresholdProjected = fn["lineThresholdProjected"];
    int lineThresholdBinarized = fn["lineThresholdBinarized"];
    int suppressNonmaxSize = fn["suppressNonmaxSize"];

    star = StarDetector( maxSize, responseThreshold, lineThresholdProjected,
              lineThresholdBinarized, suppressNonmaxSize);
}

void StarFeatureDetector::write (FileStorage& fs) const
{
    //fs << "algorithm" << getAlgorithmName ();

    fs << "maxSize" << star.maxSize;
    fs << "responseThreshold" << star.responseThreshold;
    fs << "lineThresholdProjected" << star.lineThresholdProjected;
    fs << "lineThresholdBinarized" << star.lineThresholdBinarized;
    fs << "suppressNonmaxSize" << star.suppressNonmaxSize;
}

void StarFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    star(grayImage, keypoints);
    removeInvalidPoints(mask, keypoints);
}

/*
 *   SiftFeatureDetector
 */
SiftFeatureDetector::SiftFeatureDetector( const SIFT::DetectorParams &detectorParams,
                                          const SIFT::CommonParams &commonParams )
    : sift(detectorParams.threshold, detectorParams.edgeThreshold,
           commonParams.nOctaves, commonParams.nOctaveLayers, commonParams.firstOctave, commonParams.angleMode)
{
}

SiftFeatureDetector::SiftFeatureDetector( double threshold, double edgeThreshold,
                                          int nOctaves, int nOctaveLayers, int firstOctave, int angleMode ) :
    sift(threshold, edgeThreshold, nOctaves, nOctaveLayers, firstOctave, angleMode)
{
}

void SiftFeatureDetector::read( const FileNode& fn )
{
    double threshold = fn["threshold"];
    double edgeThreshold = fn["edgeThreshold"];
    int nOctaves = fn["nOctaves"];
    int nOctaveLayers = fn["nOctaveLayers"];
    int firstOctave = fn["firstOctave"];
    int angleMode = fn["angleMode"];

    sift = SIFT(threshold, edgeThreshold, nOctaves, nOctaveLayers, firstOctave, angleMode);
}

void SiftFeatureDetector::write (FileStorage& fs) const
{
    //fs << "algorithm" << getAlgorithmName ();

    SIFT::CommonParams commParams = sift.getCommonParams ();
    SIFT::DetectorParams detectorParams = sift.getDetectorParams ();
    fs << "threshold" << detectorParams.threshold;
    fs << "edgeThreshold" << detectorParams.edgeThreshold;
    fs << "nOctaves" << commParams.nOctaves;
    fs << "nOctaveLayers" << commParams.nOctaveLayers;
    fs << "firstOctave" << commParams.firstOctave;
    fs << "angleMode" << commParams.angleMode;
}


void SiftFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    sift(grayImage, mask, keypoints);
}

/*
 *  SurfFeatureDetector
 */
SurfFeatureDetector::SurfFeatureDetector( double hessianThreshold, int octaves, int octaveLayers)
    : surf(hessianThreshold, octaves, octaveLayers)
{}

void SurfFeatureDetector::read (const FileNode& fn)
{
    double hessianThreshold = fn["hessianThreshold"];
    int octaves = fn["octaves"];
    int octaveLayers = fn["octaveLayers"];

    surf = SURF( hessianThreshold, octaves, octaveLayers );
}

void SurfFeatureDetector::write (FileStorage& fs) const
{
    //fs << "algorithm" << getAlgorithmName ();

    fs << "hessianThreshold" << surf.hessianThreshold;
    fs << "octaves" << surf.nOctaves;
    fs << "octaveLayers" << surf.nOctaveLayers;
}

void SurfFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    surf(grayImage, mask, keypoints);
}

/*
 *  DenseFeatureDetector
 */
DenseFeatureDetector::Params::Params( float _initFeatureScale, int _featureScaleLevels, 
									  float _featureScaleMul, int _initXyStep, 
									  int _initImgBound, bool _varyXyStepWithScale, 
									  bool _varyImgBoundWithScale ) :
	initFeatureScale(_initFeatureScale), featureScaleLevels(_featureScaleLevels),
	featureScaleMul(_featureScaleMul), initXyStep(_initXyStep), initImgBound(_initImgBound),
	varyXyStepWithScale(_varyXyStepWithScale), varyImgBoundWithScale(_varyImgBoundWithScale)
{}

DenseFeatureDetector::DenseFeatureDetector(const DenseFeatureDetector::Params &_params) : params(_params)
{}

void DenseFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    float curScale = params.initFeatureScale;
    int curStep = params.initXyStep;
    int curBound = params.initImgBound;
    for( int curLevel = 0; curLevel < params.featureScaleLevels; curLevel++ )
    {
        for( int x = curBound; x < image.cols - curBound; x += curStep )
        {
            for( int y = curBound; y < image.rows - curBound; y += curStep )
            {
                keypoints.push_back( KeyPoint(static_cast<float>(x), static_cast<float>(y), curScale) );
            }
        }

        curScale = curScale * params.featureScaleMul;
        if( params.varyXyStepWithScale ) curStep = static_cast<int>( curStep * params.featureScaleMul + 0.5f );
        if( params.varyImgBoundWithScale ) curBound = static_cast<int>( curBound * params.featureScaleMul + 0.5f );
    }

    removeInvalidPoints( mask, keypoints );
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
            for( std::vector<cv::KeyPoint>::iterator it = sub_keypoints.begin(), end = sub_keypoints.end();
                 it != end; ++it )
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
PyramidAdaptedFeatureDetector::PyramidAdaptedFeatureDetector( const Ptr<FeatureDetector>& _detector, int _levels )
    : detector(_detector), levels(_levels)
{}

bool PyramidAdaptedFeatureDetector::empty() const
{
    return detector.empty() || (FeatureDetector*)detector->empty();
}

void PyramidAdaptedFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    Mat src = image;
    for( int l = 0, multiplier = 1; l <= levels; ++l, multiplier *= 2 )
    {
        // Detect on current level of the pyramid
        vector<KeyPoint> new_pts;
        detector->detect(src, new_pts);
        for( vector<KeyPoint>::iterator it = new_pts.begin(), end = new_pts.end(); it != end; ++it)
        {
            it->pt.x *= multiplier;
            it->pt.y *= multiplier;
            it->size *= multiplier;
            it->octave = l;
        }
        removeInvalidPoints( mask, new_pts );
        keypoints.insert( keypoints.end(), new_pts.begin(), new_pts.end() );

        // Downsample
        if( l < levels )
        {
            Mat dst;
            pyrDown(src, dst);
            src = dst;
        }
    }
}

}
