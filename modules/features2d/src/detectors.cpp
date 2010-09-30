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
struct MaskPredicate
{
    MaskPredicate( const Mat& _mask ) : mask(_mask)
    {}
    MaskPredicate& operator=(const MaskPredicate&) { return *this; }
    bool operator() (const KeyPoint& key_pt) const
    {
      return mask.at<uchar>( (int)(key_pt.pt.y + 0.5f), (int)(key_pt.pt.x + 0.5f) ) == 0;
    }

    const Mat& mask;
};

void FeatureDetector::removeInvalidPoints( const Mat& mask, vector<KeyPoint>& keypoints )
{
    if( mask.empty() )
        return;

    keypoints.erase(remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
};

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

void FastFeatureDetector::detectImpl( const Mat& image, const Mat& mask, vector<KeyPoint>& keypoints) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );
    FAST( grayImage, keypoints, threshold, nonmaxSuppression );
    removeInvalidPoints( mask, keypoints );
}

/*
 *  GoodFeaturesToTrackDetector
 */
GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector( int _maxCorners, double _qualityLevel, \
                                                          double _minDistance, int _blockSize,
                                                          bool _useHarrisDetector, double _k )
    : maxCorners(_maxCorners), qualityLevel(_qualityLevel), minDistance(_minDistance),
      blockSize(_blockSize), useHarrisDetector(_useHarrisDetector), k(_k)
{}

void GoodFeaturesToTrackDetector::read (const FileNode& fn)
{
    maxCorners = fn["maxCorners"];
    qualityLevel = fn["qualityLevel"];
    minDistance = fn["minDistance"];
    blockSize = fn["blockSize"];
    useHarrisDetector = (int)fn["useHarrisDetector"] != 0;
    k = fn["k"];
}

void GoodFeaturesToTrackDetector::write (FileStorage& fs) const
{
    fs << "maxCorners" << maxCorners;
    fs << "qualityLevel" << qualityLevel;
    fs << "minDistance" << minDistance;
    fs << "blockSize" << blockSize;
    fs << "useHarrisDetector" << useHarrisDetector;
    fs << "k" << k;
}

void GoodFeaturesToTrackDetector::detectImpl( const Mat& image, const Mat& mask,
                                              vector<KeyPoint>& keypoints ) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    vector<Point2f> corners;
    goodFeaturesToTrack( grayImage, corners, maxCorners, qualityLevel, minDistance, mask,
                         blockSize, useHarrisDetector, k );
    keypoints.resize(corners.size());
    vector<Point2f>::const_iterator corner_it = corners.begin();
    vector<KeyPoint>::iterator keypoint_it = keypoints.begin();
    for( ; corner_it != corners.end(); ++corner_it, ++keypoint_it )
    {
        *keypoint_it = KeyPoint( *corner_it, (float)blockSize );
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


void MserFeatureDetector::detectImpl( const Mat& image, const Mat& mask, vector<KeyPoint>& keypoints ) const
{
    vector<vector<Point> > msers;

    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    mser(grayImage, msers, mask);

    keypoints.resize( msers.size() );
    vector<vector<Point> >::const_iterator contour_it = msers.begin();
    vector<KeyPoint>::iterator keypoint_it = keypoints.begin();
    for( ; contour_it != msers.end(); ++contour_it, ++keypoint_it )
    {
        // TODO check transformation from MSER region to KeyPoint
        RotatedRect rect = fitEllipse(Mat(*contour_it));
        *keypoint_it = KeyPoint( rect.center, sqrt(rect.size.height*rect.size.width), rect.angle);
    }
}

/*
 *  StarFeatureDetector
 */
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

void StarFeatureDetector::detectImpl( const Mat& image, const Mat& mask, vector<KeyPoint>& keypoints) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    star(grayImage, keypoints);
    removeInvalidPoints(mask, keypoints);
}

/*
 *   SiftFeatureDetector
 */
SiftFeatureDetector::SiftFeatureDetector(double threshold, double edgeThreshold,
                                         int nOctaves, int nOctaveLayers, int firstOctave, int angleMode) :
    sift(threshold, edgeThreshold, nOctaves, nOctaveLayers, firstOctave, angleMode)
{
}

void SiftFeatureDetector::read (const FileNode& fn)
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


void SiftFeatureDetector::detectImpl( const Mat& image, const Mat& mask,
                                      vector<KeyPoint>& keypoints) const
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

void SurfFeatureDetector::detectImpl( const Mat& image, const Mat& mask,
                                      vector<KeyPoint>& keypoints) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    surf(grayImage, mask, keypoints);
}

/*
 *  DenseFeatureDetector
 */
void DenseFeatureDetector::detectImpl( const Mat& image, const Mat& mask, vector<KeyPoint>& keypoints ) const
{
    keypoints.clear();

    float curScale = initFeatureScale;
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

        curScale = curScale * featureScaleMul;
        if( varyXyStepWithScale ) curStep = static_cast<int>( curStep * featureScaleMul + 0.5f );
        if( varyImgBoundWithScale ) curBound = static_cast<int>( curBound * featureScaleMul + 0.5f );
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

void GridAdaptedFeatureDetector::detectImpl( const Mat &image, const Mat &mask,
                                             vector<KeyPoint> &keypoints ) const
{
    keypoints.clear();
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
 *  GridAdaptedFeatureDetector
 */
PyramidAdaptedFeatureDetector::PyramidAdaptedFeatureDetector( const Ptr<FeatureDetector>& _detector, int _levels )
    : detector(_detector), levels(_levels)
{}

void PyramidAdaptedFeatureDetector::detectImpl( const Mat& image, const Mat& mask, vector<KeyPoint>& keypoints ) const
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

Ptr<FeatureDetector> createFeatureDetector( const string& detectorType )
{
    FeatureDetector* fd = 0;
    if( !detectorType.compare( "FAST" ) )
    {
        fd = new FastFeatureDetector( 10/*threshold*/, true/*nonmax_suppression*/ );
    }
    else if( !detectorType.compare( "STAR" ) )
    {
        fd = new StarFeatureDetector( 16/*max_size*/, 5/*response_threshold*/, 10/*line_threshold_projected*/,
                                      8/*line_threshold_binarized*/, 5/*suppress_nonmax_size*/ );
    }
    else if( !detectorType.compare( "SIFT" ) )
    {
        fd = new SiftFeatureDetector(SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
                                     SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD());
    }
    else if( !detectorType.compare( "SURF" ) )
    {
        fd = new SurfFeatureDetector( 400./*hessian_threshold*/, 3 /*octaves*/, 4/*octave_layers*/ );
    }
    else if( !detectorType.compare( "MSER" ) )
    {
        fd = new MserFeatureDetector( 5/*delta*/, 60/*min_area*/, 14400/*_max_area*/, 0.25f/*max_variation*/,
                0.2/*min_diversity*/, 200/*max_evolution*/, 1.01/*area_threshold*/, 0.003/*min_margin*/,
                5/*edge_blur_size*/ );
    }
    else if( !detectorType.compare( "GFTT" ) )
    {
        fd = new GoodFeaturesToTrackDetector( 1000/*maxCorners*/, 0.01/*qualityLevel*/, 1./*minDistance*/,
                                              3/*int _blockSize*/, false/*useHarrisDetector*/, 0.04/*k*/ );
    }
    else if( !detectorType.compare( "HARRIS" ) )
    {
        fd = new GoodFeaturesToTrackDetector( 1000/*maxCorners*/, 0.01/*qualityLevel*/, 1./*minDistance*/,
                                              3/*int _blockSize*/, true/*useHarrisDetector*/, 0.04/*k*/ );
    }
    return fd;
}

}
