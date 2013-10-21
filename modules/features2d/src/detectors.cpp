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

namespace cv
{

/*
 *  FeatureDetector
 */

FeatureDetector::~FeatureDetector()
{}

void FeatureDetector::detect( const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    keypoints.clear();

    if( image.empty() )
        return;

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()) );

    detectImpl( image, keypoints, mask );
}

void FeatureDetector::detect(const std::vector<Mat>& imageCollection, std::vector<std::vector<KeyPoint> >& pointCollection, const std::vector<Mat>& masks ) const
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

void FeatureDetector::removeInvalidPoints( const Mat& mask, std::vector<KeyPoint>& keypoints )
{
    KeyPointsFilter::runByPixelsMask( keypoints, mask );
}

Ptr<FeatureDetector> FeatureDetector::create( const String& detectorType )
{
    if( detectorType.find("Grid") == 0 )
    {
        return makePtr<GridAdaptedFeatureDetector>(FeatureDetector::create(
                                detectorType.substr(strlen("Grid"))));
    }

    if( detectorType.find("Pyramid") == 0 )
    {
        return makePtr<PyramidAdaptedFeatureDetector>(FeatureDetector::create(
                                detectorType.substr(strlen("Pyramid"))));
    }

    if( detectorType.find("Dynamic") == 0 )
    {
        return makePtr<DynamicAdaptedFeatureDetector>(AdjusterAdapter::create(
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

void GFTTDetector::detectImpl( const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask) const
{
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, COLOR_BGR2GRAY );

    std::vector<Point2f> corners;
    goodFeaturesToTrack( grayImage, corners, nfeatures, qualityLevel, minDistance, mask,
                         blockSize, useHarrisDetector, k );
    keypoints.resize(corners.size());
    std::vector<Point2f>::const_iterator corner_it = corners.begin();
    std::vector<KeyPoint>::iterator keypoint_it = keypoints.begin();
    for( ; corner_it != corners.end(); ++corner_it, ++keypoint_it )
    {
        *keypoint_it = KeyPoint( *corner_it, (float)blockSize );
    }
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


void DenseFeatureDetector::detectImpl( const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask ) const
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

/*
 *  GridAdaptedFeatureDetector
 */
GridAdaptedFeatureDetector::GridAdaptedFeatureDetector( const Ptr<FeatureDetector>& _detector,
                                                        int _maxTotalKeypoints, int _gridRows, int _gridCols )
    : detector(_detector), maxTotalKeypoints(_maxTotalKeypoints), gridRows(_gridRows), gridCols(_gridCols)
{}

bool GridAdaptedFeatureDetector::empty() const
{
    return !detector || detector->empty();
}

struct ResponseComparator
{
    bool operator() (const KeyPoint& a, const KeyPoint& b)
    {
        return std::abs(a.response) > std::abs(b.response);
    }
};

static void keepStrongest( int N, std::vector<KeyPoint>& keypoints )
{
    if( (int)keypoints.size() > N )
    {
        std::vector<KeyPoint>::iterator nth = keypoints.begin() + N;
        std::nth_element( keypoints.begin(), nth, keypoints.end(), ResponseComparator() );
        keypoints.erase( nth, keypoints.end() );
    }
}

namespace {
class GridAdaptedFeatureDetectorInvoker : public ParallelLoopBody
{
private:
    int gridRows_, gridCols_;
    int maxPerCell_;
    std::vector<KeyPoint>& keypoints_;
    const Mat& image_;
    const Mat& mask_;
    const Ptr<FeatureDetector>& detector_;
    Mutex* kptLock_;

    GridAdaptedFeatureDetectorInvoker& operator=(const GridAdaptedFeatureDetectorInvoker&); // to quiet MSVC

public:

    GridAdaptedFeatureDetectorInvoker(const Ptr<FeatureDetector>& detector, const Mat& image, const Mat& mask,
                                      std::vector<KeyPoint>& keypoints, int maxPerCell, int gridRows, int gridCols,
                                      cv::Mutex* kptLock)
        : gridRows_(gridRows), gridCols_(gridCols), maxPerCell_(maxPerCell),
          keypoints_(keypoints), image_(image), mask_(mask), detector_(detector),
          kptLock_(kptLock)
    {
    }

    void operator() (const Range& range) const
    {
        for (int i = range.start; i < range.end; ++i)
        {
            int celly = i / gridCols_;
            int cellx = i - celly * gridCols_;

            Range row_range((celly*image_.rows)/gridRows_, ((celly+1)*image_.rows)/gridRows_);
            Range col_range((cellx*image_.cols)/gridCols_, ((cellx+1)*image_.cols)/gridCols_);

            Mat sub_image = image_(row_range, col_range);
            Mat sub_mask;
            if (!mask_.empty()) sub_mask = mask_(row_range, col_range);

            std::vector<KeyPoint> sub_keypoints;
            sub_keypoints.reserve(maxPerCell_);

            detector_->detect( sub_image, sub_keypoints, sub_mask );
            keepStrongest( maxPerCell_, sub_keypoints );

            std::vector<cv::KeyPoint>::iterator it = sub_keypoints.begin(),
                                                end = sub_keypoints.end();
            for( ; it != end; ++it )
            {
                it->pt.x += col_range.start;
                it->pt.y += row_range.start;
            }

            cv::AutoLock join_keypoints(*kptLock_);
            keypoints_.insert( keypoints_.end(), sub_keypoints.begin(), sub_keypoints.end() );
        }
    }
};
} // namepace

void GridAdaptedFeatureDetector::detectImpl( const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    if (image.empty() || maxTotalKeypoints < gridRows * gridCols)
    {
        keypoints.clear();
        return;
    }
    keypoints.reserve(maxTotalKeypoints);
    int maxPerCell = maxTotalKeypoints / (gridRows * gridCols);

    cv::Mutex kptLock;
    cv::parallel_for_(cv::Range(0, gridRows * gridCols),
        GridAdaptedFeatureDetectorInvoker(detector, image, mask, keypoints, maxPerCell, gridRows, gridCols, &kptLock));
}

/*
 *  PyramidAdaptedFeatureDetector
 */
PyramidAdaptedFeatureDetector::PyramidAdaptedFeatureDetector( const Ptr<FeatureDetector>& _detector, int _maxLevel )
    : detector(_detector), maxLevel(_maxLevel)
{}

bool PyramidAdaptedFeatureDetector::empty() const
{
    return !detector || detector->empty();
}

void PyramidAdaptedFeatureDetector::detectImpl( const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask ) const
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
        std::vector<KeyPoint> new_pts;
        detector->detect( src, new_pts, src_mask );
        std::vector<KeyPoint>::iterator it = new_pts.begin(),
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
                resize( dilated_mask, src_mask, src.size(), 0, 0, INTER_AREA );
        }
    }

    if( !mask.empty() )
        KeyPointsFilter::runByPixelsMask( keypoints, mask );
}


}
