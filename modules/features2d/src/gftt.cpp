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

class GFTTDetector_Impl CV_FINAL : public GFTTDetector
{
public:
    GFTTDetector_Impl( int _nfeatures, double _qualityLevel,
                      double _minDistance, int _blockSize, int _gradientSize,
                      bool _useHarrisDetector, double _k )
        : nfeatures(_nfeatures), qualityLevel(_qualityLevel), minDistance(_minDistance),
        blockSize(_blockSize), gradSize(_gradientSize), useHarrisDetector(_useHarrisDetector), k(_k)
    {
    }

    void setMaxFeatures(int maxFeatures) CV_OVERRIDE { nfeatures = maxFeatures; }
    int getMaxFeatures() const CV_OVERRIDE { return nfeatures; }

    void setQualityLevel(double qlevel) CV_OVERRIDE { qualityLevel = qlevel; }
    double getQualityLevel() const CV_OVERRIDE { return qualityLevel; }

    void setMinDistance(double minDistance_) CV_OVERRIDE { minDistance = minDistance_; }
    double getMinDistance() const CV_OVERRIDE { return minDistance; }

    void setBlockSize(int blockSize_) CV_OVERRIDE { blockSize = blockSize_; }
    int getBlockSize() const CV_OVERRIDE { return blockSize; }

    //void setGradientSize(int gradientSize_) { gradSize = gradientSize_; }
    //int getGradientSize() { return gradSize; }

    void setHarrisDetector(bool val) CV_OVERRIDE { useHarrisDetector = val; }
    bool getHarrisDetector() const CV_OVERRIDE { return useHarrisDetector; }

    void setK(double k_) CV_OVERRIDE { k = k_; }
    double getK() const CV_OVERRIDE { return k; }

    void detect( InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask ) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        if(_image.empty())
        {
            keypoints.clear();
            return;
        }

        std::vector<Point2f> corners;

        if (_image.isUMat())
        {
            UMat ugrayImage;
            if( _image.type() != CV_8U )
                cvtColor( _image, ugrayImage, COLOR_BGR2GRAY );
            else
                ugrayImage = _image.getUMat();

            goodFeaturesToTrack( ugrayImage, corners, nfeatures, qualityLevel, minDistance, _mask,
                                 blockSize, gradSize, useHarrisDetector, k );
        }
        else
        {
            Mat image = _image.getMat(), grayImage = image;
            if( image.type() != CV_8U )
                cvtColor( image, grayImage, COLOR_BGR2GRAY );

            goodFeaturesToTrack( grayImage, corners, nfeatures, qualityLevel, minDistance, _mask,
                                blockSize, gradSize, useHarrisDetector, k );
        }

        keypoints.resize(corners.size());
        std::vector<Point2f>::const_iterator corner_it = corners.begin();
        std::vector<KeyPoint>::iterator keypoint_it = keypoints.begin();
        for( ; corner_it != corners.end() && keypoint_it != keypoints.end(); ++corner_it, ++keypoint_it )
            *keypoint_it = KeyPoint( *corner_it, (float)blockSize );

    }

    int nfeatures;
    double qualityLevel;
    double minDistance;
    int blockSize;
    int gradSize;
    bool useHarrisDetector;
    double k;
};


Ptr<GFTTDetector> GFTTDetector::create( int _nfeatures, double _qualityLevel,
                         double _minDistance, int _blockSize, int _gradientSize,
                         bool _useHarrisDetector, double _k )
{
    return makePtr<GFTTDetector_Impl>(_nfeatures, _qualityLevel,
                                      _minDistance, _blockSize, _gradientSize, _useHarrisDetector, _k);
}

Ptr<GFTTDetector> GFTTDetector::create( int _nfeatures, double _qualityLevel,
                         double _minDistance, int _blockSize,
                         bool _useHarrisDetector, double _k )
{
    return makePtr<GFTTDetector_Impl>(_nfeatures, _qualityLevel,
                                      _minDistance, _blockSize, 3, _useHarrisDetector, _k);
}

String GFTTDetector::getDefaultName() const
{
    return (Feature2D::getDefaultName() + ".GFTTDetector");
}

}
