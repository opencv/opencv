/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2008, Willow Garage Inc., all rights reserved.
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

struct KeypointResponseGreaterThanThreshold
{
    KeypointResponseGreaterThanThreshold(float _value) :
    value(_value)
    {
    }
    inline bool operator()(const KeyPoint& kpt) const
    {
        return kpt.response >= value;
    }
    float value;
};

struct KeypointResponseGreater
{
    inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const
    {
        return kp1.response > kp2.response;
    }
};

// takes keypoints and culls them by the response
void KeyPointsFilter::retainBest(std::vector<KeyPoint>& keypoints, int n_points)
{
    //this is only necessary if the keypoints size is greater than the number of desired points.
    if( n_points >= 0 && keypoints.size() > (size_t)n_points )
    {
        if (n_points==0)
        {
            keypoints.clear();
            return;
        }
        //first use nth element to partition the keypoints into the best and worst.
        std::nth_element(keypoints.begin(), keypoints.begin() + n_points, keypoints.end(), KeypointResponseGreater());
        //this is the boundary response, and in the case of FAST may be ambigous
        float ambiguous_response = keypoints[n_points - 1].response;
        //use std::partition to grab all of the keypoints with the boundary response.
        std::vector<KeyPoint>::const_iterator new_end =
        std::partition(keypoints.begin() + n_points, keypoints.end(),
                       KeypointResponseGreaterThanThreshold(ambiguous_response));
        //resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
        keypoints.resize(new_end - keypoints.begin());
    }
}

struct RoiPredicate
{
    RoiPredicate( const Rect& _r ) : r(_r)
    {}

    bool operator()( const KeyPoint& keyPt ) const
    {
        return !r.contains( keyPt.pt );
    }

    Rect r;
};

void KeyPointsFilter::runByImageBorder( std::vector<KeyPoint>& keypoints, Size imageSize, int borderSize )
{
    if( borderSize > 0)
    {
        if (imageSize.height <= borderSize * 2 || imageSize.width <= borderSize * 2)
            keypoints.clear();
        else
            keypoints.erase( std::remove_if(keypoints.begin(), keypoints.end(),
                                       RoiPredicate(Rect(Point(borderSize, borderSize),
                                                         Point(imageSize.width - borderSize, imageSize.height - borderSize)))),
                             keypoints.end() );
    }
}

struct SizePredicate
{
    SizePredicate( float _minSize, float _maxSize ) : minSize(_minSize), maxSize(_maxSize)
    {}

    bool operator()( const KeyPoint& keyPt ) const
    {
        float size = keyPt.size;
        return (size < minSize) || (size > maxSize);
    }

    float minSize, maxSize;
};

void KeyPointsFilter::runByKeypointSize( std::vector<KeyPoint>& keypoints, float minSize, float maxSize )
{
    CV_Assert( minSize >= 0 );
    CV_Assert( maxSize >= 0);
    CV_Assert( minSize <= maxSize );

    keypoints.erase( std::remove_if(keypoints.begin(), keypoints.end(), SizePredicate(minSize, maxSize)),
                     keypoints.end() );
}

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
    MaskPredicate& operator=(const MaskPredicate&);
};

void KeyPointsFilter::runByPixelsMask( std::vector<KeyPoint>& keypoints, const Mat& mask )
{
    if( mask.empty() )
        return;

    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
}

struct KeyPoint_LessThan
{
    KeyPoint_LessThan(const std::vector<KeyPoint>& _kp) : kp(&_kp) {}
    bool operator()(int i, int j) const
    {
        const KeyPoint& kp1 = (*kp)[i];
        const KeyPoint& kp2 = (*kp)[j];
        if( kp1.pt.x != kp2.pt.x )
            return kp1.pt.x < kp2.pt.x;
        if( kp1.pt.y != kp2.pt.y )
            return kp1.pt.y < kp2.pt.y;
        if( kp1.size != kp2.size )
            return kp1.size > kp2.size;
        if( kp1.angle != kp2.angle )
            return kp1.angle < kp2.angle;
        if( kp1.response != kp2.response )
            return kp1.response > kp2.response;
        if( kp1.octave != kp2.octave )
            return kp1.octave > kp2.octave;
        if( kp1.class_id != kp2.class_id )
            return kp1.class_id > kp2.class_id;

        return i < j;
    }
    const std::vector<KeyPoint>* kp;
};

void KeyPointsFilter::removeDuplicated( std::vector<KeyPoint>& keypoints )
{
    int i, j, n = (int)keypoints.size();
    std::vector<int> kpidx(n);
    std::vector<uchar> mask(n, (uchar)1);

    for( i = 0; i < n; i++ )
        kpidx[i] = i;
    std::sort(kpidx.begin(), kpidx.end(), KeyPoint_LessThan(keypoints));
    for( i = 1, j = 0; i < n; i++ )
    {
        KeyPoint& kp1 = keypoints[kpidx[i]];
        KeyPoint& kp2 = keypoints[kpidx[j]];
        if( kp1.pt.x != kp2.pt.x || kp1.pt.y != kp2.pt.y ||
            kp1.size != kp2.size || kp1.angle != kp2.angle )
            j = i;
        else
            mask[kpidx[i]] = 0;
    }

    for( i = j = 0; i < n; i++ )
    {
        if( mask[i] )
        {
            if( i != j )
                keypoints[j] = keypoints[i];
            j++;
        }
    }
    keypoints.resize(j);
}

}
