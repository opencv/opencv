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

size_t KeyPoint::hash() const
{
    size_t _Val = 2166136261U, scale = 16777619U;
    Cv32suf u;
    u.f = pt.x; _Val = (scale * _Val) ^ u.u;
    u.f = pt.y; _Val = (scale * _Val) ^ u.u;
    u.f = size; _Val = (scale * _Val) ^ u.u;
    u.f = angle; _Val = (scale * _Val) ^ u.u;
    u.f = response; _Val = (scale * _Val) ^ u.u;
    _Val = (scale * _Val) ^ ((size_t) octave);
    _Val = (scale * _Val) ^ ((size_t) class_id);
    return _Val;
}    

void write(FileStorage& fs, const string& objname, const vector<KeyPoint>& keypoints)
{
    WriteStructContext ws(fs, objname, CV_NODE_SEQ + CV_NODE_FLOW);
    
    int i, npoints = (int)keypoints.size();
    for( i = 0; i < npoints; i++ )
    {
        const KeyPoint& kpt = keypoints[i];
        write(fs, kpt.pt.x);
        write(fs, kpt.pt.y);
        write(fs, kpt.size);
        write(fs, kpt.angle);
        write(fs, kpt.response);
        write(fs, kpt.octave);
        write(fs, kpt.class_id);
    }
}


void read(const FileNode& node, vector<KeyPoint>& keypoints)
{
    keypoints.resize(0);
    FileNodeIterator it = node.begin(), it_end = node.end();
    for( ; it != it_end; )
    {
        KeyPoint kpt;
        it >> kpt.pt.x >> kpt.pt.y >> kpt.size >> kpt.angle >> kpt.response >> kpt.octave >> kpt.class_id;
        keypoints.push_back(kpt);
    }
}
    

void KeyPoint::convert(const std::vector<KeyPoint>& keypoints, std::vector<Point2f>& points2f,
                       const vector<int>& keypointIndexes)
{
    if( keypointIndexes.empty() )
    {
        points2f.resize( keypoints.size() );
        for( size_t i = 0; i < keypoints.size(); i++ )
            points2f[i] = keypoints[i].pt;
    }
    else
    {
        points2f.resize( keypointIndexes.size() );
        for( size_t i = 0; i < keypointIndexes.size(); i++ )
        {
            int idx = keypointIndexes[i];
            if( idx >= 0 )
                points2f[i] = keypoints[idx].pt;
            else
            {
                CV_Error( CV_StsBadArg, "keypointIndexes has element < 0. TODO: process this case" );
                //points2f[i] = Point2f(-1, -1);
            }
        }
    }
}
    
void KeyPoint::convert( const std::vector<Point2f>& points2f, std::vector<KeyPoint>& keypoints,
                        float size, float response, int octave, int class_id )
{
    for( size_t i = 0; i < points2f.size(); i++ )
        keypoints[i] = KeyPoint(points2f[i], size, -1, response, octave, class_id);
}

float KeyPoint::overlap( const KeyPoint& kp1, const KeyPoint& kp2 )
{
    float a = kp1.size * 0.5f;
    float b = kp2.size * 0.5f;
    float a_2 = a * a;
    float b_2 = b * b;

    Point2f p1 = kp1.pt;
    Point2f p2 = kp2.pt;
    float c = (float)norm( p1 - p2 );

    float ovrl = 0.f;

    // one circle is completely encovered by the other => no intersection points!
    if( min( a, b ) + c <= max( a, b ) )
        return min( a_2, b_2 ) / max( a_2, b_2 );

    if( c < a + b ) // circles intersect
    {
        float c_2 = c * c;
        float cosAlpha = ( b_2 + c_2 - a_2 ) / ( kp2.size * c );
        float cosBeta  = ( a_2 + c_2 - b_2 ) / ( kp1.size * c );
        float alpha = acos( cosAlpha );
        float beta = acos( cosBeta );
        float sinAlpha = sin(alpha);
        float sinBeta  = sin(beta);

        float segmentAreaA = a_2 * beta;
        float segmentAreaB = b_2 * alpha;

        float triangleAreaA = a_2 * sinBeta * cosBeta;
        float triangleAreaB = b_2 * sinAlpha * cosAlpha;

        float intersectionArea = segmentAreaA + segmentAreaB - triangleAreaA - triangleAreaB;
        float unionArea = (a_2 + b_2) * (float)CV_PI - intersectionArea;

        ovrl = intersectionArea / unionArea;
    }

    return ovrl;
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

void KeyPointsFilter::runByImageBorder( vector<KeyPoint>& keypoints, Size imageSize, int borderSize )
{
    if( borderSize > 0)
    {
        keypoints.erase( remove_if(keypoints.begin(), keypoints.end(),
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

void KeyPointsFilter::runByKeypointSize( vector<KeyPoint>& keypoints, float minSize, float maxSize )
{
    CV_Assert( minSize >= 0 );
    CV_Assert( maxSize >= 0);
    CV_Assert( minSize <= maxSize );

    keypoints.erase( remove_if(keypoints.begin(), keypoints.end(), SizePredicate(minSize, maxSize)),
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
};

void KeyPointsFilter::runByPixelsMask( vector<KeyPoint>& keypoints, const Mat& mask )
{
    if( mask.empty() )
        return;

    keypoints.erase(remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
}

struct KeyPoint_LessThan
{
    KeyPoint_LessThan(const vector<KeyPoint>& _kp) : kp(&_kp) {}
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
    const vector<KeyPoint>* kp;
};

void KeyPointsFilter::removeDuplicated( vector<KeyPoint>& keypoints )
{
    int i, j, n = (int)keypoints.size();
    vector<int> kpidx(n);
    vector<uchar> mask(n, (uchar)1);

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
