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

const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;

namespace cv
{

/*
 * Functions to draw keypoints and matches.
 */
static inline void _drawKeypoint( Mat& img, const KeyPoint& p, const Scalar& color, int flags )
{
    CV_Assert( !img.empty() );
    Point center( cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier) );

    if( flags & DrawMatchesFlags::DRAW_RICH_KEYPOINTS )
    {
        int radius = cvRound(p.size/2 * draw_multiplier); // KeyPoint::size is a diameter

        // draw the circles around keypoints with the keypoints size
        circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );

        // draw orientation of the keypoint, if it is applicable
        if( p.angle != -1 )
        {
            float srcAngleRad = p.angle*(float)CV_PI/180.f;
            Point orient( cvRound( cos(srcAngleRad)*radius ),
                          cvRound(-sin(srcAngleRad)*radius ) // "-" to invert orientation of axis y
                        );
            line( img, center, center+orient, color, 1, CV_AA, draw_shift_bits );
        }
#if 0
        else
        {
            // draw center with R=1
            int radius = 1 * draw_multiplier;
            circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
        }
#endif
    }
    else
    {
        // draw center with R=3
        int radius = 3 * draw_multiplier;
        circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
    }
}

void drawKeypoints( const Mat& image, const vector<KeyPoint>& keypoints, Mat& outImage,
                    const Scalar& _color, int flags )
{
    if( !(flags & DrawMatchesFlags::DRAW_OVER_OUTIMG) )
    {
        if( image.type() == CV_8UC3 )
        {
            image.copyTo( outImage );
        }
        else if( image.type() == CV_8UC1 )
        {
            cvtColor( image, outImage, CV_GRAY2BGR );
        }
        else
        {
            CV_Error( CV_StsBadArg, "Incorrect type of input image.\n" );
        }
    }

    RNG& rng=theRNG();
    bool isRandColor = _color == Scalar::all(-1);

    CV_Assert( !outImage.empty() );
    vector<KeyPoint>::const_iterator it = keypoints.begin(),
                                     end = keypoints.end();
    for( ; it != end; ++it )
    {
        Scalar color = isRandColor ? Scalar(rng(256), rng(256), rng(256)) : _color;
        _drawKeypoint( outImage, *it, color, flags );
    }
}

static void _prepareImgAndDrawKeypoints( const Mat& img1, const vector<KeyPoint>& keypoints1,
                                         const Mat& img2, const vector<KeyPoint>& keypoints2,
                                         Mat& outImg, Mat& outImg1, Mat& outImg2,
                                         const Scalar& singlePointColor, int flags )
{
    Size size( img1.cols + img2.cols, MAX(img1.rows, img2.rows) );
    if( flags & DrawMatchesFlags::DRAW_OVER_OUTIMG )
    {
        if( size.width > outImg.cols || size.height > outImg.rows )
            CV_Error( CV_StsBadSize, "outImg has size less than need to draw img1 and img2 together" );
        outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
        outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
    }
    else
    {
        outImg.create( size, CV_MAKETYPE(img1.depth(), 3) );
        outImg = Scalar::all(0);
        outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
        outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );

        if( img1.type() == CV_8U )
            cvtColor( img1, outImg1, CV_GRAY2BGR );
        else
            img1.copyTo( outImg1 );

        if( img2.type() == CV_8U )
            cvtColor( img2, outImg2, CV_GRAY2BGR );
        else
            img2.copyTo( outImg2 );
    }

    // draw keypoints
    if( !(flags & DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS) )
    {
        Mat outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
        drawKeypoints( outImg1, keypoints1, outImg1, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG );

        Mat outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
        drawKeypoints( outImg2, keypoints2, outImg2, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG );
    }
}

static inline void _drawMatch( Mat& outImg, Mat& outImg1, Mat& outImg2 ,
                          const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, int flags )
{
    RNG& rng = theRNG();
    bool isRandMatchColor = matchColor == Scalar::all(-1);
    Scalar color = isRandMatchColor ? Scalar( rng(256), rng(256), rng(256) ) : matchColor;

    _drawKeypoint( outImg1, kp1, color, flags );
    _drawKeypoint( outImg2, kp2, color, flags );

    Point2f pt1 = kp1.pt,
            pt2 = kp2.pt,
            dpt2 = Point2f( std::min(pt2.x+outImg1.cols, float(outImg.cols-1)), pt2.y );

    line( outImg, 
		  Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
		  Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
          color, 1, CV_AA, draw_shift_bits );
}

void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                  const Mat& img2, const vector<KeyPoint>& keypoints2,
                  const vector<DMatch>& matches1to2, Mat& outImg,
                  const Scalar& matchColor, const Scalar& singlePointColor,
                  const vector<char>& matchesMask, int flags )
{
    if( !matchesMask.empty() && matchesMask.size() != matches1to2.size() )
        CV_Error( CV_StsBadSize, "matchesMask must have the same size as matches1to2" );

    Mat outImg1, outImg2;
    _prepareImgAndDrawKeypoints( img1, keypoints1, img2, keypoints2,
                                 outImg, outImg1, outImg2, singlePointColor, flags );

    // draw matches
    for( size_t m = 0; m < matches1to2.size(); m++ )
    {
        int i1 = matches1to2[m].queryIdx;
        int i2 = matches1to2[m].trainIdx;
        if( matchesMask.empty() || matchesMask[m] )
        {
            const KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
            _drawMatch( outImg, outImg1, outImg2, kp1, kp2, matchColor, flags );
        }
    }
}

void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                  const Mat& img2, const vector<KeyPoint>& keypoints2,
                  const vector<vector<DMatch> >& matches1to2, Mat& outImg,
                  const Scalar& matchColor, const Scalar& singlePointColor,
                  const vector<vector<char> >& matchesMask, int flags )
{
    if( !matchesMask.empty() && matchesMask.size() != matches1to2.size() )
        CV_Error( CV_StsBadSize, "matchesMask must have the same size as matches1to2" );

    Mat outImg1, outImg2;
    _prepareImgAndDrawKeypoints( img1, keypoints1, img2, keypoints2,
                                 outImg, outImg1, outImg2, singlePointColor, flags );

    // draw matches
    for( size_t i = 0; i < matches1to2.size(); i++ )
    {
        for( size_t j = 0; j < matches1to2[i].size(); j++ )
        {
            int i1 = matches1to2[i][j].queryIdx;
            int i2 = matches1to2[i][j].trainIdx;
            if( matchesMask.empty() || matchesMask[i][j] )
            {
                const KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
                _drawMatch( outImg, outImg1, outImg2, kp1, kp2, matchColor, flags );
            }
        }
    }
}
}
