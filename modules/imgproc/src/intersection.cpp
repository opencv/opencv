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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//      Nghia Ho, nghiaho12@yahoo.com
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
//   * The name of OpenCV Foundation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
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

int rotatedRectangleIntersection( const RotatedRect& rect1, const RotatedRect& rect2, OutputArray intersectingRegion )
{
    CV_INSTRUMENT_REGION()

    const float samePointEps = 0.00001f; // used to test if two points are the same

    Point2f vec1[4], vec2[4];
    Point2f pts1[4], pts2[4];

    std::vector <Point2f> intersection;

    rect1.points(pts1);
    rect2.points(pts2);

    int ret = INTERSECT_FULL;

    // Specical case of rect1 == rect2
    {
        bool same = true;

        for( int i = 0; i < 4; i++ )
        {
            if( fabs(pts1[i].x - pts2[i].x) > samePointEps || (fabs(pts1[i].y - pts2[i].y) > samePointEps) )
            {
                same = false;
                break;
            }
        }

        if(same)
        {
            intersection.resize(4);

            for( int i = 0; i < 4; i++ )
            {
                intersection[i] = pts1[i];
            }

            Mat(intersection).copyTo(intersectingRegion);

            return INTERSECT_FULL;
        }
    }

    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    for( int i = 0; i < 4; i++ )
    {
        vec1[i].x = pts1[(i+1)%4].x - pts1[i].x;
        vec1[i].y = pts1[(i+1)%4].y - pts1[i].y;

        vec2[i].x = pts2[(i+1)%4].x - pts2[i].x;
        vec2[i].y = pts2[(i+1)%4].y - pts2[i].y;
    }

    // Line test - test all line combos for intersection
    for( int i = 0; i < 4; i++ )
    {
        for( int j = 0; j < 4; j++ )
        {
            // Solve for 2x2 Ax=b
            float x21 = pts2[j].x - pts1[i].x;
            float y21 = pts2[j].y - pts1[i].y;

            float vx1 = vec1[i].x;
            float vy1 = vec1[i].y;

            float vx2 = vec2[j].x;
            float vy2 = vec2[j].y;

            float det = vx2*vy1 - vx1*vy2;

            float t1 = (vx2*y21 - vy2*x21) / det;
            float t2 = (vx1*y21 - vy1*x21) / det;

            // This takes care of parallel lines
            if( cvIsInf(t1) || cvIsInf(t2) || cvIsNaN(t1) || cvIsNaN(t2) )
            {
                continue;
            }

            if( t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f )
            {
                float xi = pts1[i].x + vec1[i].x*t1;
                float yi = pts1[i].y + vec1[i].y*t1;

                intersection.push_back(Point2f(xi,yi));
            }
        }
    }

    if( !intersection.empty() )
    {
        ret = INTERSECT_PARTIAL;
    }

    // Check for vertices from rect1 inside recct2
    for( int i = 0; i < 4; i++ )
    {
        // We do a sign test to see which side the point lies.
        // If the point all lie on the same sign for all 4 sides of the rect,
        // then there's an intersection
        int posSign = 0;
        int negSign = 0;

        float x = pts1[i].x;
        float y = pts1[i].y;

        for( int j = 0; j < 4; j++ )
        {
            // line equation: Ax + By + C = 0
            // see which side of the line this point is at
            float A = -vec2[j].y;
            float B = vec2[j].x;
            float C = -(A*pts2[j].x + B*pts2[j].y);

            float s = A*x+ B*y+ C;

            if( s >= 0 )
            {
                posSign++;
            }
            else
            {
                negSign++;
            }
        }

        if( posSign == 4 || negSign == 4 )
        {
            intersection.push_back(pts1[i]);
        }
    }

    // Reverse the check - check for vertices from rect2 inside recct1
    for( int i = 0; i < 4; i++ )
    {
        // We do a sign test to see which side the point lies.
        // If the point all lie on the same sign for all 4 sides of the rect,
        // then there's an intersection
        int posSign = 0;
        int negSign = 0;

        float x = pts2[i].x;
        float y = pts2[i].y;

        for( int j = 0; j < 4; j++ )
        {
            // line equation: Ax + By + C = 0
            // see which side of the line this point is at
            float A = -vec1[j].y;
            float B = vec1[j].x;
            float C = -(A*pts1[j].x + B*pts1[j].y);

            float s = A*x + B*y + C;

            if( s >= 0 )
            {
                posSign++;
            }
            else
            {
                negSign++;
            }
        }

        if( posSign == 4 || negSign == 4 )
        {
            intersection.push_back(pts2[i]);
        }
    }

    // Get rid of dupes
    for( int i = 0; i < (int)intersection.size()-1; i++ )
    {
        for( size_t j = i+1; j < intersection.size(); j++ )
        {
            float dx = intersection[i].x - intersection[j].x;
            float dy = intersection[i].y - intersection[j].y;
            double d2 = dx*dx + dy*dy; // can be a really small number, need double here

            if( d2 < samePointEps*samePointEps )
            {
                // Found a dupe, remove it
                std::swap(intersection[j], intersection.back());
                intersection.pop_back();
                j--; // restart check
            }
        }
    }

    if( intersection.empty() )
    {
        return INTERSECT_NONE ;
    }

    // If this check fails then it means we're getting dupes, increase samePointEps
    CV_Assert( intersection.size() <= 8 );

    Mat(intersection).copyTo(intersectingRegion);

    return ret;
}

} // end namespace
