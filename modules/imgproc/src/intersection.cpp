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

static int _rotatedRectangleIntersection( const RotatedRect& rect1, const RotatedRect& rect2, std::vector<Point2f> &intersection )
{
    CV_INSTRUMENT_REGION();

    typedef double precision_t;
    typedef cv::Point_<precision_t> point_t;

    std::vector<point_t> __highPrecisionIntersections;
    const bool useHighPrecisionIntersections = !std::is_same<point_t, cv::Point2f>();
    std::vector<point_t>& _intersection = useHighPrecisionIntersections ? __highPrecisionIntersections : *reinterpret_cast<std::vector<point_t>*>(&intersection);


    cv::Point2f _pts1[4], _pts2[4];
    rect1.points(_pts1);
    rect2.points(_pts2);

    point_t vec1[4], vec2[4];
    point_t pts1[4], pts2[4];
    for(int i = 0 ; i<4 ;++i)
    {
        pts1[i] = _pts1[i];
        pts2[i] = _pts2[i];
    }

    // L2 metric
    precision_t samePointEps = static_cast<precision_t>(1e-6) * static_cast<precision_t>(std::max(rect1.size.area(), rect2.size.area()));

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
            _intersection.resize(4);

            for( int i = 0; i < 4; i++ )
            {
                _intersection[i] = pts1[i];
            }

            if (useHighPrecisionIntersections)
            {
                intersection.resize(_intersection.size());
                std::copy(_intersection.begin(), _intersection.end(), intersection.begin());
            }

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

    //we adapt the epsilon to the smallest dimension of the rects
    for( int i = 0; i < 4; i++ )
    {
        samePointEps = std::min(samePointEps, std::sqrt(vec1[i].x*vec1[i].x+vec1[i].y*vec1[i].y));
        samePointEps = std::min(samePointEps, std::sqrt(vec2[i].x*vec2[i].x+vec2[i].y*vec2[i].y));
    }
    samePointEps = std::max(static_cast<precision_t>(1e-16), samePointEps);

    // Line test - test all line combos for intersection
    for( int i = 0; i < 4; i++ )
    {
        for( int j = 0; j < 4; j++ )
        {
            // Solve for 2x2 Ax=b
            const precision_t x21 = pts2[j].x - pts1[i].x;
            const precision_t y21 = pts2[j].y - pts1[i].y;

            precision_t vx1 = vec1[i].x;
            precision_t vy1 = vec1[i].y;

            precision_t vx2 = vec2[j].x;
            precision_t vy2 = vec2[j].y;

            precision_t normalizationScale  = std::min(vx1*vx1+vy1*vy1, vx2*vx2+vy2*vy2);//sum of squares : this is >= 0
            //normalizationScale is a square, and we usually limit accuracy around 1e-6, so normalizationScale should be rather limited by ((1e-6)^2)=1e-12
            normalizationScale  = (normalizationScale < static_cast<precision_t>(1e-12)) ? static_cast<precision_t>(1.) : static_cast<precision_t>(1.)/normalizationScale;

            vx1 *= normalizationScale;
            vy1 *= normalizationScale;
            vx2 *= normalizationScale;
            vy2 *= normalizationScale;

            const precision_t det = vx2*vy1 - vx1*vy2;
            if (std::abs(det) < static_cast<precision_t>(1e-12))//like normalizationScale, we consider accuracy around 1e-6, i.e. 1e-12 when squared
              continue;
            const precision_t detInvScaled = normalizationScale/det;

            const precision_t t1 = (vx2*y21 - vy2*x21)*detInvScaled;
            const precision_t t2 = (vx1*y21 - vy1*x21)*detInvScaled;

            // This takes care of parallel lines
            if( cvIsInf(t1) || cvIsInf(t2) || cvIsNaN(t1) || cvIsNaN(t2) )
            {
                continue;
            }

            if( t1 >= static_cast<precision_t>(0.) && t1 <= static_cast<precision_t>(1.) && t2 >= static_cast<precision_t>(0.) && t2 <= static_cast<precision_t>(1.) )
            {
                const precision_t xi = pts1[i].x + vec1[i].x*t1;
                const precision_t yi = pts1[i].y + vec1[i].y*t1;

                _intersection.push_back(point_t(xi,yi));
            }
        }
    }

    if( !_intersection.empty() )
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

        const precision_t x = pts1[i].x;
        const precision_t y = pts1[i].y;

        for( int j = 0; j < 4; j++ )
        {
            precision_t normalizationScale  = vec2[j].x*vec2[j].x+vec2[j].y*vec2[j].y;
            normalizationScale  = (normalizationScale < static_cast<precision_t>(1e-12)) ? static_cast<precision_t>(1.) : static_cast<precision_t>(1.)/normalizationScale;
            // line equation: Ax + By + C = 0
            // see which side of the line this point is at
            const precision_t A = -vec2[j].y*normalizationScale ;
            const precision_t B = vec2[j].x*normalizationScale ;
            const precision_t C = -(A*pts2[j].x + B*pts2[j].y);

            const precision_t s = A*x + B*y + C;

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
            _intersection.push_back(pts1[i]);
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

        const precision_t x = pts2[i].x;
        const precision_t y = pts2[i].y;

        for( int j = 0; j < 4; j++ )
        {
            // line equation: Ax + By + C = 0
            // see which side of the line this point is at
            precision_t normalizationScale  = vec2[j].x*vec2[j].x+vec2[j].y*vec2[j].y;
            normalizationScale  = (normalizationScale < static_cast<precision_t>(1e-12)) ? static_cast<precision_t>(1.) : static_cast<precision_t>(1.)/normalizationScale;
            if (std::isinf(normalizationScale ))
                normalizationScale = static_cast<precision_t>(1.);
            const precision_t A = -vec1[j].y*normalizationScale ;
            const precision_t B = vec1[j].x*normalizationScale ;
            const precision_t C = -(A*pts1[j].x + B*pts1[j].y);

            const precision_t s = A*x + B*y + C;

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
            _intersection.push_back(pts2[i]);
        }
    }

    int N = (int)_intersection.size();
    if (N == 0)
    {
        if (useHighPrecisionIntersections)
          intersection.resize(0);
        return INTERSECT_NONE;
    }

    // Get rid of duplicated points
    const int Nstride = N;
    cv::AutoBuffer<precision_t, 100> distPt(N * N);
    cv::AutoBuffer<int> ptDistRemap(N);
    for (int i = 0; i < N; ++i)
    {
        const point_t pt0 = _intersection[i];
        ptDistRemap[i] = i;
        for (int j = i + 1; j < N; )
        {
            const point_t pt1 = _intersection[j];
            const precision_t d2 = normL2Sqr<precision_t>(pt1 - pt0);
            if(d2 <= samePointEps)
            {
                if (j < N - 1)
                    _intersection[j] =  _intersection[N - 1];
                N--;
                continue;
            }
            distPt[i*Nstride + j] = d2;
            ++j;
        }
    }
    while (N > 8) // we still have duplicate points after samePointEps threshold (eliminate closest points)
    {
        int minI = 0;
        int minJ = 1;
        precision_t minD = distPt[1];
        for (int i = 0; i < N - 1; ++i)
        {
            const precision_t* pDist = distPt.data() + Nstride * ptDistRemap[i];
            for (int j = i + 1; j < N; ++j)
            {
                const precision_t d = pDist[ptDistRemap[j]];
                if (d < minD)
                {
                    minD = d;
                    minI = i;
                    minJ = j;
                }
            }
        }
        CV_Assert(fabs(normL2Sqr<precision_t>(_intersection[minI] - _intersection[minJ]) - minD) < static_cast<precision_t>(1e-6));  // ptDistRemap is not corrupted
        // drop minJ point
        if (minJ < N - 1)
        {
            _intersection[minJ] =  _intersection[N - 1];
            ptDistRemap[minJ] = ptDistRemap[N - 1];
        }
        N--;
    }

    // order points
    for (int i = 0; i < N - 1; ++i)
    {
        point_t diffI = _intersection[i + 1] - _intersection[i];
        for (int j = i + 2; j < N; ++j)
        {
            point_t diffJ = _intersection[j] - _intersection[i];
            if (diffI.cross(diffJ) < 0)
            {
                std::swap(_intersection[i + 1], _intersection[j]);
                diffI = diffJ;
            }
        }
    }

    _intersection.resize(N);
    if (useHighPrecisionIntersections)
    {
        intersection.resize(_intersection.size());
        std::copy(_intersection.begin(), _intersection.end(), intersection.begin());
    }

    return ret;
}

int rotatedRectangleIntersection( const RotatedRect& rect1, const RotatedRect& rect2, OutputArray intersectingRegion )
{
    CV_INSTRUMENT_REGION();

    if (rect1.size.empty() || rect2.size.empty())
    {
        intersectingRegion.release();
        return INTERSECT_NONE;
    }

    // Shift rectangles closer to origin (0, 0) to improve the calculation of the intesection region
    // To do that, the average center of the rectangles is moved to the origin
    const Point2f averageCenter = (rect1.center + rect2.center) / 2.0f;

    RotatedRect shiftedRect1(rect1);
    RotatedRect shiftedRect2(rect2);

    // Move rectangles closer to origin
    shiftedRect1.center -= averageCenter;
    shiftedRect2.center -= averageCenter;

    std::vector <Point2f> intersection; intersection.reserve(24);

    const int ret = _rotatedRectangleIntersection(shiftedRect1, shiftedRect2, intersection);

    // If return is not None, the intersection Points are shifted back to the original position
    // and copied to the interesectingRegion
    if (ret != INTERSECT_NONE)
    {
        for (size_t i = 0; i < intersection.size(); ++i)
        {
            intersection[i] += averageCenter;
        }

        Mat(intersection).copyTo(intersectingRegion);
    }
    else
    {
        intersectingRegion.release();
    }

    return ret;
}

} // end namespace
