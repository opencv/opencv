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


#ifndef __CVVECTRACK_H__
#define __CVVECTRACK_H__

#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

#undef max
#undef min

#define max(a,b) ((a)<(b) ? (b) : (a))
#define min(a,b) ((a)>(b) ? (a) : (b))

inline int pow2(int v)
{
        return (v*v);
}

inline int operator == (const CvRect& r1, const CvRect& r2)
{
        return (r1.x == r2.x) && (r1.y == r2.y) &&
                        (r1.width == r2.width) && (r1.height == r2.height);
}

inline int operator != (const CvRect& r1, const CvRect& r2)
{
        return !(r1 == r2);
}

inline
int CmpPoints(const CvPoint& p1, const CvPoint& p2, int err)
{
        /* Simakov: modify __max to max */
        return (max(abs(p1.x - p2.x), abs(p1.y - p2.y)) < err);
}

inline
int PointInRect(const CvPoint& p, const CvRect& r)
{
        return ((p.x > r.x) && (p.x < (r.x + r.width)) &&
                        (p.y > r.y) && (p.y < (r.y + r.height)));
}

inline
int RectInRect(const CvRect& r1, const CvRect& r2)
{
        CvPoint plt = {r1.x, r1.y};
        CvPoint prb = {r1.x + r1.width, r1.y + r1.height};
        return (PointInRect(plt, r2) && PointInRect(prb, r2));
}

inline
CvRect Increase(const CvRect& r, int decr)
{
        CvRect rect;
        rect.x = r.x * decr;
        rect.y = r.y * decr;
        rect.width = r.width * decr;
        rect.height = r.height * decr;
        return rect;
}

inline
CvPoint Increase(const CvPoint& p, int decr)
{
        CvPoint point;
        point.x = p.x * decr;
        point.y = p.y * decr;
        return point;
}

inline
void Move(CvRect& r, int dx, int dy)
{
        r.x += dx;
        r.y += dy;
}

inline
void Move(CvPoint& p, int dx, int dy)
{
        p.x += dx;
        p.y += dy;
}

inline
void Extend(CvRect& r, int d)
{
        r.x -= d;
        r.y -= d;
        r.width += 2*d;
        r.height += 2*d;
}

inline
CvPoint Center(const CvRect& r)
{
        CvPoint p;
        p.x = r.x + r.width / 2;
        p.y = r.y + r.height / 2;
        return p;
}

inline void ReallocImage(IplImage** ppImage, CvSize sz, long lChNum)
{
    IplImage* pImage;
    if( ppImage == NULL )
                return;
    pImage = *ppImage;
    if( pImage != NULL )
    {
        if (pImage->width != sz.width || pImage->height != sz.height || pImage->nChannels != lChNum)
            cvReleaseImage( &pImage );
    }
    if( pImage == NULL )
        pImage = cvCreateImage( sz, IPL_DEPTH_8U, (int)lChNum);
    *ppImage = pImage;
}

#endif //__VECTRACK_H__
