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
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{

// inner product
static float innerProduct(Point2f &v1, Point2f &v2)
{
    return v1.x * v2.y - v1.y * v2.x;
}

static void findCircle3pts(Point2f *pts, Point2f &center, float &radius)
{
    // two edges of the triangle v1, v2
    Point2f v1 = pts[1] - pts[0];
    Point2f v2 = pts[2] - pts[0];

    if (innerProduct(v1, v2) == 0.0f)
    {
        // v1, v2 colineation, can not determine a unique circle
        // find the longtest distance as diameter line
        float d1 = (float)norm(pts[0] - pts[1]);
        float d2 = (float)norm(pts[0] - pts[2]);
        float d3 = (float)norm(pts[1] - pts[2]);
        if (d1 >= d2 && d1 >= d3)
        {
            center = (pts[0] + pts[1]) / 2.0f;
            radius = (d1 / 2.0f);
        }
        else if (d2 >= d1 && d2 >= d3)
        {
            center = (pts[0] + pts[2]) / 2.0f;
            radius = (d2 / 2.0f);
        }
        else if (d3 >= d1 && d3 >= d2)
        {
            center = (pts[1] + pts[2]) / 2.0f;
            radius = (d3 / 2.0f);
        }
    }
    else
    {
        // center is intersection of midperpendicular lines of the two edges v1, v2
        // a1*x + b1*y = c1 where a1 = v1.x, b1 = v1.y
        // a2*x + b2*y = c2 where a2 = v2.x, b2 = v2.y
        Point2f midPoint1 = (pts[0] + pts[1]) / 2.0f;
        float c1 = midPoint1.x * v1.x + midPoint1.y * v1.y;
        Point2f midPoint2 = (pts[0] + pts[2]) / 2.0f;
        float c2 = midPoint2.x * v2.x + midPoint2.y * v2.y;
        float det = v1.x * v2.y - v1.y * v2.x;
        float cx = (c1 * v2.y - c2 * v1.y) / det;
        float cy = (v1.x * c2 - v2.x * c1) / det;
        center.x = (float)cx;
        center.y = (float)cy;
        cx -= pts[0].x;
        cy -= pts[0].y;
        radius = (float)(std::sqrt(cx *cx + cy * cy));
    }
}

const float EPS = 1.0e-4f;

static void findEnclosingCircle3pts_orLess_32f(Point2f *pts, int count, Point2f &center, float &radius)
{
    switch (count)
    {
    case 1:
        center = pts[0];
        radius = 0.0f;
        break;
    case 2:
        center.x = (pts[0].x + pts[1].x) / 2.0f;
        center.y = (pts[0].y + pts[1].y) / 2.0f;
        radius = (float)(norm(pts[0] - pts[1]) / 2.0);
        break;
    case 3:
        findCircle3pts(pts, center, radius);
        break;
    default:
        break;
    }

    radius += EPS;
}

template<typename PT>
static void findThirdPoint(const PT *pts, int i, int j, Point2f &center, float &radius)
{
    center.x = (float)(pts[j].x + pts[i].x) / 2.0f;
    center.y = (float)(pts[j].y + pts[i].y) / 2.0f;
    float dx = (float)(pts[j].x - pts[i].x);
    float dy = (float)(pts[j].y - pts[i].y);
    radius = (float)norm(Point2f(dx, dy)) / 2.0f + EPS;

    for (int k = 0; k < j; ++k)
    {
        dx = center.x - (float)pts[k].x;
        dy = center.y - (float)pts[k].y;
        if (norm(Point2f(dx, dy)) < radius)
        {
            continue;
        }
        else
        {
            Point2f ptsf[3];
            ptsf[0] = (Point2f)pts[i];
            ptsf[1] = (Point2f)pts[j];
            ptsf[2] = (Point2f)pts[k];
            findEnclosingCircle3pts_orLess_32f(ptsf, 3, center, radius);
        }
    }
}


template<typename PT>
void findSecondPoint(const PT *pts, int i, Point2f &center, float &radius)
{
    center.x = (float)(pts[0].x + pts[i].x) / 2.0f;
    center.y = (float)(pts[0].y + pts[i].y) / 2.0f;
    float dx = (float)(pts[0].x - pts[i].x);
    float dy = (float)(pts[0].y - pts[i].y);
    radius = (float)norm(Point2f(dx, dy)) / 2.0f + EPS;

    for (int j = 1; j < i; ++j)
    {
        dx = center.x - (float)pts[j].x;
        dy = center.y - (float)pts[j].y;
        if (norm(Point2f(dx, dy)) < radius)
        {
            continue;
        }
        else
        {
            findThirdPoint(pts, i, j, center, radius);
        }
    }
}


template<typename PT>
static void findMinEnclosingCircle(const PT *pts, int count, Point2f &center, float &radius)
{
    center.x = (float)(pts[0].x + pts[1].x) / 2.0f;
    center.y = (float)(pts[0].y + pts[1].y) / 2.0f;
    float dx = (float)(pts[0].x - pts[1].x);
    float dy = (float)(pts[0].y - pts[1].y);
    radius = (float)norm(Point2f(dx, dy)) / 2.0f + EPS;

    for (int i = 2; i < count; ++i)
    {
        dx = (float)pts[i].x - center.x;
        dy = (float)pts[i].y - center.y;
        float d = (float)norm(Point2f(dx, dy));
        if (d < radius)
        {
            continue;
        }
        else
        {
            findSecondPoint(pts, i, center, radius);
        }
    }
}

// see Welzl, Emo. Smallest enclosing disks (balls and ellipsoids). Springer Berlin Heidelberg, 1991.
void minEnclosingCircle( InputArray _points, Point2f& _center, float& _radius )
{
    CV_INSTRUMENT_REGION()

    Mat points = _points.getMat();
    int count = points.checkVector(2);
    int depth = points.depth();
    Point2f center;
    float radius = 0.f;
    CV_Assert(count >= 0 && (depth == CV_32F || depth == CV_32S));

    _center.x = _center.y = 0.f;
    _radius = 0.f;

    if( count == 0 )
        return;

    bool is_float = depth == CV_32F;
    const Point* ptsi = points.ptr<Point>();
    const Point2f* ptsf = points.ptr<Point2f>();

    // point count <= 3
    if (count <= 3)
    {
        Point2f ptsf3[3];
        for (int i = 0; i < count; ++i)
        {
            ptsf3[i] = (is_float) ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        }
        findEnclosingCircle3pts_orLess_32f(ptsf3, count, center, radius);
        _center = center;
        _radius = radius;
        return;
    }

    if (is_float)
    {
        findMinEnclosingCircle<Point2f>(ptsf, count, center, radius);
#if 0
        for (size_t m = 0; m < count; ++m)
        {
            float d = (float)norm(ptsf[m] - center);
            if (d > radius)
            {
                printf("error!\n");
            }
        }
#endif
    }
    else
    {
        findMinEnclosingCircle<Point>(ptsi, count, center, radius);
#if 0
        for (size_t m = 0; m < count; ++m)
        {
            double dx = ptsi[m].x - center.x;
            double dy = ptsi[m].y - center.y;
            double d = std::sqrt(dx * dx + dy * dy);
            if (d > radius)
            {
                printf("error!\n");
            }
        }
#endif
    }
    _center = center;
    _radius = radius;
}


// calculates length of a curve (e.g. contour perimeter)
double arcLength( InputArray _curve, bool is_closed )
{
    CV_INSTRUMENT_REGION()

    Mat curve = _curve.getMat();
    int count = curve.checkVector(2);
    int depth = curve.depth();
    CV_Assert( count >= 0 && (depth == CV_32F || depth == CV_32S));
    double perimeter = 0;

    int i;

    if( count <= 1 )
        return 0.;

    bool is_float = depth == CV_32F;
    int last = is_closed ? count-1 : 0;
    const Point* pti = curve.ptr<Point>();
    const Point2f* ptf = curve.ptr<Point2f>();

    Point2f prev = is_float ? ptf[last] : Point2f((float)pti[last].x,(float)pti[last].y);

    for( i = 0; i < count; i++ )
    {
        Point2f p = is_float ? ptf[i] : Point2f((float)pti[i].x,(float)pti[i].y);
        float dx = p.x - prev.x, dy = p.y - prev.y;
        perimeter += std::sqrt(dx*dx + dy*dy);

        prev = p;
    }

    return perimeter;
}

// area of a whole sequence
double contourArea( InputArray _contour, bool oriented )
{
    CV_INSTRUMENT_REGION()

    Mat contour = _contour.getMat();
    int npoints = contour.checkVector(2);
    int depth = contour.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    if( npoints == 0 )
        return 0.;

    double a00 = 0;
    bool is_float = depth == CV_32F;
    const Point* ptsi = contour.ptr<Point>();
    const Point2f* ptsf = contour.ptr<Point2f>();
    Point2f prev = is_float ? ptsf[npoints-1] : Point2f((float)ptsi[npoints-1].x, (float)ptsi[npoints-1].y);

    for( int i = 0; i < npoints; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        a00 += (double)prev.x * p.y - (double)prev.y * p.x;
        prev = p;
    }

    a00 *= 0.5;
    if( !oriented )
        a00 = fabs(a00);

    return a00;
}

static RotatedRect fitEllipseHalir98Impl(InputArray _points)
{
    CV_INSTRUMENT_REGION()

    CV_Assert((_points.type() == CV_32SC2 || _points.type() == CV_32FC2 || _points.type() == CV_64FC2));
//        && (_points.kind() == _InputArray::MAT || _points.kind() == _InputArray::STD_VECTOR));

    if (_points.total() < 5)
        CV_Error(Error::StsBadSize, "Not enough points to fit an ellipse.");

    Mat points = _points.getMat();
    const int pointSz = (const int)points.total();
    Mat D1(pointSz, 3, CV_64FC1), D2(pointSz, 3, CV_64FC1);
    Matx33d S1, S2, S3, T, M;
    Matx31d eigenValX, eigenVecY;

    double *p_D1 = D1.ptr<double>(), *p_D2 = D2.ptr<double>();
    Point2d meanVal;
    int i = 1, j = 0;

#if CV_SIMD128
    bool haveSimd = hasSIMD128();
#endif

    if (points.depth() == CV_64F)
    {
        Point2d *p_points = points.ptr<Point2d>();
        meanVal = p_points[0];

#if CV_SIMD128_64F
        if (haveSimd)
        {
            v_float64x2 v_psum1 = v_load((const double*)p_points),
                    v_psum2 = v_load((const double*)(p_points + 1));

            for (i = 2; i <= pointSz - 2; i += 2)
            {
                v_psum1 += v_load((const double*)(p_points + i));
                v_psum2 += v_load((const double*)(p_points + i + 1));
            }

            v_psum1 += v_psum2;
            v_store((double*)&meanVal, v_psum1);
        }
#endif
        for (; i < pointSz; ++i)
            meanVal += p_points[i];

        meanVal.x /= (double)pointSz;
        meanVal.y /= (double)pointSz;

        i = 0;
#if CV_SIMD128_64F
        if (haveSimd)
        {
            v_float64x2 v_mean(meanVal.x, meanVal.y), v_p1, v_p2, v_px, v_py, v_ps1, v_ps2;

            for (; i <= pointSz - 2; i += 2, j += 6)
            {
                v_p1 = v_load((const double*)(p_points + i)) - v_mean;
                v_p2 = v_load((const double*)(p_points + i + 1)) - v_mean;

                v_zip(v_p1, v_p2, v_px, v_py);
                v_ps1 = v_p1 * v_p1;

                v_ps2 = v_p2 * v_p2;
                v_px *= v_py;

                v_store((double*)(p_D1 + j), v_combine_low(v_ps1, v_px));
                v_store((double*)(p_D1 + j + 2), v_combine_low(v_combine_high(v_ps1, v_ps1), v_ps2));
                v_store((double*)(p_D1 + j + 4), v_combine_high(v_px, v_ps2));
            }
        }
#endif
        for(; i < pointSz; ++i, j += 3)
        {
            Point2d pt = p_points[i] - meanVal;
            p_D1[j] = pt.x * pt.x;
            p_D1[j + 1] = pt.x * pt.y;
            p_D1[j + 2] = pt.y * pt.y;
        }

        j = i = 0;
#if CV_SIMD128_64F
        if (haveSimd)
        {
            v_float64x2 v_mean(meanVal.x, meanVal.y), v_1 = v_setall_f64(1.);

            for (; i <= pointSz - 2; i += 2, j += 6)
            {
                v_float64x2 v_p1 = v_load((const double*)(p_points + i)) - v_mean;
                v_float64x2 v_p2 = v_load((const double*)(p_points + i + 1)) - v_mean;

                v_store((double*)(p_D2 + j), v_p1);
                v_store((double*)(p_D2 + j + 2), v_combine_low(v_1, v_p2));
                v_store((double*)(p_D2 + j + 4), v_combine_high(v_p2, v_1));
            }
        }
#endif
        for(; i < pointSz; ++i, j += 3)
        {
            Point2d pt = p_points[i] - meanVal;
            p_D2[j] = pt.x;
            p_D2[j + 1] = pt.y;
            p_D2[j + 2] = 1.;
        }

    }
    else if (points.depth() == CV_32F)
    {
        Point2f *p_points = points.ptr<Point2f>();
        meanVal = (Point2d)p_points[0];

#if CV_SIMD128
        if (haveSimd)
        {
            v_float32x4 v_psum1 = v_load((const float*)p_points),
                    v_psum2 = v_load((const float*)(p_points + 2));

            for (i = 4; i <= pointSz - 4; i += 4)
            {
                v_psum1 += v_load((const float*)(p_points + i));
                v_psum2 += v_load((const float*)(p_points + i + 2));
            }

            v_psum1 += v_psum2;
            float sumTmp[4];
            v_store((float*)sumTmp, v_psum1);
            meanVal.x = (double)sumTmp[0] + sumTmp[2];
            meanVal.y = (double)sumTmp[1] + sumTmp[3];
        }
#endif

        for (; i < pointSz; ++i)
            meanVal += (Point2d)p_points[i];

        meanVal.x /= (double)pointSz;
        meanVal.y /= (double)pointSz;

        i = 0;
#if CV_SIMD128_64F
        if (haveSimd)
        {
            v_float64x2 v_mean(meanVal.x, meanVal.y), v_p1, v_p2, v_px, v_py, v_ps1, v_ps2;

            for (; i <= pointSz - 2; i += 2, j += 6)
            {
                v_float32x4 v_p_float = v_load((const float*)(p_points + i));
                v_p1 = v_cvt_f64(v_p_float) - v_mean;
                v_p2 = v_cvt_f64_high(v_p_float) - v_mean;

                v_zip(v_p1, v_p2, v_px, v_py);
                v_ps1 = v_p1 * v_p1;

                v_ps2 = v_p2 * v_p2;
                v_px *= v_py;

                v_store((double*)(p_D1 + j), v_combine_low(v_ps1, v_px));
                v_store((double*)(p_D1 + j + 2), v_combine_low(v_combine_high(v_ps1, v_ps1), v_ps2));
                v_store((double*)(p_D1 + j + 4), v_combine_high(v_px, v_ps2));
            }

        }
#endif
        for(; i < pointSz; ++i, j += 3)
        {
            Point2d pt = (Point2d)p_points[i] - meanVal;
            p_D1[j] = pt.x * pt.x;
            p_D1[j + 1] = pt.x * pt.y;
            p_D1[j + 2] = pt.y * pt.y;
        }

        j = i = 0;
#if CV_SIMD128_64F
        if (haveSimd)
        {
            v_float64x2 v_mean(meanVal.x, meanVal.y), v_p1, v_p2, v_1 = v_setall_f64(1.);

            for (; i <= pointSz - 2; i += 2, j += 6)
            {
                v_float32x4 v_p_float = v_load((const float*)(p_points + i));
                v_p1 = v_cvt_f64(v_p_float) - v_mean;
                v_p2 = v_cvt_f64_high(v_p_float) - v_mean;

                v_store((double*)(p_D2 + j), v_p1);
                v_store((double*)(p_D2 + j + 2), v_combine_low(v_1, v_p2));
                v_store((double*)(p_D2 + j + 4), v_combine_high(v_p2, v_1));
            }
        }
#endif
        for(; i < pointSz; ++i, j += 3)
        {
            Point2d pt = (Point2d)p_points[i] - meanVal;
            p_D2[j] = pt.x;
            p_D2[j + 1] = pt.y;
            p_D2[j + 2] = 1.;
        }
    }
    else
    {
        const Point *p_points = points.ptr<Point>();
        meanVal = (Point2d)p_points[0];

#if CV_SIMD128
        if (haveSimd)
        {
            v_float32x4 v_psum1 = v_cvt_f32(v_load((const int*)p_points)),
                    v_psum2 = v_cvt_f32(v_load((const int*)(p_points + 2)));

            for (i = 4; i <= pointSz - 4; i += 4)
            {
                v_psum1 += v_cvt_f32(v_load((const int*)(p_points + i)));
                v_psum2 += v_cvt_f32(v_load((const int*)(p_points + i + 2)));
            }

            v_psum1 += v_psum2;
            float sumTmp[4];
            v_store((float*)sumTmp, v_psum1);
            meanVal.x = (double)(sumTmp[0] + sumTmp[2]);
            meanVal.y = (double)(sumTmp[1] + sumTmp[3]);
        }
#endif

        for (; i < pointSz; ++i)
            meanVal += (Point2d)p_points[i];

        meanVal.x /= (double)pointSz;
        meanVal.y /= (double)pointSz;

        i = 0;
#if CV_SIMD128_64F
        if (haveSimd)
        {
            v_float64x2 v_mean(meanVal.x, meanVal.y), v_p1, v_p2, v_px, v_py, v_ps1, v_ps2;

            for (; i <= pointSz - 2; i += 2, j += 6)
            {
                v_int32x4 v_p_int = v_load((const int*)(p_points + i));
                v_p1 = v_cvt_f64(v_p_int) - v_mean;
                v_p2 = v_cvt_f64_high(v_p_int) - v_mean;

                v_zip(v_p1, v_p2, v_px, v_py);
                v_ps1 = v_p1 * v_p1;

                v_ps2 = v_p2 * v_p2;
                v_px *= v_py;

                v_store((double*)(p_D1 + j), v_combine_low(v_ps1, v_px));
                v_store((double*)(p_D1 + j + 2), v_combine_low(v_combine_high(v_ps1, v_ps1), v_ps2));
                v_store((double*)(p_D1 + j + 4), v_combine_high(v_px, v_ps2));
            }
        }
#endif
        for(; i < pointSz; ++i, j += 3)
        {
            Point2d pt = (Point2d)p_points[i] - meanVal;
            p_D1[j] = pt.x * pt.x;
            p_D1[j + 1] = pt.x * pt.y;
            p_D1[j + 2] = pt.y * pt.y;
        }

        j = i = 0;
#if CV_SIMD128_64F
        if (haveSimd)
        {
            v_float64x2 v_mean(meanVal.x, meanVal.y), v_p1, v_p2, v_1 = v_setall_f64(1.);

            for (; i <= pointSz - 2; i += 2, j += 6) {
                v_int32x4 v_p_int = v_load((const int*)(p_points + i));
                v_p1 = v_cvt_f64(v_p_int) - v_mean;
                v_p2 = v_cvt_f64_high(v_p_int) - v_mean;

                v_store((double*)(p_D2 + j), v_p1);
                v_store((double*)(p_D2 + j + 2), v_combine_low(v_1, v_p2));
                v_store((double*)(p_D2 + j + 4), v_combine_high(v_p2, v_1));
            }
        }
#endif
        for(; i < pointSz; ++i, j += 3)
        {
            Point2d pt = (Point2d)p_points[i] - meanVal;
            p_D2[j] = pt.x;
            p_D2[j + 1] = pt.y;
            p_D2[j + 2] = 1.;
        }
    }

    gemm(D1, D1, 1, noArray(), 0, S1, GEMM_1_T);
    gemm(D1, D2, 1, noArray(), 0, S2, GEMM_1_T);
    gemm(D2, D2, 1, noArray(), 0, S3, GEMM_1_T);

    gemm(S3.inv(DECOMP_LU), S2, -1., noArray(), 0, T, GEMM_2_T);
    gemm(S2, T, 1., S1, 1., M);

    M.val[0] /= 2.;
    M.val[1] /= 2.;
    M.val[2] /= 2.;
    M.val[3] *= -1.;
    M.val[4] *= -1.;
    M.val[5] *= -1.;
    M.val[6] /= 2.;
    M.val[7] /= 2.;
    M.val[8] /= 2.;

//    // Not stable in some situations, but sometimes results looks better than with solve() ...
//    Mat evalx, evecx;
//    Matx31d cond;
//    eigen(M, evalx, evecx);
//    cond.val[0] = 4 * evecx.at<double>(0) * evecx.at<double>(6) - evecx.at<double>(3) * evecx.at<double>(3);
//    cond.val[1] = 4 * evecx.at<double>(1) * evecx.at<double>(7) - evecx.at<double>(4) * evecx.at<double>(4);
//    cond.val[2] = 4 * evecx.at<double>(2) * evecx.at<double>(8) - evecx.at<double>(5) * evecx.at<double>(5);

//    double minVal = DBL_MAX;
//    for(i = 0, j = -1; i < 3; ++i)
//    {
//        if(cond.val[i] > 0. && cond.val[i] < minVal)
//        {
//            j = i;
//            minVal = cond.val[i];
//        }
//    }

//    eigenValX.val[0] = evecx.at<double>(j);
//    eigenValX.val[1] = evecx.at<double>(3 + j);
//    eigenValX.val[2] = evecx.at<double>(6 + j);

    // Must be DECOMP_SVD, DECOMP_LU failed in feature2d tests
    solve(M, Matx31d(1., 1., 1.), eigenValX, DECOMP_SVD);

    if(std::abs(eigenValX.val[0]) < DBL_MIN && std::abs(eigenValX.val[2]) < DBL_MIN)
    {
        // If this is true points are on a circle, add some large value to eigenVal[0] and [1], but not too large
        eigenValX.val[2] = eigenValX.val[0] = (double)FLT_MAX;
    }

    gemm(T, eigenValX, 1., noArray(), 0, eigenVecY);

    Matx61d par(eigenValX.val[0], eigenValX.val[1], eigenValX.val[2],
        -2. * eigenValX.val[0] * meanVal.x - eigenValX.val[1] * meanVal.y + eigenVecY.val[0],
        -eigenValX.val[1] * meanVal.x - 2. * eigenValX.val[2] * meanVal.y + eigenVecY.val[1],
        eigenValX.val[0] * meanVal.x * meanVal.x + eigenValX.val[1] * meanVal.x * meanVal.y +
            eigenValX.val[2] * meanVal.y * meanVal.y - eigenVecY.val[0] * meanVal.x -
            eigenVecY.val[1] * meanVal.y + eigenVecY.val[2]);

    double theta = atan2(par.val[1], par.val[0] - par.val[2]) / 2.;

    double cost = std::cos(theta), sint = std::sin(theta);
    double sint_sq = sint * sint, cost_sq = cost * cost, cost_sint = cost * sint;

    double Ao = par.val[5];
    double Au = par.val[3] * cost + par.val[4] * sint;
    double Av = par.val[4] * cost - par.val[3] * sint;
    double Auu = par.val[0] * cost_sq + par.val[2] * sint_sq + par.val[1] * cost_sint;
    double Avv = par.val[0] * sint_sq + par.val[2] * cost_sq - par.val[1] * cost_sint;

    double tuCenter = -Au / (2. * Auu), tvCenter = -Av / (2. * Avv);
    double wCenter = Ao - Auu * tuCenter * tuCenter - Avv * tvCenter * tvCenter;

    double uCenter = tuCenter * cost - tvCenter * sint, vCenter = tuCenter * sint + tvCenter * cost;
    double Ru = -wCenter / Auu, Rv = -wCenter / Avv;

    Ru = std::sqrt(std::abs(Ru));
    Rv = std::sqrt(std::abs(Rv));

    Point2f xy = Point2f((float)uCenter, (float)vCenter);
    Size2f wh = Size2f((float)(2. * Ru), (float)(2. * Rv));

    if (wh.height <= 0. || wh.width <= 0. || wh.height != wh.height ||
            wh.width != wh.width || xy.x != xy.x || xy.y != xy.y)
    {
        CV_Error(Error::StsBadSize, "Could not fit ellipse to given points.");
    }

    return RotatedRect(xy, wh, (float)(theta * 180. / CV_PI));
}

static RotatedRect fitEllipseFitz95Impl(InputArray _points )
{
    CV_INSTRUMENT_REGION()

    Mat points = _points.getMat();
    int i, n = points.checkVector(2);
    int depth = points.depth();
    CV_Assert( n >= 0 && (depth == CV_32F || depth == CV_32S));

    RotatedRect box;

    if( n < 5 )
        CV_Error( CV_StsBadSize, "There should be at least 5 points to fit the ellipse" );

    // New fitellipse algorithm, contributed by Dr. Daniel Weiss
    Point2f c(0,0);
    double gfp[5] = {0}, rp[5] = {0}, t;
    const double min_eps = 1e-8;
    bool is_float = depth == CV_32F;
    const Point* ptsi = points.ptr<Point>();
    const Point2f* ptsf = points.ptr<Point2f>();

    AutoBuffer<double> _Ad(n*5), _bd(n);
    double *Ad = _Ad, *bd = _bd;

    // first fit for parameters A - E
    Mat A( n, 5, CV_64F, Ad );
    Mat b( n, 1, CV_64F, bd );
    Mat x( 5, 1, CV_64F, gfp );

    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        c += p;
    }
    c.x /= n;
    c.y /= n;

    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        p -= c;

        bd[i] = 10000.0; // 1.0?
        Ad[i*5] = -(double)p.x * p.x; // A - C signs inverted as proposed by APP
        Ad[i*5 + 1] = -(double)p.y * p.y;
        Ad[i*5 + 2] = -(double)p.x * p.y;
        Ad[i*5 + 3] = p.x;
        Ad[i*5 + 4] = p.y;
    }

    solve(A, b, x, DECOMP_SVD);

    // now use general-form parameters A - E to find the ellipse center:
    // differentiate general form wrt x/y to get two equations for cx and cy
    A = Mat( 2, 2, CV_64F, Ad );
    b = Mat( 2, 1, CV_64F, bd );
    x = Mat( 2, 1, CV_64F, rp );
    Ad[0] = 2 * gfp[0];
    Ad[1] = Ad[2] = gfp[2];
    Ad[3] = 2 * gfp[1];
    bd[0] = gfp[3];
    bd[1] = gfp[4];
    solve( A, b, x, DECOMP_SVD );

    // re-fit for parameters A - C with those center coordinates
    A = Mat( n, 3, CV_64F, Ad );
    b = Mat( n, 1, CV_64F, bd );
    x = Mat( 3, 1, CV_64F, gfp );
    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        p -= c;
        bd[i] = 1.0;
        Ad[i * 3] = (p.x - rp[0]) * (p.x - rp[0]);
        Ad[i * 3 + 1] = (p.y - rp[1]) * (p.y - rp[1]);
        Ad[i * 3 + 2] = (p.x - rp[0]) * (p.y - rp[1]);
    }
    solve(A, b, x, DECOMP_SVD);

    // store angle and radii
    rp[4] = -0.5 * atan2(gfp[2], gfp[1] - gfp[0]); // convert from APP angle usage
    if( fabs(gfp[2]) > min_eps )
        t = gfp[2]/sin(-2.0 * rp[4]);
    else // ellipse is rotated by an integer multiple of pi/2
        t = gfp[1] - gfp[0];
    rp[2] = fabs(gfp[0] + gfp[1] - t);
    if( rp[2] > min_eps )
        rp[2] = std::sqrt(2.0 / rp[2]);
    rp[3] = fabs(gfp[0] + gfp[1] + t);
    if( rp[3] > min_eps )
        rp[3] = std::sqrt(2.0 / rp[3]);

    box.center.x = (float)rp[0] + c.x;
    box.center.y = (float)rp[1] + c.y;
    box.size.width = (float)(rp[2]*2);
    box.size.height = (float)(rp[3]*2);
    if( box.size.width > box.size.height )
    {
        float tmp;
        CV_SWAP( box.size.width, box.size.height, tmp );
        box.angle = (float)(90 + rp[4]*180/CV_PI);
    }
    if( box.angle < -180 )
        box.angle += 360;
    if( box.angle > 360 )
        box.angle -= 360;

    return box;
}

RotatedRect fitEllipse(InputArray points)
{
    return fitEllipseHalir98Impl(points);
}

RotatedRect fitEllipse(InputArray points, int method)
{
    switch(method)
    {
    case ELLIPSEFIT_HALIR:
        return fitEllipseHalir98Impl(points);
    case ELLIPSEFIT_FITZGIBBON:
        return fitEllipseFitz95Impl(points);
    default:
        CV_Error(Error::StsBadArg, "Ellipse fitting method not known/supported.");
    }

    return RotatedRect();
}

// Calculates bounding rectagnle of a point set or retrieves already calculated
static Rect pointSetBoundingRect( const Mat& points )
{
    int npoints = points.checkVector(2);
    int depth = points.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i;
    bool is_float = depth == CV_32F;

    if( npoints == 0 )
        return Rect();

    const Point* pts = points.ptr<Point>();
    Point pt = pts[0];

#if CV_SSE4_2
    if(cv::checkHardwareSupport(CV_CPU_SSE4_2))
    {
        if( !is_float )
        {
            __m128i minval, maxval;
            minval = maxval = _mm_loadl_epi64((const __m128i*)(&pt)); //min[0]=pt.x, min[1]=pt.y

            for( i = 1; i < npoints; i++ )
            {
                __m128i ptXY = _mm_loadl_epi64((const __m128i*)&pts[i]);
                minval = _mm_min_epi32(ptXY, minval);
                maxval = _mm_max_epi32(ptXY, maxval);
            }
            xmin = _mm_cvtsi128_si32(minval);
            ymin = _mm_cvtsi128_si32(_mm_srli_si128(minval, 4));
            xmax = _mm_cvtsi128_si32(maxval);
            ymax = _mm_cvtsi128_si32(_mm_srli_si128(maxval, 4));
        }
        else
        {
            __m128 minvalf, maxvalf, z = _mm_setzero_ps(), ptXY = _mm_setzero_ps();
            minvalf = maxvalf = _mm_loadl_pi(z, (const __m64*)(&pt));

            for( i = 1; i < npoints; i++ )
            {
                ptXY = _mm_loadl_pi(ptXY, (const __m64*)&pts[i]);

                minvalf = _mm_min_ps(minvalf, ptXY);
                maxvalf = _mm_max_ps(maxvalf, ptXY);
            }

            float xyminf[2], xymaxf[2];
            _mm_storel_pi((__m64*)xyminf, minvalf);
            _mm_storel_pi((__m64*)xymaxf, maxvalf);
            xmin = cvFloor(xyminf[0]);
            ymin = cvFloor(xyminf[1]);
            xmax = cvFloor(xymaxf[0]);
            ymax = cvFloor(xymaxf[1]);
        }
    }
    else
#endif
    {
        if( !is_float )
        {
            xmin = xmax = pt.x;
            ymin = ymax = pt.y;

            for( i = 1; i < npoints; i++ )
            {
                pt = pts[i];

                if( xmin > pt.x )
                    xmin = pt.x;

                if( xmax < pt.x )
                    xmax = pt.x;

                if( ymin > pt.y )
                    ymin = pt.y;

                if( ymax < pt.y )
                    ymax = pt.y;
            }
        }
        else
        {
            Cv32suf v;
            // init values
            xmin = xmax = CV_TOGGLE_FLT(pt.x);
            ymin = ymax = CV_TOGGLE_FLT(pt.y);

            for( i = 1; i < npoints; i++ )
            {
                pt = pts[i];
                pt.x = CV_TOGGLE_FLT(pt.x);
                pt.y = CV_TOGGLE_FLT(pt.y);

                if( xmin > pt.x )
                    xmin = pt.x;

                if( xmax < pt.x )
                    xmax = pt.x;

                if( ymin > pt.y )
                    ymin = pt.y;

                if( ymax < pt.y )
                    ymax = pt.y;
            }

            v.i = CV_TOGGLE_FLT(xmin); xmin = cvFloor(v.f);
            v.i = CV_TOGGLE_FLT(ymin); ymin = cvFloor(v.f);
            // because right and bottom sides of the bounding rectangle are not inclusive
            // (note +1 in width and height calculation below), cvFloor is used here instead of cvCeil
            v.i = CV_TOGGLE_FLT(xmax); xmax = cvFloor(v.f);
            v.i = CV_TOGGLE_FLT(ymax); ymax = cvFloor(v.f);
        }
    }

    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

static Rect maskBoundingRect( const Mat& img )
{
    CV_Assert( img.depth() <= CV_8S && img.channels() == 1 );

    Size size = img.size();
    int xmin = size.width, ymin = -1, xmax = -1, ymax = -1, i, j, k;

    for( i = 0; i < size.height; i++ )
    {
        const uchar* _ptr = img.ptr(i);
        const uchar* ptr = (const uchar*)alignPtr(_ptr, 4);
        int have_nz = 0, k_min, offset = (int)(ptr - _ptr);
        j = 0;
        offset = MIN(offset, size.width);
        for( ; j < offset; j++ )
            if( _ptr[j] )
            {
                have_nz = 1;
                break;
            }
        if( j < offset )
        {
            if( j < xmin )
                xmin = j;
            if( j > xmax )
                xmax = j;
        }
        if( offset < size.width )
        {
            xmin -= offset;
            xmax -= offset;
            size.width -= offset;
            j = 0;
            for( ; j <= xmin - 4; j += 4 )
                if( *((int*)(ptr+j)) )
                    break;
            for( ; j < xmin; j++ )
                if( ptr[j] )
                {
                    xmin = j;
                    if( j > xmax )
                        xmax = j;
                    have_nz = 1;
                    break;
                }
            k_min = MAX(j-1, xmax);
            k = size.width - 1;
            for( ; k > k_min && (k&3) != 3; k-- )
                if( ptr[k] )
                    break;
            if( k > k_min && (k&3) == 3 )
            {
                for( ; k > k_min+3; k -= 4 )
                    if( *((int*)(ptr+k-3)) )
                        break;
            }
            for( ; k > k_min; k-- )
                if( ptr[k] )
                {
                    xmax = k;
                    have_nz = 1;
                    break;
                }
            if( !have_nz )
            {
                j &= ~3;
                for( ; j <= k - 3; j += 4 )
                    if( *((int*)(ptr+j)) )
                        break;
                for( ; j <= k; j++ )
                    if( ptr[j] )
                    {
                        have_nz = 1;
                        break;
                    }
            }
            xmin += offset;
            xmax += offset;
            size.width += offset;
        }
        if( have_nz )
        {
            if( ymin < 0 )
                ymin = i;
            ymax = i;
        }
    }

    if( xmin >= size.width )
        xmin = ymin = 0;
    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

Rect boundingRect(InputArray array)
{
    CV_INSTRUMENT_REGION()

    Mat m = array.getMat();
    return m.depth() <= CV_8U ? maskBoundingRect(m) : pointSetBoundingRect(m);
}

} // \namespace cv
////////////////////////////////////////////// C API ///////////////////////////////////////////

CV_IMPL int
cvMinEnclosingCircle( const void* array, CvPoint2D32f * _center, float *_radius )
{
    cv::AutoBuffer<double> abuf;
    cv::Mat points = cv::cvarrToMat(array, false, false, 0, &abuf);
    cv::Point2f center;
    float radius;

    cv::minEnclosingCircle(points, center, radius);
    if(_center)
        *_center = center;
    if(_radius)
        *_radius = radius;
    return 1;
}

static void
icvMemCopy( double **buf1, double **buf2, double **buf3, int *b_max )
{
    CV_Assert( (*buf1 != NULL || *buf2 != NULL) && *buf3 != NULL );

    int bb = *b_max;
    if( *buf2 == NULL )
    {
        *b_max = 2 * (*b_max);
        *buf2 = (double *)cvAlloc( (*b_max) * sizeof( double ));

        memcpy( *buf2, *buf3, bb * sizeof( double ));

        *buf3 = *buf2;
        cvFree( buf1 );
        *buf1 = NULL;
    }
    else
    {
        *b_max = 2 * (*b_max);
        *buf1 = (double *) cvAlloc( (*b_max) * sizeof( double ));

        memcpy( *buf1, *buf3, bb * sizeof( double ));

        *buf3 = *buf1;
        cvFree( buf2 );
        *buf2 = NULL;
    }
}


/* area of a contour sector */
static double icvContourSecArea( CvSeq * contour, CvSlice slice )
{
    CvPoint pt;                 /*  pointer to points   */
    CvPoint pt_s, pt_e;         /*  first and last points  */
    CvSeqReader reader;         /*  points reader of contour   */

    int p_max = 2, p_ind;
    int lpt, flag, i;
    double a00;                 /* unnormalized moments m00    */
    double xi, yi, xi_1, yi_1, x0, y0, dxy, sk, sk1, t;
    double x_s, y_s, nx, ny, dx, dy, du, dv;
    double eps = 1.e-5;
    double *p_are1, *p_are2, *p_are;
    double area = 0;

    CV_Assert( contour != NULL && CV_IS_SEQ_POINT_SET( contour ));

    lpt = cvSliceLength( slice, contour );
    /*if( n2 >= n1 )
        lpt = n2 - n1 + 1;
    else
        lpt = contour->total - n1 + n2 + 1;*/

    if( contour->total <= 0 || lpt <= 2 )
        return 0.;

    a00 = x0 = y0 = xi_1 = yi_1 = 0;
    sk1 = 0;
    flag = 0;
    dxy = 0;
    p_are1 = (double *) cvAlloc( p_max * sizeof( double ));

    p_are = p_are1;
    p_are2 = NULL;

    cvStartReadSeq( contour, &reader, 0 );
    cvSetSeqReaderPos( &reader, slice.start_index );
    CV_READ_SEQ_ELEM( pt_s, reader );
    p_ind = 0;
    cvSetSeqReaderPos( &reader, slice.end_index );
    CV_READ_SEQ_ELEM( pt_e, reader );

/*    normal coefficients    */
    nx = pt_s.y - pt_e.y;
    ny = pt_e.x - pt_s.x;
    cvSetSeqReaderPos( &reader, slice.start_index );

    while( lpt-- > 0 )
    {
        CV_READ_SEQ_ELEM( pt, reader );

        if( flag == 0 )
        {
            xi_1 = (double) pt.x;
            yi_1 = (double) pt.y;
            x0 = xi_1;
            y0 = yi_1;
            sk1 = 0;
            flag = 1;
        }
        else
        {
            xi = (double) pt.x;
            yi = (double) pt.y;

/****************   edges intersection examination   **************************/
            sk = nx * (xi - pt_s.x) + ny * (yi - pt_s.y);
            if( (fabs( sk ) < eps && lpt > 0) || sk * sk1 < -eps )
            {
                if( fabs( sk ) < eps )
                {
                    dxy = xi_1 * yi - xi * yi_1;
                    a00 = a00 + dxy;
                    dxy = xi * y0 - x0 * yi;
                    a00 = a00 + dxy;

                    if( p_ind >= p_max )
                        icvMemCopy( &p_are1, &p_are2, &p_are, &p_max );

                    p_are[p_ind] = a00 / 2.;
                    p_ind++;
                    a00 = 0;
                    sk1 = 0;
                    x0 = xi;
                    y0 = yi;
                    dxy = 0;
                }
                else
                {
/*  define intersection point    */
                    dv = yi - yi_1;
                    du = xi - xi_1;
                    dx = ny;
                    dy = -nx;
                    if( fabs( du ) > eps )
                        t = ((yi_1 - pt_s.y) * du + dv * (pt_s.x - xi_1)) /
                            (du * dy - dx * dv);
                    else
                        t = (xi_1 - pt_s.x) / dx;
                    if( t > eps && t < 1 - eps )
                    {
                        x_s = pt_s.x + t * dx;
                        y_s = pt_s.y + t * dy;
                        dxy = xi_1 * y_s - x_s * yi_1;
                        a00 += dxy;
                        dxy = x_s * y0 - x0 * y_s;
                        a00 += dxy;
                        if( p_ind >= p_max )
                            icvMemCopy( &p_are1, &p_are2, &p_are, &p_max );

                        p_are[p_ind] = a00 / 2.;
                        p_ind++;

                        a00 = 0;
                        sk1 = 0;
                        x0 = x_s;
                        y0 = y_s;
                        dxy = x_s * yi - xi * y_s;
                    }
                }
            }
            else
                dxy = xi_1 * yi - xi * yi_1;

            a00 += dxy;
            xi_1 = xi;
            yi_1 = yi;
            sk1 = sk;

        }
    }

    xi = x0;
    yi = y0;
    dxy = xi_1 * yi - xi * yi_1;

    a00 += dxy;

    if( p_ind >= p_max )
        icvMemCopy( &p_are1, &p_are2, &p_are, &p_max );

    p_are[p_ind] = a00 / 2.;
    p_ind++;

    // common area calculation
    area = 0;
    for( i = 0; i < p_ind; i++ )
        area += fabs( p_are[i] );

    if( p_are1 != NULL )
        cvFree( &p_are1 );
    else if( p_are2 != NULL )
        cvFree( &p_are2 );

    return area;
}


/* external contour area function */
CV_IMPL double
cvContourArea( const void *array, CvSlice slice, int oriented )
{
    double area = 0;

    CvContour contour_header;
    CvSeq* contour = 0;
    CvSeqBlock block;

    if( CV_IS_SEQ( array ))
    {
        contour = (CvSeq*)array;
        if( !CV_IS_SEQ_POLYLINE( contour ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );
    }
    else
    {
        contour = cvPointSeqFromMat( CV_SEQ_KIND_CURVE, array, &contour_header, &block );
    }

    if( cvSliceLength( slice, contour ) == contour->total )
    {
        cv::AutoBuffer<double> abuf;
        cv::Mat points = cv::cvarrToMat(contour, false, false, 0, &abuf);
        return cv::contourArea( points, oriented !=0 );
    }

    if( CV_SEQ_ELTYPE( contour ) != CV_32SC2 )
        CV_Error( CV_StsUnsupportedFormat,
        "Only curves with integer coordinates are supported in case of contour slice" );
    area = icvContourSecArea( contour, slice );
    return oriented ? area : fabs(area);
}


/* calculates length of a curve (e.g. contour perimeter) */
CV_IMPL  double
cvArcLength( const void *array, CvSlice slice, int is_closed )
{
    double perimeter = 0;

    int i, j = 0, count;
    const int N = 16;
    float buf[N];
    CvMat buffer = cvMat( 1, N, CV_32F, buf );
    CvSeqReader reader;
    CvContour contour_header;
    CvSeq* contour = 0;
    CvSeqBlock block;

    if( CV_IS_SEQ( array ))
    {
        contour = (CvSeq*)array;
        if( !CV_IS_SEQ_POLYLINE( contour ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );
        if( is_closed < 0 )
            is_closed = CV_IS_SEQ_CLOSED( contour );
    }
    else
    {
        is_closed = is_closed > 0;
        contour = cvPointSeqFromMat(
                                    CV_SEQ_KIND_CURVE | (is_closed ? CV_SEQ_FLAG_CLOSED : 0),
                                    array, &contour_header, &block );
    }

    if( contour->total > 1 )
    {
        int is_float = CV_SEQ_ELTYPE( contour ) == CV_32FC2;

        cvStartReadSeq( contour, &reader, 0 );
        cvSetSeqReaderPos( &reader, slice.start_index );
        count = cvSliceLength( slice, contour );

        count -= !is_closed && count == contour->total;

        // scroll the reader by 1 point
        reader.prev_elem = reader.ptr;
        CV_NEXT_SEQ_ELEM( sizeof(CvPoint), reader );

        for( i = 0; i < count; i++ )
        {
            float dx, dy;

            if( !is_float )
            {
                CvPoint* pt = (CvPoint*)reader.ptr;
                CvPoint* prev_pt = (CvPoint*)reader.prev_elem;

                dx = (float)pt->x - (float)prev_pt->x;
                dy = (float)pt->y - (float)prev_pt->y;
            }
            else
            {
                CvPoint2D32f* pt = (CvPoint2D32f*)reader.ptr;
                CvPoint2D32f* prev_pt = (CvPoint2D32f*)reader.prev_elem;

                dx = pt->x - prev_pt->x;
                dy = pt->y - prev_pt->y;
            }

            reader.prev_elem = reader.ptr;
            CV_NEXT_SEQ_ELEM( contour->elem_size, reader );
            // Bugfix by Axel at rubico.com 2010-03-22, affects closed slices only
            // wraparound not handled by CV_NEXT_SEQ_ELEM
            if( is_closed && i == count - 2 )
                cvSetSeqReaderPos( &reader, slice.start_index );

            buffer.data.fl[j] = dx * dx + dy * dy;
            if( ++j == N || i == count - 1 )
            {
                buffer.cols = j;
                cvPow( &buffer, &buffer, 0.5 );
                for( ; j > 0; j-- )
                    perimeter += buffer.data.fl[j-1];
            }
        }
    }

    return perimeter;
}


CV_IMPL CvBox2D
cvFitEllipse2( const CvArr* array )
{
    cv::AutoBuffer<double> abuf;
    cv::Mat points = cv::cvarrToMat(array, false, false, 0, &abuf);
    return cv::fitEllipse(points);
}

/* Calculates bounding rectagnle of a point set or retrieves already calculated */
CV_IMPL  CvRect
cvBoundingRect( CvArr* array, int update )
{
    CvRect  rect;
    CvContour contour_header;
    CvSeq* ptseq = 0;
    CvSeqBlock block;

    CvMat stub, *mat = 0;
    int calculate = update;

    if( CV_IS_SEQ( array ))
    {
        ptseq = (CvSeq*)array;
        if( !CV_IS_SEQ_POINT_SET( ptseq ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );

        if( ptseq->header_size < (int)sizeof(CvContour))
        {
            update = 0;
            calculate = 1;
        }
    }
    else
    {
        mat = cvGetMat( array, &stub );
        if( CV_MAT_TYPE(mat->type) == CV_32SC2 ||
            CV_MAT_TYPE(mat->type) == CV_32FC2 )
        {
            ptseq = cvPointSeqFromMat(CV_SEQ_KIND_GENERIC, mat, &contour_header, &block);
            mat = 0;
        }
        else if( CV_MAT_TYPE(mat->type) != CV_8UC1 &&
                CV_MAT_TYPE(mat->type) != CV_8SC1 )
            CV_Error( CV_StsUnsupportedFormat,
                "The image/matrix format is not supported by the function" );
        update = 0;
        calculate = 1;
    }

    if( !calculate )
        return ((CvContour*)ptseq)->rect;

    if( mat )
    {
        rect = cv::maskBoundingRect(cv::cvarrToMat(mat));
    }
    else if( ptseq->total )
    {
        cv::AutoBuffer<double> abuf;
        rect = cv::pointSetBoundingRect(cv::cvarrToMat(ptseq, false, false, 0, &abuf));
    }
    if( update )
        ((CvContour*)ptseq)->rect = rect;
    return rect;
}


/* End of file. */
