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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)
# pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace cv
{

struct FFillSegment
{
    ushort y;
    ushort l;
    ushort r;
    ushort prevl;
    ushort prevr;
    short dir;
};

enum
{
    UP = 1,
    DOWN = -1
};

#define ICV_PUSH( Y, L, R, PREV_L, PREV_R, DIR )  \
{                                                 \
    tail->y = (ushort)(Y);                        \
    tail->l = (ushort)(L);                        \
    tail->r = (ushort)(R);                        \
    tail->prevl = (ushort)(PREV_L);               \
    tail->prevr = (ushort)(PREV_R);               \
    tail->dir = (short)(DIR);                     \
    if( ++tail == buffer_end )                    \
    {                                             \
        buffer->resize(buffer->size() * 3/2);     \
        tail = &buffer->front() + (tail - head);  \
        head = &buffer->front();                  \
        buffer_end = head + buffer->size();       \
    }                                             \
}

#define ICV_POP( Y, L, R, PREV_L, PREV_R, DIR )   \
{                                                 \
    --tail;                                       \
    Y = tail->y;                                  \
    L = tail->l;                                  \
    R = tail->r;                                  \
    PREV_L = tail->prevl;                         \
    PREV_R = tail->prevr;                         \
    DIR = tail->dir;                              \
}

struct ConnectedComp
{
    ConnectedComp();
    Rect rect;
    Point pt;
    int threshold;
    int label;
    int area;
    int harea;
    int carea;
    int perimeter;
    int nholes;
    int ninflections;
    double mx;
    double my;
    Scalar avg;
    Scalar sdv;
};

ConnectedComp::ConnectedComp()
{
    rect = Rect(0, 0, 0, 0);
    pt = Point(-1, -1);
    threshold = -1;
    label = -1;
    area = harea = carea = perimeter = nholes = ninflections = 0;
    mx = my = 0;
    avg = sdv = Scalar::all(0);
}

// Simple Floodfill (repainting single-color connected component)

template<typename _Tp>
static void
floodFill_CnIR( Mat& image, Point seed,
               _Tp newVal, ConnectedComp* region, int flags,
               std::vector<FFillSegment>* buffer )
{
    _Tp* img = (_Tp*)(image.data + image.step * seed.y);
    Size roi = image.size();
    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    FFillSegment* buffer_end = &buffer->front() + buffer->size(), *head = &buffer->front(), *tail = &buffer->front();

    L = R = XMin = XMax = seed.x;

    _Tp val0 = img[L];
    img[L] = newVal;

    while( ++R < roi.width && img[R] == val0 )
        img[R] = newVal;

    while( --L >= 0 && img[L] == val0 )
        img[L] = newVal;

    XMax = --R;
    XMin = ++L;

    ICV_PUSH( seed.y, L, R, R + 1, R, UP );

    while( head != tail )
    {
        int k, YC, PL, PR, dir;
        ICV_POP( YC, L, R, PL, PR, dir );

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        if( region )
        {
            area += R - L + 1;

            if( XMax < R ) XMax = R;
            if( XMin > L ) XMin = L;
            if( YMax < YC ) YMax = YC;
            if( YMin > YC ) YMin = YC;
        }

        for( k = 0; k < 3; k++ )
        {
            dir = data[k][0];
            img = (_Tp*)(image.data + (YC + dir) * image.step);
            int left = data[k][1];
            int right = data[k][2];

            if( (unsigned)(YC + dir) >= (unsigned)roi.height )
                continue;

            for( i = left; i <= right; i++ )
            {
                if( (unsigned)i < (unsigned)roi.width && img[i] == val0 )
                {
                    int j = i;
                    img[i] = newVal;
                    while( --j >= 0 && img[j] == val0 )
                        img[j] = newVal;

                    while( ++i < roi.width && img[i] == val0 )
                        img[i] = newVal;

                    ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                }
            }
        }
    }

    if( region )
    {
        region->pt = seed;
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
    }
}

/****************************************************************************************\
*                                   Gradient Floodfill                                   *
\****************************************************************************************/

struct Diff8uC1
{
    Diff8uC1(uchar _lo, uchar _up) : lo(_lo), interval(_lo + _up) {}
    bool operator()(const uchar* a, const uchar* b) const
    { return (unsigned)(a[0] - b[0] + lo) <= interval; }
    unsigned lo, interval;
};

struct Diff8uC3
{
    Diff8uC3(Vec3b _lo, Vec3b _up)
    {
        for( int k = 0; k < 3; k++ )
            lo[k] = _lo[k], interval[k] = _lo[k] + _up[k];
    }
    bool operator()(const Vec3b* a, const Vec3b* b) const
    {
        return (unsigned)(a[0][0] - b[0][0] + lo[0]) <= interval[0] &&
               (unsigned)(a[0][1] - b[0][1] + lo[1]) <= interval[1] &&
               (unsigned)(a[0][2] - b[0][2] + lo[2]) <= interval[2];
    }
    unsigned lo[3], interval[3];
};

template<typename _Tp>
struct DiffC1
{
    DiffC1(_Tp _lo, _Tp _up) : lo(-_lo), up(_up) {}
    bool operator()(const _Tp* a, const _Tp* b) const
    {
        _Tp d = a[0] - b[0];
        return lo <= d && d <= up;
    }
    _Tp lo, up;
};

template<typename _Tp>
struct DiffC3
{
    DiffC3(_Tp _lo, _Tp _up) : lo(-_lo), up(_up) {}
    bool operator()(const _Tp* a, const _Tp* b) const
    {
        _Tp d = *a - *b;
        return lo[0] <= d[0] && d[0] <= up[0] &&
               lo[1] <= d[1] && d[1] <= up[1] &&
               lo[2] <= d[2] && d[2] <= up[2];
    }
    _Tp lo, up;
};

typedef DiffC1<int> Diff32sC1;
typedef DiffC3<Vec3i> Diff32sC3;
typedef DiffC1<float> Diff32fC1;
typedef DiffC3<Vec3f> Diff32fC3;

template<typename _Tp, typename _MTp, typename _WTp, class Diff>
static void
floodFillGrad_CnIR( Mat& image, Mat& msk,
                   Point seed, _Tp newVal, _MTp newMaskVal,
                   Diff diff, ConnectedComp* region, int flags,
                   std::vector<FFillSegment>* buffer )
{
    int step = (int)image.step, maskStep = (int)msk.step;
    uchar* pImage = image.data;
    _Tp* img = (_Tp*)(pImage + step*seed.y);
    uchar* pMask = msk.data + maskStep + sizeof(_MTp);
    _MTp* mask = (_MTp*)(pMask + maskStep*seed.y);
    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    int fixedRange = flags & FLOODFILL_FIXED_RANGE;
    int fillImage = (flags & FLOODFILL_MASK_ONLY) == 0;
    FFillSegment* buffer_end = &buffer->front() + buffer->size(), *head = &buffer->front(), *tail = &buffer->front();

    L = R = seed.x;
    if( mask[L] )
        return;

    mask[L] = newMaskVal;
    _Tp val0 = img[L];

    if( fixedRange )
    {
        while( !mask[R + 1] && diff( img + (R+1), &val0 ))
            mask[++R] = newMaskVal;

        while( !mask[L - 1] && diff( img + (L-1), &val0 ))
            mask[--L] = newMaskVal;
    }
    else
    {
        while( !mask[R + 1] && diff( img + (R+1), img + R ))
            mask[++R] = newMaskVal;

        while( !mask[L - 1] && diff( img + (L-1), img + L ))
            mask[--L] = newMaskVal;
    }

    XMax = R;
    XMin = L;

    ICV_PUSH( seed.y, L, R, R + 1, R, UP );

    while( head != tail )
    {
        int k, YC, PL, PR, dir;
        ICV_POP( YC, L, R, PL, PR, dir );

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        unsigned length = (unsigned)(R-L);

        if( region )
        {
            area += (int)length + 1;

            if( XMax < R ) XMax = R;
            if( XMin > L ) XMin = L;
            if( YMax < YC ) YMax = YC;
            if( YMin > YC ) YMin = YC;
        }

        for( k = 0; k < 3; k++ )
        {
            dir = data[k][0];
            img = (_Tp*)(pImage + (YC + dir) * step);
            _Tp* img1 = (_Tp*)(pImage + YC * step);
            mask = (_MTp*)(pMask + (YC + dir) * maskStep);
            int left = data[k][1];
            int right = data[k][2];

            if( fixedRange )
                for( i = left; i <= right; i++ )
                {
                    if( !mask[i] && diff( img + i, &val0 ))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while( !mask[--j] && diff( img + j, &val0 ))
                            mask[j] = newMaskVal;

                        while( !mask[++i] && diff( img + i, &val0 ))
                            mask[i] = newMaskVal;

                        ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                    }
                }
            else if( !_8_connectivity )
                for( i = left; i <= right; i++ )
                {
                    if( !mask[i] && diff( img + i, img1 + i ))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while( !mask[--j] && diff( img + j, img + (j+1) ))
                            mask[j] = newMaskVal;

                        while( !mask[++i] &&
                              (diff( img + i, img + (i-1) ) ||
                               (diff( img + i, img1 + i) && i <= R)))
                            mask[i] = newMaskVal;

                        ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                    }
                }
            else
                for( i = left; i <= right; i++ )
                {
                    int idx;
                    _Tp val;

                    if( !mask[i] &&
                       (((val = img[i],
                          (unsigned)(idx = i-L-1) <= length) &&
                         diff( &val, img1 + (i-1))) ||
                        ((unsigned)(++idx) <= length &&
                         diff( &val, img1 + i )) ||
                        ((unsigned)(++idx) <= length &&
                         diff( &val, img1 + (i+1) ))))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while( !mask[--j] && diff( img + j, img + (j+1) ))
                            mask[j] = newMaskVal;

                        while( !mask[++i] &&
                              ((val = img[i],
                                diff( &val, img + (i-1) )) ||
                               (((unsigned)(idx = i-L-1) <= length &&
                                 diff( &val, img1 + (i-1) ))) ||
                               ((unsigned)(++idx) <= length &&
                                diff( &val, img1 + i )) ||
                               ((unsigned)(++idx) <= length &&
                                diff( &val, img1 + (i+1) ))))
                            mask[i] = newMaskVal;

                        ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                    }
                }
        }

        img = (_Tp*)(pImage + YC * step);
        if( fillImage )
            for( i = L; i <= R; i++ )
                img[i] = newVal;
        /*else if( region )
         for( i = L; i <= R; i++ )
         sum += img[i];*/
    }

    if( region )
    {
        region->pt = seed;
        region->label = saturate_cast<int>(newMaskVal);
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
    }
}

}

/****************************************************************************************\
*                                    External Functions                                  *
\****************************************************************************************/

int cv::floodFill( InputOutputArray _image, InputOutputArray _mask,
                  Point seedPoint, Scalar newVal, Rect* rect,
                  Scalar loDiff, Scalar upDiff, int flags )
{
    ConnectedComp comp;
    std::vector<FFillSegment> buffer;

    if( rect )
        *rect = Rect();

    int i, connectivity = flags & 255;
    union {
        uchar b[4];
        int i[4];
        float f[4];
        double _[4];
    } nv_buf;
    nv_buf._[0] = nv_buf._[1] = nv_buf._[2] = nv_buf._[3] = 0;

    struct { Vec3b b; Vec3i i; Vec3f f; } ld_buf, ud_buf;
    Mat img = _image.getMat(), mask;
    if( !_mask.empty() )
        mask = _mask.getMat();
    Size size = img.size();

    int type = img.type();
    int depth = img.depth();
    int cn = img.channels();

    if( connectivity == 0 )
        connectivity = 4;
    else if( connectivity != 4 && connectivity != 8 )
        CV_Error( CV_StsBadFlag, "Connectivity must be 4, 0(=4) or 8" );

    bool is_simple = mask.empty() && (flags & FLOODFILL_MASK_ONLY) == 0;

    for( i = 0; i < cn; i++ )
    {
        if( loDiff[i] < 0 || upDiff[i] < 0 )
            CV_Error( CV_StsBadArg, "lo_diff and up_diff must be non-negative" );
        is_simple = is_simple && fabs(loDiff[i]) < DBL_EPSILON && fabs(upDiff[i]) < DBL_EPSILON;
    }

    if( (unsigned)seedPoint.x >= (unsigned)size.width ||
       (unsigned)seedPoint.y >= (unsigned)size.height )
        CV_Error( CV_StsOutOfRange, "Seed point is outside of image" );

    scalarToRawData( newVal, &nv_buf, type, 0);
    size_t buffer_size = MAX( size.width, size.height ) * 2;
    buffer.resize( buffer_size );

    if( is_simple )
    {
        size_t elem_size = img.elemSize();
        const uchar* seed_ptr = img.data + img.step*seedPoint.y + elem_size*seedPoint.x;

        size_t k = 0;
        for(; k < elem_size; k++)
            if (seed_ptr[k] != nv_buf.b[k])
                break;

        if( k != elem_size )
        {
            if( type == CV_8UC1 )
                floodFill_CnIR(img, seedPoint, nv_buf.b[0], &comp, flags, &buffer);
            else if( type == CV_8UC3 )
                floodFill_CnIR(img, seedPoint, Vec3b(nv_buf.b), &comp, flags, &buffer);
            else if( type == CV_32SC1 )
                floodFill_CnIR(img, seedPoint, nv_buf.i[0], &comp, flags, &buffer);
            else if( type == CV_32FC1 )
                floodFill_CnIR(img, seedPoint, nv_buf.f[0], &comp, flags, &buffer);
            else if( type == CV_32SC3 )
                floodFill_CnIR(img, seedPoint, Vec3i(nv_buf.i), &comp, flags, &buffer);
            else if( type == CV_32FC3 )
                floodFill_CnIR(img, seedPoint, Vec3f(nv_buf.f), &comp, flags, &buffer);
            else
                CV_Error( CV_StsUnsupportedFormat, "" );
            if( rect )
                *rect = comp.rect;
            return comp.area;
        }
    }

    if( mask.empty() )
    {
        Mat tempMask( size.height + 2, size.width + 2, CV_8UC1 );
        tempMask.setTo(Scalar::all(0));
        mask = tempMask;
    }
    else
    {
        CV_Assert( mask.rows == size.height+2 && mask.cols == size.width+2 );
        CV_Assert( mask.type() == CV_8U );
    }

    memset( mask.data, 1, mask.cols );
    memset( mask.data + mask.step*(mask.rows-1), 1, mask.cols );

    for( i = 1; i <= size.height; i++ )
    {
        mask.at<uchar>(i, 0) = mask.at<uchar>(i, mask.cols-1) = (uchar)1;
    }

    if( depth == CV_8U )
        for( i = 0; i < cn; i++ )
        {
            ld_buf.b[i] = saturate_cast<uchar>(cvFloor(loDiff[i]));
            ud_buf.b[i] = saturate_cast<uchar>(cvFloor(upDiff[i]));
        }
    else if( depth == CV_32S )
        for( i = 0; i < cn; i++ )
        {
            ld_buf.i[i] = cvFloor(loDiff[i]);
            ud_buf.i[i] = cvFloor(upDiff[i]);
        }
    else if( depth == CV_32F )
        for( i = 0; i < cn; i++ )
        {
            ld_buf.f[i] = (float)loDiff[i];
            ud_buf.f[i] = (float)upDiff[i];
        }
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    uchar newMaskVal = (uchar)((flags & ~0xff) == 0 ? 1 : ((flags >> 8) & 255));

    if( type == CV_8UC1 )
        floodFillGrad_CnIR<uchar, uchar, int, Diff8uC1>(
                img, mask, seedPoint, nv_buf.b[0], newMaskVal,
                Diff8uC1(ld_buf.b[0], ud_buf.b[0]),
                &comp, flags, &buffer);
    else if( type == CV_8UC3 )
        floodFillGrad_CnIR<Vec3b, uchar, Vec3i, Diff8uC3>(
                img, mask, seedPoint, Vec3b(nv_buf.b), newMaskVal,
                Diff8uC3(ld_buf.b, ud_buf.b),
                &comp, flags, &buffer);
    else if( type == CV_32SC1 )
        floodFillGrad_CnIR<int, uchar, int, Diff32sC1>(
                img, mask, seedPoint, nv_buf.i[0], newMaskVal,
                Diff32sC1(ld_buf.i[0], ud_buf.i[0]),
                &comp, flags, &buffer);
    else if( type == CV_32SC3 )
        floodFillGrad_CnIR<Vec3i, uchar, Vec3i, Diff32sC3>(
                img, mask, seedPoint, Vec3i(nv_buf.i), newMaskVal,
                Diff32sC3(ld_buf.i, ud_buf.i),
                &comp, flags, &buffer);
    else if( type == CV_32FC1 )
        floodFillGrad_CnIR<float, uchar, float, Diff32fC1>(
                img, mask, seedPoint, nv_buf.f[0], newMaskVal,
                Diff32fC1(ld_buf.f[0], ud_buf.f[0]),
                &comp, flags, &buffer);
    else if( type == CV_32FC3 )
        floodFillGrad_CnIR<Vec3f, uchar, Vec3f, Diff32fC3>(
                img, mask, seedPoint, Vec3f(nv_buf.f), newMaskVal,
                Diff32fC3(ld_buf.f, ud_buf.f),
                &comp, flags, &buffer);
    else
        CV_Error(CV_StsUnsupportedFormat, "");

    if( rect )
        *rect = comp.rect;
    return comp.area;
}


int cv::floodFill( InputOutputArray _image, Point seedPoint,
                  Scalar newVal, Rect* rect,
                  Scalar loDiff, Scalar upDiff, int flags )
{
    return floodFill(_image, Mat(), seedPoint, newVal, rect, loDiff, upDiff, flags);
}


CV_IMPL void
cvFloodFill( CvArr* arr, CvPoint seed_point,
             CvScalar newVal, CvScalar lo_diff, CvScalar up_diff,
             CvConnectedComp* comp, int flags, CvArr* maskarr )
{
    if( comp )
        memset( comp, 0, sizeof(*comp) );

    cv::Mat img = cv::cvarrToMat(arr), mask = cv::cvarrToMat(maskarr);
    int area = cv::floodFill(img, mask, seed_point, newVal,
                             comp ? (cv::Rect*)&comp->rect : 0,
                             lo_diff, up_diff, flags );
    if( comp )
    {
        comp->area = area;
        comp->value = newVal;
    }
}

/* End of file. */
