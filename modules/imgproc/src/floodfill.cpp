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

typedef struct CvFFillSegment
{
    ushort y;
    ushort l;
    ushort r;
    ushort prevl;
    ushort prevr;
    short dir;
}
CvFFillSegment;

#define UP 1
#define DOWN -1

#define ICV_PUSH( Y, L, R, PREV_L, PREV_R, DIR )\
{                                               \
    tail->y = (ushort)(Y);                      \
    tail->l = (ushort)(L);                      \
    tail->r = (ushort)(R);                      \
    tail->prevl = (ushort)(PREV_L);             \
    tail->prevr = (ushort)(PREV_R);             \
    tail->dir = (short)(DIR);                   \
    if( ++tail >= buffer_end )                  \
        tail = buffer;                          \
}


#define ICV_POP( Y, L, R, PREV_L, PREV_R, DIR ) \
{                                               \
    Y = head->y;                                \
    L = head->l;                                \
    R = head->r;                                \
    PREV_L = head->prevl;                       \
    PREV_R = head->prevr;                       \
    DIR = head->dir;                            \
    if( ++head >= buffer_end )                  \
        head = buffer;                          \
}

/****************************************************************************************\
*              Simple Floodfill (repainting single-color connected component)            *
\****************************************************************************************/

template<typename _Tp>
static void
icvFloodFill_CnIR( uchar* pImage, int step, CvSize roi, CvPoint seed,
                   _Tp newVal, CvConnectedComp* region, int flags,
                   CvFFillSegment* buffer, int buffer_size )
{
    typedef typename cv::DataType<_Tp>::channel_type _CTp;
    _Tp* img = (_Tp*)(pImage + step * seed.y);
    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    CvFFillSegment* buffer_end = buffer + buffer_size, *head = buffer, *tail = buffer;

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
            img = (_Tp*)(pImage + (YC + dir) * step);
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
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
        region->value = cv::Scalar(newVal);
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
    Diff8uC3(cv::Vec3b _lo, cv::Vec3b _up)
    {
        for( int k = 0; k < 3; k++ )
            lo[k] = _lo[k], interval[k] = _lo[k] + _up[k];
    }
    bool operator()(const cv::Vec3b* a, const cv::Vec3b* b) const
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
typedef DiffC3<cv::Vec3i> Diff32sC3;
typedef DiffC1<float> Diff32fC1;
typedef DiffC3<cv::Vec3f> Diff32fC3;

cv::Vec3i& operator += (cv::Vec3i& a, const cv::Vec3b& b)
{
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
    return a;
}

template<typename _Tp, typename _WTp, class Diff>
static void
icvFloodFillGrad_CnIR( uchar* pImage, int step, uchar* pMask, int maskStep,
                       CvSize /*roi*/, CvPoint seed, _Tp newVal, Diff diff,
                       CvConnectedComp* region, int flags,
                       CvFFillSegment* buffer, int buffer_size )
{
    typedef typename cv::DataType<_Tp>::channel_type _CTp;
    _Tp* img = (_Tp*)(pImage + step*seed.y);
    uchar* mask = (pMask += maskStep + 1) + maskStep*seed.y;
    int i, L, R;
    int area = 0;
    _WTp sum = _WTp((typename cv::DataType<_Tp>::channel_type)0);
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    int fixedRange = flags & CV_FLOODFILL_FIXED_RANGE;
    int fillImage = (flags & CV_FLOODFILL_MASK_ONLY) == 0;
    uchar newMaskVal = (uchar)(flags & 0xff00 ? flags >> 8 : 1);
    CvFFillSegment* buffer_end = buffer + buffer_size, *head = buffer, *tail = buffer;

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
            mask = pMask + (YC + dir) * maskStep;
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
        else if( region )
            for( i = L; i <= R; i++ )
                sum += img[i];
    }

    if( region )
    {
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;

        if( fillImage )
            region->value = cv::Scalar(newVal);
        else
        {
            double iarea = area ? 1./area : 0;
            region->value = cv::Scalar(sum*iarea);
        }
    }
}


/****************************************************************************************\
*                                    External Functions                                  *
\****************************************************************************************/

typedef  void (*CvFloodFillFunc)(
               void* img, int step, CvSize size, CvPoint seed, void* newval,
               CvConnectedComp* comp, int flags, void* buffer, int buffer_size, int cn );

typedef  void (*CvFloodFillGradFunc)(
               void* img, int step, uchar* mask, int maskStep, CvSize size,
               CvPoint seed, void* newval, void* d_lw, void* d_up, void* ccomp,
               int flags, void* buffer, int buffer_size, int cn );

CV_IMPL void
cvFloodFill( CvArr* arr, CvPoint seed_point,
             CvScalar newVal, CvScalar lo_diff, CvScalar up_diff,
             CvConnectedComp* comp, int flags, CvArr* maskarr )
{
    cv::Ptr<CvMat> tempMask;
    cv::AutoBuffer<CvFFillSegment> buffer;
    
    if( comp )
        memset( comp, 0, sizeof(*comp) );

    int i, type, depth, cn, is_simple;
    int buffer_size, connectivity = flags & 255;
    union {
        uchar b[4];
        int i[4];
        float f[4];
        double _[4];
    } nv_buf;
    nv_buf._[0] = nv_buf._[1] = nv_buf._[2] = nv_buf._[3] = 0;

    struct { cv::Vec3b b; cv::Vec3i i; cv::Vec3f f; } ld_buf, ud_buf;
    CvMat stub, *img = cvGetMat(arr, &stub);
    CvMat maskstub, *mask = (CvMat*)maskarr;
    CvSize size;

    type = CV_MAT_TYPE( img->type );
    depth = CV_MAT_DEPTH(type);
    cn = CV_MAT_CN(type);

    if( connectivity == 0 )
        connectivity = 4;
    else if( connectivity != 4 && connectivity != 8 )
        CV_Error( CV_StsBadFlag, "Connectivity must be 4, 0(=4) or 8" );

    is_simple = mask == 0 && (flags & CV_FLOODFILL_MASK_ONLY) == 0;

    for( i = 0; i < cn; i++ )
    {
        if( lo_diff.val[i] < 0 || up_diff.val[i] < 0 )
            CV_Error( CV_StsBadArg, "lo_diff and up_diff must be non-negative" );
        is_simple &= fabs(lo_diff.val[i]) < DBL_EPSILON && fabs(up_diff.val[i]) < DBL_EPSILON;
    }

    size = cvGetMatSize( img );

    if( (unsigned)seed_point.x >= (unsigned)size.width ||
        (unsigned)seed_point.y >= (unsigned)size.height )
        CV_Error( CV_StsOutOfRange, "Seed point is outside of image" );

    cvScalarToRawData( &newVal, &nv_buf, type, 0 );
    buffer_size = MAX( size.width, size.height )*2;
    buffer.allocate( buffer_size );

    if( is_simple )
    {
        /*int elem_size = CV_ELEM_SIZE(type);
        const uchar* seed_ptr = img->data.ptr + img->step*seed_point.y + elem_size*seed_point.x;
        
        // check if the new value is different from the current value at the seed point.
        // if they are exactly the same, use the generic version with mask to avoid infinite loops.
        for( i = 0; i < elem_size; i++ )
            if( seed_ptr[i] != ((uchar*)nv_buf)[i] )
                break;
        
        if( i == elem_size )
            return;*/
        
        if( type == CV_8UC1 )
            icvFloodFill_CnIR(img->data.ptr, img->step, size, seed_point, nv_buf.b[0],
                              comp, flags, buffer, buffer_size);
        else if( type == CV_8UC3 )
            icvFloodFill_CnIR(img->data.ptr, img->step, size, seed_point, cv::Vec3b(nv_buf.b),
                              comp, flags, buffer, buffer_size);
        else if( type == CV_32SC1 )
            icvFloodFill_CnIR(img->data.ptr, img->step, size, seed_point, nv_buf.i[0],
                              comp, flags, buffer, buffer_size);
        else if( type == CV_32FC1 )
            icvFloodFill_CnIR(img->data.ptr, img->step, size, seed_point, nv_buf.f[0],
                              comp, flags, buffer, buffer_size);
        else if( type == CV_32SC3 )
            icvFloodFill_CnIR(img->data.ptr, img->step, size, seed_point, cv::Vec3i(nv_buf.i),
                              comp, flags, buffer, buffer_size);
        else if( type == CV_32FC3 )
            icvFloodFill_CnIR(img->data.ptr, img->step, size, seed_point, cv::Vec3f(nv_buf.f),
                              comp, flags, buffer, buffer_size);
        else
            CV_Error( CV_StsUnsupportedFormat, "" );
        return;
    }

    if( !mask )
    {
        /* created mask will be 8-byte aligned */
        tempMask = cvCreateMat( size.height + 2, (size.width + 9) & -8, CV_8UC1 );
        mask = tempMask;
    }
    else
    {
        mask = cvGetMat( mask, &maskstub );
        if( !CV_IS_MASK_ARR( mask ))
            CV_Error( CV_StsBadMask, "" );

        if( mask->width != size.width + 2 || mask->height != size.height + 2 )
            CV_Error( CV_StsUnmatchedSizes, "mask must be 2 pixel wider "
                                   "and 2 pixel taller than filled image" );
    }

    int width = tempMask ? mask->step : size.width + 2;
    uchar* mask_row = mask->data.ptr + mask->step;
    memset( mask_row - mask->step, 1, width );

    for( i = 1; i <= size.height; i++, mask_row += mask->step )
    {
        if( tempMask )
            memset( mask_row, 0, width );
        mask_row[0] = mask_row[size.width+1] = (uchar)1;
    }
    memset( mask_row, 1, width );

    if( depth == CV_8U )
        for( i = 0; i < cn; i++ )
        {
            int t = cvFloor(lo_diff.val[i]);
            ld_buf.b[i] = CV_CAST_8U(t);
            t = cvFloor(up_diff.val[i]);
            ud_buf.b[i] = CV_CAST_8U(t);
        }
    else if( depth == CV_32S )
        for( i = 0; i < cn; i++ )
        {
            int t = cvFloor(lo_diff.val[i]);
            ld_buf.i[i] = t;
            t = cvFloor(up_diff.val[i]);
            ud_buf.i[i] = t;
        }
    else if( depth == CV_32F )
        for( i = 0; i < cn; i++ )
        {
            ld_buf.f[i] = (float)lo_diff.val[i];
            ud_buf.f[i] = (float)up_diff.val[i];
        }
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    if( type == CV_8UC1 )
        icvFloodFillGrad_CnIR<uchar, int, Diff8uC1>(
                              img->data.ptr, img->step, mask->data.ptr, mask->step,
                              size, seed_point, nv_buf.b[0],
                              Diff8uC1(ld_buf.b[0], ud_buf.b[0]),
                              comp, flags, buffer, buffer_size);
    else if( type == CV_8UC3 )
        icvFloodFillGrad_CnIR<cv::Vec3b, cv::Vec3i, Diff8uC3>(
                              img->data.ptr, img->step, mask->data.ptr, mask->step,
                              size, seed_point, cv::Vec3b(nv_buf.b),
                              Diff8uC3(ld_buf.b, ud_buf.b),
                              comp, flags, buffer, buffer_size);
    else if( type == CV_32SC1 )
        icvFloodFillGrad_CnIR<int, int, Diff32sC1>(
                              img->data.ptr, img->step, mask->data.ptr, mask->step,
                              size, seed_point, nv_buf.i[0],
                              Diff32sC1(ld_buf.i[0], ud_buf.i[0]),
                              comp, flags, buffer, buffer_size);
    else if( type == CV_32SC3 )
        icvFloodFillGrad_CnIR<cv::Vec3i, cv::Vec3i, Diff32sC3>(
                              img->data.ptr, img->step, mask->data.ptr, mask->step,
                              size, seed_point, cv::Vec3i(nv_buf.i),
                              Diff32sC3(ld_buf.i, ud_buf.i),
                              comp, flags, buffer, buffer_size);
    else if( type == CV_32FC1 )
        icvFloodFillGrad_CnIR<float, float, Diff32fC1>(
                              img->data.ptr, img->step, mask->data.ptr, mask->step,
                              size, seed_point, nv_buf.f[0],
                              Diff32fC1(ld_buf.f[0], ud_buf.f[0]),
                              comp, flags, buffer, buffer_size);
    else if( type == CV_32FC3 )
        icvFloodFillGrad_CnIR<cv::Vec3f, cv::Vec3f, Diff32fC3>(
                              img->data.ptr, img->step, mask->data.ptr, mask->step,
                              size, seed_point, cv::Vec3f(nv_buf.f),
                              Diff32fC3(ld_buf.f, ud_buf.f),
                              comp, flags, buffer, buffer_size);
    else
        CV_Error(CV_StsUnsupportedFormat, "");
}


int cv::floodFill( InputOutputArray _image, Point seedPoint,
                   Scalar newVal, Rect* rect,
                   Scalar loDiff, Scalar upDiff, int flags )
{
    CvConnectedComp ccomp;
    CvMat c_image = _image.getMat();
    cvFloodFill(&c_image, seedPoint, newVal, loDiff, upDiff, &ccomp, flags, 0);
    if( rect )
        *rect = ccomp.rect;
    return cvRound(ccomp.area);
}

int cv::floodFill( InputOutputArray _image, InputOutputArray _mask,
                   Point seedPoint, Scalar newVal, Rect* rect, 
                   Scalar loDiff, Scalar upDiff, int flags )
{
    CvConnectedComp ccomp;
    CvMat c_image = _image.getMat(), c_mask = _mask.getMat();
    cvFloodFill(&c_image, seedPoint, newVal, loDiff, upDiff, &ccomp, flags, c_mask.data.ptr ? &c_mask : 0);
    if( rect )
        *rect = ccomp.rect;
    return cvRound(ccomp.area);
}

/* End of file. */
