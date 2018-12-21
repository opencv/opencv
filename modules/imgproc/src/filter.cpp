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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include "opencv2/core/opencl/ocl_defs.hpp"
#include "opencl_kernels_imgproc.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "filter.hpp"


/****************************************************************************************\
                                    Base Image Filter
\****************************************************************************************/

#if IPP_VERSION_X100 >= 710
#define USE_IPP_SEP_FILTERS 1
#else
#undef USE_IPP_SEP_FILTERS
#endif

namespace cv
{

BaseRowFilter::BaseRowFilter() { ksize = anchor = -1; }
BaseRowFilter::~BaseRowFilter() {}

BaseColumnFilter::BaseColumnFilter() { ksize = anchor = -1; }
BaseColumnFilter::~BaseColumnFilter() {}
void BaseColumnFilter::reset() {}

BaseFilter::BaseFilter() { ksize = Size(-1,-1); anchor = Point(-1,-1); }
BaseFilter::~BaseFilter() {}
void BaseFilter::reset() {}

FilterEngine::FilterEngine()
    : srcType(-1), dstType(-1), bufType(-1), maxWidth(0), wholeSize(-1, -1), dx1(0), dx2(0),
      rowBorderType(BORDER_REPLICATE), columnBorderType(BORDER_REPLICATE),
      borderElemSize(0), bufStep(0), startY(0), startY0(0), endY(0), rowCount(0), dstY(0)
{
}


FilterEngine::FilterEngine( const Ptr<BaseFilter>& _filter2D,
                            const Ptr<BaseRowFilter>& _rowFilter,
                            const Ptr<BaseColumnFilter>& _columnFilter,
                            int _srcType, int _dstType, int _bufType,
                            int _rowBorderType, int _columnBorderType,
                            const Scalar& _borderValue )
    : srcType(-1), dstType(-1), bufType(-1), maxWidth(0), wholeSize(-1, -1), dx1(0), dx2(0),
      rowBorderType(BORDER_REPLICATE), columnBorderType(BORDER_REPLICATE),
      borderElemSize(0), bufStep(0), startY(0), startY0(0), endY(0), rowCount(0), dstY(0)
{
    init(_filter2D, _rowFilter, _columnFilter, _srcType, _dstType, _bufType,
         _rowBorderType, _columnBorderType, _borderValue);
}

FilterEngine::~FilterEngine()
{
}


void FilterEngine::init( const Ptr<BaseFilter>& _filter2D,
                         const Ptr<BaseRowFilter>& _rowFilter,
                         const Ptr<BaseColumnFilter>& _columnFilter,
                         int _srcType, int _dstType, int _bufType,
                         int _rowBorderType, int _columnBorderType,
                         const Scalar& _borderValue )
{
    _srcType = CV_MAT_TYPE(_srcType);
    _bufType = CV_MAT_TYPE(_bufType);
    _dstType = CV_MAT_TYPE(_dstType);

    srcType = _srcType;
    int srcElemSize = (int)getElemSize(srcType);
    dstType = _dstType;
    bufType = _bufType;

    filter2D = _filter2D;
    rowFilter = _rowFilter;
    columnFilter = _columnFilter;

    if( _columnBorderType < 0 )
        _columnBorderType = _rowBorderType;

    rowBorderType = _rowBorderType;
    columnBorderType = _columnBorderType;

    CV_Assert( columnBorderType != BORDER_WRAP );

    if( isSeparable() )
    {
        CV_Assert( rowFilter && columnFilter );
        ksize = Size(rowFilter->ksize, columnFilter->ksize);
        anchor = Point(rowFilter->anchor, columnFilter->anchor);
    }
    else
    {
        CV_Assert( bufType == srcType );
        ksize = filter2D->ksize;
        anchor = filter2D->anchor;
    }

    CV_Assert( 0 <= anchor.x && anchor.x < ksize.width &&
               0 <= anchor.y && anchor.y < ksize.height );

    borderElemSize = srcElemSize/(CV_MAT_DEPTH(srcType) >= CV_32S ? sizeof(int) : 1);
    int borderLength = std::max(ksize.width - 1, 1);
    borderTab.resize(borderLength*borderElemSize);

    maxWidth = bufStep = 0;
    constBorderRow.clear();

    if( rowBorderType == BORDER_CONSTANT || columnBorderType == BORDER_CONSTANT )
    {
        constBorderValue.resize(srcElemSize*borderLength);
        int srcType1 = CV_MAKETYPE(CV_MAT_DEPTH(srcType), MIN(CV_MAT_CN(srcType), 4));
        scalarToRawData(_borderValue, &constBorderValue[0], srcType1,
                        borderLength*CV_MAT_CN(srcType));
    }

    wholeSize = Size(-1,-1);
}

#define VEC_ALIGN CV_MALLOC_ALIGN

int FilterEngine::start(const Size &_wholeSize, const Size &sz, const Point &ofs)
{
    int i, j;

    wholeSize = _wholeSize;
    roi = Rect(ofs, sz);
    CV_Assert( roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 &&
        roi.x + roi.width <= wholeSize.width &&
        roi.y + roi.height <= wholeSize.height );

    int esz = (int)getElemSize(srcType);
    int bufElemSize = (int)getElemSize(bufType);
    const uchar* constVal = !constBorderValue.empty() ? &constBorderValue[0] : 0;

    int _maxBufRows = std::max(ksize.height + 3,
                               std::max(anchor.y,
                                        ksize.height-anchor.y-1)*2+1);

    if( maxWidth < roi.width || _maxBufRows != (int)rows.size() )
    {
        rows.resize(_maxBufRows);
        maxWidth = std::max(maxWidth, roi.width);
        int cn = CV_MAT_CN(srcType);
        srcRow.resize(esz*(maxWidth + ksize.width - 1));
        if( columnBorderType == BORDER_CONSTANT )
        {
            CV_Assert(constVal != NULL);
            constBorderRow.resize(getElemSize(bufType)*(maxWidth + ksize.width - 1 + VEC_ALIGN));
            uchar *dst = alignPtr(&constBorderRow[0], VEC_ALIGN), *tdst;
            int n = (int)constBorderValue.size(), N;
            N = (maxWidth + ksize.width - 1)*esz;
            tdst = isSeparable() ? &srcRow[0] : dst;

            for( i = 0; i < N; i += n )
            {
                n = std::min( n, N - i );
                for(j = 0; j < n; j++)
                    tdst[i+j] = constVal[j];
            }

            if( isSeparable() )
                (*rowFilter)(&srcRow[0], dst, maxWidth, cn);
        }

        int maxBufStep = bufElemSize*(int)alignSize(maxWidth +
            (!isSeparable() ? ksize.width - 1 : 0),VEC_ALIGN);
        ringBuf.resize(maxBufStep*rows.size()+VEC_ALIGN);
    }

    // adjust bufstep so that the used part of the ring buffer stays compact in memory
    bufStep = bufElemSize*(int)alignSize(roi.width + (!isSeparable() ? ksize.width - 1 : 0),16);

    dx1 = std::max(anchor.x - roi.x, 0);
    dx2 = std::max(ksize.width - anchor.x - 1 + roi.x + roi.width - wholeSize.width, 0);

    // recompute border tables
    if( dx1 > 0 || dx2 > 0 )
    {
        if( rowBorderType == BORDER_CONSTANT )
        {
            CV_Assert(constVal != NULL);
            int nr = isSeparable() ? 1 : (int)rows.size();
            for( i = 0; i < nr; i++ )
            {
                uchar* dst = isSeparable() ? &srcRow[0] : alignPtr(&ringBuf[0],VEC_ALIGN) + bufStep*i;
                memcpy( dst, constVal, dx1*esz );
                memcpy( dst + (roi.width + ksize.width - 1 - dx2)*esz, constVal, dx2*esz );
            }
        }
        else
        {
            int xofs1 = std::min(roi.x, anchor.x) - roi.x;

            int btab_esz = borderElemSize, wholeWidth = wholeSize.width;
            int* btab = (int*)&borderTab[0];

            for( i = 0; i < dx1; i++ )
            {
                int p0 = (borderInterpolate(i-dx1, wholeWidth, rowBorderType) + xofs1)*btab_esz;
                for( j = 0; j < btab_esz; j++ )
                    btab[i*btab_esz + j] = p0 + j;
            }

            for( i = 0; i < dx2; i++ )
            {
                int p0 = (borderInterpolate(wholeWidth + i, wholeWidth, rowBorderType) + xofs1)*btab_esz;
                for( j = 0; j < btab_esz; j++ )
                    btab[(i + dx1)*btab_esz + j] = p0 + j;
            }
        }
    }

    rowCount = dstY = 0;
    startY = startY0 = std::max(roi.y - anchor.y, 0);
    endY = std::min(roi.y + roi.height + ksize.height - anchor.y - 1, wholeSize.height);
    if( columnFilter )
        columnFilter->reset();
    if( filter2D )
        filter2D->reset();

    return startY;
}


int FilterEngine::start(const Mat& src, const Size &wsz, const Point &ofs)
{
    start( wsz, src.size(), ofs);
    return startY - ofs.y;
}

int FilterEngine::remainingInputRows() const
{
    return endY - startY - rowCount;
}

int FilterEngine::remainingOutputRows() const
{
    return roi.height - dstY;
}

int FilterEngine::proceed( const uchar* src, int srcstep, int count,
                           uchar* dst, int dststep )
{
    CV_Assert( wholeSize.width > 0 && wholeSize.height > 0 );

    const int *btab = &borderTab[0];
    int esz = (int)getElemSize(srcType), btab_esz = borderElemSize;
    uchar** brows = &rows[0];
    int bufRows = (int)rows.size();
    int cn = CV_MAT_CN(bufType);
    int width = roi.width, kwidth = ksize.width;
    int kheight = ksize.height, ay = anchor.y;
    int _dx1 = dx1, _dx2 = dx2;
    int width1 = roi.width + kwidth - 1;
    int xofs1 = std::min(roi.x, anchor.x);
    bool isSep = isSeparable();
    bool makeBorder = (_dx1 > 0 || _dx2 > 0) && rowBorderType != BORDER_CONSTANT;
    int dy = 0, i = 0;

    src -= xofs1*esz;
    count = std::min(count, remainingInputRows());

    CV_Assert( src && dst && count > 0 );

    for(;; dst += dststep*i, dy += i)
    {
        int dcount = bufRows - ay - startY - rowCount + roi.y;
        dcount = dcount > 0 ? dcount : bufRows - kheight + 1;
        dcount = std::min(dcount, count);
        count -= dcount;
        for( ; dcount-- > 0; src += srcstep )
        {
            int bi = (startY - startY0 + rowCount) % bufRows;
            uchar* brow = alignPtr(&ringBuf[0], VEC_ALIGN) + bi*bufStep;
            uchar* row = isSep ? &srcRow[0] : brow;

            if( ++rowCount > bufRows )
            {
                --rowCount;
                ++startY;
            }

            memcpy( row + _dx1*esz, src, (width1 - _dx2 - _dx1)*esz );

            if( makeBorder )
            {
                if( btab_esz*(int)sizeof(int) == esz )
                {
                    const int* isrc = (const int*)src;
                    int* irow = (int*)row;

                    for( i = 0; i < _dx1*btab_esz; i++ )
                        irow[i] = isrc[btab[i]];
                    for( i = 0; i < _dx2*btab_esz; i++ )
                        irow[i + (width1 - _dx2)*btab_esz] = isrc[btab[i+_dx1*btab_esz]];
                }
                else
                {
                    for( i = 0; i < _dx1*esz; i++ )
                        row[i] = src[btab[i]];
                    for( i = 0; i < _dx2*esz; i++ )
                        row[i + (width1 - _dx2)*esz] = src[btab[i+_dx1*esz]];
                }
            }

            if( isSep )
                (*rowFilter)(row, brow, width, CV_MAT_CN(srcType));
        }

        int max_i = std::min(bufRows, roi.height - (dstY + dy) + (kheight - 1));
        for( i = 0; i < max_i; i++ )
        {
            int srcY = borderInterpolate(dstY + dy + i + roi.y - ay,
                            wholeSize.height, columnBorderType);
            if( srcY < 0 ) // can happen only with constant border type
                brows[i] = alignPtr(&constBorderRow[0], VEC_ALIGN);
            else
            {
                CV_Assert( srcY >= startY );
                if( srcY >= startY + rowCount )
                    break;
                int bi = (srcY - startY0) % bufRows;
                brows[i] = alignPtr(&ringBuf[0], VEC_ALIGN) + bi*bufStep;
            }
        }
        if( i < kheight )
            break;
        i -= kheight - 1;
        if( isSeparable() )
            (*columnFilter)((const uchar**)brows, dst, dststep, i, roi.width*cn);
        else
            (*filter2D)((const uchar**)brows, dst, dststep, i, roi.width, cn);
    }

    dstY += dy;
    CV_Assert( dstY <= roi.height );
    return dy;
}

void FilterEngine::apply(const Mat& src, Mat& dst, const Size & wsz, const Point & ofs)
{
    CV_INSTRUMENT_REGION();

    CV_Assert( src.type() == srcType && dst.type() == dstType );

    int y = start(src, wsz, ofs);
    proceed(src.ptr() + y*src.step,
            (int)src.step,
            endY - startY,
            dst.ptr(),
            (int)dst.step );
}

}

/****************************************************************************************\
*                                 Separable linear filter                                *
\****************************************************************************************/

int cv::getKernelType(InputArray filter_kernel, Point anchor)
{
    Mat _kernel = filter_kernel.getMat();
    CV_Assert( _kernel.channels() == 1 );
    int i, sz = _kernel.rows*_kernel.cols;

    Mat kernel;
    _kernel.convertTo(kernel, CV_64F);

    const double* coeffs = kernel.ptr<double>();
    double sum = 0;
    int type = KERNEL_SMOOTH + KERNEL_INTEGER;
    if( (_kernel.rows == 1 || _kernel.cols == 1) &&
        anchor.x*2 + 1 == _kernel.cols &&
        anchor.y*2 + 1 == _kernel.rows )
        type |= (KERNEL_SYMMETRICAL + KERNEL_ASYMMETRICAL);

    for( i = 0; i < sz; i++ )
    {
        double a = coeffs[i], b = coeffs[sz - i - 1];
        if( a != b )
            type &= ~KERNEL_SYMMETRICAL;
        if( a != -b )
            type &= ~KERNEL_ASYMMETRICAL;
        if( a < 0 )
            type &= ~KERNEL_SMOOTH;
        if( a != saturate_cast<int>(a) )
            type &= ~KERNEL_INTEGER;
        sum += a;
    }

    if( fabs(sum - 1) > FLT_EPSILON*(fabs(sum) + 1) )
        type &= ~KERNEL_SMOOTH;
    return type;
}


namespace cv
{

struct RowNoVec
{
    RowNoVec() {}
    RowNoVec(const Mat&) {}
    int operator()(const uchar*, uchar*, int, int) const { return 0; }
};

struct ColumnNoVec
{
    ColumnNoVec() {}
    ColumnNoVec(const Mat&, int, int, double) {}
    int operator()(const uchar**, uchar*, int) const { return 0; }
};

struct SymmRowSmallNoVec
{
    SymmRowSmallNoVec() {}
    SymmRowSmallNoVec(const Mat&, int) {}
    int operator()(const uchar*, uchar*, int, int) const { return 0; }
};

struct SymmColumnSmallNoVec
{
    SymmColumnSmallNoVec() {}
    SymmColumnSmallNoVec(const Mat&, int, int, double) {}
    int operator()(const uchar**, uchar*, int) const { return 0; }
};

struct FilterNoVec
{
    FilterNoVec() {}
    FilterNoVec(const Mat&, int, double) {}
    int operator()(const uchar**, uchar*, int) const { return 0; }
};


#if CV_SIMD

///////////////////////////////////// 8u-16s & 8u-8u //////////////////////////////////

struct RowVec_8u32s
{
    RowVec_8u32s() { smallValues = false; }
    RowVec_8u32s( const Mat& _kernel )
    {
        kernel = _kernel;
        smallValues = true;
        int k, ksize = kernel.rows + kernel.cols - 1;
        for( k = 0; k < ksize; k++ )
        {
            int v = kernel.ptr<int>()[k];
            if( v < SHRT_MIN || v > SHRT_MAX )
            {
                smallValues = false;
                break;
            }
        }
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        int* dst = (int*)_dst;
        const int* _kx = kernel.ptr<int>();
        width *= cn;

        if( smallValues )
        {
            for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes )
            {
                const uchar* src = _src + i;
                v_int32 s0 = vx_setzero_s32();
                v_int32 s1 = vx_setzero_s32();
                v_int32 s2 = vx_setzero_s32();
                v_int32 s3 = vx_setzero_s32();
                k = 0;
                for (; k <= _ksize - 2; k += 2, src += 2 * cn)
                {
                    v_int32 f = vx_setall_s32((_kx[k] & 0xFFFF) | (_kx[k + 1] << 16));
                    v_uint8 x0, x1;
                    v_zip(vx_load(src), vx_load(src + cn), x0, x1);
                    s0 += v_dotprod(v_reinterpret_as_s16(v_expand_low(x0)), v_reinterpret_as_s16(f));
                    s1 += v_dotprod(v_reinterpret_as_s16(v_expand_high(x0)), v_reinterpret_as_s16(f));
                    s2 += v_dotprod(v_reinterpret_as_s16(v_expand_low(x1)), v_reinterpret_as_s16(f));
                    s3 += v_dotprod(v_reinterpret_as_s16(v_expand_high(x1)), v_reinterpret_as_s16(f));
                }
                if (k < _ksize)
                {
                    v_int32 f = vx_setall_s32(_kx[k]);
                    v_uint16 x0, x1;
                    v_expand(vx_load(src), x0, x1);
                    s0 += v_dotprod(v_reinterpret_as_s16(v_expand_low(x0)), v_reinterpret_as_s16(f));
                    s1 += v_dotprod(v_reinterpret_as_s16(v_expand_high(x0)), v_reinterpret_as_s16(f));
                    s2 += v_dotprod(v_reinterpret_as_s16(v_expand_low(x1)), v_reinterpret_as_s16(f));
                    s3 += v_dotprod(v_reinterpret_as_s16(v_expand_high(x1)), v_reinterpret_as_s16(f));
                }
                v_store(dst + i, s0);
                v_store(dst + i + v_int32::nlanes, s1);
                v_store(dst + i + 2*v_int32::nlanes, s2);
                v_store(dst + i + 3*v_int32::nlanes, s3);
            }
            if( i <= width - v_uint16::nlanes )
            {
                const uchar* src = _src + i;
                v_int32 s0 = vx_setzero_s32();
                v_int32 s1 = vx_setzero_s32();
                k = 0;
                for( ; k <= _ksize - 2; k += 2, src += 2*cn )
                {
                    v_int32 f = vx_setall_s32((_kx[k] & 0xFFFF) | (_kx[k + 1] << 16));
                    v_uint16 x0, x1;
                    v_zip(vx_load_expand(src), vx_load_expand(src + cn), x0, x1);
                    s0 += v_dotprod(v_reinterpret_as_s16(x0), v_reinterpret_as_s16(f));
                    s1 += v_dotprod(v_reinterpret_as_s16(x1), v_reinterpret_as_s16(f));
                }
                if( k < _ksize )
                {
                    v_int32 f = vx_setall_s32(_kx[k]);
                    v_uint32 x0, x1;
                    v_expand(vx_load_expand(src), x0, x1);
                    s0 += v_dotprod(v_reinterpret_as_s16(x0), v_reinterpret_as_s16(f));
                    s1 += v_dotprod(v_reinterpret_as_s16(x1), v_reinterpret_as_s16(f));
                }
                v_store(dst + i, s0);
                v_store(dst + i + v_int32::nlanes, s1);
                i += v_uint16::nlanes;
            }
            if( i <= width - v_uint32::nlanes )
            {
                v_int32 d = vx_setzero_s32();
                k = 0;
                const uchar* src = _src + i;
                for (; k <= _ksize - 2; k += 2, src += 2*cn)
                {
                    v_int32 f = vx_setall_s32((_kx[k] & 0xFFFF) | (_kx[k + 1] << 16));
                    v_uint32 x0, x1;
                    v_zip(vx_load_expand_q(src), vx_load_expand_q(src + cn), x0, x1);
                    d += v_dotprod(v_pack(v_reinterpret_as_s32(x0), v_reinterpret_as_s32(x1)), v_reinterpret_as_s16(f));
                }
                if (k < _ksize)
                    d += v_dotprod(v_reinterpret_as_s16(vx_load_expand_q(src)), v_reinterpret_as_s16(vx_setall_s32(_kx[k])));
                v_store(dst + i, d);
                i += v_uint32::nlanes;
            }
        }
        vx_cleanup();
        return i;
    }

    Mat kernel;
    bool smallValues;
};


struct SymmRowSmallVec_8u32s
{
    SymmRowSmallVec_8u32s() { smallValues = false; symmetryType = 0; }
    SymmRowSmallVec_8u32s( const Mat& _kernel, int _symmetryType )
    {
        kernel = _kernel;
        symmetryType = _symmetryType;
        smallValues = true;
        int k, ksize = kernel.rows + kernel.cols - 1;
        for( k = 0; k < ksize; k++ )
        {
            int v = kernel.ptr<int>()[k];
            if( v < SHRT_MIN || v > SHRT_MAX )
            {
                smallValues = false;
                break;
            }
        }
    }

    int operator()(const uchar* src, uchar* _dst, int width, int cn) const
    {
        int i = 0, j, k, _ksize = kernel.rows + kernel.cols - 1;
        int* dst = (int*)_dst;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int* kx = kernel.ptr<int>() + _ksize/2;
        if( !smallValues )
            return 0;

        src += (_ksize/2)*cn;
        width *= cn;

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                {
                    for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                    {
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src), x1l, x1h);
                        v_expand(vx_load(src + cn), x2l, x2h);
                        x1l = v_add_wrap(v_add_wrap(x1l, x1l), v_add_wrap(x0l, x2l));
                        x1h = v_add_wrap(v_add_wrap(x1h, x1h), v_add_wrap(x0h, x2h));
                        v_store(dst + i, v_reinterpret_as_s32(v_expand_low(x1l)));
                        v_store(dst + i + v_int32::nlanes, v_reinterpret_as_s32(v_expand_high(x1l)));
                        v_store(dst + i + 2*v_int32::nlanes, v_reinterpret_as_s32(v_expand_low(x1h)));
                        v_store(dst + i + 3*v_int32::nlanes, v_reinterpret_as_s32(v_expand_high(x1h)));
                    }
                    if( i <= width - v_uint16::nlanes )
                    {
                        v_uint16 x = vx_load_expand(src);
                        x = v_add_wrap(v_add_wrap(x, x), v_add_wrap(vx_load_expand(src - cn), vx_load_expand(src + cn)));
                        v_store(dst + i, v_reinterpret_as_s32(v_expand_low(x)));
                        v_store(dst + i + v_int32::nlanes, v_reinterpret_as_s32(v_expand_high(x)));
                        i += v_uint16::nlanes; src += v_uint16::nlanes;
                    }
                    if( i <= width - v_uint32::nlanes )
                    {
                        v_uint32 x = vx_load_expand_q(src);
                        x = (x + x) + vx_load_expand_q(src - cn) + vx_load_expand_q(src + cn);
                        v_store(dst + i, v_reinterpret_as_s32(x));
                        i += v_uint32::nlanes;
                    }
                }
                else if( kx[0] == -2 && kx[1] == 1 )
                {
                    for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                    {
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src), x1l, x1h);
                        v_expand(vx_load(src + cn), x2l, x2h);
                        x1l = v_sub_wrap(v_add_wrap(x0l, x2l), v_add_wrap(x1l, x1l));
                        x1h = v_sub_wrap(v_add_wrap(x0h, x2h), v_add_wrap(x1h, x1h));
                        v_store(dst + i, v_expand_low(v_reinterpret_as_s16(x1l)));
                        v_store(dst + i + v_int32::nlanes, v_expand_high(v_reinterpret_as_s16(x1l)));
                        v_store(dst + i + 2*v_int32::nlanes, v_expand_low(v_reinterpret_as_s16(x1h)));
                        v_store(dst + i + 3*v_int32::nlanes, v_expand_high(v_reinterpret_as_s16(x1h)));
                    }
                    if( i <= width - v_uint16::nlanes )
                    {
                        v_uint16 x = vx_load_expand(src);
                        x = v_sub_wrap(v_add_wrap(vx_load_expand(src - cn), vx_load_expand(src + cn)), v_add_wrap(x, x));
                        v_store(dst + i, v_expand_low(v_reinterpret_as_s16(x)));
                        v_store(dst + i + v_int32::nlanes, v_expand_high(v_reinterpret_as_s16(x)));
                        i += v_uint16::nlanes; src += v_uint16::nlanes;
                    }
                    if( i <= width - v_uint32::nlanes )
                    {
                        v_int32 x = v_reinterpret_as_s32(vx_load_expand_q(src));
                        x = v_reinterpret_as_s32(vx_load_expand_q(src - cn) + vx_load_expand_q(src + cn)) - (x + x);
                        v_store(dst + i, x);
                        i += v_uint32::nlanes;
                    }
                }
                else
                {
                    v_int16 k0 = vx_setall_s16((short)kx[0]);
                    v_int16 k1 = vx_setall_s16((short)kx[1]);
                    for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                    {
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src), x1l, x1h);
                        v_expand(vx_load(src + cn), x2l, x2h);

                        v_int32 dl, dh;
                        v_int16 x0, x1;
                        v_mul_expand(v_reinterpret_as_s16(x1l), k0, dl, dh);
                        v_zip(v_reinterpret_as_s16(x0l), v_reinterpret_as_s16(x2l), x0, x1);
                        dl += v_dotprod(x0, k1);
                        dh += v_dotprod(x1, k1);
                        v_store(dst + i, dl);
                        v_store(dst + i + v_int32::nlanes, dh);

                        v_mul_expand(v_reinterpret_as_s16(x1h), k0, dl, dh);
                        v_zip(v_reinterpret_as_s16(x0h), v_reinterpret_as_s16(x2h), x0, x1);
                        dl += v_dotprod(x0, k1);
                        dh += v_dotprod(x1, k1);
                        v_store(dst + i + 2*v_int32::nlanes, dl);
                        v_store(dst + i + 3*v_int32::nlanes, dh);
                    }
                    if ( i <= width - v_uint16::nlanes )
                    {
                        v_int32 dl, dh;
                        v_mul_expand(v_reinterpret_as_s16(vx_load_expand(src)), k0, dl, dh);
                        v_int16 x0, x1;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src - cn)), v_reinterpret_as_s16(vx_load_expand(src + cn)), x0, x1);
                        dl += v_dotprod(x0, k1);
                        dh += v_dotprod(x1, k1);
                        v_store(dst + i, dl);
                        v_store(dst + i + v_int32::nlanes, dh);
                        i += v_uint16::nlanes; src += v_uint16::nlanes;
                    }
                    if ( i <= width - v_uint32::nlanes )
                    {
                        v_store(dst + i, v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src)), vx_setall_s32(kx[0]), v_reinterpret_as_s32(vx_load_expand_q(src - cn) + vx_load_expand_q(src + cn)) * vx_setall_s32(kx[1])));
                        i += v_uint32::nlanes;
                    }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                {
                    for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                    {
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
                        v_expand(vx_load(src - 2*cn), x0l, x0h);
                        v_expand(vx_load(src), x1l, x1h);
                        v_expand(vx_load(src + 2*cn), x2l, x2h);
                        x1l = v_sub_wrap(v_add_wrap(x0l, x2l), v_add_wrap(x1l, x1l));
                        x1h = v_sub_wrap(v_add_wrap(x0h, x2h), v_add_wrap(x1h, x1h));
                        v_store(dst + i, v_expand_low(v_reinterpret_as_s16(x1l)));
                        v_store(dst + i + v_int32::nlanes, v_expand_high(v_reinterpret_as_s16(x1l)));
                        v_store(dst + i + 2*v_int32::nlanes, v_expand_low(v_reinterpret_as_s16(x1h)));
                        v_store(dst + i + 3*v_int32::nlanes, v_expand_high(v_reinterpret_as_s16(x1h)));
                    }
                    if( i <= width - v_uint16::nlanes )
                    {
                        v_uint16 x = vx_load_expand(src);
                        x = v_sub_wrap(v_add_wrap(vx_load_expand(src - 2*cn), vx_load_expand(src + 2*cn)), v_add_wrap(x, x));
                        v_store(dst + i, v_expand_low(v_reinterpret_as_s16(x)));
                        v_store(dst + i + v_int32::nlanes, v_expand_high(v_reinterpret_as_s16(x)));
                        i += v_uint16::nlanes; src += v_uint16::nlanes;
                    }
                    if( i <= width - v_uint32::nlanes )
                    {
                        v_int32 x = v_reinterpret_as_s32(vx_load_expand_q(src));
                        x = v_reinterpret_as_s32(vx_load_expand_q(src - 2*cn) + vx_load_expand_q(src + 2*cn)) - (x + x);
                        v_store(dst + i, x);
                        i += v_uint32::nlanes;
                    }
                }
                else
                {
                    v_int16 k0 = vx_setall_s16((short)(kx[0]));
                    v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[1] & 0xFFFF) | (kx[2] << 16)));
                    for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                    {
                        v_int32 x0, x1, x2, x3;
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h, x3l, x3h;
                        v_int16 xl, xh;

                        v_expand(vx_load(src), x0l, x0h);
                        v_mul_expand(v_reinterpret_as_s16(x0l), k0, x0, x1);
                        v_mul_expand(v_reinterpret_as_s16(x0h), k0, x2, x3);

                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src + cn), x1l, x1h);
                        v_expand(vx_load(src - 2*cn), x2l, x2h);
                        v_expand(vx_load(src + 2*cn), x3l, x3h);
                        v_zip(v_reinterpret_as_s16(x0l + x1l), v_reinterpret_as_s16(x2l + x3l), xl, xh);
                        x0 += v_dotprod(xl, k12);
                        x1 += v_dotprod(xh, k12);
                        v_zip(v_reinterpret_as_s16(x0h + x1h), v_reinterpret_as_s16(x2h + x3h), xl, xh);
                        x2 += v_dotprod(xl, k12);
                        x3 += v_dotprod(xh, k12);

                        v_store(dst + i, x0);
                        v_store(dst + i + v_int32::nlanes, x1);
                        v_store(dst + i + 2*v_int32::nlanes, x2);
                        v_store(dst + i + 3*v_int32::nlanes, x3);
                    }
                    if( i <= width - v_uint16::nlanes )
                    {
                        v_int32 x1, x2;
                        v_mul_expand(v_reinterpret_as_s16(vx_load_expand(src)), k0, x1, x2);

                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src - cn) + vx_load_expand(src + cn)), v_reinterpret_as_s16(vx_load_expand(src - 2*cn) + vx_load_expand(src + 2*cn)), xl, xh);
                        x1 += v_dotprod(xl, k12);
                        x2 += v_dotprod(xh, k12);

                        v_store(dst + i, x1);
                        v_store(dst + i + v_int32::nlanes, x2);
                        i += v_uint16::nlanes, src += v_uint16::nlanes;
                    }
                    if( i <= width - v_uint32::nlanes )
                    {
                        v_store(dst + i, v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src)), vx_setall_s32(kx[0]),
                                         v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src - cn) + vx_load_expand_q(src + cn)), vx_setall_s32(kx[1]),
                                                  v_reinterpret_as_s32(vx_load_expand_q(src - 2*cn) + vx_load_expand_q(src + 2*cn)) * vx_setall_s32(kx[2]))));
                        i += v_uint32::nlanes;
                    }
                }
            }
            else
            {
                v_int16 k0 = vx_setall_s16((short)(kx[0]));
                for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                {
                    v_uint8 v_src = vx_load(src);
                    v_int32 s0, s1, s2, s3;
                    v_mul_expand(v_reinterpret_as_s16(v_expand_low(v_src)), k0, s0, s1);
                    v_mul_expand(v_reinterpret_as_s16(v_expand_high(v_src)), k0, s2, s3);
                    for (k = 1, j = cn; k <= _ksize / 2 - 1; k += 2, j += 2 * cn)
                    {
                        v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (kx[k + 1] << 16)));

                        v_uint8 v_src0 = vx_load(src - j);
                        v_uint8 v_src1 = vx_load(src - j - cn);
                        v_uint8 v_src2 = vx_load(src + j);
                        v_uint8 v_src3 = vx_load(src + j + cn);

                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_expand_low(v_src0) + v_expand_low(v_src2)), v_reinterpret_as_s16(v_expand_low(v_src1) + v_expand_low(v_src3)), xl, xh);
                        s0 += v_dotprod(xl, k12);
                        s1 += v_dotprod(xh, k12);
                        v_zip(v_reinterpret_as_s16(v_expand_high(v_src0) + v_expand_high(v_src2)), v_reinterpret_as_s16(v_expand_high(v_src1) + v_expand_high(v_src3)), xl, xh);
                        s2 += v_dotprod(xl, k12);
                        s3 += v_dotprod(xh, k12);
                    }
                    if( k < _ksize / 2 + 1 )
                    {
                        v_int16 k1 = vx_setall_s16((short)(kx[k]));

                        v_uint8 v_src0 = vx_load(src - j);
                        v_uint8 v_src1 = vx_load(src + j);

                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_expand_low(v_src0)), v_reinterpret_as_s16(v_expand_low(v_src1)), xl, xh);
                        s0 += v_dotprod(xl, k1);
                        s1 += v_dotprod(xh, k1);
                        v_zip(v_reinterpret_as_s16(v_expand_high(v_src0)), v_reinterpret_as_s16(v_expand_high(v_src1)), xl, xh);
                        s2 += v_dotprod(xl, k1);
                        s3 += v_dotprod(xh, k1);
                    }
                    v_store(dst + i, s0);
                    v_store(dst + i + v_int32::nlanes, s1);
                    v_store(dst + i + 2*v_int32::nlanes, s2);
                    v_store(dst + i + 3*v_int32::nlanes, s3);
                }
                if( i <= width - v_uint16::nlanes )
                {
                    v_int32 s0, s1;
                    v_mul_expand(v_reinterpret_as_s16(vx_load_expand(src)), k0, s0, s1);
                    for (k = 1, j = cn; k <= _ksize / 2 - 1; k+=2, j += 2*cn)
                    {
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src - j) + vx_load_expand(src + j)), v_reinterpret_as_s16(vx_load_expand(src - j - cn) + vx_load_expand(src + j + cn)), xl, xh);
                        v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (kx[k+1] << 16)));
                        s0 += v_dotprod(xl, k12);
                        s1 += v_dotprod(xh, k12);
                    }
                    if ( k < _ksize / 2 + 1 )
                    {
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src - j)), v_reinterpret_as_s16(vx_load_expand(src + j)), xl, xh);
                        v_int16 k1 = vx_setall_s16((short)(kx[k]));
                        s0 += v_dotprod(xl, k1);
                        s1 += v_dotprod(xh, k1);
                    }
                    v_store(dst + i, s0);
                    v_store(dst + i + v_int32::nlanes, s1);
                    i += v_uint16::nlanes; src += v_uint16::nlanes;
                }
                if( i <= width - v_uint32::nlanes )
                {
                    v_int32 s0 = v_reinterpret_as_s32(vx_load_expand_q(src)) * vx_setall_s32(kx[0]);
                    for( k = 1, j = cn; k < _ksize / 2 + 1; k++, j += cn )
                        s0 = v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src - j) + vx_load_expand_q(src + j)), vx_setall_s32(kx[k]), s0);
                    v_store(dst + i, s0);
                    i += v_uint32::nlanes;
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                {
                    for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                    {
                        v_uint16 x0l, x0h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src + cn), x2l, x2h);
                        v_int16 dl = v_reinterpret_as_s16(v_sub_wrap(x2l, x0l));
                        v_int16 dh = v_reinterpret_as_s16(v_sub_wrap(x2h, x0h));
                        v_store(dst + i, v_expand_low(dl));
                        v_store(dst + i + v_int32::nlanes, v_expand_high(dl));
                        v_store(dst + i + 2*v_int32::nlanes, v_expand_low(dh));
                        v_store(dst + i + 3*v_int32::nlanes, v_expand_high(dh));
                    }
                    if( i <= width - v_uint16::nlanes )
                    {
                        v_int16 dl = v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + cn), vx_load_expand(src - cn)));
                        v_store(dst + i, v_expand_low(dl));
                        v_store(dst + i + v_int32::nlanes, v_expand_high(dl));
                        i += v_uint16::nlanes; src += v_uint16::nlanes;
                    }
                    if (i <= width - v_uint32::nlanes)
                    {
                        v_store(dst + i, v_reinterpret_as_s32(vx_load_expand_q(src + cn)) - v_reinterpret_as_s32(vx_load_expand_q(src - cn)));
                        i += v_uint32::nlanes;
                    }
                }
                else
                {
                    v_int16 k0 = v_reinterpret_as_s16(vx_setall_s32((kx[1] & 0xFFFF) | (-kx[1] << 16)));
                    for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                    {
                        v_uint16 x0l, x0h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src + cn), x2l, x2h);
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(x2l), v_reinterpret_as_s16(x0l), xl, xh);
                        v_store(dst + i, v_dotprod(xl, k0));
                        v_store(dst + i + v_int32::nlanes, v_dotprod(xh, k0));
                        v_zip(v_reinterpret_as_s16(x2h), v_reinterpret_as_s16(x0h), xl, xh);
                        v_store(dst + i + 2*v_int32::nlanes, v_dotprod(xl, k0));
                        v_store(dst + i + 3*v_int32::nlanes, v_dotprod(xh, k0));
                    }
                    if( i <= width - v_uint16::nlanes )
                    {
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src + cn)), v_reinterpret_as_s16(vx_load_expand(src - cn)), xl, xh);
                        v_store(dst + i, v_dotprod(xl, k0));
                        v_store(dst + i + v_int32::nlanes, v_dotprod(xh, k0));
                        i += v_uint16::nlanes; src += v_uint16::nlanes;
                    }
                    if (i <= width - v_uint32::nlanes)
                    {
                        v_store(dst + i, v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src + cn)), vx_setall_s32(kx[1]), v_reinterpret_as_s32(vx_load_expand_q(src - cn)) * vx_setall_s32(-kx[1])));
                        i += v_uint32::nlanes;
                    }
                }
            }
            else if( _ksize == 5 )
            {
                v_int16 k0 = v_reinterpret_as_s16(vx_setall_s32((kx[1] & 0xFFFF) | (kx[2] << 16)));
                for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                {
                    v_uint16 x0l, x0h, x1l, x1h, x2l, x2h, x3l, x3h;
                    v_expand(vx_load(src - cn), x0l, x0h);
                    v_expand(vx_load(src - 2*cn), x1l, x1h);
                    v_expand(vx_load(src + cn), x2l, x2h);
                    v_expand(vx_load(src + 2*cn), x3l, x3h);
                    v_int16 x0, x1;
                    v_zip(v_reinterpret_as_s16(v_sub_wrap(x2l, x0l)), v_reinterpret_as_s16(v_sub_wrap(x3l, x1l)), x0, x1);
                    v_store(dst + i, v_dotprod(x0, k0));
                    v_store(dst + i + v_int32::nlanes, v_dotprod(x1, k0));
                    v_zip(v_reinterpret_as_s16(v_sub_wrap(x2h, x0h)), v_reinterpret_as_s16(v_sub_wrap(x3h, x1h)), x0, x1);
                    v_store(dst + i + 2*v_int32::nlanes, v_dotprod(x0, k0));
                    v_store(dst + i + 3*v_int32::nlanes, v_dotprod(x1, k0));
                }
                if( i <= width - v_uint16::nlanes )
                {
                    v_int16 x0, x1;
                    v_zip(v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + cn), vx_load_expand(src - cn))),
                          v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + 2*cn), vx_load_expand(src - 2*cn))), x0, x1);
                    v_store(dst + i, v_dotprod(x0, k0));
                    v_store(dst + i + v_int32::nlanes, v_dotprod(x1, k0));
                    i += v_uint16::nlanes; src += v_uint16::nlanes;
                }
                if( i <= width - v_uint32::nlanes )
                {
                    v_store(dst + i, v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src + cn)) - v_reinterpret_as_s32(vx_load_expand_q(src - cn)), vx_setall_s32(kx[1]),
                                             (v_reinterpret_as_s32(vx_load_expand_q(src + 2*cn)) - v_reinterpret_as_s32(vx_load_expand_q(src - 2*cn))) * vx_setall_s32(kx[2])));
                    i += v_uint32::nlanes;
                }
            }
            else
            {
                v_int16 k0 = vx_setall_s16((short)(kx[0]));
                for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes, src += v_uint8::nlanes )
                {
                    v_uint8 v_src = vx_load(src);
                    v_int32 s0, s1, s2, s3;
                    v_mul_expand(v_reinterpret_as_s16(v_expand_low(v_src)), k0, s0, s1);
                    v_mul_expand(v_reinterpret_as_s16(v_expand_high(v_src)), k0, s2, s3);
                    for( k = 1, j = cn; k <= _ksize / 2 - 1; k += 2, j += 2 * cn )
                    {
                        v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (kx[k + 1] << 16)));

                        v_uint8 v_src0 = vx_load(src - j);
                        v_uint8 v_src1 = vx_load(src - j - cn);
                        v_uint8 v_src2 = vx_load(src + j);
                        v_uint8 v_src3 = vx_load(src + j + cn);

                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_sub_wrap(v_expand_low(v_src2), v_expand_low(v_src0))), v_reinterpret_as_s16(v_sub_wrap(v_expand_low(v_src3), v_expand_low(v_src1))), xl, xh);
                        s0 += v_dotprod(xl, k12);
                        s1 += v_dotprod(xh, k12);
                        v_zip(v_reinterpret_as_s16(v_sub_wrap(v_expand_high(v_src2), v_expand_high(v_src0))), v_reinterpret_as_s16(v_sub_wrap(v_expand_high(v_src3), v_expand_high(v_src1))), xl, xh);
                        s2 += v_dotprod(xl, k12);
                        s3 += v_dotprod(xh, k12);
                    }
                    if( k < _ksize / 2 + 1 )
                    {
                        v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (-kx[k] << 16)));
                        v_uint8 v_src0 = vx_load(src - j);
                        v_uint8 v_src1 = vx_load(src + j);

                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_expand_low(v_src1)), v_reinterpret_as_s16(v_expand_low(v_src0)), xl, xh);
                        s0 += v_dotprod(xl, k12);
                        s1 += v_dotprod(xh, k12);
                        v_zip(v_reinterpret_as_s16(v_expand_high(v_src1)), v_reinterpret_as_s16(v_expand_high(v_src0)), xl, xh);
                        s2 += v_dotprod(xl, k12);
                        s3 += v_dotprod(xh, k12);
                    }
                    v_store(dst + i, s0);
                    v_store(dst + i + v_int32::nlanes, s1);
                    v_store(dst + i + 2*v_int32::nlanes, s2);
                    v_store(dst + i + 3*v_int32::nlanes, s3);
                }
                if( i <= width - v_uint16::nlanes )
                {
                    v_int32 s0, s1;
                    v_mul_expand(v_reinterpret_as_s16(vx_load_expand(src)), k0, s0, s1);
                    for( k = 1, j = cn; k <= _ksize / 2 - 1; k += 2, j += 2 * cn )
                    {
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + j), vx_load_expand(src - j))), v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + j + cn), vx_load_expand(src - j - cn))), xl, xh);
                        v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (kx[k + 1] << 16)));
                        s0 += v_dotprod(xl, k12);
                        s1 += v_dotprod(xh, k12);
                    }
                    if( k < _ksize / 2 + 1 )
                    {
                        v_int16 k1 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (-kx[k] << 16)));
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src + j)), v_reinterpret_as_s16(vx_load_expand(src - j)), xl, xh);
                        s0 += v_dotprod(xl, k1);
                        s1 += v_dotprod(xh, k1);
                    }
                    v_store(dst + i, s0);
                    v_store(dst + i + v_int32::nlanes, s1);
                    i += v_uint16::nlanes; src += v_uint16::nlanes;
                }
                if( i <= width - v_uint32::nlanes )
                {
                    v_int32 s0 = v_reinterpret_as_s32(vx_load_expand_q(src)) * vx_setall_s32(kx[0]);
                    for (k = 1, j = cn; k < _ksize / 2 + 1; k++, j += cn)
                        s0 = v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src + j)) - v_reinterpret_as_s32(vx_load_expand_q(src - j)), vx_setall_s32(kx[k]), s0);
                    v_store(dst + i, s0);
                    i += v_uint32::nlanes;
                }
            }
        }

        vx_cleanup();
        return i;
    }

    Mat kernel;
    int symmetryType;
    bool smallValues;
};


struct SymmColumnVec_32s8u
{
    SymmColumnVec_32s8u() { symmetryType=0; delta = 0; }
    SymmColumnVec_32s8u(const Mat& _kernel, int _symmetryType, int _bits, double _delta)
    {
        symmetryType = _symmetryType;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* dst, int width) const
    {
        int _ksize = kernel.rows + kernel.cols - 1;
        if( _ksize == 1 )
            return 0;
        int ksize2 = _ksize/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;

        v_float32 d4 = vx_setall_f32(delta);
        if( symmetrical )
        {
            v_float32 f0 = vx_setall_f32(ky[0]);
            v_float32 f1 = vx_setall_f32(ky[1]);
            for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes )
            {
                const int* S = src[0] + i;
                v_float32 s0 = v_muladd(v_cvt_f32(vx_load(S)), f0, d4);
                v_float32 s1 = v_muladd(v_cvt_f32(vx_load(S + v_int32::nlanes)), f0, d4);
                v_float32 s2 = v_muladd(v_cvt_f32(vx_load(S + 2*v_int32::nlanes)), f0, d4);
                v_float32 s3 = v_muladd(v_cvt_f32(vx_load(S + 3*v_int32::nlanes)), f0, d4);
                const int* S0 = src[1] + i;
                const int* S1 = src[-1] + i;
                s0 = v_muladd(v_cvt_f32(vx_load(S0) + vx_load(S1)), f1, s0);
                s1 = v_muladd(v_cvt_f32(vx_load(S0 + v_int32::nlanes) + vx_load(S1 + v_int32::nlanes)), f1, s1);
                s2 = v_muladd(v_cvt_f32(vx_load(S0 + 2 * v_int32::nlanes) + vx_load(S1 + 2 * v_int32::nlanes)), f1, s2);
                s3 = v_muladd(v_cvt_f32(vx_load(S0 + 3 * v_int32::nlanes) + vx_load(S1 + 3 * v_int32::nlanes)), f1, s3);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 f = vx_setall_f32(ky[k]);
                    S0 = src[k] + i;
                    S1 = src[-k] + i;
                    s0 = v_muladd(v_cvt_f32(vx_load(S0) + vx_load(S1)), f, s0);
                    s1 = v_muladd(v_cvt_f32(vx_load(S0 + v_int32::nlanes) + vx_load(S1 + v_int32::nlanes)), f, s1);
                    s2 = v_muladd(v_cvt_f32(vx_load(S0 + 2*v_int32::nlanes) + vx_load(S1 + 2*v_int32::nlanes)), f, s2);
                    s3 = v_muladd(v_cvt_f32(vx_load(S0 + 3*v_int32::nlanes) + vx_load(S1 + 3*v_int32::nlanes)), f, s3);
                }
                v_store(dst + i, v_pack_u(v_pack(v_round(s0), v_round(s1)), v_pack(v_round(s2), v_round(s3))));
            }
            if( i <= width - v_uint16::nlanes )
            {
                const int* S = src[0] + i;
                v_float32 s0 = v_muladd(v_cvt_f32(vx_load(S)), f0, d4);
                v_float32 s1 = v_muladd(v_cvt_f32(vx_load(S + v_int32::nlanes)), f0, d4);
                const int* S0 = src[1] + i;
                const int* S1 = src[-1] + i;
                s0 = v_muladd(v_cvt_f32(vx_load(S0) + vx_load(S1)), f1, s0);
                s1 = v_muladd(v_cvt_f32(vx_load(S0 + v_int32::nlanes) + vx_load(S1 + v_int32::nlanes)), f1, s1);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 f = vx_setall_f32(ky[k]);
                    S0 = src[k] + i;
                    S1 = src[-k] + i;
                    s0 = v_muladd(v_cvt_f32(vx_load(S0) + vx_load(S1)), f, s0);
                    s1 = v_muladd(v_cvt_f32(vx_load(S0 + v_int32::nlanes) + vx_load(S1 + v_int32::nlanes)), f, s1);
                }
                v_pack_u_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                i += v_uint16::nlanes;
            }
#if CV_SIMD_WIDTH > 16
            while( i <= width - v_int32x4::nlanes )
#else
            if( i <= width - v_int32x4::nlanes )
#endif
            {
                v_float32x4 s0 = v_muladd(v_cvt_f32(v_load(src[0] + i)), v_setall_f32(ky[0]), v_setall_f32(delta));
                s0 = v_muladd(v_cvt_f32(v_load(src[1] + i) + v_load(src[-1] + i)), v_setall_f32(ky[1]), s0);
                for( k = 2; k <= ksize2; k++ )
                    s0 = v_muladd(v_cvt_f32(v_load(src[k] + i) + v_load(src[-k] + i)), v_setall_f32(ky[k]), s0);
                v_int32x4 s32 = v_round(s0);
                v_int16x8 s16 = v_pack(s32, s32);
                *(int*)(dst + i) = v_reinterpret_as_s32(v_pack_u(s16, s16)).get0();
                i += v_int32x4::nlanes;
            }
        }
        else
        {
            v_float32 f1 = vx_setall_f32(ky[1]);
            for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes )
            {
                const int* S0 = src[1] + i;
                const int* S1 = src[-1] + i;
                v_float32 s0 = v_muladd(v_cvt_f32(vx_load(S0) - vx_load(S1)), f1, d4);
                v_float32 s1 = v_muladd(v_cvt_f32(vx_load(S0 + v_int32::nlanes) - vx_load(S1 + v_int32::nlanes)), f1, d4);
                v_float32 s2 = v_muladd(v_cvt_f32(vx_load(S0 + 2 * v_int32::nlanes) - vx_load(S1 + 2 * v_int32::nlanes)), f1, d4);
                v_float32 s3 = v_muladd(v_cvt_f32(vx_load(S0 + 3 * v_int32::nlanes) - vx_load(S1 + 3 * v_int32::nlanes)), f1, d4);
                for ( k = 2; k <= ksize2; k++ )
                {
                    v_float32 f = vx_setall_f32(ky[k]);
                    S0 = src[k] + i;
                    S1 = src[-k] + i;
                    s0 = v_muladd(v_cvt_f32(vx_load(S0) - vx_load(S1)), f, s0);
                    s1 = v_muladd(v_cvt_f32(vx_load(S0 + v_int32::nlanes) - vx_load(S1 + v_int32::nlanes)), f, s1);
                    s2 = v_muladd(v_cvt_f32(vx_load(S0 + 2*v_int32::nlanes) - vx_load(S1 + 2*v_int32::nlanes)), f, s2);
                    s3 = v_muladd(v_cvt_f32(vx_load(S0 + 3*v_int32::nlanes) - vx_load(S1 + 3*v_int32::nlanes)), f, s3);
                }
                v_store(dst + i, v_pack_u(v_pack(v_round(s0), v_round(s1)), v_pack(v_round(s2), v_round(s3))));
            }
            if( i <= width - v_uint16::nlanes )
            {
                const int* S0 = src[1] + i;
                const int* S1 = src[-1] + i;
                v_float32 s0 = v_muladd(v_cvt_f32(vx_load(S0) - vx_load(S1)), f1, d4);
                v_float32 s1 = v_muladd(v_cvt_f32(vx_load(S0 + v_int32::nlanes) - vx_load(S1 + v_int32::nlanes)), f1, d4);
                for ( k = 2; k <= ksize2; k++ )
                {
                    v_float32 f = vx_setall_f32(ky[k]);
                    S0 = src[k] + i;
                    S1 = src[-k] + i;
                    s0 = v_muladd(v_cvt_f32(vx_load(S0) - vx_load(S1)), f, s0);
                    s1 = v_muladd(v_cvt_f32(vx_load(S0 + v_int32::nlanes) - vx_load(S1 + v_int32::nlanes)), f, s1);
                }
                v_pack_u_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                i += v_uint16::nlanes;
            }
#if CV_SIMD_WIDTH > 16
            while( i <= width - v_int32x4::nlanes )
#else
            if( i <= width - v_int32x4::nlanes )
#endif
            {
                v_float32x4 s0 = v_muladd(v_cvt_f32(v_load(src[1] + i) - v_load(src[-1] + i)), v_setall_f32(ky[1]), v_setall_f32(delta));
                for (k = 2; k <= ksize2; k++)
                    s0 = v_muladd(v_cvt_f32(v_load(src[k] + i) - v_load(src[-k] + i)), v_setall_f32(ky[k]), s0);
                v_int32x4 s32 = v_round(s0);
                v_int16x8 s16 = v_pack(s32, s32);
                *(int*)(dst + i) = v_reinterpret_as_s32(v_pack_u(s16, s16)).get0();
                i += v_int32x4::nlanes;
            }
        }

        vx_cleanup();
        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


struct SymmColumnSmallVec_32s16s
{
    SymmColumnSmallVec_32s16s() { symmetryType=0; delta = 0; }
    SymmColumnSmallVec_32s16s(const Mat& _kernel, int _symmetryType, int _bits, double _delta)
    {
        symmetryType = _symmetryType;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;
        const int *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        short* dst = (short*)_dst;

        v_float32 df4 = vx_setall_f32(delta);
        int d = cvRound(delta);
        v_int16 d8 = vx_setall_s16((short)d);
        if( symmetrical )
        {
            if( ky[0] == 2 && ky[1] == 1 )
            {
                for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
                {
                    v_int32 s0 = vx_load(S1 + i);
                    v_int32 s1 = vx_load(S1 + i + v_int32::nlanes);
                    v_int32 s2 = vx_load(S1 + i + 2*v_int32::nlanes);
                    v_int32 s3 = vx_load(S1 + i + 3*v_int32::nlanes);
                    v_store(dst + i, v_pack(vx_load(S0 + i) + vx_load(S2 + i) + (s0 + s0), vx_load(S0 + i + v_int32::nlanes) + vx_load(S2 + i + v_int32::nlanes) + (s1 + s1)) + d8);
                    v_store(dst + i + v_int16::nlanes, v_pack(vx_load(S0 + i + 2*v_int32::nlanes) + vx_load(S2 + i + 2*v_int32::nlanes) + (s2 + s2),
                                                              vx_load(S0 + i + 3*v_int32::nlanes) + vx_load(S2 + i + 3*v_int32::nlanes) + (s3 + s3)) + d8);
                }
                if( i <= width - v_int16::nlanes )
                {
                    v_int32 sl = vx_load(S1 + i);
                    v_int32 sh = vx_load(S1 + i + v_int32::nlanes);
                    v_store(dst + i, v_pack(vx_load(S0 + i) + vx_load(S2 + i) + (sl + sl), vx_load(S0 + i + v_int32::nlanes) + vx_load(S2 + i + v_int32::nlanes) + (sh + sh)) + d8);
                    i += v_int16::nlanes;
                }
                if( i <= width - v_int32::nlanes )
                {
                    v_int32 s = vx_load(S1 + i);
                    v_pack_store(dst + i, vx_load(S0 + i) + vx_load(S2 + i) + vx_setall_s32(d) + (s + s));
                    i += v_int32::nlanes;
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
                {
                    v_int32 s0 = vx_load(S1 + i);
                    v_int32 s1 = vx_load(S1 + i + v_int32::nlanes);
                    v_int32 s2 = vx_load(S1 + i + 2*v_int32::nlanes);
                    v_int32 s3 = vx_load(S1 + i + 3*v_int32::nlanes);
                    v_store(dst + i, v_pack(vx_load(S0 + i) + vx_load(S2 + i) - (s0 + s0),
                                            vx_load(S0 + i + v_int32::nlanes) + vx_load(S2 + i + v_int32::nlanes) - (s1 + s1)) + d8);
                    v_store(dst + i + v_int16::nlanes, v_pack(vx_load(S0 + i + 2*v_int32::nlanes) + vx_load(S2 + i + 2*v_int32::nlanes) - (s2 + s2),
                                                              vx_load(S0 + i + 3*v_int32::nlanes) + vx_load(S2 + i + 3*v_int32::nlanes) - (s3 + s3)) + d8);
                }
                if( i <= width - v_int16::nlanes )
                {
                    v_int32 sl = vx_load(S1 + i);
                    v_int32 sh = vx_load(S1 + i + v_int32::nlanes);
                    v_store(dst + i, v_pack(vx_load(S0 + i) + vx_load(S2 + i) - (sl + sl), vx_load(S0 + i + v_int32::nlanes) + vx_load(S2 + i + v_int32::nlanes) - (sh + sh)) + d8);
                    i += v_int16::nlanes;
                }
                if( i <= width - v_int32::nlanes )
                {
                    v_int32 s = vx_load(S1 + i);
                    v_pack_store(dst + i, vx_load(S0 + i) + vx_load(S2 + i) + vx_setall_s32(d) - (s + s));
                    i += v_int32::nlanes;
                }
            }
#if CV_NEON
            else if( ky[0] == (float)((int)ky[0]) && ky[1] == (float)((int)ky[1]) )
            {
                v_int32 k0 = vx_setall_s32((int)ky[0]), k1 = vx_setall_s32((int)ky[1]);
                v_int32 d4 = vx_setall_s32(d);
                for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
                {
                    v_store(dst + i, v_pack(v_muladd(vx_load(S0 + i) + vx_load(S2 + i), k1, v_muladd(vx_load(S1 + i), k0, d4)),
                                            v_muladd(vx_load(S0 + i + v_int32::nlanes) + vx_load(S2 + i + v_int32::nlanes), k1, v_muladd(vx_load(S1 + i + v_int32::nlanes), k0, d4))));
                    v_store(dst + i + v_int16::nlanes, v_pack(v_muladd(vx_load(S0 + i + 2*v_int32::nlanes) + vx_load(S2 + i + 2*v_int32::nlanes), k1, v_muladd(vx_load(S1 + i + 2*v_int32::nlanes), k0, d4)),
                                                              v_muladd(vx_load(S0 + i + 3*v_int32::nlanes) + vx_load(S2 + i + 3*v_int32::nlanes), k1, v_muladd(vx_load(S1 + i + 3*v_int32::nlanes), k0, d4))));
                }
                if( i <= width - v_int16::nlanes )
                {
                    v_store(dst + i, v_pack(v_muladd(vx_load(S0 + i) + vx_load(S2 + i), k1, v_muladd(vx_load(S1 + i), k0, d4)),
                                            v_muladd(vx_load(S0 + i + v_int32::nlanes) + vx_load(S2 + i + v_int32::nlanes), k1, v_muladd(vx_load(S1 + i + v_int32::nlanes), k0, d4))));
                    i += v_int16::nlanes;
                }
                if( i <= width - v_int32::nlanes )
                {
                    v_pack_store(dst + i, v_muladd(vx_load(S0 + i) + vx_load(S2 + i), k1, v_muladd(vx_load(S1 + i), k0, d4)));
                    i += v_int32::nlanes;
                }
            }
#endif
            else
            {
                v_float32 k0 = vx_setall_f32(ky[0]), k1 = vx_setall_f32(ky[1]);
                for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
                {
                    v_store(dst + i, v_pack(v_round(v_muladd(v_cvt_f32(vx_load(S0 + i) + vx_load(S2 + i)), k1, v_muladd(v_cvt_f32(vx_load(S1 + i)), k0, df4))),
                                            v_round(v_muladd(v_cvt_f32(vx_load(S0 + i + v_int32::nlanes) + vx_load(S2 + i + v_int32::nlanes)), k1, v_muladd(v_cvt_f32(vx_load(S1 + i + v_int32::nlanes)), k0, df4)))));
                    v_store(dst + i + v_int16::nlanes, v_pack(v_round(v_muladd(v_cvt_f32(vx_load(S0 + i + 2*v_int32::nlanes) + vx_load(S2 + i + 2*v_int32::nlanes)), k1, v_muladd(v_cvt_f32(vx_load(S1 + i + 2*v_int32::nlanes)), k0, df4))),
                                                              v_round(v_muladd(v_cvt_f32(vx_load(S0 + i + 3*v_int32::nlanes) + vx_load(S2 + i + 3*v_int32::nlanes)), k1, v_muladd(v_cvt_f32(vx_load(S1 + i + 3*v_int32::nlanes)), k0, df4)))));
                }
                if( i <= width - v_int16::nlanes )
                {
                    v_store(dst + i, v_pack(v_round(v_muladd(v_cvt_f32(vx_load(S0 + i) + vx_load(S2 + i)), k1, v_muladd(v_cvt_f32(vx_load(S1 + i)), k0, df4))),
                                            v_round(v_muladd(v_cvt_f32(vx_load(S0 + i + v_int32::nlanes) + vx_load(S2 + i + v_int32::nlanes)), k1, v_muladd(v_cvt_f32(vx_load(S1 + i + v_int32::nlanes)), k0, df4)))));
                    i += v_int16::nlanes;
                }
                if( i <= width - v_int32::nlanes )
                {
                    v_pack_store(dst + i, v_round(v_muladd(v_cvt_f32(vx_load(S0 + i) + vx_load(S2 + i)), k1, v_muladd(v_cvt_f32(vx_load(S1 + i)), k0, df4))));
                    i += v_int32::nlanes;
                }
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
                {
                    v_store(dst + i, v_pack(vx_load(S2 + i) - vx_load(S0 + i), vx_load(S2 + i + v_int32::nlanes) - vx_load(S0 + i + v_int32::nlanes)) + d8);
                    v_store(dst + i + v_int16::nlanes, v_pack(vx_load(S2 + i + 2*v_int32::nlanes) - vx_load(S0 + i + 2*v_int32::nlanes), vx_load(S2 + i + 3*v_int32::nlanes) - vx_load(S0 + i + 3*v_int32::nlanes)) + d8);
                }
                if( i <= width - v_int16::nlanes )
                {
                    v_store(dst + i, v_pack(vx_load(S2 + i) - vx_load(S0 + i), vx_load(S2 + i + v_int32::nlanes) - vx_load(S0 + i + v_int32::nlanes)) + d8);
                    i += v_int16::nlanes;
                }
                if( i <= width - v_int32::nlanes )
                {
                    v_pack_store(dst + i, vx_load(S2 + i) - vx_load(S0 + i) + vx_setall_s32(d));
                    i += v_int32::nlanes;
                }
            }
            else
            {
                v_float32 k1 = vx_setall_f32(ky[1]);
                for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
                {
                    v_store(dst + i, v_pack(v_round(v_muladd(v_cvt_f32(vx_load(S2 + i) - vx_load(S0 + i)), k1, df4)),
                                            v_round(v_muladd(v_cvt_f32(vx_load(S2 + i + v_int32::nlanes) - vx_load(S0 + i + v_int32::nlanes)), k1, df4))));
                    v_store(dst + i + v_int16::nlanes, v_pack(v_round(v_muladd(v_cvt_f32(vx_load(S2 + i + 2*v_int32::nlanes) - vx_load(S0 + i + 2*v_int32::nlanes)), k1, df4)),
                                                              v_round(v_muladd(v_cvt_f32(vx_load(S2 + i + 3*v_int32::nlanes) - vx_load(S0 + i + 3*v_int32::nlanes)), k1, df4))));
                }
                if( i <= width - v_int16::nlanes )
                {
                    v_store(dst + i, v_pack(v_round(v_muladd(v_cvt_f32(vx_load(S2 + i) - vx_load(S0 + i)), k1, df4)),
                                            v_round(v_muladd(v_cvt_f32(vx_load(S2 + i + v_int32::nlanes) - vx_load(S0 + i + v_int32::nlanes)), k1, df4))));
                    i += v_int16::nlanes;
                }
                if( i <= width - v_int32::nlanes )
                {
                    v_pack_store(dst + i, v_round(v_muladd(v_cvt_f32(vx_load(S2 + i) - vx_load(S0 + i)), k1, df4)));
                    i += v_int32::nlanes;
                }
            }
        }

        vx_cleanup();
        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


/////////////////////////////////////// 16s //////////////////////////////////

struct RowVec_16s32f
{
    RowVec_16s32f() {}
    RowVec_16s32f( const Mat& _kernel )
    {
        kernel = _kernel;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* _kx = kernel.ptr<float>();
        width *= cn;

        for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
        {
            const short* src = (const short*)_src + i;
            v_float32 s0 = vx_setzero_f32();
            v_float32 s1 = vx_setzero_f32();
            v_float32 s2 = vx_setzero_f32();
            v_float32 s3 = vx_setzero_f32();
            for( k = 0; k < _ksize; k++, src += cn )
            {
                v_float32 f = vx_setall_f32(_kx[k]);
                v_int16 xl = vx_load(src);
                v_int16 xh = vx_load(src + v_int16::nlanes);
                s0 = v_muladd(v_cvt_f32(v_expand_low(xl)), f, s0);
                s1 = v_muladd(v_cvt_f32(v_expand_high(xl)), f, s1);
                s2 = v_muladd(v_cvt_f32(v_expand_low(xh)), f, s2);
                s3 = v_muladd(v_cvt_f32(v_expand_high(xh)), f, s3);
            }
            v_store(dst + i, s0);
            v_store(dst + i + v_float32::nlanes, s1);
            v_store(dst + i + 2*v_float32::nlanes, s2);
            v_store(dst + i + 3*v_float32::nlanes, s3);
        }
        if( i <= width - v_int16::nlanes )
        {
            const short* src = (const short*)_src + i;
            v_float32 s0 = vx_setzero_f32();
            v_float32 s1 = vx_setzero_f32();
            for( k = 0; k < _ksize; k++, src += cn )
            {
                v_float32 f = vx_setall_f32(_kx[k]);
                v_int16 x = vx_load(src);
                s0 = v_muladd(v_cvt_f32(v_expand_low(x)), f, s0);
                s1 = v_muladd(v_cvt_f32(v_expand_high(x)), f, s1);
            }
            v_store(dst + i, s0);
            v_store(dst + i + v_float32::nlanes, s1);
            i += v_int16::nlanes;
        }
        if( i <= width - v_float32::nlanes )
        {
            const short* src = (const short*)_src + i;
            v_float32 s0 = vx_setzero_f32();
            for( k = 0; k < _ksize; k++, src += cn )
                s0 = v_muladd(v_cvt_f32(vx_load_expand(src)), vx_setall_f32(_kx[k]), s0);
            v_store(dst + i, s0);
            i += v_float32::nlanes;
        }
        vx_cleanup();
        return i;
    }

    Mat kernel;
};


struct SymmColumnVec_32f16s
{
    SymmColumnVec_32f16s() { symmetryType=0; delta = 0; }
    SymmColumnVec_32f16s(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        int _ksize = kernel.rows + kernel.cols - 1;
        if( _ksize == 1 )
            return 0;
        int ksize2 = _ksize / 2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        short* dst = (short*)_dst;

        v_float32 d4 = vx_setall_f32(delta);
        if( symmetrical )
        {
            v_float32 k0 = vx_setall_f32(ky[0]);
            v_float32 k1 = vx_setall_f32(ky[1]);
            for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                v_float32 s1 = v_muladd(vx_load(src[0] + i + v_float32::nlanes), k0, d4);
                v_float32 s2 = v_muladd(vx_load(src[0] + i + 2*v_float32::nlanes), k0, d4);
                v_float32 s3 = v_muladd(vx_load(src[0] + i + 3*v_float32::nlanes), k0, d4);
                s0 = v_muladd(vx_load(src[1] + i) + vx_load(src[-1] + i), k1, s0);
                s1 = v_muladd(vx_load(src[1] + i + v_float32::nlanes) + vx_load(src[-1] + i + v_float32::nlanes), k1, s1);
                s2 = v_muladd(vx_load(src[1] + i + 2*v_float32::nlanes) + vx_load(src[-1] + i + 2*v_float32::nlanes), k1, s2);
                s3 = v_muladd(vx_load(src[1] + i + 3*v_float32::nlanes) + vx_load(src[-1] + i + 3*v_float32::nlanes), k1, s3);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(vx_load(src[k] + i) + vx_load(src[-k] + i), k2, s0);
                    s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes) + vx_load(src[-k] + i + v_float32::nlanes), k2, s1);
                    s2 = v_muladd(vx_load(src[k] + i + 2*v_float32::nlanes) + vx_load(src[-k] + i + 2*v_float32::nlanes), k2, s2);
                    s3 = v_muladd(vx_load(src[k] + i + 3*v_float32::nlanes) + vx_load(src[-k] + i + 3*v_float32::nlanes), k2, s3);
                }
                v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                v_store(dst + i + v_int16::nlanes, v_pack(v_round(s2), v_round(s3)));
            }
            if( i <= width - v_int16::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                v_float32 s1 = v_muladd(vx_load(src[0] + i + v_float32::nlanes), k0, d4);
                s0 = v_muladd(vx_load(src[1] + i) + vx_load(src[-1] + i), k1, s0);
                s1 = v_muladd(vx_load(src[1] + i + v_float32::nlanes) + vx_load(src[-1] + i + v_float32::nlanes), k1, s1);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(vx_load(src[k] + i) + vx_load(src[-k] + i), k2, s0);
                    s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes) + vx_load(src[-k] + i + v_float32::nlanes), k2, s1);
                }
                v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                i += v_int16::nlanes;
            }
            if( i <= width - v_float32::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                s0 = v_muladd(vx_load(src[1] + i) + vx_load(src[-1] + i), k1, s0);
                for( k = 2; k <= ksize2; k++ )
                    s0 = v_muladd(vx_load(src[k] + i) + vx_load(src[-k] + i), vx_setall_f32(ky[k]), s0);
                v_pack_store(dst + i, v_round(s0));
                i += v_float32::nlanes;
            }
        }
        else
        {
            v_float32 k1 = vx_setall_f32(ky[1]);
            for( ; i <= width - 2*v_int16::nlanes; i += 2*v_int16::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[1] + i) - vx_load(src[-1] + i), k1, d4);
                v_float32 s1 = v_muladd(vx_load(src[1] + i + v_float32::nlanes) - vx_load(src[-1] + i + v_float32::nlanes), k1, d4);
                v_float32 s2 = v_muladd(vx_load(src[1] + i + 2*v_float32::nlanes) - vx_load(src[-1] + i + 2*v_float32::nlanes), k1, d4);
                v_float32 s3 = v_muladd(vx_load(src[1] + i + 3*v_float32::nlanes) - vx_load(src[-1] + i + 3*v_float32::nlanes), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(vx_load(src[k] + i) - vx_load(src[-k] + i), k2, s0);
                    s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes) - vx_load(src[-k] + i + v_float32::nlanes), k2, s1);
                    s0 = v_muladd(vx_load(src[k] + i + 2*v_float32::nlanes) - vx_load(src[-k] + i + 2*v_float32::nlanes), k2, s2);
                    s1 = v_muladd(vx_load(src[k] + i + 3*v_float32::nlanes) - vx_load(src[-k] + i + 3*v_float32::nlanes), k2, s3);
                }
                v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                v_store(dst + i + v_int16::nlanes, v_pack(v_round(s2), v_round(s3)));
            }
            if( i <= width - v_int16::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[1] + i) - vx_load(src[-1] + i), k1, d4);
                v_float32 s1 = v_muladd(vx_load(src[1] + i + v_float32::nlanes) - vx_load(src[-1] + i + v_float32::nlanes), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(vx_load(src[k] + i) - vx_load(src[-k] + i), k2, s0);
                    s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes) - vx_load(src[-k] + i + v_float32::nlanes), k2, s1);
                }
                v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                i += v_int16::nlanes;
            }
            if( i <= width - v_float32::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[1] + i) - vx_load(src[-1] + i), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                    s0 = v_muladd(vx_load(src[k] + i) - vx_load(src[-k] + i), vx_setall_f32(ky[k]), s0);
                v_pack_store(dst + i, v_round(s0));
                i += v_float32::nlanes;
            }
        }

        vx_cleanup();
        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


/////////////////////////////////////// 32f //////////////////////////////////

struct RowVec_32f
{
    RowVec_32f()
    {
        haveAVX2 = CV_CPU_HAS_SUPPORT_AVX2;
#if defined USE_IPP_SEP_FILTERS
        bufsz = -1;
#endif
    }

    RowVec_32f( const Mat& _kernel )
    {
        kernel = _kernel;
        haveAVX2 = CV_CPU_HAS_SUPPORT_AVX2;
#if defined USE_IPP_SEP_FILTERS
        bufsz = -1;
#endif
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
#if defined USE_IPP_SEP_FILTERS
        CV_IPP_CHECK()
        {
            int ret = ippiOperator(_src, _dst, width, cn);
            if (ret > 0)
                return ret;
        }
#endif
        int _ksize = kernel.rows + kernel.cols - 1;
        CV_DbgAssert(_ksize > 0);
        const float* src0 = (const float*)_src;
        float* dst = (float*)_dst;
        const float* _kx = kernel.ptr<float>();

        int i = 0, k;
        width *= cn;

#if CV_TRY_AVX2
        if (haveAVX2)
            return RowVec_32f_AVX(src0, _kx, dst, width, cn, _ksize);
#endif
        v_float32 k0 = vx_setall_f32(_kx[0]);
        for( ; i <= width - 4*v_float32::nlanes; i += 4*v_float32::nlanes )
        {
            const float* src = src0 + i;
            v_float32 s0 = vx_load(src) * k0;
            v_float32 s1 = vx_load(src + v_float32::nlanes) * k0;
            v_float32 s2 = vx_load(src + 2*v_float32::nlanes) * k0;
            v_float32 s3 = vx_load(src + 3*v_float32::nlanes) * k0;
            src += cn;
            for( k = 1; k < _ksize; k++, src += cn )
            {
                v_float32 k1 = vx_setall_f32(_kx[k]);
                s0 = v_muladd(vx_load(src), k1, s0);
                s1 = v_muladd(vx_load(src + v_float32::nlanes), k1, s1);
                s2 = v_muladd(vx_load(src + 2*v_float32::nlanes), k1, s2);
                s3 = v_muladd(vx_load(src + 3*v_float32::nlanes), k1, s3);
            }
            v_store(dst + i, s0);
            v_store(dst + i + v_float32::nlanes, s1);
            v_store(dst + i + 2*v_float32::nlanes, s2);
            v_store(dst + i + 3*v_float32::nlanes, s3);
        }
        if( i <= width - 2*v_float32::nlanes )
        {
            const float* src = src0 + i;
            v_float32 s0 = vx_load(src) * k0;
            v_float32 s1 = vx_load(src + v_float32::nlanes) * k0;
            src += cn;
            for( k = 1; k < _ksize; k++, src += cn )
            {
                v_float32 k1 = vx_setall_f32(_kx[k]);
                s0 = v_muladd(vx_load(src), k1, s0);
                s1 = v_muladd(vx_load(src + v_float32::nlanes), k1, s1);
            }
            v_store(dst + i, s0);
            v_store(dst + i + v_float32::nlanes, s1);
            i += 2*v_float32::nlanes;
        }
        if( i <= width - v_float32::nlanes )
        {
            const float* src = src0 + i;
            v_float32 s0 = vx_load(src) * k0;
            src += cn;
            for( k = 1; k < _ksize; k++, src += cn )
                s0 = v_muladd(vx_load(src), vx_setall_f32(_kx[k]), s0);
            v_store(dst + i, s0);
            i += v_float32::nlanes;
        }
        vx_cleanup();
        return i;
    }

    Mat kernel;
    bool haveAVX2;
#if defined USE_IPP_SEP_FILTERS
private:
    mutable int bufsz;
    int ippiOperator(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        CV_INSTRUMENT_REGION_IPP();

        int _ksize = kernel.rows + kernel.cols - 1;
        if ((1 != cn && 3 != cn) || width < _ksize*8)
            return 0;

        const float* src = (const float*)_src;
        float* dst = (float*)_dst;
        const float* _kx = (const float*)kernel.data;

        IppiSize roisz = { width, 1 };
        if( bufsz < 0 )
        {
            if( (cn == 1 && ippiFilterRowBorderPipelineGetBufferSize_32f_C1R(roisz, _ksize, &bufsz) < 0) ||
                (cn == 3 && ippiFilterRowBorderPipelineGetBufferSize_32f_C3R(roisz, _ksize, &bufsz) < 0))
                return 0;
        }
        AutoBuffer<uchar> buf(bufsz + 64);
        uchar* bufptr = alignPtr(buf.data(), 32);
        int step = (int)(width*sizeof(dst[0])*cn);
        float borderValue[] = {0.f, 0.f, 0.f};
        // here is the trick. IPP needs border type and extrapolates the row. We did it already.
        // So we pass anchor=0 and ignore the right tail of results since they are incorrect there.
        if( (cn == 1 && CV_INSTRUMENT_FUN_IPP(ippiFilterRowBorderPipeline_32f_C1R, src, step, &dst, roisz, _kx, _ksize, 0,
                                                            ippBorderRepl, borderValue[0], bufptr) < 0) ||
            (cn == 3 && CV_INSTRUMENT_FUN_IPP(ippiFilterRowBorderPipeline_32f_C3R, src, step, &dst, roisz, _kx, _ksize, 0,
                                                            ippBorderRepl, borderValue, bufptr) < 0))
        {
            setIppErrorStatus();
            return 0;
        }
        CV_IMPL_ADD(CV_IMPL_IPP);
        return width - _ksize + 1;
    }
#endif
};


struct SymmRowSmallVec_32f
{
    SymmRowSmallVec_32f() { symmetryType = 0; }
    SymmRowSmallVec_32f( const Mat& _kernel, int _symmetryType )
    {
        kernel = _kernel;
        symmetryType = _symmetryType;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        int i = 0, _ksize = kernel.rows + kernel.cols - 1;
        if( _ksize == 1 )
            return 0;
        float* dst = (float*)_dst;
        const float* src = (const float*)_src + (_ksize/2)*cn;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float* kx = kernel.ptr<float>() + _ksize/2;
        width *= cn;

        if( symmetrical )
        {
            if( _ksize == 3 )
            {
                if( fabs(kx[0]) == 2 && kx[1] == 1 )
                {
#if CV_FMA3 || CV_AVX2
                    v_float32 k0 = vx_setall_f32(kx[0]);
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                        v_store(dst + i, v_muladd(vx_load(src), k0, vx_load(src - cn) + vx_load(src + cn)));
#else
                    if( kx[0] > 0 )
                        for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                        {
                            v_float32 x = vx_load(src);
                            v_store(dst + i, vx_load(src - cn) + vx_load(src + cn) + (x + x));
                        }
                    else
                        for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                        {
                            v_float32 x = vx_load(src);
                            v_store(dst + i, vx_load(src - cn) + vx_load(src + cn) - (x + x));
                        }
#endif
                }
                else
                {
                    v_float32 k0 = vx_setall_f32(kx[0]), k1 = vx_setall_f32(kx[1]);
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                        v_store(dst + i, v_muladd(vx_load(src), k0, (vx_load(src - cn) + vx_load(src + cn)) * k1));
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                {
#if CV_FMA3 || CV_AVX2
                    v_float32 k0 = vx_setall_f32(-2);
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                        v_store(dst + i, v_muladd(vx_load(src), k0, vx_load(src - 2*cn) + vx_load(src + 2*cn)));
#else
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                    {
                        v_float32 x = vx_load(src);
                        v_store(dst + i, vx_load(src - 2*cn) + vx_load(src + 2*cn) - (x + x));
                    }
#endif
                }
                else
                {
                    v_float32 k0 = vx_setall_f32(kx[0]), k1 = vx_setall_f32(kx[1]), k2 = vx_setall_f32(kx[2]);
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                        v_store(dst + i, v_muladd(vx_load(src + 2*cn) + vx_load(src - 2*cn), k2, v_muladd(vx_load(src), k0, (vx_load(src - cn) + vx_load(src + cn)) * k1)));
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                        v_store(dst + i, vx_load(src + cn) - vx_load(src - cn));
                else
                {
                    v_float32 k1 = vx_setall_f32(kx[1]);
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                        v_store(dst + i, (vx_load(src + cn) - vx_load(src - cn)) * k1);
                }
            }
            else if( _ksize == 5 )
            {
                v_float32 k1 = vx_setall_f32(kx[1]), k2 = vx_setall_f32(kx[2]);
                for ( ; i <= width - v_float32::nlanes; i += v_float32::nlanes, src += v_float32::nlanes )
                    v_store(dst + i, v_muladd(vx_load(src + 2*cn) - vx_load(src - 2*cn), k2, (vx_load(src + cn) - vx_load(src - cn)) * k1));
            }
        }

        vx_cleanup();
        return i;
    }

    Mat kernel;
    int symmetryType;
};


struct SymmColumnVec_32f
{
    SymmColumnVec_32f() {
        symmetryType=0;
        haveAVX2 = CV_CPU_HAS_SUPPORT_AVX2;
        delta = 0;
    }
    SymmColumnVec_32f(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        haveAVX2 = CV_CPU_HAS_SUPPORT_AVX2;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        float* dst = (float*)_dst;

        if( symmetrical )
        {

#if CV_TRY_AVX2
            if (haveAVX2)
                return SymmColumnVec_32f_Symm_AVX(src, ky, dst, delta, width, ksize2);
#endif
            const v_float32 d4 = vx_setall_f32(delta);
            const v_float32 k0 = vx_setall_f32(ky[0]);
            for( ; i <= width - 4*v_float32::nlanes; i += 4*v_float32::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                v_float32 s1 = v_muladd(vx_load(src[0] + i + v_float32::nlanes), k0, d4);
                v_float32 s2 = v_muladd(vx_load(src[0] + i + 2*v_float32::nlanes), k0, d4);
                v_float32 s3 = v_muladd(vx_load(src[0] + i + 3*v_float32::nlanes), k0, d4);
                for( k = 1; k <= ksize2; k++ )
                {
                    v_float32 k1 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(vx_load(src[k] + i) + vx_load(src[-k] + i), k1, s0);
                    s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes) + vx_load(src[-k] + i + v_float32::nlanes), k1, s1);
                    s2 = v_muladd(vx_load(src[k] + i + 2*v_float32::nlanes) + vx_load(src[-k] + i + 2*v_float32::nlanes), k1, s2);
                    s3 = v_muladd(vx_load(src[k] + i + 3*v_float32::nlanes) + vx_load(src[-k] + i + 3*v_float32::nlanes), k1, s3);
                }
                v_store(dst + i, s0);
                v_store(dst + i + v_float32::nlanes, s1);
                v_store(dst + i + 2*v_float32::nlanes, s2);
                v_store(dst + i + 3*v_float32::nlanes, s3);
            }
            if( i <= width - 2*v_float32::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                v_float32 s1 = v_muladd(vx_load(src[0] + i + v_float32::nlanes), k0, d4);
                for( k = 1; k <= ksize2; k++ )
                {
                    v_float32 k1 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(vx_load(src[k] + i) + vx_load(src[-k] + i), k1, s0);
                    s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes) + vx_load(src[-k] + i + v_float32::nlanes), k1, s1);
                }
                v_store(dst + i, s0);
                v_store(dst + i + v_float32::nlanes, s1);
                i += 2*v_float32::nlanes;
            }
            if( i <= width - v_float32::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                for( k = 1; k <= ksize2; k++ )
                    s0 = v_muladd(vx_load(src[k] + i) + vx_load(src[-k] + i), vx_setall_f32(ky[k]), s0);
                v_store(dst + i, s0);
                i += v_float32::nlanes;
            }
        }
        else
        {
#if CV_TRY_AVX2
            if (haveAVX2)
                return SymmColumnVec_32f_Unsymm_AVX(src, ky, dst, delta, width, ksize2);
#endif
            CV_DbgAssert(ksize2 > 0);
            const v_float32 d4 = vx_setall_f32(delta);
            const v_float32 k1 = vx_setall_f32(ky[1]);
            for( ; i <= width - 4*v_float32::nlanes; i += 4*v_float32::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[1] + i) - vx_load(src[-1] + i), k1, d4);
                v_float32 s1 = v_muladd(vx_load(src[1] + i + v_float32::nlanes) - vx_load(src[-1] + i + v_float32::nlanes), k1, d4);
                v_float32 s2 = v_muladd(vx_load(src[1] + i + 2*v_float32::nlanes) - vx_load(src[-1] + i + 2*v_float32::nlanes), k1, d4);
                v_float32 s3 = v_muladd(vx_load(src[1] + i + 3*v_float32::nlanes) - vx_load(src[-1] + i + 3*v_float32::nlanes), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(vx_load(src[k] + i) - vx_load(src[-k] + i), k2, s0);
                    s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes) - vx_load(src[-k] + i + v_float32::nlanes), k2, s1);
                    s2 = v_muladd(vx_load(src[k] + i + 2*v_float32::nlanes) - vx_load(src[-k] + i + 2*v_float32::nlanes), k2, s2);
                    s3 = v_muladd(vx_load(src[k] + i + 3*v_float32::nlanes) - vx_load(src[-k] + i + 3*v_float32::nlanes), k2, s3);
                }
                v_store(dst + i, s0);
                v_store(dst + i + v_float32::nlanes, s1);
                v_store(dst + i + 2*v_float32::nlanes, s2);
                v_store(dst + i + 3*v_float32::nlanes, s3);
            }
            if( i <= width - 2*v_float32::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[1] + i) - vx_load(src[-1] + i), k1, d4);
                v_float32 s1 = v_muladd(vx_load(src[1] + i + v_float32::nlanes) - vx_load(src[-1] + i + v_float32::nlanes), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(vx_load(src[k] + i) - vx_load(src[-k] + i), k2, s0);
                    s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes) - vx_load(src[-k] + i + v_float32::nlanes), k2, s1);
                }
                v_store(dst + i, s0);
                v_store(dst + i + v_float32::nlanes, s1);
                i += 2*v_float32::nlanes;
            }
            if( i <= width - v_float32::nlanes )
            {
                v_float32 s0 = v_muladd(vx_load(src[1] + i) - vx_load(src[-1] + i), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                    s0 = v_muladd(vx_load(src[k] + i) - vx_load(src[-k] + i), vx_setall_f32(ky[k]), s0);
                v_store(dst + i, s0);
                i += v_float32::nlanes;
            }
        }

        vx_cleanup();
        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
    bool haveAVX2;
};


struct SymmColumnSmallVec_32f
{
    SymmColumnSmallVec_32f() { symmetryType=0; delta = 0; }
    SymmColumnSmallVec_32f(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        float* dst = (float*)_dst;

        v_float32 d4 = vx_setall_f32(delta);
        if( symmetrical )
        {
            if( fabs(ky[0]) == 2 && ky[1] == 1 )
            {
#if CV_FMA3 || CV_AVX2
                v_float32 k0 = vx_setall_f32(ky[0]);
                for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes )
                    v_store(dst + i, v_muladd(vx_load(S1 + i), k0, vx_load(S0 + i) + vx_load(S2 + i) + d4));
#else
                if(ky[0] > 0)
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes )
                    {
                        v_float32 x = vx_load(S1 + i);
                        v_store(dst + i, vx_load(S0 + i) + vx_load(S2 + i) + d4 + (x + x));
                    }
                else
                    for( ; i <= width - v_float32::nlanes; i += v_float32::nlanes )
                    {
                        v_float32 x = vx_load(S1 + i);
                        v_store(dst + i, vx_load(S0 + i) + vx_load(S2 + i) + d4 - (x + x));
                    }
#endif
            }
            else
            {
                v_float32 k0 = vx_setall_f32(ky[0]), k1 = vx_setall_f32(ky[1]);
                for ( ; i <= width - v_float32::nlanes; i += v_float32::nlanes )
                    v_store(dst + i, v_muladd(vx_load(S0 + i) + vx_load(S2 + i), k1, v_muladd(vx_load(S1 + i), k0, d4)));
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for ( ; i <= width - v_float32::nlanes; i += v_float32::nlanes )
                    v_store(dst + i, vx_load(S2 + i) - vx_load(S0 + i) + d4);
            }
            else
            {
                v_float32 k1 = vx_setall_f32(ky[1]);
                for ( ; i <= width - v_float32::nlanes; i += v_float32::nlanes )
                    v_store(dst + i, v_muladd(vx_load(S2 + i) - vx_load(S0 + i), k1, d4));
            }
        }

        vx_cleanup();
        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


/////////////////////////////// non-separable filters ///////////////////////////////

///////////////////////////////// 8u<->8u, 8u<->16s /////////////////////////////////

struct FilterVec_8u
{
    FilterVec_8u() { delta = 0; _nz = 0; }
    FilterVec_8u(const Mat& _kernel, int _bits, double _delta)
    {
        Mat kernel;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        std::vector<Point> coords;
        preprocess2DKernel(kernel, coords, coeffs);
        _nz = (int)coords.size();
    }

    int operator()(const uchar** src, uchar* dst, int width) const
    {
        CV_DbgAssert(_nz > 0);
        const float* kf = (const float*)&coeffs[0];
        int i = 0, k, nz = _nz;

        v_float32 d4 = vx_setall_f32(delta);
        v_float32 f0 = vx_setall_f32(kf[0]);
        for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes )
        {
            v_uint16 xl, xh;
            v_expand(vx_load(src[0] + i), xl, xh);
            v_uint32 x0, x1, x2, x3;
            v_expand(xl, x0, x1);
            v_expand(xh, x2, x3);
            v_float32 s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x0)), f0, d4);
            v_float32 s1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x1)), f0, d4);
            v_float32 s2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x2)), f0, d4);
            v_float32 s3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x3)), f0, d4);
            for( k = 1; k < nz; k++ )
            {
                v_float32 f = vx_setall_f32(kf[k]);
                v_expand(vx_load(src[k] + i), xl, xh);
                v_expand(xl, x0, x1);
                v_expand(xh, x2, x3);
                s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x0)), f, s0);
                s1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x1)), f, s1);
                s2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x2)), f, s2);
                s3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x3)), f, s3);
            }
            v_store(dst + i, v_pack_u(v_pack(v_round(s0), v_round(s1)), v_pack(v_round(s2), v_round(s3))));
        }
        if( i <= width - v_uint16::nlanes )
        {
            v_uint32 x0, x1;
            v_expand(vx_load_expand(src[0] + i), x0, x1);
            v_float32 s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x0)), f0, d4);
            v_float32 s1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x1)), f0, d4);
            for( k = 1; k < nz; k++ )
            {
                v_float32 f = vx_setall_f32(kf[k]);
                v_expand(vx_load_expand(src[k] + i), x0, x1);
                s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x0)), f, s0);
                s1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(x1)), f, s1);
            }
            v_pack_u_store(dst + i, v_pack(v_round(s0), v_round(s1)));
            i += v_uint16::nlanes;
        }
#if CV_SIMD_WIDTH > 16
        while( i <= width - v_int32x4::nlanes )
#else
        if( i <= width - v_int32x4::nlanes )
#endif
        {
            v_float32x4 s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(src[0] + i))), v_setall_f32(kf[0]), v_setall_f32(delta));
            for( k = 1; k < nz; k++ )
                s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(src[k] + i))), v_setall_f32(kf[k]), s0);
            v_int32x4 s32 = v_round(s0);
            v_int16x8 s16 = v_pack(s32, s32);
            *(int*)(dst + i) = v_reinterpret_as_s32(v_pack_u(s16, s16)).get0();
            i += v_int32x4::nlanes;
        }

        vx_cleanup();
        return i;
    }

    int _nz;
    std::vector<uchar> coeffs;
    float delta;
};


struct FilterVec_8u16s
{
    FilterVec_8u16s() { delta = 0; _nz = 0; }
    FilterVec_8u16s(const Mat& _kernel, int _bits, double _delta)
    {
        Mat kernel;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        std::vector<Point> coords;
        preprocess2DKernel(kernel, coords, coeffs);
        _nz = (int)coords.size();
    }

    int operator()(const uchar** src, uchar* _dst, int width) const
    {
        CV_DbgAssert(_nz > 0);
        const float* kf = (const float*)&coeffs[0];
        short* dst = (short*)_dst;
        int i = 0, k, nz = _nz;

        v_float32 d4 = vx_setall_f32(delta);
        v_float32 f0 = vx_setall_f32(kf[0]);
        for( ; i <= width - v_uint8::nlanes; i += v_uint8::nlanes )
        {
            v_uint16 xl, xh;
            v_expand(vx_load(src[0] + i), xl, xh);
            v_float32 s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_low(xl))), f0, d4);
            v_float32 s1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_high(xl))), f0, d4);
            v_float32 s2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_low(xh))), f0, d4);
            v_float32 s3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_high(xh))), f0, d4);
            for( k = 1; k < nz; k++ )
            {
                v_float32 f = vx_setall_f32(kf[k]);
                v_expand(vx_load(src[k] + i), xl, xh);
                s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_low(xl))), f, s0);
                s1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_high(xl))), f, s1);
                s2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_low(xh))), f, s2);
                s3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_high(xh))), f, s3);
            }
            v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
            v_store(dst + i + v_int16::nlanes, v_pack(v_round(s2), v_round(s3)));
        }
        if( i <= width - v_uint16::nlanes )
        {
            v_uint16 x = vx_load_expand(src[0] + i);
            v_float32 s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_low(x))), f0, d4);
            v_float32 s1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_high(x))), f0, d4);
            for( k = 1; k < nz; k++ )
            {
                v_float32 f = vx_setall_f32(kf[k]);
                x = vx_load_expand(src[k] + i);
                s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_low(x))), f, s0);
                s1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(v_expand_high(x))), f, s1);
            }
            v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
            i += v_uint16::nlanes;
        }
        if( i <= width - v_int32::nlanes )
        {
            v_float32 s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(src[0] + i))), f0, d4);
            for( k = 1; k < nz; k++ )
                s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(src[k] + i))), vx_setall_f32(kf[k]), s0);
            v_pack_store(dst + i, v_round(s0));
            i += v_int32::nlanes;
        }

        vx_cleanup();
        return i;
    }

    int _nz;
    std::vector<uchar> coeffs;
    float delta;
};


struct FilterVec_32f
{
    FilterVec_32f() { delta = 0; _nz = 0; }
    FilterVec_32f(const Mat& _kernel, int, double _delta)
    {
        delta = (float)_delta;
        std::vector<Point> coords;
        preprocess2DKernel(_kernel, coords, coeffs);
        _nz = (int)coords.size();
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        const float* kf = (const float*)&coeffs[0];
        const float** src = (const float**)_src;
        float* dst = (float*)_dst;
        int i = 0, k, nz = _nz;

        v_float32 d4 = vx_setall_f32(delta);
        v_float32 f0 = vx_setall_f32(kf[0]);
        for( ; i <= width - 4*v_float32::nlanes; i += 4*v_float32::nlanes )
        {
            v_float32 s0 = v_muladd(vx_load(src[0] + i), f0, d4);
            v_float32 s1 = v_muladd(vx_load(src[0] + i + v_float32::nlanes), f0, d4);
            v_float32 s2 = v_muladd(vx_load(src[0] + i + 2*v_float32::nlanes), f0, d4);
            v_float32 s3 = v_muladd(vx_load(src[0] + i + 3*v_float32::nlanes), f0, d4);
            for( k = 1; k < nz; k++ )
            {
                v_float32 f1 = vx_setall_f32(kf[k]);
                s0 = v_muladd(vx_load(src[k] + i), f1, s0);
                s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes), f1, s1);
                s2 = v_muladd(vx_load(src[k] + i + 2*v_float32::nlanes), f1, s2);
                s3 = v_muladd(vx_load(src[k] + i + 3*v_float32::nlanes), f1, s3);
            }
            v_store(dst + i, s0);
            v_store(dst + i + v_float32::nlanes, s1);
            v_store(dst + i + 2*v_float32::nlanes, s2);
            v_store(dst + i + 3*v_float32::nlanes, s3);
        }
        if( i <= width - 2*v_float32::nlanes )
        {
            v_float32 s0 = v_muladd(vx_load(src[0] + i), f0, d4);
            v_float32 s1 = v_muladd(vx_load(src[0] + i + v_float32::nlanes), f0, d4);
            for( k = 1; k < nz; k++ )
            {
                v_float32 f1 = vx_setall_f32(kf[k]);
                s0 = v_muladd(vx_load(src[k] + i), f1, s0);
                s1 = v_muladd(vx_load(src[k] + i + v_float32::nlanes), f1, s1);
            }
            v_store(dst + i, s0);
            v_store(dst + i + v_float32::nlanes, s1);
            i += 2*v_float32::nlanes;
        }
        if( i <= width - v_float32::nlanes )
        {
            v_float32 s0 = v_muladd(vx_load(src[0] + i), f0, d4);
            for( k = 1; k < nz; k++ )
                s0 = v_muladd(vx_load(src[k] + i), vx_setall_f32(kf[k]), s0);
            v_store(dst + i, s0);
            i += v_float32::nlanes;
        }

        vx_cleanup();
        return i;
    }

    int _nz;
    std::vector<uchar> coeffs;
    float delta;
};

#else

typedef RowNoVec RowVec_8u32s;
typedef RowNoVec RowVec_16s32f;
typedef RowNoVec RowVec_32f;
typedef SymmRowSmallNoVec SymmRowSmallVec_8u32s;
typedef SymmRowSmallNoVec SymmRowSmallVec_32f;
typedef ColumnNoVec SymmColumnVec_32s8u;
typedef ColumnNoVec SymmColumnVec_32f16s;
typedef ColumnNoVec SymmColumnVec_32f;
typedef SymmColumnSmallNoVec SymmColumnSmallVec_32s16s;
typedef SymmColumnSmallNoVec SymmColumnSmallVec_32f;
typedef FilterNoVec FilterVec_8u;
typedef FilterNoVec FilterVec_8u16s;
typedef FilterNoVec FilterVec_32f;

#endif


template<typename ST, typename DT, class VecOp> struct RowFilter : public BaseRowFilter
{
    RowFilter( const Mat& _kernel, int _anchor, const VecOp& _vecOp=VecOp() )
    {
        if( _kernel.isContinuous() )
            kernel = _kernel;
        else
            _kernel.copyTo(kernel);
        anchor = _anchor;
        ksize = kernel.rows + kernel.cols - 1;
        CV_Assert( kernel.type() == DataType<DT>::type &&
                   (kernel.rows == 1 || kernel.cols == 1));
        vecOp = _vecOp;
    }

    void operator()(const uchar* src, uchar* dst, int width, int cn) CV_OVERRIDE
    {
        int _ksize = ksize;
        const DT* kx = kernel.ptr<DT>();
        const ST* S;
        DT* D = (DT*)dst;
        int i, k;

        i = vecOp(src, dst, width, cn);
        width *= cn;
        #if CV_ENABLE_UNROLLED
        for( ; i <= width - 4; i += 4 )
        {
            S = (const ST*)src + i;
            DT f = kx[0];
            DT s0 = f*S[0], s1 = f*S[1], s2 = f*S[2], s3 = f*S[3];

            for( k = 1; k < _ksize; k++ )
            {
                S += cn;
                f = kx[k];
                s0 += f*S[0]; s1 += f*S[1];
                s2 += f*S[2]; s3 += f*S[3];
            }

            D[i] = s0; D[i+1] = s1;
            D[i+2] = s2; D[i+3] = s3;
        }
        #endif
        for( ; i < width; i++ )
        {
            S = (const ST*)src + i;
            DT s0 = kx[0]*S[0];
            for( k = 1; k < _ksize; k++ )
            {
                S += cn;
                s0 += kx[k]*S[0];
            }
            D[i] = s0;
        }
    }

    Mat kernel;
    VecOp vecOp;
};


template<typename ST, typename DT, class VecOp> struct SymmRowSmallFilter :
    public RowFilter<ST, DT, VecOp>
{
    SymmRowSmallFilter( const Mat& _kernel, int _anchor, int _symmetryType,
                        const VecOp& _vecOp = VecOp())
        : RowFilter<ST, DT, VecOp>( _kernel, _anchor, _vecOp )
    {
        symmetryType = _symmetryType;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 && this->ksize <= 5 );
    }

    void operator()(const uchar* src, uchar* dst, int width, int cn) CV_OVERRIDE
    {
        int ksize2 = this->ksize/2, ksize2n = ksize2*cn;
        const DT* kx = this->kernel.template ptr<DT>() + ksize2;
        bool symmetrical = (this->symmetryType & KERNEL_SYMMETRICAL) != 0;
        DT* D = (DT*)dst;
        int i = this->vecOp(src, dst, width, cn), j, k;
        const ST* S = (const ST*)src + i + ksize2n;
        width *= cn;

        if( symmetrical )
        {
            if( this->ksize == 1 && kx[0] == 1 )
            {
                for( ; i <= width - 2; i += 2 )
                {
                    DT s0 = S[i], s1 = S[i+1];
                    D[i] = s0; D[i+1] = s1;
                }
                S += i;
            }
            else if( this->ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[-cn] + S[0]*2 + S[cn], s1 = S[1-cn] + S[1]*2 + S[1+cn];
                        D[i] = s0; D[i+1] = s1;
                    }
                else if( kx[0] == -2 && kx[1] == 1 )
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[-cn] - S[0]*2 + S[cn], s1 = S[1-cn] - S[1]*2 + S[1+cn];
                        D[i] = s0; D[i+1] = s1;
                    }
                else
                {
                    DT k0 = kx[0], k1 = kx[1];
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[0]*k0 + (S[-cn] + S[cn])*k1, s1 = S[1]*k0 + (S[1-cn] + S[1+cn])*k1;
                        D[i] = s0; D[i+1] = s1;
                    }
                }
            }
            else if( this->ksize == 5 )
            {
                DT k0 = kx[0], k1 = kx[1], k2 = kx[2];
                if( k0 == -2 && k1 == 0 && k2 == 1 )
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = -2*S[0] + S[-cn*2] + S[cn*2];
                        DT s1 = -2*S[1] + S[1-cn*2] + S[1+cn*2];
                        D[i] = s0; D[i+1] = s1;
                    }
                else
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[0]*k0 + (S[-cn] + S[cn])*k1 + (S[-cn*2] + S[cn*2])*k2;
                        DT s1 = S[1]*k0 + (S[1-cn] + S[1+cn])*k1 + (S[1-cn*2] + S[1+cn*2])*k2;
                        D[i] = s0; D[i+1] = s1;
                    }
            }

            for( ; i < width; i++, S++ )
            {
                DT s0 = kx[0]*S[0];
                for( k = 1, j = cn; k <= ksize2; k++, j += cn )
                    s0 += kx[k]*(S[j] + S[-j]);
                D[i] = s0;
            }
        }
        else
        {
            if( this->ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[cn] - S[-cn], s1 = S[1+cn] - S[1-cn];
                        D[i] = s0; D[i+1] = s1;
                    }
                else
                {
                    DT k1 = kx[1];
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = (S[cn] - S[-cn])*k1, s1 = (S[1+cn] - S[1-cn])*k1;
                        D[i] = s0; D[i+1] = s1;
                    }
                }
            }
            else if( this->ksize == 5 )
            {
                DT k1 = kx[1], k2 = kx[2];
                for( ; i <= width - 2; i += 2, S += 2 )
                {
                    DT s0 = (S[cn] - S[-cn])*k1 + (S[cn*2] - S[-cn*2])*k2;
                    DT s1 = (S[1+cn] - S[1-cn])*k1 + (S[1+cn*2] - S[1-cn*2])*k2;
                    D[i] = s0; D[i+1] = s1;
                }
            }

            for( ; i < width; i++, S++ )
            {
                DT s0 = kx[0]*S[0];
                for( k = 1, j = cn; k <= ksize2; k++, j += cn )
                    s0 += kx[k]*(S[j] - S[-j]);
                D[i] = s0;
            }
        }
    }

    int symmetryType;
};


template<class CastOp, class VecOp> struct ColumnFilter : public BaseColumnFilter
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    ColumnFilter( const Mat& _kernel, int _anchor,
        double _delta, const CastOp& _castOp=CastOp(),
        const VecOp& _vecOp=VecOp() )
    {
        if( _kernel.isContinuous() )
            kernel = _kernel;
        else
            _kernel.copyTo(kernel);
        anchor = _anchor;
        ksize = kernel.rows + kernel.cols - 1;
        delta = saturate_cast<ST>(_delta);
        castOp0 = _castOp;
        vecOp = _vecOp;
        CV_Assert( kernel.type() == DataType<ST>::type &&
                   (kernel.rows == 1 || kernel.cols == 1));
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        const ST* ky = kernel.template ptr<ST>();
        ST _delta = delta;
        int _ksize = ksize;
        int i, k;
        CastOp castOp = castOp0;

        for( ; count--; dst += dststep, src++ )
        {
            DT* D = (DT*)dst;
            i = vecOp(src, dst, width);
            #if CV_ENABLE_UNROLLED
            for( ; i <= width - 4; i += 4 )
            {
                ST f = ky[0];
                const ST* S = (const ST*)src[0] + i;
                ST s0 = f*S[0] + _delta, s1 = f*S[1] + _delta,
                    s2 = f*S[2] + _delta, s3 = f*S[3] + _delta;

                for( k = 1; k < _ksize; k++ )
                {
                    S = (const ST*)src[k] + i; f = ky[k];
                    s0 += f*S[0]; s1 += f*S[1];
                    s2 += f*S[2]; s3 += f*S[3];
                }

                D[i] = castOp(s0); D[i+1] = castOp(s1);
                D[i+2] = castOp(s2); D[i+3] = castOp(s3);
            }
            #endif
            for( ; i < width; i++ )
            {
                ST s0 = ky[0]*((const ST*)src[0])[i] + _delta;
                for( k = 1; k < _ksize; k++ )
                    s0 += ky[k]*((const ST*)src[k])[i];
                D[i] = castOp(s0);
            }
        }
    }

    Mat kernel;
    CastOp castOp0;
    VecOp vecOp;
    ST delta;
};


template<class CastOp, class VecOp> struct SymmColumnFilter : public ColumnFilter<CastOp, VecOp>
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    SymmColumnFilter( const Mat& _kernel, int _anchor,
        double _delta, int _symmetryType,
        const CastOp& _castOp=CastOp(),
        const VecOp& _vecOp=VecOp())
        : ColumnFilter<CastOp, VecOp>( _kernel, _anchor, _delta, _castOp, _vecOp )
    {
        symmetryType = _symmetryType;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        int ksize2 = this->ksize/2;
        const ST* ky = this->kernel.template ptr<ST>() + ksize2;
        int i, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        ST _delta = this->delta;
        CastOp castOp = this->castOp0;
        src += ksize2;

        if( symmetrical )
        {
            for( ; count--; dst += dststep, src++ )
            {
                DT* D = (DT*)dst;
                i = (this->vecOp)(src, dst, width);
                #if CV_ENABLE_UNROLLED
                for( ; i <= width - 4; i += 4 )
                {
                    ST f = ky[0];
                    const ST* S = (const ST*)src[0] + i, *S2;
                    ST s0 = f*S[0] + _delta, s1 = f*S[1] + _delta,
                        s2 = f*S[2] + _delta, s3 = f*S[3] + _delta;

                    for( k = 1; k <= ksize2; k++ )
                    {
                        S = (const ST*)src[k] + i;
                        S2 = (const ST*)src[-k] + i;
                        f = ky[k];
                        s0 += f*(S[0] + S2[0]);
                        s1 += f*(S[1] + S2[1]);
                        s2 += f*(S[2] + S2[2]);
                        s3 += f*(S[3] + S2[3]);
                    }

                    D[i] = castOp(s0); D[i+1] = castOp(s1);
                    D[i+2] = castOp(s2); D[i+3] = castOp(s3);
                }
                #endif
                for( ; i < width; i++ )
                {
                    ST s0 = ky[0]*((const ST*)src[0])[i] + _delta;
                    for( k = 1; k <= ksize2; k++ )
                        s0 += ky[k]*(((const ST*)src[k])[i] + ((const ST*)src[-k])[i]);
                    D[i] = castOp(s0);
                }
            }
        }
        else
        {
            for( ; count--; dst += dststep, src++ )
            {
                DT* D = (DT*)dst;
                i = this->vecOp(src, dst, width);
                #if CV_ENABLE_UNROLLED
                for( ; i <= width - 4; i += 4 )
                {
                    ST f = ky[0];
                    const ST *S, *S2;
                    ST s0 = _delta, s1 = _delta, s2 = _delta, s3 = _delta;

                    for( k = 1; k <= ksize2; k++ )
                    {
                        S = (const ST*)src[k] + i;
                        S2 = (const ST*)src[-k] + i;
                        f = ky[k];
                        s0 += f*(S[0] - S2[0]);
                        s1 += f*(S[1] - S2[1]);
                        s2 += f*(S[2] - S2[2]);
                        s3 += f*(S[3] - S2[3]);
                    }

                    D[i] = castOp(s0); D[i+1] = castOp(s1);
                    D[i+2] = castOp(s2); D[i+3] = castOp(s3);
                }
                #endif
                for( ; i < width; i++ )
                {
                    ST s0 = _delta;
                    for( k = 1; k <= ksize2; k++ )
                        s0 += ky[k]*(((const ST*)src[k])[i] - ((const ST*)src[-k])[i]);
                    D[i] = castOp(s0);
                }
            }
        }
    }

    int symmetryType;
};


template<class CastOp, class VecOp>
struct SymmColumnSmallFilter : public SymmColumnFilter<CastOp, VecOp>
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    SymmColumnSmallFilter( const Mat& _kernel, int _anchor,
                           double _delta, int _symmetryType,
                           const CastOp& _castOp=CastOp(),
                           const VecOp& _vecOp=VecOp())
        : SymmColumnFilter<CastOp, VecOp>( _kernel, _anchor, _delta, _symmetryType, _castOp, _vecOp )
    {
        CV_Assert( this->ksize == 3 );
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        int ksize2 = this->ksize/2;
        const ST* ky = this->kernel.template ptr<ST>() + ksize2;
        int i;
        bool symmetrical = (this->symmetryType & KERNEL_SYMMETRICAL) != 0;
        bool is_1_2_1 = ky[0] == 2 && ky[1] == 1;
        bool is_1_m2_1 = ky[0] == -2 && ky[1] == 1;
        bool is_m1_0_1 = ky[0] == 0 && (ky[1] == 1 || ky[1] == -1);
        ST f0 = ky[0], f1 = ky[1];
        ST _delta = this->delta;
        CastOp castOp = this->castOp0;
        src += ksize2;

        for( ; count--; dst += dststep, src++ )
        {
            DT* D = (DT*)dst;
            i = (this->vecOp)(src, dst, width);
            const ST* S0 = (const ST*)src[-1];
            const ST* S1 = (const ST*)src[0];
            const ST* S2 = (const ST*)src[1];

            if( symmetrical )
            {
                if( is_1_2_1 )
                {
                    #if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = S0[i] + S1[i]*2 + S2[i] + _delta;
                        ST s1 = S0[i+1] + S1[i+1]*2 + S2[i+1] + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = S0[i+2] + S1[i+2]*2 + S2[i+2] + _delta;
                        s1 = S0[i+3] + S1[i+3]*2 + S2[i+3] + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #endif
                    for( ; i < width; i ++ )
                    {
                        ST s0 = S0[i] + S1[i]*2 + S2[i] + _delta;
                        D[i] = castOp(s0);
                    }
                }
                else if( is_1_m2_1 )
                {
                    #if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = S0[i] - S1[i]*2 + S2[i] + _delta;
                        ST s1 = S0[i+1] - S1[i+1]*2 + S2[i+1] + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = S0[i+2] - S1[i+2]*2 + S2[i+2] + _delta;
                        s1 = S0[i+3] - S1[i+3]*2 + S2[i+3] + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #endif
                    for( ; i < width; i ++ )
                    {
                        ST s0 = S0[i] - S1[i]*2 + S2[i] + _delta;
                        D[i] = castOp(s0);
                    }
                }
                else
                {
                    #if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = (S0[i] + S2[i])*f1 + S1[i]*f0 + _delta;
                        ST s1 = (S0[i+1] + S2[i+1])*f1 + S1[i+1]*f0 + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = (S0[i+2] + S2[i+2])*f1 + S1[i+2]*f0 + _delta;
                        s1 = (S0[i+3] + S2[i+3])*f1 + S1[i+3]*f0 + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #endif
                    for( ; i < width; i ++ )
                    {
                        ST s0 = (S0[i] + S2[i])*f1 + S1[i]*f0 + _delta;
                        D[i] = castOp(s0);
                    }
                }
            }
            else
            {
                if( is_m1_0_1 )
                {
                    if( f1 < 0 )
                        std::swap(S0, S2);
                    #if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = S2[i] - S0[i] + _delta;
                        ST s1 = S2[i+1] - S0[i+1] + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = S2[i+2] - S0[i+2] + _delta;
                        s1 = S2[i+3] - S0[i+3] + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #endif
                    for( ; i < width; i ++ )
                    {
                        ST s0 = S2[i] - S0[i] + _delta;
                        D[i] = castOp(s0);
                    }
                    if( f1 < 0 )
                        std::swap(S0, S2);
                }
                else
                {
                    #if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = (S2[i] - S0[i])*f1 + _delta;
                        ST s1 = (S2[i+1] - S0[i+1])*f1 + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = (S2[i+2] - S0[i+2])*f1 + _delta;
                        s1 = (S2[i+3] - S0[i+3])*f1 + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #endif
                    for( ; i < width; i++ )
                        D[i] = castOp((S2[i] - S0[i])*f1 + _delta);
                }
            }
        }
    }
};

template<typename ST, typename DT> struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits-1) };

    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
};

template<typename ST, typename DT> struct FixedPtCastEx
{
    typedef ST type1;
    typedef DT rtype;

    FixedPtCastEx() : SHIFT(0), DELTA(0) {}
    FixedPtCastEx(int bits) : SHIFT(bits), DELTA(bits ? 1 << (bits-1) : 0) {}
    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
    int SHIFT, DELTA;
};

}

cv::Ptr<cv::BaseRowFilter> cv::getLinearRowFilter( int srcType, int bufType,
                                                   InputArray _kernel, int anchor,
                                                   int symmetryType )
{
    Mat kernel = _kernel.getMat();
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(bufType);
    int cn = CV_MAT_CN(srcType);
    CV_Assert( cn == CV_MAT_CN(bufType) &&
        ddepth >= std::max(sdepth, CV_32S) &&
        kernel.type() == ddepth );
    int ksize = kernel.rows + kernel.cols - 1;

    if( (symmetryType & (KERNEL_SYMMETRICAL|KERNEL_ASYMMETRICAL)) != 0 && ksize <= 5 )
    {
        if( sdepth == CV_8U && ddepth == CV_32S )
            return makePtr<SymmRowSmallFilter<uchar, int, SymmRowSmallVec_8u32s> >
                (kernel, anchor, symmetryType, SymmRowSmallVec_8u32s(kernel, symmetryType));
        if( sdepth == CV_32F && ddepth == CV_32F )
            return makePtr<SymmRowSmallFilter<float, float, SymmRowSmallVec_32f> >
                (kernel, anchor, symmetryType, SymmRowSmallVec_32f(kernel, symmetryType));
    }

    if( sdepth == CV_8U && ddepth == CV_32S )
        return makePtr<RowFilter<uchar, int, RowVec_8u32s> >
            (kernel, anchor, RowVec_8u32s(kernel));
    if( sdepth == CV_8U && ddepth == CV_32F )
        return makePtr<RowFilter<uchar, float, RowNoVec> >(kernel, anchor);
    if( sdepth == CV_8U && ddepth == CV_64F )
        return makePtr<RowFilter<uchar, double, RowNoVec> >(kernel, anchor);
    if( sdepth == CV_16U && ddepth == CV_32F )
        return makePtr<RowFilter<ushort, float, RowNoVec> >(kernel, anchor);
    if( sdepth == CV_16U && ddepth == CV_64F )
        return makePtr<RowFilter<ushort, double, RowNoVec> >(kernel, anchor);
    if( sdepth == CV_16S && ddepth == CV_32F )
        return makePtr<RowFilter<short, float, RowVec_16s32f> >
                                  (kernel, anchor, RowVec_16s32f(kernel));
    if( sdepth == CV_16S && ddepth == CV_64F )
        return makePtr<RowFilter<short, double, RowNoVec> >(kernel, anchor);
    if( sdepth == CV_32F && ddepth == CV_32F )
        return makePtr<RowFilter<float, float, RowVec_32f> >
            (kernel, anchor, RowVec_32f(kernel));
    if( sdepth == CV_32F && ddepth == CV_64F )
        return makePtr<RowFilter<float, double, RowNoVec> >(kernel, anchor);
    if( sdepth == CV_64F && ddepth == CV_64F )
        return makePtr<RowFilter<double, double, RowNoVec> >(kernel, anchor);

    CV_Error_( CV_StsNotImplemented,
        ("Unsupported combination of source format (=%d), and buffer format (=%d)",
        srcType, bufType));
}


cv::Ptr<cv::BaseColumnFilter> cv::getLinearColumnFilter( int bufType, int dstType,
                                             InputArray _kernel, int anchor,
                                             int symmetryType, double delta,
                                             int bits )
{
    Mat kernel = _kernel.getMat();
    int sdepth = CV_MAT_DEPTH(bufType), ddepth = CV_MAT_DEPTH(dstType);
    int cn = CV_MAT_CN(dstType);
    CV_Assert( cn == CV_MAT_CN(bufType) &&
        sdepth >= std::max(ddepth, CV_32S) &&
        kernel.type() == sdepth );

    if( !(symmetryType & (KERNEL_SYMMETRICAL|KERNEL_ASYMMETRICAL)) )
    {
        if( ddepth == CV_8U && sdepth == CV_32S )
            return makePtr<ColumnFilter<FixedPtCastEx<int, uchar>, ColumnNoVec> >
            (kernel, anchor, delta, FixedPtCastEx<int, uchar>(bits));
        if( ddepth == CV_8U && sdepth == CV_32F )
            return makePtr<ColumnFilter<Cast<float, uchar>, ColumnNoVec> >(kernel, anchor, delta);
        if( ddepth == CV_8U && sdepth == CV_64F )
            return makePtr<ColumnFilter<Cast<double, uchar>, ColumnNoVec> >(kernel, anchor, delta);
        if( ddepth == CV_16U && sdepth == CV_32F )
            return makePtr<ColumnFilter<Cast<float, ushort>, ColumnNoVec> >(kernel, anchor, delta);
        if( ddepth == CV_16U && sdepth == CV_64F )
            return makePtr<ColumnFilter<Cast<double, ushort>, ColumnNoVec> >(kernel, anchor, delta);
        if( ddepth == CV_16S && sdepth == CV_32F )
            return makePtr<ColumnFilter<Cast<float, short>, ColumnNoVec> >(kernel, anchor, delta);
        if( ddepth == CV_16S && sdepth == CV_64F )
            return makePtr<ColumnFilter<Cast<double, short>, ColumnNoVec> >(kernel, anchor, delta);
        if( ddepth == CV_32F && sdepth == CV_32F )
            return makePtr<ColumnFilter<Cast<float, float>, ColumnNoVec> >(kernel, anchor, delta);
        if( ddepth == CV_64F && sdepth == CV_64F )
            return makePtr<ColumnFilter<Cast<double, double>, ColumnNoVec> >(kernel, anchor, delta);
    }
    else
    {
        int ksize = kernel.rows + kernel.cols - 1;
        if( ksize == 3 )
        {
            if( ddepth == CV_8U && sdepth == CV_32S )
                return makePtr<SymmColumnSmallFilter<
                    FixedPtCastEx<int, uchar>, SymmColumnVec_32s8u> >
                    (kernel, anchor, delta, symmetryType, FixedPtCastEx<int, uchar>(bits),
                    SymmColumnVec_32s8u(kernel, symmetryType, bits, delta));
            if( ddepth == CV_16S && sdepth == CV_32S && bits == 0 )
                return makePtr<SymmColumnSmallFilter<Cast<int, short>,
                    SymmColumnSmallVec_32s16s> >(kernel, anchor, delta, symmetryType,
                        Cast<int, short>(), SymmColumnSmallVec_32s16s(kernel, symmetryType, bits, delta));
            if( ddepth == CV_32F && sdepth == CV_32F )
                return makePtr<SymmColumnSmallFilter<
                    Cast<float, float>,SymmColumnSmallVec_32f> >
                    (kernel, anchor, delta, symmetryType, Cast<float, float>(),
                    SymmColumnSmallVec_32f(kernel, symmetryType, 0, delta));
        }
        if( ddepth == CV_8U && sdepth == CV_32S )
            return makePtr<SymmColumnFilter<FixedPtCastEx<int, uchar>, SymmColumnVec_32s8u> >
                (kernel, anchor, delta, symmetryType, FixedPtCastEx<int, uchar>(bits),
                SymmColumnVec_32s8u(kernel, symmetryType, bits, delta));
        if( ddepth == CV_8U && sdepth == CV_32F )
            return makePtr<SymmColumnFilter<Cast<float, uchar>, ColumnNoVec> >
                (kernel, anchor, delta, symmetryType);
        if( ddepth == CV_8U && sdepth == CV_64F )
            return makePtr<SymmColumnFilter<Cast<double, uchar>, ColumnNoVec> >
                (kernel, anchor, delta, symmetryType);
        if( ddepth == CV_16U && sdepth == CV_32F )
            return makePtr<SymmColumnFilter<Cast<float, ushort>, ColumnNoVec> >
                (kernel, anchor, delta, symmetryType);
        if( ddepth == CV_16U && sdepth == CV_64F )
            return makePtr<SymmColumnFilter<Cast<double, ushort>, ColumnNoVec> >
                (kernel, anchor, delta, symmetryType);
        if( ddepth == CV_16S && sdepth == CV_32S )
            return makePtr<SymmColumnFilter<Cast<int, short>, ColumnNoVec> >
                (kernel, anchor, delta, symmetryType);
        if( ddepth == CV_16S && sdepth == CV_32F )
            return makePtr<SymmColumnFilter<Cast<float, short>, SymmColumnVec_32f16s> >
                 (kernel, anchor, delta, symmetryType, Cast<float, short>(),
                  SymmColumnVec_32f16s(kernel, symmetryType, 0, delta));
        if( ddepth == CV_16S && sdepth == CV_64F )
            return makePtr<SymmColumnFilter<Cast<double, short>, ColumnNoVec> >
                (kernel, anchor, delta, symmetryType);
        if( ddepth == CV_32F && sdepth == CV_32F )
            return makePtr<SymmColumnFilter<Cast<float, float>, SymmColumnVec_32f> >
                (kernel, anchor, delta, symmetryType, Cast<float, float>(),
                SymmColumnVec_32f(kernel, symmetryType, 0, delta));
        if( ddepth == CV_64F && sdepth == CV_64F )
            return makePtr<SymmColumnFilter<Cast<double, double>, ColumnNoVec> >
                (kernel, anchor, delta, symmetryType);
    }

    CV_Error_( CV_StsNotImplemented,
        ("Unsupported combination of buffer format (=%d), and destination format (=%d)",
        bufType, dstType));
}


cv::Ptr<cv::FilterEngine> cv::createSeparableLinearFilter(
    int _srcType, int _dstType,
    InputArray __rowKernel, InputArray __columnKernel,
    Point _anchor, double _delta,
    int _rowBorderType, int _columnBorderType,
    const Scalar& _borderValue )
{
    Mat _rowKernel = __rowKernel.getMat(), _columnKernel = __columnKernel.getMat();
    _srcType = CV_MAT_TYPE(_srcType);
    _dstType = CV_MAT_TYPE(_dstType);
    int sdepth = CV_MAT_DEPTH(_srcType), ddepth = CV_MAT_DEPTH(_dstType);
    int cn = CV_MAT_CN(_srcType);
    CV_Assert( cn == CV_MAT_CN(_dstType) );
    int rsize = _rowKernel.rows + _rowKernel.cols - 1;
    int csize = _columnKernel.rows + _columnKernel.cols - 1;
    if( _anchor.x < 0 )
        _anchor.x = rsize/2;
    if( _anchor.y < 0 )
        _anchor.y = csize/2;
    int rtype = getKernelType(_rowKernel,
        _rowKernel.rows == 1 ? Point(_anchor.x, 0) : Point(0, _anchor.x));
    int ctype = getKernelType(_columnKernel,
        _columnKernel.rows == 1 ? Point(_anchor.y, 0) : Point(0, _anchor.y));
    Mat rowKernel, columnKernel;

    int bdepth = std::max(CV_32F,std::max(sdepth, ddepth));
    int bits = 0;

    if( sdepth == CV_8U &&
        ((rtype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
          ctype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
          ddepth == CV_8U) ||
         ((rtype & (KERNEL_SYMMETRICAL+KERNEL_ASYMMETRICAL)) &&
          (ctype & (KERNEL_SYMMETRICAL+KERNEL_ASYMMETRICAL)) &&
          (rtype & ctype & KERNEL_INTEGER) &&
          ddepth == CV_16S)) )
    {
        bdepth = CV_32S;
        bits = ddepth == CV_8U ? 8 : 0;
        _rowKernel.convertTo( rowKernel, CV_32S, 1 << bits );
        _columnKernel.convertTo( columnKernel, CV_32S, 1 << bits );
        bits *= 2;
        _delta *= (1 << bits);
    }
    else
    {
        if( _rowKernel.type() != bdepth )
            _rowKernel.convertTo( rowKernel, bdepth );
        else
            rowKernel = _rowKernel;
        if( _columnKernel.type() != bdepth )
            _columnKernel.convertTo( columnKernel, bdepth );
        else
            columnKernel = _columnKernel;
    }

    int _bufType = CV_MAKETYPE(bdepth, cn);
    Ptr<BaseRowFilter> _rowFilter = getLinearRowFilter(
        _srcType, _bufType, rowKernel, _anchor.x, rtype);
    Ptr<BaseColumnFilter> _columnFilter = getLinearColumnFilter(
        _bufType, _dstType, columnKernel, _anchor.y, ctype, _delta, bits );

    return Ptr<FilterEngine>( new FilterEngine(Ptr<BaseFilter>(), _rowFilter, _columnFilter,
        _srcType, _dstType, _bufType, _rowBorderType, _columnBorderType, _borderValue ));
}


/****************************************************************************************\
*                               Non-separable linear filter                              *
\****************************************************************************************/

namespace cv
{

void preprocess2DKernel( const Mat& kernel, std::vector<Point>& coords, std::vector<uchar>& coeffs )
{
    int i, j, k, nz = countNonZero(kernel), ktype = kernel.type();
    if(nz == 0)
        nz = 1;
    CV_Assert( ktype == CV_8U || ktype == CV_32S || ktype == CV_32F || ktype == CV_64F );
    coords.resize(nz);
    coeffs.resize(nz*getElemSize(ktype));
    uchar* _coeffs = &coeffs[0];

    for( i = k = 0; i < kernel.rows; i++ )
    {
        const uchar* krow = kernel.ptr(i);
        for( j = 0; j < kernel.cols; j++ )
        {
            if( ktype == CV_8U )
            {
                uchar val = krow[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                _coeffs[k++] = val;
            }
            else if( ktype == CV_32S )
            {
                int val = ((const int*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((int*)_coeffs)[k++] = val;
            }
            else if( ktype == CV_32F )
            {
                float val = ((const float*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((float*)_coeffs)[k++] = val;
            }
            else
            {
                double val = ((const double*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((double*)_coeffs)[k++] = val;
            }
        }
    }
}


template<typename ST, class CastOp, class VecOp> struct Filter2D : public BaseFilter
{
    typedef typename CastOp::type1 KT;
    typedef typename CastOp::rtype DT;

    Filter2D( const Mat& _kernel, Point _anchor,
        double _delta, const CastOp& _castOp=CastOp(),
        const VecOp& _vecOp=VecOp() )
    {
        anchor = _anchor;
        ksize = _kernel.size();
        delta = saturate_cast<KT>(_delta);
        castOp0 = _castOp;
        vecOp = _vecOp;
        CV_Assert( _kernel.type() == DataType<KT>::type );
        preprocess2DKernel( _kernel, coords, coeffs );
        ptrs.resize( coords.size() );
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int cn) CV_OVERRIDE
    {
        KT _delta = delta;
        const Point* pt = &coords[0];
        const KT* kf = (const KT*)&coeffs[0];
        const ST** kp = (const ST**)&ptrs[0];
        int i, k, nz = (int)coords.size();
        CastOp castOp = castOp0;

        width *= cn;
        for( ; count > 0; count--, dst += dststep, src++ )
        {
            DT* D = (DT*)dst;

            for( k = 0; k < nz; k++ )
                kp[k] = (const ST*)src[pt[k].y] + pt[k].x*cn;

            i = vecOp((const uchar**)kp, dst, width);
            #if CV_ENABLE_UNROLLED
            for( ; i <= width - 4; i += 4 )
            {
                KT s0 = _delta, s1 = _delta, s2 = _delta, s3 = _delta;

                for( k = 0; k < nz; k++ )
                {
                    const ST* sptr = kp[k] + i;
                    KT f = kf[k];
                    s0 += f*sptr[0];
                    s1 += f*sptr[1];
                    s2 += f*sptr[2];
                    s3 += f*sptr[3];
                }

                D[i] = castOp(s0); D[i+1] = castOp(s1);
                D[i+2] = castOp(s2); D[i+3] = castOp(s3);
            }
            #endif
            for( ; i < width; i++ )
            {
                KT s0 = _delta;
                for( k = 0; k < nz; k++ )
                    s0 += kf[k]*kp[k][i];
                D[i] = castOp(s0);
            }
        }
    }

    std::vector<Point> coords;
    std::vector<uchar> coeffs;
    std::vector<uchar*> ptrs;
    KT delta;
    CastOp castOp0;
    VecOp vecOp;
};

#ifdef HAVE_OPENCL

#define DIVUP(total, grain) (((total) + (grain) - 1) / (grain))
#define ROUNDUP(sz, n)      ((sz) + (n) - 1 - (((sz) + (n) - 1) % (n)))

// prepare kernel: transpose and make double rows (+align). Returns size of aligned row
// Samples:
//        a b c
// Input: d e f
//        g h i
// Output, last two zeros is the alignment:
// a d g a d g 0 0
// b e h b e h 0 0
// c f i c f i 0 0
template <typename T>
static int _prepareKernelFilter2D(std::vector<T> & data, const Mat & kernel)
{
    Mat _kernel; kernel.convertTo(_kernel, DataDepth<T>::value);
    int size_y_aligned = ROUNDUP(kernel.rows * 2, 4);
    data.clear(); data.resize(size_y_aligned * kernel.cols, 0);
    for (int x = 0; x < kernel.cols; x++)
    {
        for (int y = 0; y < kernel.rows; y++)
        {
            data[x * size_y_aligned + y] = _kernel.at<T>(y, x);
            data[x * size_y_aligned + y + kernel.rows] = _kernel.at<T>(y, x);
        }
    }
    return size_y_aligned;
}

static bool ocl_filter2D( InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor,
                   double delta, int borderType )
{
    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    ddepth = ddepth < 0 ? sdepth : ddepth;
    int dtype = CV_MAKE_TYPE(ddepth, cn), wdepth = std::max(std::max(sdepth, ddepth), CV_32F),
            wtype = CV_MAKE_TYPE(wdepth, cn);
    if (cn > 4)
        return false;

    Size ksize = _kernel.size();
    if (anchor.x < 0)
        anchor.x = ksize.width / 2;
    if (anchor.y < 0)
        anchor.y = ksize.height / 2;

    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    borderType &= ~BORDER_ISOLATED;
    const cv::ocl::Device &device = cv::ocl::Device::getDefault();
    bool doubleSupport = device.doubleFPConfig() > 0;
    if (wdepth == CV_64F && !doubleSupport)
        return false;

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
                                       "BORDER_WRAP", "BORDER_REFLECT_101" };

    cv::Mat kernelMat = _kernel.getMat();
    cv::Size sz = _src.size(), wholeSize;
    size_t globalsize[2] = { (size_t)sz.width, (size_t)sz.height };
    size_t localsize_general[2] = {0, 1};
    size_t* localsize = NULL;

    ocl::Kernel k;
    UMat src = _src.getUMat();
    if (!isolated)
    {
        Point ofs;
        src.locateROI(wholeSize, ofs);
    }

    size_t tryWorkItems = device.maxWorkGroupSize();
    if (device.isIntel() && 128 < tryWorkItems)
        tryWorkItems = 128;
    char cvt[2][40];

    // For smaller filter kernels, there is a special kernel that is more
    // efficient than the general one.
    UMat kernalDataUMat;
    if (device.isIntel() && (device.type() & ocl::Device::TYPE_GPU) &&
        ((ksize.width < 5 && ksize.height < 5) ||
        (ksize.width == 5 && ksize.height == 5 && cn == 1)))
    {
        kernelMat = kernelMat.reshape(0, 1);
        String kerStr = ocl::kernelToStr(kernelMat, CV_32F);
        int h = isolated ? sz.height : wholeSize.height;
        int w = isolated ? sz.width : wholeSize.width;

        if (w < ksize.width || h < ksize.height)
            return false;

        // Figure out what vector size to use for loading the pixels.
        int pxLoadNumPixels = cn != 1 || sz.width % 4 ? 1 : 4;
        int pxLoadVecSize = cn * pxLoadNumPixels;

        // Figure out how many pixels per work item to compute in X and Y
        // directions.  Too many and we run out of registers.
        int pxPerWorkItemX = 1;
        int pxPerWorkItemY = 1;
        if (cn <= 2 && ksize.width <= 4 && ksize.height <= 4)
        {
            pxPerWorkItemX = sz.width % 8 ? sz.width % 4 ? sz.width % 2 ? 1 : 2 : 4 : 8;
            pxPerWorkItemY = sz.height % 2 ? 1 : 2;
        }
        else if (cn < 4 || (ksize.width <= 4 && ksize.height <= 4))
        {
            pxPerWorkItemX = sz.width % 2 ? 1 : 2;
            pxPerWorkItemY = sz.height % 2 ? 1 : 2;
        }
        globalsize[0] = sz.width / pxPerWorkItemX;
        globalsize[1] = sz.height / pxPerWorkItemY;

        // Need some padding in the private array for pixels
        int privDataWidth = ROUNDUP(pxPerWorkItemX + ksize.width - 1, pxLoadNumPixels);

        // Make the global size a nice round number so the runtime can pick
        // from reasonable choices for the workgroup size
        const int wgRound = 256;
        globalsize[0] = ROUNDUP(globalsize[0], wgRound);

        char build_options[1024];
        sprintf(build_options, "-D cn=%d "
                "-D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d "
                "-D PX_LOAD_VEC_SIZE=%d -D PX_LOAD_NUM_PX=%d "
                "-D PX_PER_WI_X=%d -D PX_PER_WI_Y=%d -D PRIV_DATA_WIDTH=%d -D %s -D %s "
                "-D PX_LOAD_X_ITERATIONS=%d -D PX_LOAD_Y_ITERATIONS=%d "
                "-D srcT=%s -D srcT1=%s -D dstT=%s -D dstT1=%s -D WT=%s -D WT1=%s "
                "-D convertToWT=%s -D convertToDstT=%s %s",
                cn, anchor.x, anchor.y, ksize.width, ksize.height,
                pxLoadVecSize, pxLoadNumPixels,
                pxPerWorkItemX, pxPerWorkItemY, privDataWidth, borderMap[borderType],
                isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
                privDataWidth / pxLoadNumPixels, pxPerWorkItemY + ksize.height - 1,
                ocl::typeToStr(type), ocl::typeToStr(sdepth), ocl::typeToStr(dtype),
                ocl::typeToStr(ddepth), ocl::typeToStr(wtype), ocl::typeToStr(wdepth),
                ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
                ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]), kerStr.c_str());

        if (!k.create("filter2DSmall", cv::ocl::imgproc::filter2DSmall_oclsrc, build_options))
            return false;
    }
    else
    {
        localsize = localsize_general;
        std::vector<float> kernelMatDataFloat;
        int kernel_size_y2_aligned = _prepareKernelFilter2D<float>(kernelMatDataFloat, kernelMat);
        String kerStr = ocl::kernelToStr(kernelMatDataFloat, CV_32F);

        for ( ; ; )
        {
            size_t BLOCK_SIZE = tryWorkItems;
            while (BLOCK_SIZE > 32 && BLOCK_SIZE >= (size_t)ksize.width * 2 && BLOCK_SIZE > (size_t)sz.width * 2)
                BLOCK_SIZE /= 2;

            if ((size_t)ksize.width > BLOCK_SIZE)
                return false;

            int requiredTop = anchor.y;
            int requiredLeft = (int)BLOCK_SIZE; // not this: anchor.x;
            int requiredBottom = ksize.height - 1 - anchor.y;
            int requiredRight = (int)BLOCK_SIZE; // not this: ksize.width - 1 - anchor.x;
            int h = isolated ? sz.height : wholeSize.height;
            int w = isolated ? sz.width : wholeSize.width;
            bool extra_extrapolation = h < requiredTop || h < requiredBottom || w < requiredLeft || w < requiredRight;

            if ((w < ksize.width) || (h < ksize.height))
                return false;

            String opts = format("-D LOCAL_SIZE=%d -D cn=%d "
                                 "-D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d "
                                 "-D KERNEL_SIZE_Y2_ALIGNED=%d -D %s -D %s -D %s%s%s "
                                 "-D srcT=%s -D srcT1=%s -D dstT=%s -D dstT1=%s -D WT=%s -D WT1=%s "
                                 "-D convertToWT=%s -D convertToDstT=%s",
                                 (int)BLOCK_SIZE, cn, anchor.x, anchor.y,
                                 ksize.width, ksize.height, kernel_size_y2_aligned, borderMap[borderType],
                                 extra_extrapolation ? "EXTRA_EXTRAPOLATION" : "NO_EXTRA_EXTRAPOLATION",
                                 isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
                                 doubleSupport ? " -D DOUBLE_SUPPORT" : "", kerStr.c_str(),
                                 ocl::typeToStr(type), ocl::typeToStr(sdepth), ocl::typeToStr(dtype),
                                 ocl::typeToStr(ddepth), ocl::typeToStr(wtype), ocl::typeToStr(wdepth),
                                 ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
                                 ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]));

            localsize[0] = BLOCK_SIZE;
            globalsize[0] = DIVUP(sz.width, BLOCK_SIZE - (ksize.width - 1)) * BLOCK_SIZE;
            globalsize[1] = sz.height;

            if (!k.create("filter2D", cv::ocl::imgproc::filter2D_oclsrc, opts))
                return false;

            size_t kernelWorkGroupSize = k.workGroupSize();
            if (localsize[0] <= kernelWorkGroupSize)
                break;
            if (BLOCK_SIZE < kernelWorkGroupSize)
                return false;
            tryWorkItems = kernelWorkGroupSize;
        }
    }

    _dst.create(sz, dtype);
    UMat dst = _dst.getUMat();

    int srcOffsetX = (int)((src.offset % src.step) / src.elemSize());
    int srcOffsetY = (int)(src.offset / src.step);
    int srcEndX = (isolated ? (srcOffsetX + sz.width) : wholeSize.width);
    int srcEndY = (isolated ? (srcOffsetY + sz.height) : wholeSize.height);

    k.args(ocl::KernelArg::PtrReadOnly(src), (int)src.step, srcOffsetX, srcOffsetY,
           srcEndX, srcEndY, ocl::KernelArg::WriteOnly(dst), (float)delta);

    return k.run(2, globalsize, localsize, false);
}

const int shift_bits = 8;

static bool ocl_sepRowFilter2D(const UMat & src, UMat & buf, const Mat & kernelX, int anchor,
                               int borderType, int ddepth, bool fast8uc1, bool int_arithm)
{
    int type = src.type(), cn = CV_MAT_CN(type), sdepth = CV_MAT_DEPTH(type);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;
    Size bufSize = buf.size();
    int buf_type = buf.type(), bdepth = CV_MAT_DEPTH(buf_type);

    if (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        return false;

#ifdef __ANDROID__
    size_t localsize[2] = {16, 10};
#else
    size_t localsize[2] = {16, 16};
#endif

    size_t globalsize[2] = {DIVUP(bufSize.width, localsize[0]) * localsize[0], DIVUP(bufSize.height, localsize[1]) * localsize[1]};
    if (fast8uc1)
        globalsize[0] = DIVUP((bufSize.width + 3) >> 2, localsize[0]) * localsize[0];

    int radiusX = anchor, radiusY = (buf.rows - src.rows) >> 1;

    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP", "BORDER_REFLECT_101" },
        * const btype = borderMap[borderType & ~BORDER_ISOLATED];

    bool extra_extrapolation = src.rows < (int)((-radiusY + globalsize[1]) >> 1) + 1;
    extra_extrapolation |= src.rows < radiusY;
    extra_extrapolation |= src.cols < (int)((-radiusX + globalsize[0] + 8 * localsize[0] + 3) >> 1) + 1;
    extra_extrapolation |= src.cols < radiusX;

    char cvt[40];
    cv::String build_options = cv::format("-D RADIUSX=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D %s -D %s"
                                          " -D srcT=%s -D dstT=%s -D convertToDstT=%s -D srcT1=%s -D dstT1=%s%s%s",
                                          radiusX, (int)localsize[0], (int)localsize[1], cn, btype,
                                          extra_extrapolation ? "EXTRA_EXTRAPOLATION" : "NO_EXTRA_EXTRAPOLATION",
                                          isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
                                          ocl::typeToStr(type), ocl::typeToStr(buf_type),
                                          ocl::convertTypeStr(sdepth, bdepth, cn, cvt),
                                          ocl::typeToStr(sdepth), ocl::typeToStr(bdepth),
                                          doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                                          int_arithm ? " -D INTEGER_ARITHMETIC" : "");
    build_options += ocl::kernelToStr(kernelX, bdepth);

    Size srcWholeSize; Point srcOffset;
    src.locateROI(srcWholeSize, srcOffset);

    String kernelName("row_filter");
    if (fast8uc1)
        kernelName += "_C1_D0";

    ocl::Kernel k(kernelName.c_str(), cv::ocl::imgproc::filterSepRow_oclsrc,
                  build_options);
    if (k.empty())
        return false;

    if (fast8uc1)
        k.args(ocl::KernelArg::PtrReadOnly(src), (int)(src.step / src.elemSize()), srcOffset.x,
               srcOffset.y, src.cols, src.rows, srcWholeSize.width, srcWholeSize.height,
               ocl::KernelArg::PtrWriteOnly(buf), (int)(buf.step / buf.elemSize()),
               buf.cols, buf.rows, radiusY);
    else
        k.args(ocl::KernelArg::PtrReadOnly(src), (int)src.step, srcOffset.x,
               srcOffset.y, src.cols, src.rows, srcWholeSize.width, srcWholeSize.height,
               ocl::KernelArg::PtrWriteOnly(buf), (int)buf.step, buf.cols, buf.rows, radiusY);

    return k.run(2, globalsize, localsize, false);
}

static bool ocl_sepColFilter2D(const UMat & buf, UMat & dst, const Mat & kernelY, double delta, int anchor, bool int_arithm)
{
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;
    if (dst.depth() == CV_64F && !doubleSupport)
        return false;

#ifdef __ANDROID__
    size_t localsize[2] = { 16, 10 };
#else
    size_t localsize[2] = { 16, 16 };
#endif
    size_t globalsize[2] = { 0, 0 };

    int dtype = dst.type(), cn = CV_MAT_CN(dtype), ddepth = CV_MAT_DEPTH(dtype);
    Size sz = dst.size();
    int buf_type = buf.type(), bdepth = CV_MAT_DEPTH(buf_type);

    globalsize[1] = DIVUP(sz.height, localsize[1]) * localsize[1];
    globalsize[0] = DIVUP(sz.width, localsize[0]) * localsize[0];

    char cvt[40];
    cv::String build_options = cv::format("-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d"
                                          " -D srcT=%s -D dstT=%s -D convertToDstT=%s"
                                          " -D srcT1=%s -D dstT1=%s -D SHIFT_BITS=%d%s%s",
                                          anchor, (int)localsize[0], (int)localsize[1], cn,
                                          ocl::typeToStr(buf_type), ocl::typeToStr(dtype),
                                          ocl::convertTypeStr(bdepth, ddepth, cn, cvt),
                                          ocl::typeToStr(bdepth), ocl::typeToStr(ddepth),
                                          2*shift_bits, doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                                          int_arithm ? " -D INTEGER_ARITHMETIC" : "");
    build_options += ocl::kernelToStr(kernelY, bdepth);

    ocl::Kernel k("col_filter", cv::ocl::imgproc::filterSepCol_oclsrc,
                  build_options);
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnly(buf), ocl::KernelArg::WriteOnly(dst),
           static_cast<float>(delta));

    return k.run(2, globalsize, localsize, false);
}

const int optimizedSepFilterLocalWidth  = 16;
const int optimizedSepFilterLocalHeight = 8;

static bool ocl_sepFilter2D_SinglePass(InputArray _src, OutputArray _dst,
                                       Mat row_kernel, Mat col_kernel,
                                       double delta, int borderType, int ddepth, int bdepth, bool int_arithm)
{
    Size size = _src.size(), wholeSize;
    Point origin;
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype),
            esz = CV_ELEM_SIZE(stype), wdepth = std::max(std::max(sdepth, ddepth), bdepth),
            dtype = CV_MAKE_TYPE(ddepth, cn);
    size_t src_step = _src.step(), src_offset = _src.offset();
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if (esz == 0 || src_step == 0
        || (src_offset % src_step) % esz != 0
        || (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        || !(borderType == BORDER_CONSTANT
             || borderType == BORDER_REPLICATE
             || borderType == BORDER_REFLECT
             || borderType == BORDER_WRAP
             || borderType == BORDER_REFLECT_101))
        return false;

    size_t lt2[2] = { optimizedSepFilterLocalWidth, optimizedSepFilterLocalHeight };
    size_t gt2[2] = { lt2[0] * (1 + (size.width - 1) / lt2[0]), lt2[1]};

    char cvt[2][40];
    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP",
                                       "BORDER_REFLECT_101" };

    String opts = cv::format("-D BLK_X=%d -D BLK_Y=%d -D RADIUSX=%d -D RADIUSY=%d%s%s"
                             " -D srcT=%s -D convertToWT=%s -D WT=%s -D dstT=%s -D convertToDstT=%s"
                             " -D %s -D srcT1=%s -D dstT1=%s -D WT1=%s -D CN=%d -D SHIFT_BITS=%d%s",
                             (int)lt2[0], (int)lt2[1], row_kernel.cols / 2, col_kernel.cols / 2,
                             ocl::kernelToStr(row_kernel, wdepth, "KERNEL_MATRIX_X").c_str(),
                             ocl::kernelToStr(col_kernel, wdepth, "KERNEL_MATRIX_Y").c_str(),
                             ocl::typeToStr(stype), ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
                             ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)), ocl::typeToStr(dtype),
                             ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]), borderMap[borderType],
                             ocl::typeToStr(sdepth), ocl::typeToStr(ddepth), ocl::typeToStr(wdepth),
                             cn, 2*shift_bits, int_arithm ? " -D INTEGER_ARITHMETIC" : "");

    ocl::Kernel k("sep_filter", ocl::imgproc::filterSep_singlePass_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(size, dtype);
    UMat dst = _dst.getUMat();

    int src_offset_x = static_cast<int>((src_offset % src_step) / esz);
    int src_offset_y = static_cast<int>(src_offset / src_step);

    src.locateROI(wholeSize, origin);

    k.args(ocl::KernelArg::PtrReadOnly(src), (int)src_step, src_offset_x, src_offset_y,
           wholeSize.height, wholeSize.width, ocl::KernelArg::WriteOnly(dst),
           static_cast<float>(delta));

    return k.run(2, gt2, lt2, false);
}

bool ocl_sepFilter2D( InputArray _src, OutputArray _dst, int ddepth,
                      InputArray _kernelX, InputArray _kernelY, Point anchor,
                      double delta, int borderType )
{
    const ocl::Device & d = ocl::Device::getDefault();
    Size imgSize = _src.size();

    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if (cn > 4)
        return false;

    Mat kernelX = _kernelX.getMat().reshape(1, 1);
    if (kernelX.cols % 2 != 1)
        return false;
    Mat kernelY = _kernelY.getMat().reshape(1, 1);
    if (kernelY.cols % 2 != 1)
        return false;

    if (ddepth < 0)
        ddepth = sdepth;

    if (anchor.x < 0)
        anchor.x = kernelX.cols >> 1;
    if (anchor.y < 0)
        anchor.y = kernelY.cols >> 1;

    int rtype = getKernelType(kernelX,
        kernelX.rows == 1 ? Point(anchor.x, 0) : Point(0, anchor.x));
    int ctype = getKernelType(kernelY,
        kernelY.rows == 1 ? Point(anchor.y, 0) : Point(0, anchor.y));

    int bdepth = CV_32F;
    bool int_arithm = false;
    if( sdepth == CV_8U && ddepth == CV_8U &&
        rtype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
        ctype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL)
    {
        if (ocl::Device::getDefault().isIntel())
        {
            for (int i=0; i<kernelX.cols; i++)
                kernelX.at<float>(0, i) = (float) cvRound(kernelX.at<float>(0, i) * (1 << shift_bits));
            if (kernelX.data != kernelY.data)
                for (int i=0; i<kernelX.cols; i++)
                    kernelY.at<float>(0, i) = (float) cvRound(kernelY.at<float>(0, i) * (1 << shift_bits));
        } else
        {
            bdepth = CV_32S;
            kernelX.convertTo( kernelX, bdepth, 1 << shift_bits );
            kernelY.convertTo( kernelY, bdepth, 1 << shift_bits );
        }
        int_arithm = true;
    }

    CV_OCL_RUN_(kernelY.cols <= 21 && kernelX.cols <= 21 &&
                imgSize.width > optimizedSepFilterLocalWidth + anchor.x &&
                imgSize.height > optimizedSepFilterLocalHeight + anchor.y &&
                (!(borderType & BORDER_ISOLATED) || _src.offset() == 0) &&
                anchor == Point(kernelX.cols >> 1, kernelY.cols >> 1) &&
                OCL_PERFORMANCE_CHECK(d.isIntel()),  // TODO FIXIT
                ocl_sepFilter2D_SinglePass(_src, _dst, kernelX, kernelY, delta,
                                           borderType & ~BORDER_ISOLATED, ddepth, bdepth, int_arithm), true)

    UMat src = _src.getUMat();
    Size srcWholeSize; Point srcOffset;
    src.locateROI(srcWholeSize, srcOffset);

    bool fast8uc1 = type == CV_8UC1 && srcOffset.x % 4 == 0 &&
            src.cols % 4 == 0 && src.step % 4 == 0;

    Size srcSize = src.size();
    Size bufSize(srcSize.width, srcSize.height + kernelY.cols - 1);
    UMat buf(bufSize, CV_MAKETYPE(bdepth, cn));
    if (!ocl_sepRowFilter2D(src, buf, kernelX, anchor.x, borderType, ddepth, fast8uc1, int_arithm))
        return false;

    _dst.create(srcSize, CV_MAKETYPE(ddepth, cn));
    UMat dst = _dst.getUMat();

    return ocl_sepColFilter2D(buf, dst, kernelY, delta, anchor.y, int_arithm);
}

#endif

}

cv::Ptr<cv::BaseFilter> cv::getLinearFilter(int srcType, int dstType,
                                InputArray filter_kernel, Point anchor,
                                double delta, int bits)
{
    Mat _kernel = filter_kernel.getMat();
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);
    int cn = CV_MAT_CN(srcType), kdepth = _kernel.depth();
    CV_Assert( cn == CV_MAT_CN(dstType) && ddepth >= sdepth );

    anchor = normalizeAnchor(anchor, _kernel.size());

    /*if( sdepth == CV_8U && ddepth == CV_8U && kdepth == CV_32S )
        return makePtr<Filter2D<uchar, FixedPtCastEx<int, uchar>, FilterVec_8u> >
            (_kernel, anchor, delta, FixedPtCastEx<int, uchar>(bits),
            FilterVec_8u(_kernel, bits, delta));
    if( sdepth == CV_8U && ddepth == CV_16S && kdepth == CV_32S )
        return makePtr<Filter2D<uchar, FixedPtCastEx<int, short>, FilterVec_8u16s> >
            (_kernel, anchor, delta, FixedPtCastEx<int, short>(bits),
            FilterVec_8u16s(_kernel, bits, delta));*/

    kdepth = sdepth == CV_64F || ddepth == CV_64F ? CV_64F : CV_32F;
    Mat kernel;
    if( _kernel.type() == kdepth )
        kernel = _kernel;
    else
        _kernel.convertTo(kernel, kdepth, _kernel.type() == CV_32S ? 1./(1 << bits) : 1.);

    if( sdepth == CV_8U && ddepth == CV_8U )
        return makePtr<Filter2D<uchar, Cast<float, uchar>, FilterVec_8u> >
            (kernel, anchor, delta, Cast<float, uchar>(), FilterVec_8u(kernel, 0, delta));
    if( sdepth == CV_8U && ddepth == CV_16U )
        return makePtr<Filter2D<uchar,
            Cast<float, ushort>, FilterNoVec> >(kernel, anchor, delta);
    if( sdepth == CV_8U && ddepth == CV_16S )
        return makePtr<Filter2D<uchar, Cast<float, short>, FilterVec_8u16s> >
            (kernel, anchor, delta, Cast<float, short>(), FilterVec_8u16s(kernel, 0, delta));
    if( sdepth == CV_8U && ddepth == CV_32F )
        return makePtr<Filter2D<uchar,
            Cast<float, float>, FilterNoVec> >(kernel, anchor, delta);
    if( sdepth == CV_8U && ddepth == CV_64F )
        return makePtr<Filter2D<uchar,
            Cast<double, double>, FilterNoVec> >(kernel, anchor, delta);

    if( sdepth == CV_16U && ddepth == CV_16U )
        return makePtr<Filter2D<ushort,
            Cast<float, ushort>, FilterNoVec> >(kernel, anchor, delta);
    if( sdepth == CV_16U && ddepth == CV_32F )
        return makePtr<Filter2D<ushort,
            Cast<float, float>, FilterNoVec> >(kernel, anchor, delta);
    if( sdepth == CV_16U && ddepth == CV_64F )
        return makePtr<Filter2D<ushort,
            Cast<double, double>, FilterNoVec> >(kernel, anchor, delta);

    if( sdepth == CV_16S && ddepth == CV_16S )
        return makePtr<Filter2D<short,
            Cast<float, short>, FilterNoVec> >(kernel, anchor, delta);
    if( sdepth == CV_16S && ddepth == CV_32F )
        return makePtr<Filter2D<short,
            Cast<float, float>, FilterNoVec> >(kernel, anchor, delta);
    if( sdepth == CV_16S && ddepth == CV_64F )
        return makePtr<Filter2D<short,
            Cast<double, double>, FilterNoVec> >(kernel, anchor, delta);

    if( sdepth == CV_32F && ddepth == CV_32F )
        return makePtr<Filter2D<float, Cast<float, float>, FilterVec_32f> >
            (kernel, anchor, delta, Cast<float, float>(), FilterVec_32f(kernel, 0, delta));
    if( sdepth == CV_64F && ddepth == CV_64F )
        return makePtr<Filter2D<double,
            Cast<double, double>, FilterNoVec> >(kernel, anchor, delta);

    CV_Error_( CV_StsNotImplemented,
        ("Unsupported combination of source format (=%d), and destination format (=%d)",
        srcType, dstType));
}


cv::Ptr<cv::FilterEngine> cv::createLinearFilter( int _srcType, int _dstType,
                                              InputArray filter_kernel,
                                              Point _anchor, double _delta,
                                              int _rowBorderType, int _columnBorderType,
                                              const Scalar& _borderValue )
{
    Mat _kernel = filter_kernel.getMat();
    _srcType = CV_MAT_TYPE(_srcType);
    _dstType = CV_MAT_TYPE(_dstType);
    int cn = CV_MAT_CN(_srcType);
    CV_Assert( cn == CV_MAT_CN(_dstType) );

    Mat kernel = _kernel;
    int bits = 0;

    /*int sdepth = CV_MAT_DEPTH(_srcType), ddepth = CV_MAT_DEPTH(_dstType);
    int ktype = _kernel.depth() == CV_32S ? KERNEL_INTEGER : getKernelType(_kernel, _anchor);
    if( sdepth == CV_8U && (ddepth == CV_8U || ddepth == CV_16S) &&
        _kernel.rows*_kernel.cols <= (1 << 10) )
    {
        bits = (ktype & KERNEL_INTEGER) ? 0 : 11;
        _kernel.convertTo(kernel, CV_32S, 1 << bits);
    }*/

    Ptr<BaseFilter> _filter2D = getLinearFilter(_srcType, _dstType,
        kernel, _anchor, _delta, bits);

    return makePtr<FilterEngine>(_filter2D, Ptr<BaseRowFilter>(),
        Ptr<BaseColumnFilter>(), _srcType, _dstType, _srcType,
        _rowBorderType, _columnBorderType, _borderValue );
}


//================================================================
// HAL interface
//================================================================

using namespace cv;

static bool replacementFilter2D(int stype, int dtype, int kernel_type,
                                uchar * src_data, size_t src_step,
                                uchar * dst_data, size_t dst_step,
                                int width, int height,
                                int full_width, int full_height,
                                int offset_x, int offset_y,
                                uchar * kernel_data, size_t kernel_step,
                                int kernel_width, int kernel_height,
                                int anchor_x, int anchor_y,
                                double delta, int borderType, bool isSubmatrix)
{
    cvhalFilter2D* ctx;
    int res = cv_hal_filterInit(&ctx, kernel_data, kernel_step, kernel_type, kernel_width, kernel_height, width, height,
                                stype, dtype, borderType, delta, anchor_x, anchor_y, isSubmatrix, src_data == dst_data);
    if (res != CV_HAL_ERROR_OK)
        return false;
    res = cv_hal_filter(ctx, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    bool success = (res == CV_HAL_ERROR_OK);
    res = cv_hal_filterFree(ctx);
    if (res != CV_HAL_ERROR_OK)
        return false;
    return success;
}

#ifdef HAVE_IPP
static bool ippFilter2D(int stype, int dtype, int kernel_type,
              uchar * src_data, size_t src_step,
              uchar * dst_data, size_t dst_step,
              int width, int height,
              int full_width, int full_height,
              int offset_x, int offset_y,
              uchar * kernel_data, size_t kernel_step,
              int kernel_width, int kernel_height,
              int anchor_x, int anchor_y,
              double delta, int borderType,
              bool isSubmatrix)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

    ::ipp::IwiSize  iwSize(width, height);
    ::ipp::IwiSize  kernelSize(kernel_width, kernel_height);
    IppDataType     type        = ippiGetDataType(CV_MAT_DEPTH(stype));
    int             channels    = CV_MAT_CN(stype);

    CV_UNUSED(isSubmatrix);

#if IPP_VERSION_X100 >= 201700 && IPP_VERSION_X100 <= 201702 // IPP bug with 1x1 kernel
    if(kernel_width == 1 && kernel_height == 1)
        return false;
#endif

#if IPP_DISABLE_FILTER2D_BIG_MASK
    // Too big difference compared to OpenCV FFT-based convolution
    if(kernel_type == CV_32FC1 && (type == ipp16s || type == ipp16u) && (kernel_width > 7 || kernel_height > 7))
        return false;

    // Poor optimization for big kernels
    if(kernel_width > 7 || kernel_height > 7)
        return false;
#endif

    if(src_data == dst_data)
        return false;

    if(stype != dtype)
        return false;

    if(kernel_type != CV_16SC1 && kernel_type != CV_32FC1)
        return false;

    // TODO: Implement offset for 8u, 16u
    if(std::fabs(delta) >= DBL_EPSILON)
        return false;

    if(!ippiCheckAnchor(anchor_x, anchor_y, kernel_width, kernel_height))
        return false;

    try
    {
        ::ipp::IwiBorderSize    iwBorderSize;
        ::ipp::IwiBorderType    iwBorderType;
        ::ipp::IwiImage         iwKernel(ippiSize(kernel_width, kernel_height), ippiGetDataType(CV_MAT_DEPTH(kernel_type)), CV_MAT_CN(kernel_type), 0, (void*)kernel_data, kernel_step);
        ::ipp::IwiImage         iwSrc(iwSize, type, channels, ::ipp::IwiBorderSize(offset_x, offset_y, full_width-offset_x-width, full_height-offset_y-height), (void*)src_data, src_step);
        ::ipp::IwiImage         iwDst(iwSize, type, channels, ::ipp::IwiBorderSize(offset_x, offset_y, full_width-offset_x-width, full_height-offset_y-height), (void*)dst_data, dst_step);

        iwBorderSize = ::ipp::iwiSizeToBorderSize(kernelSize);
        iwBorderType = ippiGetBorder(iwSrc, borderType, iwBorderSize);
        if(!iwBorderType)
            return false;

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilter, iwSrc, iwDst, iwKernel, ::ipp::IwiFilterParams(1, 0, ippAlgHintNone, ippRndFinancial), iwBorderType);
    }
    catch(const ::ipp::IwException& ex)
    {
        CV_UNUSED(ex);
        return false;
    }

    return true;
#else
    CV_UNUSED(stype); CV_UNUSED(dtype); CV_UNUSED(kernel_type); CV_UNUSED(src_data); CV_UNUSED(src_step);
    CV_UNUSED(dst_data); CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(full_width);
    CV_UNUSED(full_height); CV_UNUSED(offset_x); CV_UNUSED(offset_y); CV_UNUSED(kernel_data); CV_UNUSED(kernel_step);
    CV_UNUSED(kernel_width); CV_UNUSED(kernel_height); CV_UNUSED(anchor_x); CV_UNUSED(anchor_y); CV_UNUSED(delta);
    CV_UNUSED(borderType); CV_UNUSED(isSubmatrix);
    return false;
#endif
}
#endif

static bool dftFilter2D(int stype, int dtype, int kernel_type,
                        uchar * src_data, size_t src_step,
                        uchar * dst_data, size_t dst_step,
                        int full_width, int full_height,
                        int offset_x, int offset_y,
                        uchar * kernel_data, size_t kernel_step,
                        int kernel_width, int kernel_height,
                        int anchor_x, int anchor_y,
                        double delta, int borderType)
{
    {
        int sdepth = CV_MAT_DEPTH(stype);
        int ddepth = CV_MAT_DEPTH(dtype);
        int dft_filter_size = checkHardwareSupport(CV_CPU_SSE3) && ((sdepth == CV_8U && (ddepth == CV_8U || ddepth == CV_16S)) || (sdepth == CV_32F && ddepth == CV_32F)) ? 130 : 50;
        if (kernel_width * kernel_height < dft_filter_size)
            return false;
    }

    Point anchor = Point(anchor_x, anchor_y);
    Mat kernel = Mat(Size(kernel_width, kernel_height), kernel_type, kernel_data, kernel_step);

    Mat src(Size(full_width-offset_x, full_height-offset_y), stype, src_data, src_step);
    Mat dst(Size(full_width, full_height), dtype, dst_data, dst_step);
    Mat temp;
    int src_channels = CV_MAT_CN(stype);
    int dst_channels = CV_MAT_CN(dtype);
    int ddepth = CV_MAT_DEPTH(dtype);
    // crossCorr doesn't accept non-zero delta with multiple channels
    if (src_channels != 1 && delta != 0) {
        // The semantics of filter2D require that the delta be applied
        // as floating-point math.  So wee need an intermediate Mat
        // with a float datatype.  If the dest is already floats,
        // we just use that.
        int corrDepth = ddepth;
        if ((ddepth == CV_32F || ddepth == CV_64F) && src_data != dst_data) {
            temp = Mat(Size(full_width, full_height), dtype, dst_data, dst_step);
        } else {
            corrDepth = ddepth == CV_64F ? CV_64F : CV_32F;
            temp.create(Size(full_width, full_height), CV_MAKETYPE(corrDepth, dst_channels));
        }
        crossCorr(src, kernel, temp, src.size(),
                  CV_MAKETYPE(corrDepth, src_channels),
                  anchor, 0, borderType);
        add(temp, delta, temp);
        if (temp.data != dst_data) {
            temp.convertTo(dst, dst.type());
        }
    } else {
        if (src_data != dst_data)
            temp = Mat(Size(full_width, full_height), dtype, dst_data, dst_step);
        else
            temp.create(Size(full_width, full_height), dtype);
        crossCorr(src, kernel, temp, src.size(),
                  CV_MAKETYPE(ddepth, src_channels),
                  anchor, delta, borderType);
        if (temp.data != dst_data)
            temp.copyTo(dst);
    }
    return true;
}

static void ocvFilter2D(int stype, int dtype, int kernel_type,
                        uchar * src_data, size_t src_step,
                        uchar * dst_data, size_t dst_step,
                        int width, int height,
                        int full_width, int full_height,
                        int offset_x, int offset_y,
                        uchar * kernel_data, size_t kernel_step,
                        int kernel_width, int kernel_height,
                        int anchor_x, int anchor_y,
                        double delta, int borderType)
{
    int borderTypeValue = borderType & ~BORDER_ISOLATED;
    Mat kernel = Mat(Size(kernel_width, kernel_height), kernel_type, kernel_data, kernel_step);
    Ptr<FilterEngine> f = createLinearFilter(stype, dtype, kernel, Point(anchor_x, anchor_y), delta,
                                             borderTypeValue);
    Mat src(Size(width, height), stype, src_data, src_step);
    Mat dst(Size(width, height), dtype, dst_data, dst_step);
    f->apply(src, dst, Size(full_width, full_height), Point(offset_x, offset_y));
}

static bool replacementSepFilter(int stype, int dtype, int ktype,
                                 uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                                 int width, int height, int full_width, int full_height,
                                 int offset_x, int offset_y,
                                 uchar * kernelx_data, int kernelx_len,
                                 uchar * kernely_data, int kernely_len,
                                 int anchor_x, int anchor_y, double delta, int borderType)
{
    cvhalFilter2D *ctx;
    int res = cv_hal_sepFilterInit(&ctx, stype, dtype, ktype,
                                   kernelx_data, kernelx_len,
                                   kernely_data, kernely_len,
                                   anchor_x, anchor_y, delta, borderType);
    if (res != CV_HAL_ERROR_OK)
        return false;
    res = cv_hal_sepFilter(ctx, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    bool success = (res == CV_HAL_ERROR_OK);
    res = cv_hal_sepFilterFree(ctx);
    if (res != CV_HAL_ERROR_OK)
        return false;
    return success;
}

static void ocvSepFilter(int stype, int dtype, int ktype,
                         uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                         int width, int height, int full_width, int full_height,
                         int offset_x, int offset_y,
                         uchar * kernelx_data, int kernelx_len,
                         uchar * kernely_data, int kernely_len,
                         int anchor_x, int anchor_y, double delta, int borderType)
{
    Mat kernelX(Size(kernelx_len, 1), ktype, kernelx_data);
    Mat kernelY(Size(kernely_len, 1), ktype, kernely_data);
    Ptr<FilterEngine> f = createSeparableLinearFilter(stype, dtype, kernelX, kernelY,
                                                      Point(anchor_x, anchor_y),
                                                      delta, borderType & ~BORDER_ISOLATED);
    Mat src(Size(width, height), stype, src_data, src_step);
    Mat dst(Size(width, height), dtype, dst_data, dst_step);
    f->apply(src, dst, Size(full_width, full_height), Point(offset_x, offset_y));
};

//===================================================================
//       HAL functions
//===================================================================

namespace cv {
namespace hal {


CV_DEPRECATED Ptr<hal::Filter2D> Filter2D::create(uchar * , size_t , int ,
                                 int , int ,
                                 int , int ,
                                 int , int ,
                                 int , double ,
                                 int , int ,
                                 bool , bool ) { return Ptr<hal::Filter2D>(); }

CV_DEPRECATED Ptr<hal::SepFilter2D> SepFilter2D::create(int , int , int ,
                                    uchar * , int ,
                                    uchar * , int ,
                                    int , int ,
                                    double , int )  { return Ptr<hal::SepFilter2D>(); }


void filter2D(int stype, int dtype, int kernel_type,
              uchar * src_data, size_t src_step,
              uchar * dst_data, size_t dst_step,
              int width, int height,
              int full_width, int full_height,
              int offset_x, int offset_y,
              uchar * kernel_data, size_t kernel_step,
              int kernel_width, int kernel_height,
              int anchor_x, int anchor_y,
              double delta, int borderType,
              bool isSubmatrix)
{
    bool res;
    res = replacementFilter2D(stype, dtype, kernel_type,
                              src_data, src_step,
                              dst_data, dst_step,
                              width, height,
                              full_width, full_height,
                              offset_x, offset_y,
                              kernel_data, kernel_step,
                              kernel_width, kernel_height,
                              anchor_x, anchor_y,
                              delta, borderType, isSubmatrix);
    if (res)
        return;

    CV_IPP_RUN_FAST(ippFilter2D(stype, dtype, kernel_type,
                              src_data, src_step,
                              dst_data, dst_step,
                              width, height,
                              full_width, full_height,
                              offset_x, offset_y,
                              kernel_data, kernel_step,
                              kernel_width, kernel_height,
                              anchor_x, anchor_y,
                              delta, borderType, isSubmatrix))

    res = dftFilter2D(stype, dtype, kernel_type,
                      src_data, src_step,
                      dst_data, dst_step,
                      full_width, full_height,
                      offset_x, offset_y,
                      kernel_data, kernel_step,
                      kernel_width, kernel_height,
                      anchor_x, anchor_y,
                      delta, borderType);
    if (res)
        return;
    ocvFilter2D(stype, dtype, kernel_type,
                src_data, src_step,
                dst_data, dst_step,
                width, height,
                full_width, full_height,
                offset_x, offset_y,
                kernel_data, kernel_step,
                kernel_width, kernel_height,
                anchor_x, anchor_y,
                delta, borderType);
}

//---------------------------------------------------------------

void sepFilter2D(int stype, int dtype, int ktype,
                 uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                 int width, int height, int full_width, int full_height,
                 int offset_x, int offset_y,
                 uchar * kernelx_data, int kernelx_len,
                 uchar * kernely_data, int kernely_len,
                 int anchor_x, int anchor_y, double delta, int borderType)
{

    bool res = replacementSepFilter(stype, dtype, ktype,
                                    src_data, src_step, dst_data, dst_step,
                                    width, height, full_width, full_height,
                                    offset_x, offset_y,
                                    kernelx_data, kernelx_len,
                                    kernely_data, kernely_len,
                                    anchor_x, anchor_y, delta, borderType);
    if (res)
        return;
    ocvSepFilter(stype, dtype, ktype,
                 src_data, src_step, dst_data, dst_step,
                 width, height, full_width, full_height,
                 offset_x, offset_y,
                 kernelx_data, kernelx_len,
                 kernely_data, kernely_len,
                 anchor_x, anchor_y, delta, borderType);
}

} // cv::hal::
} // cv::

//================================================================
//   Main interface
//================================================================

void cv::filter2D( InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor0,
                   double delta, int borderType )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_filter2D(_src, _dst, ddepth, _kernel, anchor0, delta, borderType))

    Mat src = _src.getMat(), kernel = _kernel.getMat();

    if( ddepth < 0 )
        ddepth = src.depth();

    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();
    Point anchor = normalizeAnchor(anchor0, kernel.size());

    Point ofs;
    Size wsz(src.cols, src.rows);
    if( (borderType & BORDER_ISOLATED) == 0 )
        src.locateROI( wsz, ofs );

    hal::filter2D(src.type(), dst.type(), kernel.type(),
                  src.data, src.step, dst.data, dst.step,
                  dst.cols, dst.rows, wsz.width, wsz.height, ofs.x, ofs.y,
                  kernel.data, kernel.step,  kernel.cols, kernel.rows,
                  anchor.x, anchor.y,
                  delta, borderType, src.isSubmatrix());
}

void cv::sepFilter2D( InputArray _src, OutputArray _dst, int ddepth,
                      InputArray _kernelX, InputArray _kernelY, Point anchor,
                      double delta, int borderType )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2 && (size_t)_src.rows() > _kernelY.total() && (size_t)_src.cols() > _kernelX.total(),
               ocl_sepFilter2D(_src, _dst, ddepth, _kernelX, _kernelY, anchor, delta, borderType))

    Mat src = _src.getMat(), kernelX = _kernelX.getMat(), kernelY = _kernelY.getMat();

    if( ddepth < 0 )
        ddepth = src.depth();

    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();

    Point ofs;
    Size wsz(src.cols, src.rows);
    if( (borderType & BORDER_ISOLATED) == 0 )
        src.locateROI( wsz, ofs );

    CV_Assert( kernelX.type() == kernelY.type() &&
               (kernelX.cols == 1 || kernelX.rows == 1) &&
               (kernelY.cols == 1 || kernelY.rows == 1) );

    Mat contKernelX = kernelX.isContinuous() ? kernelX : kernelX.clone();
    Mat contKernelY = kernelY.isContinuous() ? kernelY : kernelY.clone();

    hal::sepFilter2D(src.type(), dst.type(), kernelX.type(),
                     src.data, src.step, dst.data, dst.step,
                     dst.cols, dst.rows, wsz.width, wsz.height, ofs.x, ofs.y,
                     contKernelX.data, kernelX.cols + kernelX.rows - 1,
                     contKernelY.data, kernelY.cols + kernelY.rows - 1,
                     anchor.x, anchor.y, delta, borderType & ~BORDER_ISOLATED);
}


CV_IMPL void
cvFilter2D( const CvArr* srcarr, CvArr* dstarr, const CvMat* _kernel, CvPoint anchor )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    cv::Mat kernel = cv::cvarrToMat(_kernel);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() );

    cv::filter2D( src, dst, dst.depth(), kernel, anchor, 0, cv::BORDER_REPLICATE );
}

/* End of file. */
