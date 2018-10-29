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


#if CV_SSE2

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
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        int* dst = (int*)_dst;
        const int* _kx = kernel.ptr<int>();
        width *= cn;

        if( smallValues )
        {
            __m128i z = _mm_setzero_si128();
            for( ; i <= width - 8; i += 8 )
            {
                const uchar* src = _src + i;
                __m128i s0 = z, s1 = z;

                for( k = 0; k < _ksize; k++, src += cn )
                {
                    __m128i f = _mm_cvtsi32_si128(_kx[k]);
                    f = _mm_shuffle_epi32(f, 0);

                    __m128i x0 = _mm_loadl_epi64((const __m128i*)src);
                    x0 = _mm_unpacklo_epi8(x0, z);

                    __m128i x1 = _mm_unpackhi_epi16(x0, z);
                    x0 = _mm_unpacklo_epi16(x0, z);

                    x0 = _mm_madd_epi16(x0, f);
                    x1 = _mm_madd_epi16(x1, f);

                    s0 = _mm_add_epi32(s0, x0);
                    s1 = _mm_add_epi32(s1, x1);
                }

                _mm_store_si128((__m128i*)(dst + i), s0);
                _mm_store_si128((__m128i*)(dst + i + 4), s1);
            }

            if( i <= width - 4 )
            {
                const uchar* src = _src + i;
                __m128i s0 = z;

                for( k = 0; k < _ksize; k++, src += cn )
                {
                    __m128i f = _mm_cvtsi32_si128(_kx[k]);
                    f = _mm_shuffle_epi32(f, 0);

                    __m128i x0 = _mm_cvtsi32_si128(*(const int*)src);
                    x0 = _mm_unpacklo_epi8(x0, z);
                    x0 = _mm_unpacklo_epi16(x0, z);
                    x0 = _mm_madd_epi16(x0, f);
                    s0 = _mm_add_epi32(s0, x0);
                }
                _mm_store_si128((__m128i*)(dst + i), s0);
                i += 4;
            }
        }
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
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        int i = 0, j, k, _ksize = kernel.rows + kernel.cols - 1;
        int* dst = (int*)_dst;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int* kx = kernel.ptr<int>() + _ksize/2;
        if( !smallValues )
            return 0;

        src += (_ksize/2)*cn;
        width *= cn;

        __m128i z = _mm_setzero_si128();
        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_si128((__m128i*)(src - cn));
                        x1 = _mm_loadu_si128((__m128i*)src);
                        x2 = _mm_loadu_si128((__m128i*)(src + cn));
                        y0 = _mm_unpackhi_epi8(x0, z);
                        x0 = _mm_unpacklo_epi8(x0, z);
                        y1 = _mm_unpackhi_epi8(x1, z);
                        x1 = _mm_unpacklo_epi8(x1, z);
                        y2 = _mm_unpackhi_epi8(x2, z);
                        x2 = _mm_unpacklo_epi8(x2, z);
                        x0 = _mm_add_epi16(x0, _mm_add_epi16(_mm_add_epi16(x1, x1), x2));
                        y0 = _mm_add_epi16(y0, _mm_add_epi16(_mm_add_epi16(y1, y1), y2));
                        _mm_store_si128((__m128i*)(dst + i), _mm_unpacklo_epi16(x0, z));
                        _mm_store_si128((__m128i*)(dst + i + 4), _mm_unpackhi_epi16(x0, z));
                        _mm_store_si128((__m128i*)(dst + i + 8), _mm_unpacklo_epi16(y0, z));
                        _mm_store_si128((__m128i*)(dst + i + 12), _mm_unpackhi_epi16(y0, z));
                    }
                else if( kx[0] == -2 && kx[1] == 1 )
                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_si128((__m128i*)(src - cn));
                        x1 = _mm_loadu_si128((__m128i*)src);
                        x2 = _mm_loadu_si128((__m128i*)(src + cn));
                        y0 = _mm_unpackhi_epi8(x0, z);
                        x0 = _mm_unpacklo_epi8(x0, z);
                        y1 = _mm_unpackhi_epi8(x1, z);
                        x1 = _mm_unpacklo_epi8(x1, z);
                        y2 = _mm_unpackhi_epi8(x2, z);
                        x2 = _mm_unpacklo_epi8(x2, z);
                        x0 = _mm_add_epi16(x0, _mm_sub_epi16(x2, _mm_add_epi16(x1, x1)));
                        y0 = _mm_add_epi16(y0, _mm_sub_epi16(y2, _mm_add_epi16(y1, y1)));
                        _mm_store_si128((__m128i*)(dst + i), _mm_srai_epi32(_mm_unpacklo_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 4), _mm_srai_epi32(_mm_unpackhi_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 8), _mm_srai_epi32(_mm_unpacklo_epi16(y0, y0),16));
                        _mm_store_si128((__m128i*)(dst + i + 12), _mm_srai_epi32(_mm_unpackhi_epi16(y0, y0),16));
                    }
                else
                {
                    __m128i k0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[0]), 0),
                            k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0);
                    k1 = _mm_packs_epi32(k1, k1);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128i x0 = _mm_loadl_epi64((__m128i*)(src - cn));
                        __m128i x1 = _mm_loadl_epi64((__m128i*)src);
                        __m128i x2 = _mm_loadl_epi64((__m128i*)(src + cn));

                        x0 = _mm_unpacklo_epi8(x0, z);
                        x1 = _mm_unpacklo_epi8(x1, z);
                        x2 = _mm_unpacklo_epi8(x2, z);
                        __m128i x3 = _mm_unpacklo_epi16(x0, x2);
                        __m128i x4 = _mm_unpackhi_epi16(x0, x2);
                        __m128i x5 = _mm_unpacklo_epi16(x1, z);
                        __m128i x6 = _mm_unpackhi_epi16(x1, z);
                        x3 = _mm_madd_epi16(x3, k1);
                        x4 = _mm_madd_epi16(x4, k1);
                        x5 = _mm_madd_epi16(x5, k0);
                        x6 = _mm_madd_epi16(x6, k0);
                        x3 = _mm_add_epi32(x3, x5);
                        x4 = _mm_add_epi32(x4, x6);

                        _mm_store_si128((__m128i*)(dst + i), x3);
                        _mm_store_si128((__m128i*)(dst + i + 4), x4);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_si128((__m128i*)(src - cn*2));
                        x1 = _mm_loadu_si128((__m128i*)src);
                        x2 = _mm_loadu_si128((__m128i*)(src + cn*2));
                        y0 = _mm_unpackhi_epi8(x0, z);
                        x0 = _mm_unpacklo_epi8(x0, z);
                        y1 = _mm_unpackhi_epi8(x1, z);
                        x1 = _mm_unpacklo_epi8(x1, z);
                        y2 = _mm_unpackhi_epi8(x2, z);
                        x2 = _mm_unpacklo_epi8(x2, z);
                        x0 = _mm_add_epi16(x0, _mm_sub_epi16(x2, _mm_add_epi16(x1, x1)));
                        y0 = _mm_add_epi16(y0, _mm_sub_epi16(y2, _mm_add_epi16(y1, y1)));
                        _mm_store_si128((__m128i*)(dst + i), _mm_srai_epi32(_mm_unpacklo_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 4), _mm_srai_epi32(_mm_unpackhi_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 8), _mm_srai_epi32(_mm_unpacklo_epi16(y0, y0),16));
                        _mm_store_si128((__m128i*)(dst + i + 12), _mm_srai_epi32(_mm_unpackhi_epi16(y0, y0),16));
                    }
                else
                {
                    __m128i k0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[0]), 0),
                            k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0),
                            k2 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[2]), 0);
                    k1 = _mm_packs_epi32(k1, k1);
                    k2 = _mm_packs_epi32(k2, k2);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128i x0 = _mm_loadl_epi64((__m128i*)src);

                        x0 = _mm_unpacklo_epi8(x0, z);
                        __m128i x1 = _mm_unpacklo_epi16(x0, z);
                        __m128i x2 = _mm_unpackhi_epi16(x0, z);
                        x1 = _mm_madd_epi16(x1, k0);
                        x2 = _mm_madd_epi16(x2, k0);

                        __m128i x3 = _mm_loadl_epi64((__m128i*)(src - cn));
                        __m128i x4 = _mm_loadl_epi64((__m128i*)(src + cn));

                        x3 = _mm_unpacklo_epi8(x3, z);
                        x4 = _mm_unpacklo_epi8(x4, z);
                        __m128i x5 = _mm_unpacklo_epi16(x3, x4);
                        __m128i x6 = _mm_unpackhi_epi16(x3, x4);
                        x5 = _mm_madd_epi16(x5, k1);
                        x6 = _mm_madd_epi16(x6, k1);
                        x1 = _mm_add_epi32(x1, x5);
                        x2 = _mm_add_epi32(x2, x6);

                        x3 = _mm_loadl_epi64((__m128i*)(src - cn*2));
                        x4 = _mm_loadl_epi64((__m128i*)(src + cn*2));

                        x3 = _mm_unpacklo_epi8(x3, z);
                        x4 = _mm_unpacklo_epi8(x4, z);
                        x5 = _mm_unpacklo_epi16(x3, x4);
                        x6 = _mm_unpackhi_epi16(x3, x4);
                        x5 = _mm_madd_epi16(x5, k2);
                        x6 = _mm_madd_epi16(x6, k2);
                        x1 = _mm_add_epi32(x1, x5);
                        x2 = _mm_add_epi32(x2, x6);

                        _mm_store_si128((__m128i*)(dst + i), x1);
                        _mm_store_si128((__m128i*)(dst + i + 4), x2);
                    }
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, y0;
                        x0 = _mm_loadu_si128((__m128i*)(src + cn));
                        x1 = _mm_loadu_si128((__m128i*)(src - cn));
                        y0 = _mm_sub_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x1, z));
                        x0 = _mm_sub_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x1, z));
                        _mm_store_si128((__m128i*)(dst + i), _mm_srai_epi32(_mm_unpacklo_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 4), _mm_srai_epi32(_mm_unpackhi_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 8), _mm_srai_epi32(_mm_unpacklo_epi16(y0, y0),16));
                        _mm_store_si128((__m128i*)(dst + i + 12), _mm_srai_epi32(_mm_unpackhi_epi16(y0, y0),16));
                    }
                else
                {
                    __m128i k0 = _mm_set_epi32(-kx[1], kx[1], -kx[1], kx[1]);
                    k0 = _mm_packs_epi32(k0, k0);

                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0 = _mm_loadu_si128((__m128i*)(src + cn));
                        __m128i x1 = _mm_loadu_si128((__m128i*)(src - cn));

                        __m128i x2 = _mm_unpacklo_epi8(x0, z);
                        __m128i x3 = _mm_unpacklo_epi8(x1, z);
                        __m128i x4 = _mm_unpackhi_epi8(x0, z);
                        __m128i x5 = _mm_unpackhi_epi8(x1, z);
                        __m128i x6 = _mm_unpacklo_epi16(x2, x3);
                        __m128i x7 = _mm_unpacklo_epi16(x4, x5);
                        __m128i x8 = _mm_unpackhi_epi16(x2, x3);
                        __m128i x9 = _mm_unpackhi_epi16(x4, x5);
                        x6 = _mm_madd_epi16(x6, k0);
                        x7 = _mm_madd_epi16(x7, k0);
                        x8 = _mm_madd_epi16(x8, k0);
                        x9 = _mm_madd_epi16(x9, k0);

                        _mm_store_si128((__m128i*)(dst + i), x6);
                        _mm_store_si128((__m128i*)(dst + i + 4), x8);
                        _mm_store_si128((__m128i*)(dst + i + 8), x7);
                        _mm_store_si128((__m128i*)(dst + i + 12), x9);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                __m128i k0 = _mm_loadl_epi64((__m128i*)(kx + 1));
                k0 = _mm_unpacklo_epi64(k0, k0);
                k0 = _mm_packs_epi32(k0, k0);

                for( ; i <= width - 16; i += 16, src += 16 )
                {
                    __m128i x0 = _mm_loadu_si128((__m128i*)(src + cn));
                    __m128i x1 = _mm_loadu_si128((__m128i*)(src - cn));

                    __m128i x2 = _mm_unpackhi_epi8(x0, z);
                    __m128i x3 = _mm_unpackhi_epi8(x1, z);
                    x0 = _mm_unpacklo_epi8(x0, z);
                    x1 = _mm_unpacklo_epi8(x1, z);
                    __m128i x5 = _mm_sub_epi16(x2, x3);
                    __m128i x4 = _mm_sub_epi16(x0, x1);

                    __m128i x6 = _mm_loadu_si128((__m128i*)(src + cn * 2));
                    __m128i x7 = _mm_loadu_si128((__m128i*)(src - cn * 2));

                    __m128i x8 = _mm_unpackhi_epi8(x6, z);
                    __m128i x9 = _mm_unpackhi_epi8(x7, z);
                    x6 = _mm_unpacklo_epi8(x6, z);
                    x7 = _mm_unpacklo_epi8(x7, z);
                    __m128i x11 = _mm_sub_epi16(x8, x9);
                    __m128i x10 = _mm_sub_epi16(x6, x7);

                    __m128i x13 = _mm_unpackhi_epi16(x5, x11);
                    __m128i x12 = _mm_unpackhi_epi16(x4, x10);
                    x5 = _mm_unpacklo_epi16(x5, x11);
                    x4 = _mm_unpacklo_epi16(x4, x10);
                    x5 = _mm_madd_epi16(x5, k0);
                    x4 = _mm_madd_epi16(x4, k0);
                    x13 = _mm_madd_epi16(x13, k0);
                    x12 = _mm_madd_epi16(x12, k0);

                    _mm_store_si128((__m128i*)(dst + i), x4);
                    _mm_store_si128((__m128i*)(dst + i + 4), x12);
                    _mm_store_si128((__m128i*)(dst + i + 8), x5);
                    _mm_store_si128((__m128i*)(dst + i + 12), x13);
                }
            }
        }

        src -= (_ksize/2)*cn;
        kx -= _ksize/2;
        for( ; i <= width - 4; i += 4, src += 4 )
        {
            __m128i s0 = z;

            for( k = j = 0; k < _ksize; k++, j += cn )
            {
                __m128i f = _mm_cvtsi32_si128(kx[k]);
                f = _mm_shuffle_epi32(f, 0);

                __m128i x0 = _mm_cvtsi32_si128(*(const int*)(src + j));
                x0 = _mm_unpacklo_epi8(x0, z);
                x0 = _mm_unpacklo_epi16(x0, z);
                x0 = _mm_madd_epi16(x0, f);
                s0 = _mm_add_epi32(s0, x0);
            }
            _mm_store_si128((__m128i*)(dst + i), s0);
        }

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
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;
        const __m128i *S, *S2;
        __m128 d4 = _mm_set1_ps(delta);

        if( symmetrical )
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 s0, s1, s2, s3;
                __m128i x0, x1;
                S = (const __m128i*)(src[0] + i);
                s0 = _mm_cvtepi32_ps(_mm_load_si128(S));
                s1 = _mm_cvtepi32_ps(_mm_load_si128(S+1));
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);
                s1 = _mm_add_ps(_mm_mul_ps(s1, f), d4);
                s2 = _mm_cvtepi32_ps(_mm_load_si128(S+2));
                s3 = _mm_cvtepi32_ps(_mm_load_si128(S+3));
                s2 = _mm_add_ps(_mm_mul_ps(s2, f), d4);
                s3 = _mm_add_ps(_mm_mul_ps(s3, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = (const __m128i*)(src[k] + i);
                    S2 = (const __m128i*)(src[-k] + i);
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_add_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                    x1 = _mm_add_epi32(_mm_load_si128(S+1), _mm_load_si128(S2+1));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                    x0 = _mm_add_epi32(_mm_load_si128(S+2), _mm_load_si128(S2+2));
                    x1 = _mm_add_epi32(_mm_load_si128(S+3), _mm_load_si128(S2+3));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                }

                x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
                x0 = _mm_packus_epi16(x0, x1);
                _mm_storeu_si128((__m128i*)(dst + i), x0);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128i x0;
                __m128 s0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[0] + i)));
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = (const __m128i*)(src[k] + i);
                    S2 = (const __m128i*)(src[-k] + i);
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_add_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                }

                x0 = _mm_cvtps_epi32(s0);
                x0 = _mm_packs_epi32(x0, x0);
                x0 = _mm_packus_epi16(x0, x0);
                *(int*)(dst + i) = _mm_cvtsi128_si32(x0);
            }
        }
        else
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f, s0 = d4, s1 = d4, s2 = d4, s3 = d4;
                __m128i x0, x1;

                for( k = 1; k <= ksize2; k++ )
                {
                    S = (const __m128i*)(src[k] + i);
                    S2 = (const __m128i*)(src[-k] + i);
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                    x1 = _mm_sub_epi32(_mm_load_si128(S+1), _mm_load_si128(S2+1));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                    x0 = _mm_sub_epi32(_mm_load_si128(S+2), _mm_load_si128(S2+2));
                    x1 = _mm_sub_epi32(_mm_load_si128(S+3), _mm_load_si128(S2+3));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                }

                x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
                x0 = _mm_packus_epi16(x0, x1);
                _mm_storeu_si128((__m128i*)(dst + i), x0);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f, s0 = d4;
                __m128i x0;

                for( k = 1; k <= ksize2; k++ )
                {
                    S = (const __m128i*)(src[k] + i);
                    S2 = (const __m128i*)(src[-k] + i);
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                }

                x0 = _mm_cvtps_epi32(s0);
                x0 = _mm_packs_epi32(x0, x0);
                x0 = _mm_packus_epi16(x0, x0);
                *(int*)(dst + i) = _mm_cvtsi128_si32(x0);
            }
        }

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
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;
        const int *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        short* dst = (short*)_dst;
        __m128 df4 = _mm_set1_ps(delta);
        __m128i d4 = _mm_cvtps_epi32(df4);

        if( symmetrical )
        {
            if( ky[0] == 2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128i s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_si128((__m128i*)(S0 + i));
                    s1 = _mm_load_si128((__m128i*)(S0 + i + 4));
                    s2 = _mm_load_si128((__m128i*)(S1 + i));
                    s3 = _mm_load_si128((__m128i*)(S1 + i + 4));
                    s4 = _mm_load_si128((__m128i*)(S2 + i));
                    s5 = _mm_load_si128((__m128i*)(S2 + i + 4));
                    s0 = _mm_add_epi32(s0, _mm_add_epi32(s4, _mm_add_epi32(s2, s2)));
                    s1 = _mm_add_epi32(s1, _mm_add_epi32(s5, _mm_add_epi32(s3, s3)));
                    s0 = _mm_add_epi32(s0, d4);
                    s1 = _mm_add_epi32(s1, d4);
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0, s1));
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128i s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_si128((__m128i*)(S0 + i));
                    s1 = _mm_load_si128((__m128i*)(S0 + i + 4));
                    s2 = _mm_load_si128((__m128i*)(S1 + i));
                    s3 = _mm_load_si128((__m128i*)(S1 + i + 4));
                    s4 = _mm_load_si128((__m128i*)(S2 + i));
                    s5 = _mm_load_si128((__m128i*)(S2 + i + 4));
                    s0 = _mm_add_epi32(s0, _mm_sub_epi32(s4, _mm_add_epi32(s2, s2)));
                    s1 = _mm_add_epi32(s1, _mm_sub_epi32(s5, _mm_add_epi32(s3, s3)));
                    s0 = _mm_add_epi32(s0, d4);
                    s1 = _mm_add_epi32(s1, d4);
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0, s1));
                }
            }
            else
            {
                __m128 k0 = _mm_set1_ps(ky[0]), k1 = _mm_set1_ps(ky[1]);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1;
                    s0 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(S1 + i)));
                    s1 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(S1 + i + 4)));
                    s0 = _mm_add_ps(_mm_mul_ps(s0, k0), df4);
                    s1 = _mm_add_ps(_mm_mul_ps(s1, k0), df4);
                    __m128i x0, x1;
                    x0 = _mm_add_epi32(_mm_load_si128((__m128i*)(S0 + i)),
                                       _mm_load_si128((__m128i*)(S2 + i)));
                    x1 = _mm_add_epi32(_mm_load_si128((__m128i*)(S0 + i + 4)),
                                       _mm_load_si128((__m128i*)(S2 + i + 4)));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0),k1));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1),k1));
                    x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                    _mm_storeu_si128((__m128i*)(dst + i), x0);
                }
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128i s0, s1, s2, s3;
                    s0 = _mm_load_si128((__m128i*)(S2 + i));
                    s1 = _mm_load_si128((__m128i*)(S2 + i + 4));
                    s2 = _mm_load_si128((__m128i*)(S0 + i));
                    s3 = _mm_load_si128((__m128i*)(S0 + i + 4));
                    s0 = _mm_add_epi32(_mm_sub_epi32(s0, s2), d4);
                    s1 = _mm_add_epi32(_mm_sub_epi32(s1, s3), d4);
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0, s1));
                }
            }
            else
            {
                __m128 k1 = _mm_set1_ps(ky[1]);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0 = df4, s1 = df4;
                    __m128i x0, x1;
                    x0 = _mm_sub_epi32(_mm_load_si128((__m128i*)(S2 + i)),
                                       _mm_load_si128((__m128i*)(S0 + i)));
                    x1 = _mm_sub_epi32(_mm_load_si128((__m128i*)(S2 + i + 4)),
                                       _mm_load_si128((__m128i*)(S0 + i + 4)));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0),k1));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1),k1));
                    x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                    _mm_storeu_si128((__m128i*)(dst + i), x0);
                }
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


/////////////////////////////////////// 16s //////////////////////////////////

struct RowVec_16s32f
{
    RowVec_16s32f() { sse2_supported = false; }
    RowVec_16s32f( const Mat& _kernel )
    {
        kernel = _kernel;
        sse2_supported = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        if( !sse2_supported )
            return 0;

        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* _kx = kernel.ptr<float>();
        width *= cn;

        for( ; i <= width - 8; i += 8 )
        {
            const short* src = (const short*)_src + i;
            __m128 f, s0 = _mm_setzero_ps(), s1 = s0, x0, x1;
            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = _mm_load_ss(_kx+k);
                f = _mm_shuffle_ps(f, f, 0);

                __m128i x0i = _mm_loadu_si128((const __m128i*)src);
                __m128i x1i = _mm_srai_epi32(_mm_unpackhi_epi16(x0i, x0i), 16);
                x0i = _mm_srai_epi32(_mm_unpacklo_epi16(x0i, x0i), 16);
                x0 = _mm_cvtepi32_ps(x0i);
                x1 = _mm_cvtepi32_ps(x1i);
                s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
            }
            _mm_store_ps(dst + i, s0);
            _mm_store_ps(dst + i + 4, s1);
        }
        return i;
    }

    Mat kernel;
    bool sse2_supported;
};


struct SymmColumnVec_32f16s
{
    SymmColumnVec_32f16s() { symmetryType=0; delta = 0; sse2_supported = false; }
    SymmColumnVec_32f16s(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
        sse2_supported = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        if( !sse2_supported )
            return 0;

        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S, *S2;
        short* dst = (short*)_dst;
        __m128 d4 = _mm_set1_ps(delta);

        if( symmetrical )
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 s0, s1, s2, s3;
                __m128 x0, x1;
                S = src[0] + i;
                s0 = _mm_load_ps(S);
                s1 = _mm_load_ps(S+4);
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);
                s1 = _mm_add_ps(_mm_mul_ps(s1, f), d4);
                s2 = _mm_load_ps(S+8);
                s3 = _mm_load_ps(S+12);
                s2 = _mm_add_ps(_mm_mul_ps(s2, f), d4);
                s3 = _mm_add_ps(_mm_mul_ps(s3, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_add_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_add_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_add_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_add_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }

                __m128i s0i = _mm_cvtps_epi32(s0);
                __m128i s1i = _mm_cvtps_epi32(s1);
                __m128i s2i = _mm_cvtps_epi32(s2);
                __m128i s3i = _mm_cvtps_epi32(s3);

                _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0i, s1i));
                _mm_storeu_si128((__m128i*)(dst + i + 8), _mm_packs_epi32(s2i, s3i));
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 x0, s0 = _mm_load_ps(src[0] + i);
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    x0 = _mm_add_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                }

                __m128i s0i = _mm_cvtps_epi32(s0);
                _mm_storel_epi64((__m128i*)(dst + i), _mm_packs_epi32(s0i, s0i));
            }
        }
        else
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f, s0 = d4, s1 = d4, s2 = d4, s3 = d4;
                __m128 x0, x1;
                S = src[0] + i;

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_sub_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_sub_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_sub_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }

                __m128i s0i = _mm_cvtps_epi32(s0);
                __m128i s1i = _mm_cvtps_epi32(s1);
                __m128i s2i = _mm_cvtps_epi32(s2);
                __m128i s3i = _mm_cvtps_epi32(s3);

                _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0i, s1i));
                _mm_storeu_si128((__m128i*)(dst + i + 8), _mm_packs_epi32(s2i, s3i));
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f, x0, s0 = d4;

                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                }

                __m128i s0i = _mm_cvtps_epi32(s0);
                _mm_storel_epi64((__m128i*)(dst + i), _mm_packs_epi32(s0i, s0i));
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
    bool sse2_supported;
};


/////////////////////////////////////// 32f //////////////////////////////////

struct RowVec_32f
{
    RowVec_32f()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE);
        haveAVX2 = CV_CPU_HAS_SUPPORT_AVX2;
#if defined USE_IPP_SEP_FILTERS
        bufsz = -1;
#endif
    }

    RowVec_32f( const Mat& _kernel )
    {
        kernel = _kernel;
        haveSSE = checkHardwareSupport(CV_CPU_SSE);
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
        const float* src0 = (const float*)_src;
        float* dst = (float*)_dst;
        const float* _kx = kernel.ptr<float>();

        if( !haveSSE )
            return 0;

        int i = 0, k;
        width *= cn;

#if CV_TRY_AVX2
        if (haveAVX2)
            return RowVec_32f_AVX(src0, _kx, dst, width, cn, _ksize);
#endif
        for( ; i <= width - 8; i += 8 )
        {
            const float* src = src0 + i;
            __m128 f, s0 = _mm_setzero_ps(), s1 = s0, x0, x1;
            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = _mm_set1_ps(_kx[k]);

                x0 = _mm_loadu_ps(src);
                x1 = _mm_loadu_ps(src + 4);
                s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
            }
            _mm_store_ps(dst + i, s0);
            _mm_store_ps(dst + i + 4, s1);
        }
        return i;
    }

    Mat kernel;
    bool haveSSE;
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
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;

        int i = 0, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* src = (const float*)_src + (_ksize/2)*cn;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float* kx = kernel.ptr<float>() + _ksize/2;
        width *= cn;

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn);
                        y0 = _mm_loadu_ps(src - cn + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn + 4);
                        x0 = _mm_add_ps(x0, _mm_add_ps(_mm_add_ps(x1, x1), x2));
                        y0 = _mm_add_ps(y0, _mm_add_ps(_mm_add_ps(y1, y1), y2));
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                else if( kx[0] == -2 && kx[1] == 1 )
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn);
                        y0 = _mm_loadu_ps(src - cn + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn + 4);
                        x0 = _mm_add_ps(x0, _mm_sub_ps(x2, _mm_add_ps(x1, x1)));
                        y0 = _mm_add_ps(y0, _mm_sub_ps(y2, _mm_add_ps(y1, y1)));
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                else
                {
                    __m128 k0 = _mm_set1_ps(kx[0]), k1 = _mm_set1_ps(kx[1]);
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn);
                        y0 = _mm_loadu_ps(src - cn + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn + 4);

                        x0 = _mm_mul_ps(_mm_add_ps(x0, x2), k1);
                        y0 = _mm_mul_ps(_mm_add_ps(y0, y2), k1);
                        x0 = _mm_add_ps(x0, _mm_mul_ps(x1, k0));
                        y0 = _mm_add_ps(y0, _mm_mul_ps(y1, k0));
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn*2);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn*2);
                        y0 = _mm_loadu_ps(src - cn*2 + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn*2 + 4);
                        x0 = _mm_add_ps(x0, _mm_sub_ps(x2, _mm_add_ps(x1, x1)));
                        y0 = _mm_add_ps(y0, _mm_sub_ps(y2, _mm_add_ps(y1, y1)));
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                else
                {
                    __m128 k0 = _mm_set1_ps(kx[0]), k1 = _mm_set1_ps(kx[1]), k2 = _mm_set1_ps(kx[2]);
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn);
                        y0 = _mm_loadu_ps(src - cn + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn + 4);

                        x0 = _mm_mul_ps(_mm_add_ps(x0, x2), k1);
                        y0 = _mm_mul_ps(_mm_add_ps(y0, y2), k1);
                        x0 = _mm_add_ps(x0, _mm_mul_ps(x1, k0));
                        y0 = _mm_add_ps(y0, _mm_mul_ps(y1, k0));

                        x2 = _mm_add_ps(_mm_loadu_ps(src + cn*2), _mm_loadu_ps(src - cn*2));
                        y2 = _mm_add_ps(_mm_loadu_ps(src + cn*2 + 4), _mm_loadu_ps(src - cn*2 + 4));
                        x0 = _mm_add_ps(x0, _mm_mul_ps(x2, k2));
                        y0 = _mm_add_ps(y0, _mm_mul_ps(y2, k2));

                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x2, y0, y2;
                        x0 = _mm_loadu_ps(src + cn);
                        x2 = _mm_loadu_ps(src - cn);
                        y0 = _mm_loadu_ps(src + cn + 4);
                        y2 = _mm_loadu_ps(src - cn + 4);
                        x0 = _mm_sub_ps(x0, x2);
                        y0 = _mm_sub_ps(y0, y2);
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                else
                {
                    __m128 k1 = _mm_set1_ps(kx[1]);
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x2, y0, y2;
                        x0 = _mm_loadu_ps(src + cn);
                        x2 = _mm_loadu_ps(src - cn);
                        y0 = _mm_loadu_ps(src + cn + 4);
                        y2 = _mm_loadu_ps(src - cn + 4);

                        x0 = _mm_mul_ps(_mm_sub_ps(x0, x2), k1);
                        y0 = _mm_mul_ps(_mm_sub_ps(y0, y2), k1);
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                __m128 k1 = _mm_set1_ps(kx[1]), k2 = _mm_set1_ps(kx[2]);
                for( ; i <= width - 8; i += 8, src += 8 )
                {
                    __m128 x0, x2, y0, y2;
                    x0 = _mm_loadu_ps(src + cn);
                    x2 = _mm_loadu_ps(src - cn);
                    y0 = _mm_loadu_ps(src + cn + 4);
                    y2 = _mm_loadu_ps(src - cn + 4);

                    x0 = _mm_mul_ps(_mm_sub_ps(x0, x2), k1);
                    y0 = _mm_mul_ps(_mm_sub_ps(y0, y2), k1);

                    x2 = _mm_sub_ps(_mm_loadu_ps(src + cn*2), _mm_loadu_ps(src - cn*2));
                    y2 = _mm_sub_ps(_mm_loadu_ps(src + cn*2 + 4), _mm_loadu_ps(src - cn*2 + 4));
                    x0 = _mm_add_ps(x0, _mm_mul_ps(x2, k2));
                    y0 = _mm_add_ps(y0, _mm_mul_ps(y2, k2));

                    _mm_store_ps(dst + i, x0);
                    _mm_store_ps(dst + i + 4, y0);
                }
            }
        }

        return i;
    }

    Mat kernel;
    int symmetryType;
};


struct SymmColumnVec_32f
{
    SymmColumnVec_32f() {
        symmetryType=0;
        haveSSE = checkHardwareSupport(CV_CPU_SSE);
        haveAVX2 = CV_CPU_HAS_SUPPORT_AVX2;
        delta = 0;
    }
    SymmColumnVec_32f(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        haveSSE = checkHardwareSupport(CV_CPU_SSE);
        haveAVX2 = CV_CPU_HAS_SUPPORT_AVX2;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        if( !haveSSE )
            return 0;

        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S, *S2;
        float* dst = (float*)_dst;

        if( symmetrical )
        {

#if CV_TRY_AVX2
            if (haveAVX2)
                return SymmColumnVec_32f_Symm_AVX(src, ky, dst, delta, width, ksize2);
#endif
            const __m128 d4 = _mm_set1_ps(delta);
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f = _mm_set1_ps(ky[0]);
                __m128 s0, s1, s2, s3;
                __m128 x0, x1;
                S = src[0] + i;
                s0 = _mm_load_ps(S);
                s1 = _mm_load_ps(S+4);
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);
                s1 = _mm_add_ps(_mm_mul_ps(s1, f), d4);
                s2 = _mm_load_ps(S+8);
                s3 = _mm_load_ps(S+12);
                s2 = _mm_add_ps(_mm_mul_ps(s2, f), d4);
                s3 = _mm_add_ps(_mm_mul_ps(s3, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_set1_ps(ky[k]);
                    x0 = _mm_add_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_add_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_add_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_add_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }

                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
                _mm_storeu_ps(dst + i + 8, s2);
                _mm_storeu_ps(dst + i + 12, s3);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f = _mm_set1_ps(ky[0]);
                __m128 x0, s0 = _mm_load_ps(src[0] + i);
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_set1_ps(ky[k]);
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    x0 = _mm_add_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                }

                _mm_storeu_ps(dst + i, s0);
            }
        }
        else
        {
#if CV_TRY_AVX2
            if (haveAVX2)
                return SymmColumnVec_32f_Unsymm_AVX(src, ky, dst, delta, width, ksize2);
#endif
            const __m128 d4 = _mm_set1_ps(delta);
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f, s0 = d4, s1 = d4, s2 = d4, s3 = d4;
                __m128 x0, x1;
                S = src[0] + i;

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_set1_ps(ky[k]);
                    x0 = _mm_sub_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_sub_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_sub_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_sub_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }

                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
                _mm_storeu_ps(dst + i + 8, s2);
                _mm_storeu_ps(dst + i + 12, s3);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f, x0, s0 = d4;

                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_set1_ps(ky[k]);
                    x0 = _mm_sub_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                }

                _mm_storeu_ps(dst + i, s0);
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
    bool haveSSE;
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
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;

        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        float* dst = (float*)_dst;
        __m128 d4 = _mm_set1_ps(delta);

        if( symmetrical )
        {
            if( ky[0] == 2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_ps(S0 + i);
                    s1 = _mm_load_ps(S0 + i + 4);
                    s2 = _mm_load_ps(S1 + i);
                    s3 = _mm_load_ps(S1 + i + 4);
                    s4 = _mm_load_ps(S2 + i);
                    s5 = _mm_load_ps(S2 + i + 4);
                    s0 = _mm_add_ps(s0, _mm_add_ps(s4, _mm_add_ps(s2, s2)));
                    s1 = _mm_add_ps(s1, _mm_add_ps(s5, _mm_add_ps(s3, s3)));
                    s0 = _mm_add_ps(s0, d4);
                    s1 = _mm_add_ps(s1, d4);
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_ps(S0 + i);
                    s1 = _mm_load_ps(S0 + i + 4);
                    s2 = _mm_load_ps(S1 + i);
                    s3 = _mm_load_ps(S1 + i + 4);
                    s4 = _mm_load_ps(S2 + i);
                    s5 = _mm_load_ps(S2 + i + 4);
                    s0 = _mm_add_ps(s0, _mm_sub_ps(s4, _mm_add_ps(s2, s2)));
                    s1 = _mm_add_ps(s1, _mm_sub_ps(s5, _mm_add_ps(s3, s3)));
                    s0 = _mm_add_ps(s0, d4);
                    s1 = _mm_add_ps(s1, d4);
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
            else
            {
                __m128 k0 = _mm_set1_ps(ky[0]), k1 = _mm_set1_ps(ky[1]);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1, x0, x1;
                    s0 = _mm_load_ps(S1 + i);
                    s1 = _mm_load_ps(S1 + i + 4);
                    s0 = _mm_add_ps(_mm_mul_ps(s0, k0), d4);
                    s1 = _mm_add_ps(_mm_mul_ps(s1, k0), d4);
                    x0 = _mm_add_ps(_mm_load_ps(S0 + i), _mm_load_ps(S2 + i));
                    x1 = _mm_add_ps(_mm_load_ps(S0 + i + 4), _mm_load_ps(S2 + i + 4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0,k1));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1,k1));
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1, s2, s3;
                    s0 = _mm_load_ps(S2 + i);
                    s1 = _mm_load_ps(S2 + i + 4);
                    s2 = _mm_load_ps(S0 + i);
                    s3 = _mm_load_ps(S0 + i + 4);
                    s0 = _mm_add_ps(_mm_sub_ps(s0, s2), d4);
                    s1 = _mm_add_ps(_mm_sub_ps(s1, s3), d4);
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
            else
            {
                __m128 k1 = _mm_set1_ps(ky[1]);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0 = d4, s1 = d4, x0, x1;
                    x0 = _mm_sub_ps(_mm_load_ps(S2 + i), _mm_load_ps(S0 + i));
                    x1 = _mm_sub_ps(_mm_load_ps(S2 + i + 4), _mm_load_ps(S0 + i + 4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0,k1));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1,k1));
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
        }

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
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        const float* kf = (const float*)&coeffs[0];
        int i = 0, k, nz = _nz;
        __m128 d4 = _mm_set1_ps(delta);

        for( ; i <= width - 16; i += 16 )
        {
            __m128 s0 = d4, s1 = d4, s2 = d4, s3 = d4;
            __m128i x0, x1, z = _mm_setzero_si128();

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0, t1;
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_loadu_si128((const __m128i*)(src[k] + i));
                x1 = _mm_unpackhi_epi8(x0, z);
                x0 = _mm_unpacklo_epi8(x0, z);

                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x0, z));
                t1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(x0, z));
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(t1, f));

                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x1, z));
                t1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(x1, z));
                s2 = _mm_add_ps(s2, _mm_mul_ps(t0, f));
                s3 = _mm_add_ps(s3, _mm_mul_ps(t1, f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
            x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
            x0 = _mm_packus_epi16(x0, x1);
            _mm_storeu_si128((__m128i*)(dst + i), x0);
        }

        for( ; i <= width - 4; i += 4 )
        {
            __m128 s0 = d4;
            __m128i x0, z = _mm_setzero_si128();

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0;
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_cvtsi32_si128(*(const int*)(src[k] + i));
                x0 = _mm_unpacklo_epi8(x0, z);
                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x0, z));
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), z);
            x0 = _mm_packus_epi16(x0, x0);
            *(int*)(dst + i) = _mm_cvtsi128_si32(x0);
        }

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
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        const float* kf = (const float*)&coeffs[0];
        short* dst = (short*)_dst;
        int i = 0, k, nz = _nz;
        __m128 d4 = _mm_set1_ps(delta);

        for( ; i <= width - 16; i += 16 )
        {
            __m128 s0 = d4, s1 = d4, s2 = d4, s3 = d4;
            __m128i x0, x1, z = _mm_setzero_si128();

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0, t1;
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_loadu_si128((const __m128i*)(src[k] + i));
                x1 = _mm_unpackhi_epi8(x0, z);
                x0 = _mm_unpacklo_epi8(x0, z);

                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x0, z));
                t1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(x0, z));
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(t1, f));

                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x1, z));
                t1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(x1, z));
                s2 = _mm_add_ps(s2, _mm_mul_ps(t0, f));
                s3 = _mm_add_ps(s3, _mm_mul_ps(t1, f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
            x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
            _mm_storeu_si128((__m128i*)(dst + i), x0);
            _mm_storeu_si128((__m128i*)(dst + i + 8), x1);
        }

        for( ; i <= width - 4; i += 4 )
        {
            __m128 s0 = d4;
            __m128i x0, z = _mm_setzero_si128();

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0;
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_cvtsi32_si128(*(const int*)(src[k] + i));
                x0 = _mm_unpacklo_epi8(x0, z);
                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x0, z));
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), z);
            _mm_storel_epi64((__m128i*)(dst + i), x0);
        }

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
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;

        const float* kf = (const float*)&coeffs[0];
        const float** src = (const float**)_src;
        float* dst = (float*)_dst;
        int i = 0, k, nz = _nz;
        __m128 d4 = _mm_set1_ps(delta);

        for( ; i <= width - 16; i += 16 )
        {
            __m128 s0 = d4, s1 = d4, s2 = d4, s3 = d4;

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0, t1;
                f = _mm_shuffle_ps(f, f, 0);
                const float* S = src[k] + i;

                t0 = _mm_loadu_ps(S);
                t1 = _mm_loadu_ps(S + 4);
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(t1, f));

                t0 = _mm_loadu_ps(S + 8);
                t1 = _mm_loadu_ps(S + 12);
                s2 = _mm_add_ps(s2, _mm_mul_ps(t0, f));
                s3 = _mm_add_ps(s3, _mm_mul_ps(t1, f));
            }

            _mm_storeu_ps(dst + i, s0);
            _mm_storeu_ps(dst + i + 4, s1);
            _mm_storeu_ps(dst + i + 8, s2);
            _mm_storeu_ps(dst + i + 12, s3);
        }

        for( ; i <= width - 4; i += 4 )
        {
            __m128 s0 = d4;

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0;
                f = _mm_shuffle_ps(f, f, 0);
                t0 = _mm_loadu_ps(src[k] + i);
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
            }
            _mm_storeu_ps(dst + i, s0);
        }

        return i;
    }

    int _nz;
    std::vector<uchar> coeffs;
    float delta;
};


#elif CV_NEON

struct SymmRowSmallVec_8u32s
{
    SymmRowSmallVec_8u32s() { smallValues = false; }
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
         if( !checkHardwareSupport(CV_CPU_NEON) )
             return 0;

        int i = 0, _ksize = kernel.rows + kernel.cols - 1;
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
                    uint16x8_t zq = vdupq_n_u16(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1, x2;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src) );
                        x2 = vld1_u8( (uint8_t *) (src + cn) );

                        uint16x8_t y0, y1, y2;
                        y0 = vaddl_u8(x0, x2);
                        y1 = vshll_n_u8(x1, 1);
                        y2 = vaddq_u16(y0, y1);

                        uint16x8x2_t str;
                        str.val[0] = y2; str.val[1] = zq;
                        vst2q_u16( (uint16_t *) (dst + i), str );
                    }
                }
                else if( kx[0] == -2 && kx[1] == 1 )
                    return 0;
                else
                {
                    int32x4_t k32 = vdupq_n_s32(0);
                    k32 = vld1q_lane_s32(kx, k32, 0);
                    k32 = vld1q_lane_s32(kx + 1, k32, 1);

                    int16x4_t k = vqmovn_s32(k32);

                    uint8x8_t z = vdup_n_u8(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1, x2;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src) );
                        x2 = vld1_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0, y1;
                        int32x4_t y2, y3;
                        y0 = vreinterpretq_s16_u16(vaddl_u8(x1, z));
                        y1 = vreinterpretq_s16_u16(vaddl_u8(x0, x2));
                        y2 = vmull_lane_s16(vget_low_s16(y0), k, 0);
                        y2 = vmlal_lane_s16(y2, vget_low_s16(y1), k, 1);
                        y3 = vmull_lane_s16(vget_high_s16(y0), k, 0);
                        y3 = vmlal_lane_s16(y3, vget_high_s16(y1), k, 1);

                        vst1q_s32((int32_t *)(dst + i), y2);
                        vst1q_s32((int32_t *)(dst + i + 4), y3);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    return 0;
                else
                {
                    int32x4_t k32 = vdupq_n_s32(0);
                    k32 = vld1q_lane_s32(kx, k32, 0);
                    k32 = vld1q_lane_s32(kx + 1, k32, 1);
                    k32 = vld1q_lane_s32(kx + 2, k32, 2);

                    int16x4_t k = vqmovn_s32(k32);

                    uint8x8_t z = vdup_n_u8(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1, x2, x3, x4;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src) );
                        x2 = vld1_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0, y1;
                        int32x4_t accl, acch;
                        y0 = vreinterpretq_s16_u16(vaddl_u8(x1, z));
                        y1 = vreinterpretq_s16_u16(vaddl_u8(x0, x2));
                        accl = vmull_lane_s16(vget_low_s16(y0), k, 0);
                        accl = vmlal_lane_s16(accl, vget_low_s16(y1), k, 1);
                        acch = vmull_lane_s16(vget_high_s16(y0), k, 0);
                        acch = vmlal_lane_s16(acch, vget_high_s16(y1), k, 1);

                        int16x8_t y2;
                        x3 = vld1_u8( (uint8_t *) (src - cn*2) );
                        x4 = vld1_u8( (uint8_t *) (src + cn*2) );
                        y2 = vreinterpretq_s16_u16(vaddl_u8(x3, x4));
                        accl = vmlal_lane_s16(accl, vget_low_s16(y2), k, 2);
                        acch = vmlal_lane_s16(acch, vget_high_s16(y2), k, 2);

                        vst1q_s32((int32_t *)(dst + i), accl);
                        vst1q_s32((int32_t *)(dst + i + 4), acch);
                    }
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                {
                    uint8x8_t z = vdup_n_u8(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0;
                        y0 = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x1, z)),
                                vreinterpretq_s16_u16(vaddl_u8(x0, z)));

                        vst1q_s32((int32_t *)(dst + i), vmovl_s16(vget_low_s16(y0)));
                        vst1q_s32((int32_t *)(dst + i + 4), vmovl_s16(vget_high_s16(y0)));
                    }
                }
                else
                {
                    int32x4_t k32 = vdupq_n_s32(0);
                    k32 = vld1q_lane_s32(kx + 1, k32, 1);

                    int16x4_t k = vqmovn_s32(k32);

                    uint8x8_t z = vdup_n_u8(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0;
                        int32x4_t y1, y2;
                        y0 = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x1, z)),
                            vreinterpretq_s16_u16(vaddl_u8(x0, z)));
                        y1 = vmull_lane_s16(vget_low_s16(y0), k, 1);
                        y2 = vmull_lane_s16(vget_high_s16(y0), k, 1);

                        vst1q_s32((int32_t *)(dst + i), y1);
                        vst1q_s32((int32_t *)(dst + i + 4), y2);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                int32x4_t k32 = vdupq_n_s32(0);
                k32 = vld1q_lane_s32(kx + 1, k32, 1);
                k32 = vld1q_lane_s32(kx + 2, k32, 2);

                int16x4_t k = vqmovn_s32(k32);

                uint8x8_t z = vdup_n_u8(0);

                for( ; i <= width - 8; i += 8, src += 8 )
                {
                    uint8x8_t x0, x1;
                    x0 = vld1_u8( (uint8_t *) (src - cn) );
                    x1 = vld1_u8( (uint8_t *) (src + cn) );

                    int32x4_t accl, acch;
                    int16x8_t y0;
                    y0 = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x1, z)),
                        vreinterpretq_s16_u16(vaddl_u8(x0, z)));
                    accl = vmull_lane_s16(vget_low_s16(y0), k, 1);
                    acch = vmull_lane_s16(vget_high_s16(y0), k, 1);

                    uint8x8_t x2, x3;
                    x2 = vld1_u8( (uint8_t *) (src - cn*2) );
                    x3 = vld1_u8( (uint8_t *) (src + cn*2) );

                    int16x8_t y1;
                    y1 = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x3, z)),
                        vreinterpretq_s16_u16(vaddl_u8(x2, z)));
                    accl = vmlal_lane_s16(accl, vget_low_s16(y1), k, 2);
                    acch = vmlal_lane_s16(acch, vget_high_s16(y1), k, 2);

                    vst1q_s32((int32_t *)(dst + i), accl);
                    vst1q_s32((int32_t *)(dst + i + 4), acch);
                }
            }
        }

        return i;
    }

    Mat kernel;
    int symmetryType;
    bool smallValues;
};


struct SymmColumnVec_32s8u
{
    SymmColumnVec_32s8u() { symmetryType=0; }
    SymmColumnVec_32s8u(const Mat& _kernel, int _symmetryType, int _bits, double _delta)
    {
        symmetryType = _symmetryType;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* dst, int width) const
    {
         if( !checkHardwareSupport(CV_CPU_NEON) )
             return 0;

        int _ksize = kernel.rows + kernel.cols - 1;
        int ksize2 = _ksize / 2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;
        const int *S, *S2;

        float32x4_t d4 = vdupq_n_f32(delta);

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;


            float32x2_t k32;
            k32 = vdup_n_f32(0);
            k32 = vld1_lane_f32(ky, k32, 0);
            k32 = vld1_lane_f32(ky + 1, k32, 1);

            for( ; i <= width - 8; i += 8 )
            {
                float32x4_t accl, acch;
                float32x4_t f0l, f0h, f1l, f1h, f2l, f2h;

                S = src[0] + i;

                f0l = vcvtq_f32_s32( vld1q_s32(S) );
                f0h = vcvtq_f32_s32( vld1q_s32(S + 4) );

                S = src[1] + i;
                S2 = src[-1] + i;

                f1l = vcvtq_f32_s32( vld1q_s32(S) );
                f1h = vcvtq_f32_s32( vld1q_s32(S + 4) );
                f2l = vcvtq_f32_s32( vld1q_s32(S2) );
                f2h = vcvtq_f32_s32( vld1q_s32(S2 + 4) );

                accl = acch = d4;
                accl = vmlaq_lane_f32(accl, f0l, k32, 0);
                acch = vmlaq_lane_f32(acch, f0h, k32, 0);
                accl = vmlaq_lane_f32(accl, vaddq_f32(f1l, f2l), k32, 1);
                acch = vmlaq_lane_f32(acch, vaddq_f32(f1h, f2h), k32, 1);

                for( k = 2; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;

                    float32x4_t f3l, f3h, f4l, f4h;
                    f3l = vcvtq_f32_s32( vld1q_s32(S) );
                    f3h = vcvtq_f32_s32( vld1q_s32(S + 4) );
                    f4l = vcvtq_f32_s32( vld1q_s32(S2) );
                    f4h = vcvtq_f32_s32( vld1q_s32(S2 + 4) );

                    accl = vmlaq_n_f32(accl, vaddq_f32(f3l, f4l), ky[k]);
                    acch = vmlaq_n_f32(acch, vaddq_f32(f3h, f4h), ky[k]);
                }

                int32x4_t s32l, s32h;
                s32l = vcvtq_s32_f32(accl);
                s32h = vcvtq_s32_f32(acch);

                int16x4_t s16l, s16h;
                s16l = vqmovn_s32(s32l);
                s16h = vqmovn_s32(s32h);

                uint8x8_t u8;
                u8 =  vqmovun_s16(vcombine_s16(s16l, s16h));

                vst1_u8((uint8_t *)(dst + i), u8);
            }
        }
        else
        {
            float32x2_t k32;
            k32 = vdup_n_f32(0);
            k32 = vld1_lane_f32(ky + 1, k32, 1);

            for( ; i <= width - 8; i += 8 )
            {
                float32x4_t accl, acch;
                float32x4_t f1l, f1h, f2l, f2h;

                S = src[1] + i;
                S2 = src[-1] + i;

                f1l = vcvtq_f32_s32( vld1q_s32(S) );
                f1h = vcvtq_f32_s32( vld1q_s32(S + 4) );
                f2l = vcvtq_f32_s32( vld1q_s32(S2) );
                f2h = vcvtq_f32_s32( vld1q_s32(S2 + 4) );

                accl = acch = d4;
                accl = vmlaq_lane_f32(accl, vsubq_f32(f1l, f2l), k32, 1);
                acch = vmlaq_lane_f32(acch, vsubq_f32(f1h, f2h), k32, 1);

                for( k = 2; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;

                    float32x4_t f3l, f3h, f4l, f4h;
                    f3l = vcvtq_f32_s32( vld1q_s32(S) );
                    f3h = vcvtq_f32_s32( vld1q_s32(S + 4) );
                    f4l = vcvtq_f32_s32( vld1q_s32(S2) );
                    f4h = vcvtq_f32_s32( vld1q_s32(S2 + 4) );

                    accl = vmlaq_n_f32(accl, vsubq_f32(f3l, f4l), ky[k]);
                    acch = vmlaq_n_f32(acch, vsubq_f32(f3h, f4h), ky[k]);
                }

                int32x4_t s32l, s32h;
                s32l = vcvtq_s32_f32(accl);
                s32h = vcvtq_s32_f32(acch);

                int16x4_t s16l, s16h;
                s16l = vqmovn_s32(s32l);
                s16h = vqmovn_s32(s32h);

                uint8x8_t u8;
                u8 =  vqmovun_s16(vcombine_s16(s16l, s16h));

                vst1_u8((uint8_t *)(dst + i), u8);
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


struct SymmColumnSmallVec_32s16s
{
    SymmColumnSmallVec_32s16s() { symmetryType=0; }
    SymmColumnSmallVec_32s16s(const Mat& _kernel, int _symmetryType, int _bits, double _delta)
    {
        symmetryType = _symmetryType;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
         if( !checkHardwareSupport(CV_CPU_NEON) )
             return 0;

        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;
        const int *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        short* dst = (short*)_dst;
        float32x4_t df4 = vdupq_n_f32(delta);
        int32x4_t d4 = vcvtq_s32_f32(df4);

        if( symmetrical )
        {
            if( ky[0] == 2 && ky[1] == 1 )
            {
                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S1 + i));
                    x2 = vld1q_s32((int32_t const *)(S2 + i));

                    int32x4_t y0, y1, y2, y3;
                    y0 = vaddq_s32(x0, x2);
                    y1 = vqshlq_n_s32(x1, 1);
                    y2 = vaddq_s32(y0, y1);
                    y3 = vaddq_s32(y2, d4);

                    int16x4_t t;
                    t = vqmovn_s32(y3);

                    vst1_s16((int16_t *)(dst + i), t);
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S1 + i));
                    x2 = vld1q_s32((int32_t const *)(S2 + i));

                    int32x4_t y0, y1, y2, y3;
                    y0 = vaddq_s32(x0, x2);
                    y1 = vqshlq_n_s32(x1, 1);
                    y2 = vsubq_s32(y0, y1);
                    y3 = vaddq_s32(y2, d4);

                    int16x4_t t;
                    t = vqmovn_s32(y3);

                    vst1_s16((int16_t *)(dst + i), t);
                }
            }
            else if( ky[0] == 10 && ky[1] == 3 )
            {
                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2, x3;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S1 + i));
                    x2 = vld1q_s32((int32_t const *)(S2 + i));

                    x3 = vaddq_s32(x0, x2);

                    int32x4_t y0;
                    y0 = vmlaq_n_s32(d4, x1, 10);
                    y0 = vmlaq_n_s32(y0, x3, 3);

                    int16x4_t t;
                    t = vqmovn_s32(y0);

                    vst1_s16((int16_t *)(dst + i), t);
                }
            }
            else
            {
                float32x2_t k32 = vdup_n_f32(0);
                k32 = vld1_lane_f32(ky, k32, 0);
                k32 = vld1_lane_f32(ky + 1, k32, 1);

                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2, x3, x4;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S1 + i));
                    x2 = vld1q_s32((int32_t const *)(S2 + i));

                    x3 = vaddq_s32(x0, x2);

                    float32x4_t s0, s1, s2;
                    s0 = vcvtq_f32_s32(x1);
                    s1 = vcvtq_f32_s32(x3);
                    s2 = vmlaq_lane_f32(df4, s0, k32, 0);
                    s2 = vmlaq_lane_f32(s2, s1, k32, 1);

                    x4 = vcvtq_s32_f32(s2);

                    int16x4_t x5;
                    x5 = vqmovn_s32(x4);

                    vst1_s16((int16_t *)(dst + i), x5);
                }
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S2 + i));

                    int32x4_t y0, y1;
                    y0 = vsubq_s32(x1, x0);
                    y1 = vqaddq_s32(y0, d4);

                    int16x4_t t;
                    t = vqmovn_s32(y1);

                    vst1_s16((int16_t *)(dst + i), t);
                }
            }
            else
            {
                float32x2_t k32 = vdup_n_f32(0);
                k32 = vld1_lane_f32(ky + 1, k32, 1);

                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2, x3;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S2 + i));

                    x2 = vsubq_s32(x1, x0);

                    float32x4_t s0, s1;
                    s0 = vcvtq_f32_s32(x2);
                    s1 = vmlaq_lane_f32(df4, s0, k32, 1);

                    x3 = vcvtq_s32_f32(s1);

                    int16x4_t x4;
                    x4 = vqmovn_s32(x3);

                    vst1_s16((int16_t *)(dst + i), x4);
                }
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


struct SymmColumnVec_32f16s
{
    SymmColumnVec_32f16s() { symmetryType=0; }
    SymmColumnVec_32f16s(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
         neon_supported = checkHardwareSupport(CV_CPU_NEON);
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
         if( !neon_supported )
             return 0;

        int _ksize = kernel.rows + kernel.cols - 1;
        int ksize2 = _ksize / 2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S, *S2;
        short* dst = (short*)_dst;

        float32x4_t d4 = vdupq_n_f32(delta);

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;


            float32x2_t k32;
            k32 = vdup_n_f32(0);
            k32 = vld1_lane_f32(ky, k32, 0);
            k32 = vld1_lane_f32(ky + 1, k32, 1);

            for( ; i <= width - 8; i += 8 )
            {
                float32x4_t x0l, x0h, x1l, x1h, x2l, x2h;
                float32x4_t accl, acch;

                S = src[0] + i;

                x0l = vld1q_f32(S);
                x0h = vld1q_f32(S + 4);

                S = src[1] + i;
                S2 = src[-1] + i;

                x1l = vld1q_f32(S);
                x1h = vld1q_f32(S + 4);
                x2l = vld1q_f32(S2);
                x2h = vld1q_f32(S2 + 4);

                accl = acch = d4;
                accl = vmlaq_lane_f32(accl, x0l, k32, 0);
                acch = vmlaq_lane_f32(acch, x0h, k32, 0);
                accl = vmlaq_lane_f32(accl, vaddq_f32(x1l, x2l), k32, 1);
                acch = vmlaq_lane_f32(acch, vaddq_f32(x1h, x2h), k32, 1);

                for( k = 2; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;

                    float32x4_t x3l, x3h, x4l, x4h;
                    x3l = vld1q_f32(S);
                    x3h = vld1q_f32(S + 4);
                    x4l = vld1q_f32(S2);
                    x4h = vld1q_f32(S2 + 4);

                    accl = vmlaq_n_f32(accl, vaddq_f32(x3l, x4l), ky[k]);
                    acch = vmlaq_n_f32(acch, vaddq_f32(x3h, x4h), ky[k]);
                }

                int32x4_t s32l, s32h;
                s32l = vcvtq_s32_f32(accl);
                s32h = vcvtq_s32_f32(acch);

                int16x4_t s16l, s16h;
                s16l = vqmovn_s32(s32l);
                s16h = vqmovn_s32(s32h);

                vst1_s16((int16_t *)(dst + i), s16l);
                vst1_s16((int16_t *)(dst + i + 4), s16h);
            }
        }
        else
        {
            float32x2_t k32;
            k32 = vdup_n_f32(0);
            k32 = vld1_lane_f32(ky + 1, k32, 1);

            for( ; i <= width - 8; i += 8 )
            {
                float32x4_t x1l, x1h, x2l, x2h;
                float32x4_t accl, acch;

                S = src[1] + i;
                S2 = src[-1] + i;

                x1l = vld1q_f32(S);
                x1h = vld1q_f32(S + 4);
                x2l = vld1q_f32(S2);
                x2h = vld1q_f32(S2 + 4);

                accl = acch = d4;
                accl = vmlaq_lane_f32(accl, vsubq_f32(x1l, x2l), k32, 1);
                acch = vmlaq_lane_f32(acch, vsubq_f32(x1h, x2h), k32, 1);

                for( k = 2; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;

                    float32x4_t x3l, x3h, x4l, x4h;
                    x3l = vld1q_f32(S);
                    x3h = vld1q_f32(S + 4);
                    x4l = vld1q_f32(S2);
                    x4h = vld1q_f32(S2 + 4);

                    accl = vmlaq_n_f32(accl, vsubq_f32(x3l, x4l), ky[k]);
                    acch = vmlaq_n_f32(acch, vsubq_f32(x3h, x4h), ky[k]);
                }

                int32x4_t s32l, s32h;
                s32l = vcvtq_s32_f32(accl);
                s32h = vcvtq_s32_f32(acch);

                int16x4_t s16l, s16h;
                s16l = vqmovn_s32(s32l);
                s16h = vqmovn_s32(s32h);

                vst1_s16((int16_t *)(dst + i), s16l);
                vst1_s16((int16_t *)(dst + i + 4), s16h);
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
    bool neon_supported;
};


struct SymmRowSmallVec_32f
{
    SymmRowSmallVec_32f() {}
    SymmRowSmallVec_32f( const Mat& _kernel, int _symmetryType )
    {
        kernel = _kernel;
        symmetryType = _symmetryType;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
         if( !checkHardwareSupport(CV_CPU_NEON) )
             return 0;

        int i = 0, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* src = (const float*)_src + (_ksize/2)*cn;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float* kx = kernel.ptr<float>() + _ksize/2;
        width *= cn;

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                    return 0;
                else if( kx[0] == -2 && kx[1] == 1 )
                    return 0;
                else
                {
                    return 0;
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    return 0;
                else
                {
                    float32x2_t k0, k1;
                    k0 = k1 = vdup_n_f32(0);
                    k0 = vld1_lane_f32(kx + 0, k0, 0);
                    k0 = vld1_lane_f32(kx + 1, k0, 1);
                    k1 = vld1_lane_f32(kx + 2, k1, 0);

                    for( ; i <= width - 4; i += 4, src += 4 )
                    {
                        float32x4_t x0, x1, x2, x3, x4;
                        x0 = vld1q_f32(src);
                        x1 = vld1q_f32(src - cn);
                        x2 = vld1q_f32(src + cn);
                        x3 = vld1q_f32(src - cn*2);
                        x4 = vld1q_f32(src + cn*2);

                        float32x4_t y0;
                        y0 = vmulq_lane_f32(x0, k0, 0);
                        y0 = vmlaq_lane_f32(y0, vaddq_f32(x1, x2), k0, 1);
                        y0 = vmlaq_lane_f32(y0, vaddq_f32(x3, x4), k1, 0);

                        vst1q_f32(dst + i, y0);
                    }
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    return 0;
                else
                {
                    return 0;
                }
            }
            else if( _ksize == 5 )
            {
                float32x2_t k;
                k = vdup_n_f32(0);
                k = vld1_lane_f32(kx + 1, k, 0);
                k = vld1_lane_f32(kx + 2, k, 1);

                for( ; i <= width - 4; i += 4, src += 4 )
                {
                    float32x4_t x0, x1, x2, x3;
                    x0 = vld1q_f32(src - cn);
                    x1 = vld1q_f32(src + cn);
                    x2 = vld1q_f32(src - cn*2);
                    x3 = vld1q_f32(src + cn*2);

                    float32x4_t y0;
                    y0 = vmulq_lane_f32(vsubq_f32(x1, x0), k, 0);
                    y0 = vmlaq_lane_f32(y0, vsubq_f32(x3, x2), k, 1);

                    vst1q_f32(dst + i, y0);
                }
            }
        }

        return i;
    }

    Mat kernel;
    int symmetryType;
};


typedef RowNoVec RowVec_8u32s;
typedef RowNoVec RowVec_16s32f;
typedef RowNoVec RowVec_32f;
typedef ColumnNoVec SymmColumnVec_32f;
typedef SymmColumnSmallNoVec SymmColumnSmallVec_32f;
typedef FilterNoVec FilterVec_8u;
typedef FilterNoVec FilterVec_8u16s;
typedef FilterNoVec FilterVec_32f;


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
#if CV_SSE2
        int sdepth = CV_MAT_DEPTH(stype);
        int ddepth = CV_MAT_DEPTH(dtype);
        int dft_filter_size = ((sdepth == CV_8U && (ddepth == CV_8U || ddepth == CV_16S)) || (sdepth == CV_32F && ddepth == CV_32F)) && checkHardwareSupport(CV_CPU_SSE3) ? 130 : 50;
#else
        CV_UNUSED(stype);
        CV_UNUSED(dtype);
        int dft_filter_size = 50;
#endif
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
