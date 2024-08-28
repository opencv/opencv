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
#include "opencv2/core/hal/intrin.hpp"
#include "filter.hpp"

#include <cstddef>

#if defined(CV_CPU_BASELINE_MODE)
#if IPP_VERSION_X100 >= 710
#define USE_IPP_SEP_FILTERS 1
#else
#undef USE_IPP_SEP_FILTERS
#endif
#endif


/****************************************************************************************\
                                    Base Image Filter
\****************************************************************************************/

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
int FilterEngine__start(FilterEngine& this_, const Size &_wholeSize, const Size &sz, const Point &ofs);
int FilterEngine__proceed(FilterEngine& this_, const uchar* src, int srcstep, int count,
                          uchar* dst, int dststep);
void FilterEngine__apply(FilterEngine& this_, const Mat& src, Mat& dst, const Size& wsz, const Point& ofs);

Ptr<BaseRowFilter> getLinearRowFilter(
        int srcType, int bufType,
        const Mat& kernel, int anchor,
        int symmetryType);

Ptr<BaseColumnFilter> getLinearColumnFilter(
        int bufType, int dstType,
        const Mat& kernel, int anchor,
        int symmetryType, double delta,
        int bits);

Ptr<BaseFilter> getLinearFilter(
        int srcType, int dstType,
        const Mat& filter_kernel, Point anchor,
        double delta, int bits);


#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#define VEC_ALIGN CV_MALLOC_ALIGN

int FilterEngine__start(FilterEngine& this_, const Size &_wholeSize, const Size &sz, const Point &ofs)
{
    CV_INSTRUMENT_REGION();

    int i, j;

    this_.wholeSize = _wholeSize;
    this_.roi = Rect(ofs, sz);
    CV_Assert( this_.roi.x >= 0 && this_.roi.y >= 0 && this_.roi.width >= 0 && this_.roi.height >= 0 &&
        this_.roi.x + this_.roi.width <= this_.wholeSize.width &&
        this_.roi.y + this_.roi.height <= this_.wholeSize.height );

    int esz = (int)getElemSize(this_.srcType);
    int bufElemSize = (int)getElemSize(this_.bufType);
    const uchar* constVal = !this_.constBorderValue.empty() ? &this_.constBorderValue[0] : 0;

    int _maxBufRows = std::max(this_.ksize.height + 3,
                               std::max(this_.anchor.y,
                                        this_.ksize.height-this_.anchor.y-1)*2+1);

    if (this_.maxWidth < this_.roi.width || _maxBufRows != (int)this_.rows.size() )
    {
        this_.rows.resize(_maxBufRows);
        this_.maxWidth = std::max(this_.maxWidth, this_.roi.width);
        int cn = CV_MAT_CN(this_.srcType);
        this_.srcRow.resize(esz*(this_.maxWidth + this_.ksize.width - 1));
        if (this_.columnBorderType == BORDER_CONSTANT)
        {
            CV_Assert(constVal != NULL);
            this_.constBorderRow.resize(getElemSize(this_.bufType)*(this_.maxWidth + this_.ksize.width - 1 + VEC_ALIGN));
            uchar *dst = alignPtr(&this_.constBorderRow[0], VEC_ALIGN);
            int n = (int)this_.constBorderValue.size();
            int N = (this_.maxWidth + this_.ksize.width - 1)*esz;
            uchar *tdst = this_.isSeparable() ? &this_.srcRow[0] : dst;

            for( i = 0; i < N; i += n )
            {
                n = std::min( n, N - i );
                for(j = 0; j < n; j++)
                    tdst[i+j] = constVal[j];
            }

            if (this_.isSeparable())
                (*this_.rowFilter)(&this_.srcRow[0], dst, this_.maxWidth, cn);
        }

        int maxBufStep = bufElemSize*(int)alignSize(this_.maxWidth +
            (!this_.isSeparable() ? this_.ksize.width - 1 : 0), VEC_ALIGN);
        this_.ringBuf.resize(maxBufStep*this_.rows.size()+VEC_ALIGN);
    }

    // adjust bufstep so that the used part of the ring buffer stays compact in memory
    this_.bufStep = bufElemSize*(int)alignSize(this_.roi.width + (!this_.isSeparable() ? this_.ksize.width - 1 : 0), VEC_ALIGN);

    this_.dx1 = std::max(this_.anchor.x - this_.roi.x, 0);
    this_.dx2 = std::max(this_.ksize.width - this_.anchor.x - 1 + this_.roi.x + this_.roi.width - this_.wholeSize.width, 0);

    // recompute border tables
    if (this_.dx1 > 0 || this_.dx2 > 0)
    {
        if (this_.rowBorderType == BORDER_CONSTANT )
        {
            CV_Assert(constVal != NULL);
            int nr = this_.isSeparable() ? 1 : (int)this_.rows.size();
            for( i = 0; i < nr; i++ )
            {
                uchar* dst = this_.isSeparable() ? &this_.srcRow[0] : alignPtr(&this_.ringBuf[0], VEC_ALIGN) + this_.bufStep*i;
                memcpy(dst, constVal, this_.dx1*esz);
                memcpy(dst + (this_.roi.width + this_.ksize.width - 1 - this_.dx2)*esz, constVal, this_.dx2*esz);
            }
        }
        else
        {
            int xofs1 = std::min(this_.roi.x, this_.anchor.x) - this_.roi.x;

            int btab_esz = this_.borderElemSize, wholeWidth = this_.wholeSize.width;
            int* btab = (int*)&this_.borderTab[0];

            for( i = 0; i < this_.dx1; i++ )
            {
                int p0 = (borderInterpolate(i-this_.dx1, wholeWidth, this_.rowBorderType) + xofs1)*btab_esz;
                for( j = 0; j < btab_esz; j++ )
                    btab[i*btab_esz + j] = p0 + j;
            }

            for( i = 0; i < this_.dx2; i++ )
            {
                int p0 = (borderInterpolate(wholeWidth + i, wholeWidth, this_.rowBorderType) + xofs1)*btab_esz;
                for( j = 0; j < btab_esz; j++ )
                    btab[(i + this_.dx1)*btab_esz + j] = p0 + j;
            }
        }
    }

    this_.rowCount = this_.dstY = 0;
    this_.startY = this_.startY0 = std::max(this_.roi.y - this_.anchor.y, 0);
    this_.endY = std::min(this_.roi.y + this_.roi.height + this_.ksize.height - this_.anchor.y - 1, this_.wholeSize.height);

    if (this_.columnFilter)
        this_.columnFilter->reset();
    if (this_.filter2D)
        this_.filter2D->reset();

    return this_.startY;
}


int FilterEngine__proceed(FilterEngine& this_, const uchar* src, int srcstep, int count,
                          uchar* dst, int dststep)
{
    CV_INSTRUMENT_REGION();

    CV_DbgAssert(this_.wholeSize.width > 0 && this_.wholeSize.height > 0 );

    const int *btab = &this_.borderTab[0];
    int esz = (int)getElemSize(this_.srcType), btab_esz = this_.borderElemSize;
    uchar** brows = &this_.rows[0];
    int bufRows = (int)this_.rows.size();
    int cn = CV_MAT_CN(this_.bufType);
    int width = this_.roi.width, kwidth = this_.ksize.width;
    int kheight = this_.ksize.height, ay = this_.anchor.y;
    int _dx1 = this_.dx1, _dx2 = this_.dx2;
    int width1 = this_.roi.width + kwidth - 1;
    int xofs1 = std::min(this_.roi.x, this_.anchor.x);
    bool isSep = this_.isSeparable();
    bool makeBorder = (_dx1 > 0 || _dx2 > 0) && this_.rowBorderType != BORDER_CONSTANT;
    int dy = 0, i = 0;

    src -= xofs1*esz;
    count = std::min(count, this_.remainingInputRows());

    CV_Assert(src && dst && count > 0);

    for(;; dst += dststep*i, dy += i)
    {
        int dcount = bufRows - ay - this_.startY - this_.rowCount + this_.roi.y;
        dcount = dcount > 0 ? dcount : bufRows - kheight + 1;
        dcount = std::min(dcount, count);
        count -= dcount;
        for( ; dcount-- > 0; src += srcstep )
        {
            int bi = (this_.startY - this_.startY0 + this_.rowCount) % bufRows;
            uchar* brow = alignPtr(&this_.ringBuf[0], VEC_ALIGN) + bi*this_.bufStep;
            uchar* row = isSep ? &this_.srcRow[0] : brow;

            if (++this_.rowCount > bufRows)
            {
                --this_.rowCount;
                ++this_.startY;
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
                (*this_.rowFilter)(row, brow, width, CV_MAT_CN(this_.srcType));
        }

        int max_i = std::min(bufRows, this_.roi.height - (this_.dstY + dy) + (kheight - 1));
        for( i = 0; i < max_i; i++ )
        {
            int srcY = borderInterpolate(this_.dstY + dy + i + this_.roi.y - ay,
                    this_.wholeSize.height, this_.columnBorderType);
            if( srcY < 0 ) // can happen only with constant border type
                brows[i] = alignPtr(&this_.constBorderRow[0], VEC_ALIGN);
            else
            {
                CV_Assert(srcY >= this_.startY);
                if( srcY >= this_.startY + this_.rowCount)
                    break;
                int bi = (srcY - this_.startY0) % bufRows;
                brows[i] = alignPtr(&this_.ringBuf[0], VEC_ALIGN) + bi*this_.bufStep;
            }
        }
        if( i < kheight )
            break;
        i -= kheight - 1;
        if (isSep)
            (*this_.columnFilter)((const uchar**)brows, dst, dststep, i, this_.roi.width*cn);
        else
            (*this_.filter2D)((const uchar**)brows, dst, dststep, i, this_.roi.width, cn);
    }

    this_.dstY += dy;
    CV_Assert(this_.dstY <= this_.roi.height);
    return dy;
}

void FilterEngine__apply(FilterEngine& this_, const Mat& src, Mat& dst, const Size& wsz, const Point& ofs)
{
    CV_INSTRUMENT_REGION();

    CV_DbgAssert(src.type() == this_.srcType && dst.type() == this_.dstType);

    FilterEngine__start(this_, wsz, src.size(), ofs);
    int y = this_.startY - ofs.y;
    FilterEngine__proceed(this_,
            src.ptr() + y * (ptrdiff_t)src.step,
            (int)src.step,
            this_.endY - this_.startY,
            dst.ptr(),
            (int)dst.step );
}

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


#if (CV_SIMD || CV_SIMD_SCALABLE)

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
        CV_INSTRUMENT_REGION();

        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        int* dst = (int*)_dst;
        const int* _kx = kernel.ptr<int>();
        width *= cn;

        if( smallValues )
        {
            for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes() )
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
                    s0 = v_add(s0, v_dotprod(v_reinterpret_as_s16(v_expand_low(x0)), v_reinterpret_as_s16(f)));
                    s1 = v_add(s1, v_dotprod(v_reinterpret_as_s16(v_expand_high(x0)), v_reinterpret_as_s16(f)));
                    s2 = v_add(s2, v_dotprod(v_reinterpret_as_s16(v_expand_low(x1)), v_reinterpret_as_s16(f)));
                    s3 = v_add(s3, v_dotprod(v_reinterpret_as_s16(v_expand_high(x1)), v_reinterpret_as_s16(f)));
                }
                if (k < _ksize)
                {
                    v_int32 f = vx_setall_s32(_kx[k]);
                    v_uint16 x0, x1;
                    v_expand(vx_load(src), x0, x1);
                    s0 = v_add(s0, v_dotprod(v_reinterpret_as_s16(v_expand_low(x0)), v_reinterpret_as_s16(f)));
                    s1 = v_add(s1, v_dotprod(v_reinterpret_as_s16(v_expand_high(x0)), v_reinterpret_as_s16(f)));
                    s2 = v_add(s2, v_dotprod(v_reinterpret_as_s16(v_expand_low(x1)), v_reinterpret_as_s16(f)));
                    s3 = v_add(s3, v_dotprod(v_reinterpret_as_s16(v_expand_high(x1)), v_reinterpret_as_s16(f)));
                }
                v_store(dst + i, s0);
                v_store(dst + i + VTraits<v_int32>::vlanes(), s1);
                v_store(dst + i + 2*VTraits<v_int32>::vlanes(), s2);
                v_store(dst + i + 3*VTraits<v_int32>::vlanes(), s3);
            }
            if( i <= width - VTraits<v_uint16>::vlanes() )
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
                    s0 = v_add(s0, v_dotprod(v_reinterpret_as_s16(x0), v_reinterpret_as_s16(f)));
                    s1 = v_add(s1, v_dotprod(v_reinterpret_as_s16(x1), v_reinterpret_as_s16(f)));
                }
                if( k < _ksize )
                {
                    v_int32 f = vx_setall_s32(_kx[k]);
                    v_uint32 x0, x1;
                    v_expand(vx_load_expand(src), x0, x1);
                    s0 = v_add(s0, v_dotprod(v_reinterpret_as_s16(x0), v_reinterpret_as_s16(f)));
                    s1 = v_add(s1, v_dotprod(v_reinterpret_as_s16(x1), v_reinterpret_as_s16(f)));
                }
                v_store(dst + i, s0);
                v_store(dst + i + VTraits<v_int32>::vlanes(), s1);
                i += VTraits<v_uint16>::vlanes();
            }
            if( i <= width - VTraits<v_uint32>::vlanes() )
            {
                v_int32 d = vx_setzero_s32();
                k = 0;
                const uchar* src = _src + i;
                for (; k <= _ksize - 2; k += 2, src += 2*cn)
                {
                    v_int32 f = vx_setall_s32((_kx[k] & 0xFFFF) | (_kx[k + 1] << 16));
                    v_uint32 x0, x1;
                    v_zip(vx_load_expand_q(src), vx_load_expand_q(src + cn), x0, x1);
                    d = v_add(d, v_dotprod(v_pack(v_reinterpret_as_s32(x0), v_reinterpret_as_s32(x1)), v_reinterpret_as_s16(f)));
                }
                if (k < _ksize)
                    d = v_add(d, v_dotprod(v_reinterpret_as_s16(vx_load_expand_q(src)), v_reinterpret_as_s16(vx_setall_s32(_kx[k]))));
                v_store(dst + i, d);
                i += VTraits<v_uint32>::vlanes();
            }
        }
        return i;
    }

    Mat kernel;
    bool smallValues;
};

struct RowVec_8u32f
{
    RowVec_8u32f() {}
    RowVec_8u32f( const Mat& _kernel ) : kernel(_kernel) {}

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* _kx = kernel.ptr<float>();
        width *= cn;
        for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes() )
        {
            v_float32 s0 = vx_setzero_f32();
            v_float32 s1 = vx_setzero_f32();
            v_float32 s2 = vx_setzero_f32();
            v_float32 s3 = vx_setzero_f32();
            k = 0;
            for( ; k < _ksize ; k++ )
            {
                v_float32 f = vx_setall_f32(_kx[k]);
                const uchar* src = (const uchar*)_src + i + k * cn;
                v_uint8 v_src = vx_load(src);
                v_uint16 v_src_d0;
                v_uint16 v_src_d1;
                v_expand(v_src, v_src_d0, v_src_d1);
                v_uint32 v_src_q0;
                v_uint32 v_src_q1;
                v_uint32 v_src_q2;
                v_uint32 v_src_q3;
                v_expand(v_src_d0, v_src_q0, v_src_q1);
                v_expand(v_src_d1, v_src_q2, v_src_q3);
                v_float32 vs_ll = v_cvt_f32(v_reinterpret_as_s32(v_src_q0));
                v_float32 vs_lh = v_cvt_f32(v_reinterpret_as_s32(v_src_q1));
                v_float32 vs_hl = v_cvt_f32(v_reinterpret_as_s32(v_src_q2));
                v_float32 vs_hh = v_cvt_f32(v_reinterpret_as_s32(v_src_q3));
                s0 = v_muladd(vs_ll, f, s0);
                s1 = v_muladd(vs_lh, f, s1);
                s2 = v_muladd(vs_hl, f, s2);
                s3 = v_muladd(vs_hh, f, s3);
            }
            v_store(dst + i, s0);
            v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
            v_store(dst + i + 2*VTraits<v_float32>::vlanes(), s2);
            v_store(dst + i + 3*VTraits<v_float32>::vlanes(), s3);
        }
        return i;
    }

    Mat kernel;
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
        CV_INSTRUMENT_REGION();

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
                    for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
                    {
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src), x1l, x1h);
                        v_expand(vx_load(src + cn), x2l, x2h);
                        x1l = v_add_wrap(v_add_wrap(x1l, x1l), v_add_wrap(x0l, x2l));
                        x1h = v_add_wrap(v_add_wrap(x1h, x1h), v_add_wrap(x0h, x2h));
                        v_store(dst + i, v_reinterpret_as_s32(v_expand_low(x1l)));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_reinterpret_as_s32(v_expand_high(x1l)));
                        v_store(dst + i + 2*VTraits<v_int32>::vlanes(), v_reinterpret_as_s32(v_expand_low(x1h)));
                        v_store(dst + i + 3*VTraits<v_int32>::vlanes(), v_reinterpret_as_s32(v_expand_high(x1h)));
                    }
                    if( i <= width - VTraits<v_uint16>::vlanes() )
                    {
                        v_uint16 x = vx_load_expand(src);
                        x = v_add_wrap(v_add_wrap(x, x), v_add_wrap(vx_load_expand(src - cn), vx_load_expand(src + cn)));
                        v_store(dst + i, v_reinterpret_as_s32(v_expand_low(x)));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_reinterpret_as_s32(v_expand_high(x)));
                        i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                    }
                    if( i <= width - VTraits<v_uint32>::vlanes() )
                    {
                        v_uint32 x = vx_load_expand_q(src);
                        x = v_add(v_add(v_add(x, x), vx_load_expand_q(src - cn)), vx_load_expand_q(src + cn));
                        v_store(dst + i, v_reinterpret_as_s32(x));
                        i += VTraits<v_uint32>::vlanes();
                    }
                }
                else if( kx[0] == -2 && kx[1] == 1 )
                {
                    for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
                    {
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src), x1l, x1h);
                        v_expand(vx_load(src + cn), x2l, x2h);
                        x1l = v_sub_wrap(v_add_wrap(x0l, x2l), v_add_wrap(x1l, x1l));
                        x1h = v_sub_wrap(v_add_wrap(x0h, x2h), v_add_wrap(x1h, x1h));
                        v_store(dst + i, v_expand_low(v_reinterpret_as_s16(x1l)));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_expand_high(v_reinterpret_as_s16(x1l)));
                        v_store(dst + i + 2*VTraits<v_int32>::vlanes(), v_expand_low(v_reinterpret_as_s16(x1h)));
                        v_store(dst + i + 3*VTraits<v_int32>::vlanes(), v_expand_high(v_reinterpret_as_s16(x1h)));
                    }
                    if( i <= width - VTraits<v_uint16>::vlanes() )
                    {
                        v_uint16 x = vx_load_expand(src);
                        x = v_sub_wrap(v_add_wrap(vx_load_expand(src - cn), vx_load_expand(src + cn)), v_add_wrap(x, x));
                        v_store(dst + i, v_expand_low(v_reinterpret_as_s16(x)));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_expand_high(v_reinterpret_as_s16(x)));
                        i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                    }
                    if( i <= width - VTraits<v_uint32>::vlanes() )
                    {
                        v_int32 x = v_reinterpret_as_s32(vx_load_expand_q(src));
                        x = v_sub(v_reinterpret_as_s32(v_add(vx_load_expand_q(src - cn), vx_load_expand_q(src + cn))), v_add(x, x));
                        v_store(dst + i, x);
                        i += VTraits<v_uint32>::vlanes();
                    }
                }
                else
                {
                    v_int16 k0 = vx_setall_s16((short)kx[0]);
                    v_int16 k1 = vx_setall_s16((short)kx[1]);
                    for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
                    {
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src), x1l, x1h);
                        v_expand(vx_load(src + cn), x2l, x2h);

                        v_int32 dl, dh;
                        v_int16 x0, x1;
                        v_mul_expand(v_reinterpret_as_s16(x1l), k0, dl, dh);
                        v_zip(v_reinterpret_as_s16(x0l), v_reinterpret_as_s16(x2l), x0, x1);
                        dl = v_add(dl, v_dotprod(x0, k1));
                        dh = v_add(dh, v_dotprod(x1, k1));
                        v_store(dst + i, dl);
                        v_store(dst + i + VTraits<v_int32>::vlanes(), dh);

                        v_mul_expand(v_reinterpret_as_s16(x1h), k0, dl, dh);
                        v_zip(v_reinterpret_as_s16(x0h), v_reinterpret_as_s16(x2h), x0, x1);
                        dl = v_add(dl, v_dotprod(x0, k1));
                        dh = v_add(dh, v_dotprod(x1, k1));
                        v_store(dst + i + 2*VTraits<v_int32>::vlanes(), dl);
                        v_store(dst + i + 3*VTraits<v_int32>::vlanes(), dh);
                    }
                    if ( i <= width - VTraits<v_uint16>::vlanes() )
                    {
                        v_int32 dl, dh;
                        v_mul_expand(v_reinterpret_as_s16(vx_load_expand(src)), k0, dl, dh);
                        v_int16 x0, x1;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src - cn)), v_reinterpret_as_s16(vx_load_expand(src + cn)), x0, x1);
                        dl = v_add(dl, v_dotprod(x0, k1));
                        dh = v_add(dh, v_dotprod(x1, k1));
                        v_store(dst + i, dl);
                        v_store(dst + i + VTraits<v_int32>::vlanes(), dh);
                        i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                    }
                    if ( i <= width - VTraits<v_uint32>::vlanes() )
                    {
                        v_store(dst + i, v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src)), vx_setall_s32(kx[0]), v_mul(v_reinterpret_as_s32(v_add(vx_load_expand_q(src - cn), vx_load_expand_q(src + cn))), vx_setall_s32(kx[1]))));
                        i += VTraits<v_uint32>::vlanes();
                    }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                {
                    for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
                    {
                        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
                        v_expand(vx_load(src - 2*cn), x0l, x0h);
                        v_expand(vx_load(src), x1l, x1h);
                        v_expand(vx_load(src + 2*cn), x2l, x2h);
                        x1l = v_sub_wrap(v_add_wrap(x0l, x2l), v_add_wrap(x1l, x1l));
                        x1h = v_sub_wrap(v_add_wrap(x0h, x2h), v_add_wrap(x1h, x1h));
                        v_store(dst + i, v_expand_low(v_reinterpret_as_s16(x1l)));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_expand_high(v_reinterpret_as_s16(x1l)));
                        v_store(dst + i + 2*VTraits<v_int32>::vlanes(), v_expand_low(v_reinterpret_as_s16(x1h)));
                        v_store(dst + i + 3*VTraits<v_int32>::vlanes(), v_expand_high(v_reinterpret_as_s16(x1h)));
                    }
                    if( i <= width - VTraits<v_uint16>::vlanes() )
                    {
                        v_uint16 x = vx_load_expand(src);
                        x = v_sub_wrap(v_add_wrap(vx_load_expand(src - 2*cn), vx_load_expand(src + 2*cn)), v_add_wrap(x, x));
                        v_store(dst + i, v_expand_low(v_reinterpret_as_s16(x)));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_expand_high(v_reinterpret_as_s16(x)));
                        i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                    }
                    if( i <= width - VTraits<v_uint32>::vlanes() )
                    {
                        v_int32 x = v_reinterpret_as_s32(vx_load_expand_q(src));
                        x = v_sub(v_reinterpret_as_s32(v_add(vx_load_expand_q(src - 2 * cn), vx_load_expand_q(src + 2 * cn))), v_add(x, x));
                        v_store(dst + i, x);
                        i += VTraits<v_uint32>::vlanes();
                    }
                }
                else
                {
                    v_int16 k0 = vx_setall_s16((short)(kx[0]));
                    v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[1] & 0xFFFF) | (kx[2] << 16)));
                    for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
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
                        v_zip(v_reinterpret_as_s16(v_add(x0l, x1l)), v_reinterpret_as_s16(v_add(x2l, x3l)), xl, xh);
                        x0 = v_add(x0, v_dotprod(xl, k12));
                        x1 = v_add(x1, v_dotprod(xh, k12));
                        v_zip(v_reinterpret_as_s16(v_add(x0h, x1h)), v_reinterpret_as_s16(v_add(x2h, x3h)), xl, xh);
                        x2 = v_add(x2, v_dotprod(xl, k12));
                        x3 = v_add(x3, v_dotprod(xh, k12));

                        v_store(dst + i, x0);
                        v_store(dst + i + VTraits<v_int32>::vlanes(), x1);
                        v_store(dst + i + 2*VTraits<v_int32>::vlanes(), x2);
                        v_store(dst + i + 3*VTraits<v_int32>::vlanes(), x3);
                    }
                    if( i <= width - VTraits<v_uint16>::vlanes() )
                    {
                        v_int32 x1, x2;
                        v_mul_expand(v_reinterpret_as_s16(vx_load_expand(src)), k0, x1, x2);

                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_add(vx_load_expand(src - cn), vx_load_expand(src + cn))), v_reinterpret_as_s16(v_add(vx_load_expand(src - 2 * cn), vx_load_expand(src + 2 * cn))), xl, xh);
                        x1 = v_add(x1, v_dotprod(xl, k12));
                        x2 = v_add(x2, v_dotprod(xh, k12));

                        v_store(dst + i, x1);
                        v_store(dst + i + VTraits<v_int32>::vlanes(), x2);
                        i += VTraits<v_uint16>::vlanes(), src += VTraits<v_uint16>::vlanes();
                    }
                    if( i <= width - VTraits<v_uint32>::vlanes() )
                    {
                        v_store(dst + i, v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src)), vx_setall_s32(kx[0]),
                                         v_muladd(v_reinterpret_as_s32(v_add(vx_load_expand_q(src - cn), vx_load_expand_q(src + cn))), vx_setall_s32(kx[1]),
                                                  v_mul(v_reinterpret_as_s32(v_add(vx_load_expand_q(src - 2 * cn), vx_load_expand_q(src + 2 * cn))), vx_setall_s32(kx[2])))));
                        i += VTraits<v_uint32>::vlanes();
                    }
                }
            }
            else
            {
                v_int16 k0 = vx_setall_s16((short)(kx[0]));
                for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
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
                        v_zip(v_reinterpret_as_s16(v_add(v_expand_low(v_src0), v_expand_low(v_src2))), v_reinterpret_as_s16(v_add(v_expand_low(v_src1), v_expand_low(v_src3))), xl, xh);
                        s0 = v_add(s0, v_dotprod(xl, k12));
                        s1 = v_add(s1, v_dotprod(xh, k12));
                        v_zip(v_reinterpret_as_s16(v_add(v_expand_high(v_src0), v_expand_high(v_src2))), v_reinterpret_as_s16(v_add(v_expand_high(v_src1), v_expand_high(v_src3))), xl, xh);
                        s2 = v_add(s2, v_dotprod(xl, k12));
                        s3 = v_add(s3, v_dotprod(xh, k12));
                    }
                    if( k < _ksize / 2 + 1 )
                    {
                        v_int16 k1 = vx_setall_s16((short)(kx[k]));

                        v_uint8 v_src0 = vx_load(src - j);
                        v_uint8 v_src1 = vx_load(src + j);

                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_expand_low(v_src0)), v_reinterpret_as_s16(v_expand_low(v_src1)), xl, xh);
                        s0 = v_add(s0, v_dotprod(xl, k1));
                        s1 = v_add(s1, v_dotprod(xh, k1));
                        v_zip(v_reinterpret_as_s16(v_expand_high(v_src0)), v_reinterpret_as_s16(v_expand_high(v_src1)), xl, xh);
                        s2 = v_add(s2, v_dotprod(xl, k1));
                        s3 = v_add(s3, v_dotprod(xh, k1));
                    }
                    v_store(dst + i, s0);
                    v_store(dst + i + VTraits<v_int32>::vlanes(), s1);
                    v_store(dst + i + 2*VTraits<v_int32>::vlanes(), s2);
                    v_store(dst + i + 3*VTraits<v_int32>::vlanes(), s3);
                }
                if( i <= width - VTraits<v_uint16>::vlanes() )
                {
                    v_int32 s0, s1;
                    v_mul_expand(v_reinterpret_as_s16(vx_load_expand(src)), k0, s0, s1);
                    for (k = 1, j = cn; k <= _ksize / 2 - 1; k+=2, j += 2*cn)
                    {
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_add(vx_load_expand(src - j), vx_load_expand(src + j))), v_reinterpret_as_s16(v_add(vx_load_expand(src - j - cn), vx_load_expand(src + j + cn))), xl, xh);
                        v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (kx[k+1] << 16)));
                        s0 = v_add(s0, v_dotprod(xl, k12));
                        s1 = v_add(s1, v_dotprod(xh, k12));
                    }
                    if ( k < _ksize / 2 + 1 )
                    {
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src - j)), v_reinterpret_as_s16(vx_load_expand(src + j)), xl, xh);
                        v_int16 k1 = vx_setall_s16((short)(kx[k]));
                        s0 = v_add(s0, v_dotprod(xl, k1));
                        s1 = v_add(s1, v_dotprod(xh, k1));
                    }
                    v_store(dst + i, s0);
                    v_store(dst + i + VTraits<v_int32>::vlanes(), s1);
                    i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                }
                if( i <= width - VTraits<v_uint32>::vlanes() )
                {
                    v_int32 s0 = v_mul(v_reinterpret_as_s32(vx_load_expand_q(src)), vx_setall_s32(kx[0]));
                    for( k = 1, j = cn; k < _ksize / 2 + 1; k++, j += cn )
                        s0 = v_muladd(v_reinterpret_as_s32(v_add(vx_load_expand_q(src - j), vx_load_expand_q(src + j))), vx_setall_s32(kx[k]), s0);
                    v_store(dst + i, s0);
                    i += VTraits<v_uint32>::vlanes();
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                {
                    for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
                    {
                        v_uint16 x0l, x0h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src + cn), x2l, x2h);
                        v_int16 dl = v_reinterpret_as_s16(v_sub_wrap(x2l, x0l));
                        v_int16 dh = v_reinterpret_as_s16(v_sub_wrap(x2h, x0h));
                        v_store(dst + i, v_expand_low(dl));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_expand_high(dl));
                        v_store(dst + i + 2*VTraits<v_int32>::vlanes(), v_expand_low(dh));
                        v_store(dst + i + 3*VTraits<v_int32>::vlanes(), v_expand_high(dh));
                    }
                    if( i <= width - VTraits<v_uint16>::vlanes() )
                    {
                        v_int16 dl = v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + cn), vx_load_expand(src - cn)));
                        v_store(dst + i, v_expand_low(dl));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_expand_high(dl));
                        i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                    }
                    if (i <= width - VTraits<v_uint32>::vlanes())
                    {
                        v_store(dst + i, v_sub(v_reinterpret_as_s32(vx_load_expand_q(src + cn)), v_reinterpret_as_s32(vx_load_expand_q(src - cn))));
                        i += VTraits<v_uint32>::vlanes();
                    }
                }
                else
                {
                    v_int16 k0 = v_reinterpret_as_s16(vx_setall_s32((kx[1] & 0xFFFF) | (-kx[1] << 16)));
                    for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
                    {
                        v_uint16 x0l, x0h, x2l, x2h;
                        v_expand(vx_load(src - cn), x0l, x0h);
                        v_expand(vx_load(src + cn), x2l, x2h);
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(x2l), v_reinterpret_as_s16(x0l), xl, xh);
                        v_store(dst + i, v_dotprod(xl, k0));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_dotprod(xh, k0));
                        v_zip(v_reinterpret_as_s16(x2h), v_reinterpret_as_s16(x0h), xl, xh);
                        v_store(dst + i + 2*VTraits<v_int32>::vlanes(), v_dotprod(xl, k0));
                        v_store(dst + i + 3*VTraits<v_int32>::vlanes(), v_dotprod(xh, k0));
                    }
                    if( i <= width - VTraits<v_uint16>::vlanes() )
                    {
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src + cn)), v_reinterpret_as_s16(vx_load_expand(src - cn)), xl, xh);
                        v_store(dst + i, v_dotprod(xl, k0));
                        v_store(dst + i + VTraits<v_int32>::vlanes(), v_dotprod(xh, k0));
                        i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                    }
                    if (i <= width - VTraits<v_uint32>::vlanes())
                    {
                        v_store(dst + i, v_muladd(v_reinterpret_as_s32(vx_load_expand_q(src + cn)), vx_setall_s32(kx[1]), v_mul(v_reinterpret_as_s32(vx_load_expand_q(src - cn)), vx_setall_s32(-kx[1]))));
                        i += VTraits<v_uint32>::vlanes();
                    }
                }
            }
            else if( _ksize == 5 )
            {
                v_int16 k0 = v_reinterpret_as_s16(vx_setall_s32((kx[1] & 0xFFFF) | (kx[2] << 16)));
                for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
                {
                    v_uint16 x0l, x0h, x1l, x1h, x2l, x2h, x3l, x3h;
                    v_expand(vx_load(src - cn), x0l, x0h);
                    v_expand(vx_load(src - 2*cn), x1l, x1h);
                    v_expand(vx_load(src + cn), x2l, x2h);
                    v_expand(vx_load(src + 2*cn), x3l, x3h);
                    v_int16 x0, x1;
                    v_zip(v_reinterpret_as_s16(v_sub_wrap(x2l, x0l)), v_reinterpret_as_s16(v_sub_wrap(x3l, x1l)), x0, x1);
                    v_store(dst + i, v_dotprod(x0, k0));
                    v_store(dst + i + VTraits<v_int32>::vlanes(), v_dotprod(x1, k0));
                    v_zip(v_reinterpret_as_s16(v_sub_wrap(x2h, x0h)), v_reinterpret_as_s16(v_sub_wrap(x3h, x1h)), x0, x1);
                    v_store(dst + i + 2*VTraits<v_int32>::vlanes(), v_dotprod(x0, k0));
                    v_store(dst + i + 3*VTraits<v_int32>::vlanes(), v_dotprod(x1, k0));
                }
                if( i <= width - VTraits<v_uint16>::vlanes() )
                {
                    v_int16 x0, x1;
                    v_zip(v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + cn), vx_load_expand(src - cn))),
                          v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + 2*cn), vx_load_expand(src - 2*cn))), x0, x1);
                    v_store(dst + i, v_dotprod(x0, k0));
                    v_store(dst + i + VTraits<v_int32>::vlanes(), v_dotprod(x1, k0));
                    i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                }
                if( i <= width - VTraits<v_uint32>::vlanes() )
                {
                    v_store(dst + i, v_muladd(v_sub(v_reinterpret_as_s32(vx_load_expand_q(src + cn)), v_reinterpret_as_s32(vx_load_expand_q(src - cn))), vx_setall_s32(kx[1]),
                                             v_mul(v_sub(v_reinterpret_as_s32(vx_load_expand_q(src + 2 * cn)), v_reinterpret_as_s32(vx_load_expand_q(src - 2 * cn))), vx_setall_s32(kx[2]))));
                    i += VTraits<v_uint32>::vlanes();
                }
            }
            else
            {
                v_int16 k0 = vx_setall_s16((short)(kx[0]));
                for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(), src += VTraits<v_uint8>::vlanes() )
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
                        s0 = v_add(s0, v_dotprod(xl, k12));
                        s1 = v_add(s1, v_dotprod(xh, k12));
                        v_zip(v_reinterpret_as_s16(v_sub_wrap(v_expand_high(v_src2), v_expand_high(v_src0))), v_reinterpret_as_s16(v_sub_wrap(v_expand_high(v_src3), v_expand_high(v_src1))), xl, xh);
                        s2 = v_add(s2, v_dotprod(xl, k12));
                        s3 = v_add(s3, v_dotprod(xh, k12));
                    }
                    if( k < _ksize / 2 + 1 )
                    {
                        v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (-kx[k] << 16)));
                        v_uint8 v_src0 = vx_load(src - j);
                        v_uint8 v_src1 = vx_load(src + j);

                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_expand_low(v_src1)), v_reinterpret_as_s16(v_expand_low(v_src0)), xl, xh);
                        s0 = v_add(s0, v_dotprod(xl, k12));
                        s1 = v_add(s1, v_dotprod(xh, k12));
                        v_zip(v_reinterpret_as_s16(v_expand_high(v_src1)), v_reinterpret_as_s16(v_expand_high(v_src0)), xl, xh);
                        s2 = v_add(s2, v_dotprod(xl, k12));
                        s3 = v_add(s3, v_dotprod(xh, k12));
                    }
                    v_store(dst + i, s0);
                    v_store(dst + i + VTraits<v_int32>::vlanes(), s1);
                    v_store(dst + i + 2*VTraits<v_int32>::vlanes(), s2);
                    v_store(dst + i + 3*VTraits<v_int32>::vlanes(), s3);
                }
                if( i <= width - VTraits<v_uint16>::vlanes() )
                {
                    v_int32 s0, s1;
                    v_mul_expand(v_reinterpret_as_s16(vx_load_expand(src)), k0, s0, s1);
                    for( k = 1, j = cn; k <= _ksize / 2 - 1; k += 2, j += 2 * cn )
                    {
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + j), vx_load_expand(src - j))), v_reinterpret_as_s16(v_sub_wrap(vx_load_expand(src + j + cn), vx_load_expand(src - j - cn))), xl, xh);
                        v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (kx[k + 1] << 16)));
                        s0 = v_add(s0, v_dotprod(xl, k12));
                        s1 = v_add(s1, v_dotprod(xh, k12));
                    }
                    if( k < _ksize / 2 + 1 )
                    {
                        v_int16 k1 = v_reinterpret_as_s16(vx_setall_s32((kx[k] & 0xFFFF) | (-kx[k] << 16)));
                        v_int16 xl, xh;
                        v_zip(v_reinterpret_as_s16(vx_load_expand(src + j)), v_reinterpret_as_s16(vx_load_expand(src - j)), xl, xh);
                        s0 = v_add(s0, v_dotprod(xl, k1));
                        s1 = v_add(s1, v_dotprod(xh, k1));
                    }
                    v_store(dst + i, s0);
                    v_store(dst + i + VTraits<v_int32>::vlanes(), s1);
                    i += VTraits<v_uint16>::vlanes(); src += VTraits<v_uint16>::vlanes();
                }
                if( i <= width - VTraits<v_uint32>::vlanes() )
                {
                    v_int32 s0 = v_mul(v_reinterpret_as_s32(vx_load_expand_q(src)), vx_setall_s32(kx[0]));
                    for (k = 1, j = cn; k < _ksize / 2 + 1; k++, j += cn)
                        s0 = v_muladd(v_sub(v_reinterpret_as_s32(vx_load_expand_q(src + j)), v_reinterpret_as_s32(vx_load_expand_q(src - j))), vx_setall_s32(kx[k]), s0);
                    v_store(dst + i, s0);
                    i += VTraits<v_uint32>::vlanes();
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
        CV_INSTRUMENT_REGION();

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
            for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes() )
            {
                const int* S = src[0] + i;
                v_float32 s0 = v_muladd(v_cvt_f32(vx_load(S)), f0, d4);
                v_float32 s1 = v_muladd(v_cvt_f32(vx_load(S + VTraits<v_int32>::vlanes())), f0, d4);
                v_float32 s2 = v_muladd(v_cvt_f32(vx_load(S + 2*VTraits<v_int32>::vlanes())), f0, d4);
                v_float32 s3 = v_muladd(v_cvt_f32(vx_load(S + 3*VTraits<v_int32>::vlanes())), f0, d4);
                const int* S0 = src[1] + i;
                const int* S1 = src[-1] + i;
                s0 = v_muladd(v_cvt_f32(v_add(vx_load(S0), vx_load(S1))), f1, s0);
                s1 = v_muladd(v_cvt_f32(v_add(vx_load(S0 + VTraits<v_int32>::vlanes()), vx_load(S1 + VTraits<v_int32>::vlanes()))), f1, s1);
                s2 = v_muladd(v_cvt_f32(v_add(vx_load(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load(S1 + 2 * VTraits<v_int32>::vlanes()))), f1, s2);
                s3 = v_muladd(v_cvt_f32(v_add(vx_load(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load(S1 + 3 * VTraits<v_int32>::vlanes()))), f1, s3);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 f = vx_setall_f32(ky[k]);
                    S0 = src[k] + i;
                    S1 = src[-k] + i;
                    s0 = v_muladd(v_cvt_f32(v_add(vx_load(S0), vx_load(S1))), f, s0);
                    s1 = v_muladd(v_cvt_f32(v_add(vx_load(S0 + VTraits<v_int32>::vlanes()), vx_load(S1 + VTraits<v_int32>::vlanes()))), f, s1);
                    s2 = v_muladd(v_cvt_f32(v_add(vx_load(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load(S1 + 2 * VTraits<v_int32>::vlanes()))), f, s2);
                    s3 = v_muladd(v_cvt_f32(v_add(vx_load(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load(S1 + 3 * VTraits<v_int32>::vlanes()))), f, s3);
                }
                v_store(dst + i, v_pack_u(v_pack(v_round(s0), v_round(s1)), v_pack(v_round(s2), v_round(s3))));
            }
            if( i <= width - VTraits<v_uint16>::vlanes() )
            {
                const int* S = src[0] + i;
                v_float32 s0 = v_muladd(v_cvt_f32(vx_load(S)), f0, d4);
                v_float32 s1 = v_muladd(v_cvt_f32(vx_load(S + VTraits<v_int32>::vlanes())), f0, d4);
                const int* S0 = src[1] + i;
                const int* S1 = src[-1] + i;
                s0 = v_muladd(v_cvt_f32(v_add(vx_load(S0), vx_load(S1))), f1, s0);
                s1 = v_muladd(v_cvt_f32(v_add(vx_load(S0 + VTraits<v_int32>::vlanes()), vx_load(S1 + VTraits<v_int32>::vlanes()))), f1, s1);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 f = vx_setall_f32(ky[k]);
                    S0 = src[k] + i;
                    S1 = src[-k] + i;
                    s0 = v_muladd(v_cvt_f32(v_add(vx_load(S0), vx_load(S1))), f, s0);
                    s1 = v_muladd(v_cvt_f32(v_add(vx_load(S0 + VTraits<v_int32>::vlanes()), vx_load(S1 + VTraits<v_int32>::vlanes()))), f, s1);
                }
                v_pack_u_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                i += VTraits<v_uint16>::vlanes();
            }
        }
        else
        {
            v_float32 f1 = vx_setall_f32(ky[1]);
            for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes() )
            {
                const int* S0 = src[1] + i;
                const int* S1 = src[-1] + i;
                v_float32 s0 = v_muladd(v_cvt_f32(v_sub(vx_load(S0), vx_load(S1))), f1, d4);
                v_float32 s1 = v_muladd(v_cvt_f32(v_sub(vx_load(S0 + VTraits<v_int32>::vlanes()), vx_load(S1 + VTraits<v_int32>::vlanes()))), f1, d4);
                v_float32 s2 = v_muladd(v_cvt_f32(v_sub(vx_load(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load(S1 + 2 * VTraits<v_int32>::vlanes()))), f1, d4);
                v_float32 s3 = v_muladd(v_cvt_f32(v_sub(vx_load(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load(S1 + 3 * VTraits<v_int32>::vlanes()))), f1, d4);
                for ( k = 2; k <= ksize2; k++ )
                {
                    v_float32 f = vx_setall_f32(ky[k]);
                    S0 = src[k] + i;
                    S1 = src[-k] + i;
                    s0 = v_muladd(v_cvt_f32(v_sub(vx_load(S0), vx_load(S1))), f, s0);
                    s1 = v_muladd(v_cvt_f32(v_sub(vx_load(S0 + VTraits<v_int32>::vlanes()), vx_load(S1 + VTraits<v_int32>::vlanes()))), f, s1);
                    s2 = v_muladd(v_cvt_f32(v_sub(vx_load(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load(S1 + 2 * VTraits<v_int32>::vlanes()))), f, s2);
                    s3 = v_muladd(v_cvt_f32(v_sub(vx_load(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load(S1 + 3 * VTraits<v_int32>::vlanes()))), f, s3);
                }
                v_store(dst + i, v_pack_u(v_pack(v_round(s0), v_round(s1)), v_pack(v_round(s2), v_round(s3))));
            }
            if( i <= width - VTraits<v_uint16>::vlanes() )
            {
                const int* S0 = src[1] + i;
                const int* S1 = src[-1] + i;
                v_float32 s0 = v_muladd(v_cvt_f32(v_sub(vx_load(S0), vx_load(S1))), f1, d4);
                v_float32 s1 = v_muladd(v_cvt_f32(v_sub(vx_load(S0 + VTraits<v_int32>::vlanes()), vx_load(S1 + VTraits<v_int32>::vlanes()))), f1, d4);
                for ( k = 2; k <= ksize2; k++ )
                {
                    v_float32 f = vx_setall_f32(ky[k]);
                    S0 = src[k] + i;
                    S1 = src[-k] + i;
                    s0 = v_muladd(v_cvt_f32(v_sub(vx_load(S0), vx_load(S1))), f, s0);
                    s1 = v_muladd(v_cvt_f32(v_sub(vx_load(S0 + VTraits<v_int32>::vlanes()), vx_load(S1 + VTraits<v_int32>::vlanes()))), f, s1);
                }
                v_pack_u_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                i += VTraits<v_uint16>::vlanes();
            }
        }
        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};

struct SymmColumnVec_32f8u
{
    SymmColumnVec_32f8u() { symmetryType = 0; delta = 0; }
    SymmColumnVec_32f8u(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        CV_INSTRUMENT_REGION();

        int _ksize = kernel.rows + kernel.cols - 1;
        if( _ksize == 1 ) return 0;
        const int ksize2 = _ksize / 2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;

        if( symmetrical )
        {
            for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes() )
            {
                v_float32 v_ky0 = vx_setall_f32(ky[0]);
                v_float32 v32_delta = vx_setall_f32(delta);
                const float* S = src[0] + i;
                v_float32 s0 = v_muladd(v_ky0, vx_load(S), v32_delta);
                v_float32 s1 = v_muladd(v_ky0, vx_load(S + VTraits<v_float32>::vlanes()), v32_delta);
                v_float32 s2 = v_muladd(v_ky0, vx_load(S + 2*VTraits<v_float32>::vlanes()), v32_delta);
                v_float32 s3 = v_muladd(v_ky0, vx_load(S + 3*VTraits<v_float32>::vlanes()), v32_delta);
                for( k = 1; k <= ksize2; k++ )
                {
                    v_float32 v_kyk = vx_setall_f32(ky[k]);
                    const float* S0 = src[k] + i;
                    const float* S1 = src[-k] + i;
                    s0 = v_muladd(v_kyk, v_add(vx_load(S0), vx_load(S1)), s0);
                    s1 = v_muladd(v_kyk, v_add(vx_load(S0 + VTraits<v_float32>::vlanes()), vx_load(S1 + VTraits<v_float32>::vlanes())), s1);
                    s2 = v_muladd(v_kyk, v_add(vx_load(S0 + 2 * VTraits<v_float32>::vlanes()), vx_load(S1 + 2 * VTraits<v_float32>::vlanes())), s2);
                    s3 = v_muladd(v_kyk, v_add(vx_load(S0 + 3 * VTraits<v_float32>::vlanes()), vx_load(S1 + 3 * VTraits<v_float32>::vlanes())), s3);
                }
                v_store(_dst + i, v_pack_u(v_pack(v_round(s0), v_round(s1)), v_pack(v_round(s2), v_round(s3))));
            }
        }
        else
        {
            for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes() )
            {
                v_float32 s0 = vx_setall_f32(delta);
                v_float32 s1 = vx_setall_f32(delta);
                v_float32 s2 = vx_setall_f32(delta);
                v_float32 s3 = vx_setall_f32(delta);
                for( k = 1; k <= ksize2; k++ )
                {
                    v_float32 v_kyk = vx_setall_f32(ky[k]);
                    const float* S0 = src[k] + i;
                    const float* S1 = src[-k] + i;
                    s0 = v_muladd(v_kyk, v_sub(vx_load(S0), vx_load(S1)), s0);
                    s1 = v_muladd(v_kyk, v_sub(vx_load(S0 + VTraits<v_float32>::vlanes()), vx_load(S1 + VTraits<v_float32>::vlanes())), s1);
                    s2 = v_muladd(v_kyk, v_sub(vx_load(S0 + 2 * VTraits<v_float32>::vlanes()), vx_load(S1 + 2 * VTraits<v_float32>::vlanes())), s2);
                    s3 = v_muladd(v_kyk, v_sub(vx_load(S0 + 3 * VTraits<v_float32>::vlanes()), vx_load(S1 + 3 * VTraits<v_float32>::vlanes())), s3);
                }
                v_store(_dst + i, v_pack_u(v_pack(v_round(s0), v_round(s1)), v_pack(v_round(s2), v_round(s3))));
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
        CV_INSTRUMENT_REGION();

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
                for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
                {
                    v_int32 s0 = vx_load(S1 + i);
                    v_int32 s1 = vx_load(S1 + i + VTraits<v_int32>::vlanes());
                    v_int32 s2 = vx_load(S1 + i + 2*VTraits<v_int32>::vlanes());
                    v_int32 s3 = vx_load(S1 + i + 3*VTraits<v_int32>::vlanes());
                    v_store(dst + i, v_add(v_pack(v_add(v_add(vx_load(S0 + i), vx_load(S2 + i)), v_add(s0, s0)), v_add(v_add(vx_load(S0 + i + VTraits<v_int32>::vlanes()), vx_load(S2 + i + VTraits<v_int32>::vlanes())), v_add(s1, s1))), d8));
                    v_store(dst + i + VTraits<v_int16>::vlanes(), v_add(v_pack(v_add(v_add(vx_load(S0 + i + 2 * VTraits<v_int32>::vlanes()), vx_load(S2 + i + 2 * VTraits<v_int32>::vlanes())), v_add(s2, s2)), v_add(v_add(vx_load(S0 + i + 3 * VTraits<v_int32>::vlanes()), vx_load(S2 + i + 3 * VTraits<v_int32>::vlanes())), v_add(s3, s3))), d8));
                }
                if( i <= width - VTraits<v_int16>::vlanes() )
                {
                    v_int32 sl = vx_load(S1 + i);
                    v_int32 sh = vx_load(S1 + i + VTraits<v_int32>::vlanes());
                    v_store(dst + i, v_add(v_pack(v_add(v_add(vx_load(S0 + i), vx_load(S2 + i)), v_add(sl, sl)), v_add(v_add(vx_load(S0 + i + VTraits<v_int32>::vlanes()), vx_load(S2 + i + VTraits<v_int32>::vlanes())), v_add(sh, sh))), d8));
                    i += VTraits<v_int16>::vlanes();
                }
                if( i <= width - VTraits<v_int32>::vlanes() )
                {
                    v_int32 s = vx_load(S1 + i);
                    v_pack_store(dst + i, v_add(v_add(v_add(vx_load(S0 + i), vx_load(S2 + i)), vx_setall_s32(d)), v_add(s, s)));
                    i += VTraits<v_int32>::vlanes();
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
                {
                    v_int32 s0 = vx_load(S1 + i);
                    v_int32 s1 = vx_load(S1 + i + VTraits<v_int32>::vlanes());
                    v_int32 s2 = vx_load(S1 + i + 2*VTraits<v_int32>::vlanes());
                    v_int32 s3 = vx_load(S1 + i + 3*VTraits<v_int32>::vlanes());
                    v_store(dst + i, v_add(v_pack(v_sub(v_add(vx_load(S0 + i), vx_load(S2 + i)), v_add(s0, s0)), v_sub(v_add(vx_load(S0 + i + VTraits<v_int32>::vlanes()), vx_load(S2 + i + VTraits<v_int32>::vlanes())), v_add(s1, s1))), d8));
                    v_store(dst + i + VTraits<v_int16>::vlanes(), v_add(v_pack(v_sub(v_add(vx_load(S0 + i + 2 * VTraits<v_int32>::vlanes()), vx_load(S2 + i + 2 * VTraits<v_int32>::vlanes())), v_add(s2, s2)), v_sub(v_add(vx_load(S0 + i + 3 * VTraits<v_int32>::vlanes()), vx_load(S2 + i + 3 * VTraits<v_int32>::vlanes())), v_add(s3, s3))), d8));
                }
                if( i <= width - VTraits<v_int16>::vlanes() )
                {
                    v_int32 sl = vx_load(S1 + i);
                    v_int32 sh = vx_load(S1 + i + VTraits<v_int32>::vlanes());
                    v_store(dst + i, v_add(v_pack(v_sub(v_add(vx_load(S0 + i), vx_load(S2 + i)), v_add(sl, sl)), v_sub(v_add(vx_load(S0 + i + VTraits<v_int32>::vlanes()), vx_load(S2 + i + VTraits<v_int32>::vlanes())), v_add(sh, sh))), d8));
                    i += VTraits<v_int16>::vlanes();
                }
                if( i <= width - VTraits<v_int32>::vlanes() )
                {
                    v_int32 s = vx_load(S1 + i);
                    v_pack_store(dst + i, v_sub(v_add(v_add(vx_load(S0 + i), vx_load(S2 + i)), vx_setall_s32(d)), v_add(s, s)));
                    i += VTraits<v_int32>::vlanes();
                }
            }
#if CV_NEON
            else if( ky[0] == (float)((int)ky[0]) && ky[1] == (float)((int)ky[1]) )
            {
                v_int32 k0 = vx_setall_s32((int)ky[0]), k1 = vx_setall_s32((int)ky[1]);
                v_int32 d4 = vx_setall_s32(d);
                for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
                {
                    v_store(dst + i, v_pack(v_muladd(v_add(vx_load(S0 + i), vx_load(S2 + i)), k1, v_muladd(vx_load(S1 + i), k0, d4)),
                                            v_muladd(v_add(vx_load(S0 + i + VTraits<v_int32>::vlanes()), vx_load(S2 + i + VTraits<v_int32>::vlanes())), k1, v_muladd(vx_load(S1 + i + VTraits<v_int32>::vlanes()), k0, d4))));
                    v_store(dst + i + VTraits<v_int16>::vlanes(), v_pack(v_muladd(v_add(vx_load(S0 + i + 2 * VTraits<v_int32>::vlanes()), vx_load(S2 + i + 2 * VTraits<v_int32>::vlanes())), k1, v_muladd(vx_load(S1 + i + 2*VTraits<v_int32>::vlanes()), k0, d4)),
                                                              v_muladd(v_add(vx_load(S0 + i + 3 * VTraits<v_int32>::vlanes()), vx_load(S2 + i + 3 * VTraits<v_int32>::vlanes())), k1, v_muladd(vx_load(S1 + i + 3*VTraits<v_int32>::vlanes()), k0, d4))));
                }
                if( i <= width - VTraits<v_int16>::vlanes() )
                {
                    v_store(dst + i, v_pack(v_muladd(v_add(vx_load(S0 + i), vx_load(S2 + i)), k1, v_muladd(vx_load(S1 + i), k0, d4)),
                                            v_muladd(v_add(vx_load(S0 + i + VTraits<v_int32>::vlanes()), vx_load(S2 + i + VTraits<v_int32>::vlanes())), k1, v_muladd(vx_load(S1 + i + VTraits<v_int32>::vlanes()), k0, d4))));
                    i += VTraits<v_int16>::vlanes();
                }
                if( i <= width - VTraits<v_int32>::vlanes() )
                {
                    v_pack_store(dst + i, v_muladd(v_add(vx_load(S0 + i), vx_load(S2 + i)), k1, v_muladd(vx_load(S1 + i), k0, d4)));
                    i += VTraits<v_int32>::vlanes();
                }
            }
#endif
            else
            {
                v_float32 k0 = vx_setall_f32(ky[0]), k1 = vx_setall_f32(ky[1]);
                for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
                {
                    v_store(dst + i, v_pack(v_round(v_muladd(v_cvt_f32(v_add(vx_load(S0 + i), vx_load(S2 + i))), k1, v_muladd(v_cvt_f32(vx_load(S1 + i)), k0, df4))),
                                            v_round(v_muladd(v_cvt_f32(v_add(vx_load(S0 + i + VTraits<v_int32>::vlanes()), vx_load(S2 + i + VTraits<v_int32>::vlanes()))), k1, v_muladd(v_cvt_f32(vx_load(S1 + i + VTraits<v_int32>::vlanes())), k0, df4)))));
                    v_store(dst + i + VTraits<v_int16>::vlanes(), v_pack(v_round(v_muladd(v_cvt_f32(v_add(vx_load(S0 + i + 2 * VTraits<v_int32>::vlanes()), vx_load(S2 + i + 2 * VTraits<v_int32>::vlanes()))), k1, v_muladd(v_cvt_f32(vx_load(S1 + i + 2*VTraits<v_int32>::vlanes())), k0, df4))),
                                                              v_round(v_muladd(v_cvt_f32(v_add(vx_load(S0 + i + 3 * VTraits<v_int32>::vlanes()), vx_load(S2 + i + 3 * VTraits<v_int32>::vlanes()))), k1, v_muladd(v_cvt_f32(vx_load(S1 + i + 3*VTraits<v_int32>::vlanes())), k0, df4)))));
                }
                if( i <= width - VTraits<v_int16>::vlanes() )
                {
                    v_store(dst + i, v_pack(v_round(v_muladd(v_cvt_f32(v_add(vx_load(S0 + i), vx_load(S2 + i))), k1, v_muladd(v_cvt_f32(vx_load(S1 + i)), k0, df4))),
                                            v_round(v_muladd(v_cvt_f32(v_add(vx_load(S0 + i + VTraits<v_int32>::vlanes()), vx_load(S2 + i + VTraits<v_int32>::vlanes()))), k1, v_muladd(v_cvt_f32(vx_load(S1 + i + VTraits<v_int32>::vlanes())), k0, df4)))));
                    i += VTraits<v_int16>::vlanes();
                }
                if( i <= width - VTraits<v_int32>::vlanes() )
                {
                    v_pack_store(dst + i, v_round(v_muladd(v_cvt_f32(v_add(vx_load(S0 + i), vx_load(S2 + i))), k1, v_muladd(v_cvt_f32(vx_load(S1 + i)), k0, df4))));
                    i += VTraits<v_int32>::vlanes();
                }
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
                {
                    v_store(dst + i, v_add(v_pack(v_sub(vx_load(S2 + i), vx_load(S0 + i)), v_sub(vx_load(S2 + i + VTraits<v_int32>::vlanes()), vx_load(S0 + i + VTraits<v_int32>::vlanes()))), d8));
                    v_store(dst + i + VTraits<v_int16>::vlanes(), v_add(v_pack(v_sub(vx_load(S2 + i + 2 * VTraits<v_int32>::vlanes()), vx_load(S0 + i + 2 * VTraits<v_int32>::vlanes())), v_sub(vx_load(S2 + i + 3 * VTraits<v_int32>::vlanes()), vx_load(S0 + i + 3 * VTraits<v_int32>::vlanes()))), d8));
                }
                if( i <= width - VTraits<v_int16>::vlanes() )
                {
                    v_store(dst + i, v_add(v_pack(v_sub(vx_load(S2 + i), vx_load(S0 + i)), v_sub(vx_load(S2 + i + VTraits<v_int32>::vlanes()), vx_load(S0 + i + VTraits<v_int32>::vlanes()))), d8));
                    i += VTraits<v_int16>::vlanes();
                }
                if( i <= width - VTraits<v_int32>::vlanes() )
                {
                    v_pack_store(dst + i, v_add(v_sub(vx_load(S2 + i), vx_load(S0 + i)), vx_setall_s32(d)));
                    i += VTraits<v_int32>::vlanes();
                }
            }
            else
            {
                v_float32 k1 = vx_setall_f32(ky[1]);
                for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
                {
                    v_store(dst + i, v_pack(v_round(v_muladd(v_cvt_f32(v_sub(vx_load(S2 + i), vx_load(S0 + i))), k1, df4)),
                                            v_round(v_muladd(v_cvt_f32(v_sub(vx_load(S2 + i + VTraits<v_int32>::vlanes()), vx_load(S0 + i + VTraits<v_int32>::vlanes()))), k1, df4))));
                    v_store(dst + i + VTraits<v_int16>::vlanes(), v_pack(v_round(v_muladd(v_cvt_f32(v_sub(vx_load(S2 + i + 2 * VTraits<v_int32>::vlanes()), vx_load(S0 + i + 2 * VTraits<v_int32>::vlanes()))), k1, df4)),
                                                              v_round(v_muladd(v_cvt_f32(v_sub(vx_load(S2 + i + 3 * VTraits<v_int32>::vlanes()), vx_load(S0 + i + 3 * VTraits<v_int32>::vlanes()))), k1, df4))));
                }
                if( i <= width - VTraits<v_int16>::vlanes() )
                {
                    v_store(dst + i, v_pack(v_round(v_muladd(v_cvt_f32(v_sub(vx_load(S2 + i), vx_load(S0 + i))), k1, df4)),
                                            v_round(v_muladd(v_cvt_f32(v_sub(vx_load(S2 + i + VTraits<v_int32>::vlanes()), vx_load(S0 + i + VTraits<v_int32>::vlanes()))), k1, df4))));
                    i += VTraits<v_int16>::vlanes();
                }
                if( i <= width - VTraits<v_int32>::vlanes() )
                {
                    v_pack_store(dst + i, v_round(v_muladd(v_cvt_f32(v_sub(vx_load(S2 + i), vx_load(S0 + i))), k1, df4)));
                    i += VTraits<v_int32>::vlanes();
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
    RowVec_16s32f() {}
    RowVec_16s32f( const Mat& _kernel )
    {
        kernel = _kernel;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* _kx = kernel.ptr<float>();
        width *= cn;

        for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
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
                v_int16 xh = vx_load(src + VTraits<v_int16>::vlanes());
                s0 = v_muladd(v_cvt_f32(v_expand_low(xl)), f, s0);
                s1 = v_muladd(v_cvt_f32(v_expand_high(xl)), f, s1);
                s2 = v_muladd(v_cvt_f32(v_expand_low(xh)), f, s2);
                s3 = v_muladd(v_cvt_f32(v_expand_high(xh)), f, s3);
            }
            v_store(dst + i, s0);
            v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
            v_store(dst + i + 2*VTraits<v_float32>::vlanes(), s2);
            v_store(dst + i + 3*VTraits<v_float32>::vlanes(), s3);
        }
        if( i <= width - VTraits<v_int16>::vlanes() )
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
            v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
            i += VTraits<v_int16>::vlanes();
        }
        if( i <= width - VTraits<v_float32>::vlanes() )
        {
            const short* src = (const short*)_src + i;
            v_float32 s0 = vx_setzero_f32();
            for( k = 0; k < _ksize; k++, src += cn )
                s0 = v_muladd(v_cvt_f32(vx_load_expand(src)), vx_setall_f32(_kx[k]), s0);
            v_store(dst + i, s0);
            i += VTraits<v_float32>::vlanes();
        }
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
        CV_INSTRUMENT_REGION();

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
            for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                v_float32 s1 = v_muladd(vx_load(src[0] + i + VTraits<v_float32>::vlanes()), k0, d4);
                v_float32 s2 = v_muladd(vx_load(src[0] + i + 2*VTraits<v_float32>::vlanes()), k0, d4);
                v_float32 s3 = v_muladd(vx_load(src[0] + i + 3*VTraits<v_float32>::vlanes()), k0, d4);
                s0 = v_muladd(v_add(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, s0);
                s1 = v_muladd(v_add(vx_load(src[1] + i + VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + VTraits<v_float32>::vlanes())), k1, s1);
                s2 = v_muladd(v_add(vx_load(src[1] + i + 2 * VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + 2 * VTraits<v_float32>::vlanes())), k1, s2);
                s3 = v_muladd(v_add(vx_load(src[1] + i + 3 * VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + 3 * VTraits<v_float32>::vlanes())), k1, s3);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(v_add(vx_load(src[k] + i), vx_load(src[-k] + i)), k2, s0);
                    s1 = v_muladd(v_add(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + VTraits<v_float32>::vlanes())), k2, s1);
                    s2 = v_muladd(v_add(vx_load(src[k] + i + 2 * VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + 2 * VTraits<v_float32>::vlanes())), k2, s2);
                    s3 = v_muladd(v_add(vx_load(src[k] + i + 3 * VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + 3 * VTraits<v_float32>::vlanes())), k2, s3);
                }
                v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                v_store(dst + i + VTraits<v_int16>::vlanes(), v_pack(v_round(s2), v_round(s3)));
            }
            if( i <= width - VTraits<v_int16>::vlanes() )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                v_float32 s1 = v_muladd(vx_load(src[0] + i + VTraits<v_float32>::vlanes()), k0, d4);
                s0 = v_muladd(v_add(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, s0);
                s1 = v_muladd(v_add(vx_load(src[1] + i + VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + VTraits<v_float32>::vlanes())), k1, s1);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(v_add(vx_load(src[k] + i), vx_load(src[-k] + i)), k2, s0);
                    s1 = v_muladd(v_add(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + VTraits<v_float32>::vlanes())), k2, s1);
                }
                v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                i += VTraits<v_int16>::vlanes();
            }
            if( i <= width - VTraits<v_float32>::vlanes() )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                s0 = v_muladd(v_add(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, s0);
                for( k = 2; k <= ksize2; k++ )
                    s0 = v_muladd(v_add(vx_load(src[k] + i), vx_load(src[-k] + i)), vx_setall_f32(ky[k]), s0);
                v_pack_store(dst + i, v_round(s0));
                i += VTraits<v_float32>::vlanes();
            }
        }
        else
        {
            v_float32 k1 = vx_setall_f32(ky[1]);
            for( ; i <= width - 2*VTraits<v_int16>::vlanes(); i += 2*VTraits<v_int16>::vlanes() )
            {
                v_float32 s0 = v_muladd(v_sub(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, d4);
                v_float32 s1 = v_muladd(v_sub(vx_load(src[1] + i + VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + VTraits<v_float32>::vlanes())), k1, d4);
                v_float32 s2 = v_muladd(v_sub(vx_load(src[1] + i + 2 * VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + 2 * VTraits<v_float32>::vlanes())), k1, d4);
                v_float32 s3 = v_muladd(v_sub(vx_load(src[1] + i + 3 * VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + 3 * VTraits<v_float32>::vlanes())), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(v_sub(vx_load(src[k] + i), vx_load(src[-k] + i)), k2, s0);
                    s1 = v_muladd(v_sub(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + VTraits<v_float32>::vlanes())), k2, s1);
                    s2 = v_muladd(v_sub(vx_load(src[k] + i + 2 * VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + 2 * VTraits<v_float32>::vlanes())), k2, s2);
                    s3 = v_muladd(v_sub(vx_load(src[k] + i + 3 * VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + 3 * VTraits<v_float32>::vlanes())), k2, s3);
                }
                v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                v_store(dst + i + VTraits<v_int16>::vlanes(), v_pack(v_round(s2), v_round(s3)));
            }
            if( i <= width - VTraits<v_int16>::vlanes() )
            {
                v_float32 s0 = v_muladd(v_sub(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, d4);
                v_float32 s1 = v_muladd(v_sub(vx_load(src[1] + i + VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + VTraits<v_float32>::vlanes())), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(v_sub(vx_load(src[k] + i), vx_load(src[-k] + i)), k2, s0);
                    s1 = v_muladd(v_sub(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + VTraits<v_float32>::vlanes())), k2, s1);
                }
                v_store(dst + i, v_pack(v_round(s0), v_round(s1)));
                i += VTraits<v_int16>::vlanes();
            }
            if( i <= width - VTraits<v_float32>::vlanes() )
            {
                v_float32 s0 = v_muladd(v_sub(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                    s0 = v_muladd(v_sub(vx_load(src[k] + i), vx_load(src[-k] + i)), vx_setall_f32(ky[k]), s0);
                v_pack_store(dst + i, v_round(s0));
                i += VTraits<v_float32>::vlanes();
            }
        }

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
#if defined USE_IPP_SEP_FILTERS
        bufsz = -1;
#endif
    }

    RowVec_32f( const Mat& _kernel )
    {
        kernel = _kernel;
#if defined USE_IPP_SEP_FILTERS
        bufsz = -1;
#endif
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        CV_INSTRUMENT_REGION();

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

#if CV_AVX
        for (; i <= width - 8; i += 8)
        {
            const float* src = src0 + i;
            __m256 f, x0;
            __m256 s0 = _mm256_set1_ps(0.0f);
            for (k = 0; k < _ksize; k++, src += cn)
            {
                f = _mm256_set1_ps(_kx[k]);
                x0 = _mm256_loadu_ps(src);
#if CV_FMA3
                s0 = _mm256_fmadd_ps(x0, f, s0);
#else
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
#endif
            }
            _mm256_storeu_ps(dst + i, s0);
        }
#endif
        v_float32 k0 = vx_setall_f32(_kx[0]);
        for( ; i <= width - 4*VTraits<v_float32>::vlanes(); i += 4*VTraits<v_float32>::vlanes() )
        {
            const float* src = src0 + i;
            v_float32 s0 = v_mul(vx_load(src), k0);
            v_float32 s1 = v_mul(vx_load(src + VTraits<v_float32>::vlanes()), k0);
            v_float32 s2 = v_mul(vx_load(src + 2 * VTraits<v_float32>::vlanes()), k0);
            v_float32 s3 = v_mul(vx_load(src + 3 * VTraits<v_float32>::vlanes()), k0);
            src += cn;
            for( k = 1; k < _ksize; k++, src += cn )
            {
                v_float32 k1 = vx_setall_f32(_kx[k]);
                s0 = v_muladd(vx_load(src), k1, s0);
                s1 = v_muladd(vx_load(src + VTraits<v_float32>::vlanes()), k1, s1);
                s2 = v_muladd(vx_load(src + 2*VTraits<v_float32>::vlanes()), k1, s2);
                s3 = v_muladd(vx_load(src + 3*VTraits<v_float32>::vlanes()), k1, s3);
            }
            v_store(dst + i, s0);
            v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
            v_store(dst + i + 2*VTraits<v_float32>::vlanes(), s2);
            v_store(dst + i + 3*VTraits<v_float32>::vlanes(), s3);
        }
        if( i <= width - 2*VTraits<v_float32>::vlanes() )
        {
            const float* src = src0 + i;
            v_float32 s0 = v_mul(vx_load(src), k0);
            v_float32 s1 = v_mul(vx_load(src + VTraits<v_float32>::vlanes()), k0);
            src += cn;
            for( k = 1; k < _ksize; k++, src += cn )
            {
                v_float32 k1 = vx_setall_f32(_kx[k]);
                s0 = v_muladd(vx_load(src), k1, s0);
                s1 = v_muladd(vx_load(src + VTraits<v_float32>::vlanes()), k1, s1);
            }
            v_store(dst + i, s0);
            v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
            i += 2*VTraits<v_float32>::vlanes();
        }
        if( i <= width - VTraits<v_float32>::vlanes() )
        {
            const float* src = src0 + i;
            v_float32 s0 = v_mul(vx_load(src), k0);
            src += cn;
            for( k = 1; k < _ksize; k++, src += cn )
                s0 = v_muladd(vx_load(src), vx_setall_f32(_kx[k]), s0);
            v_store(dst + i, s0);
            i += VTraits<v_float32>::vlanes();
        }
        return i;
    }

    Mat kernel;
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
        CV_INSTRUMENT_REGION();

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
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                        v_store(dst + i, v_muladd(vx_load(src), k0, v_add(vx_load(src - cn), vx_load(src + cn))));
#else
                    if( kx[0] > 0 )
                        for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                        {
                            v_float32 x = vx_load(src);
                            v_store(dst + i, v_add(vx_load(src - cn), vx_load(src + cn), x , x));
                        }
                    else
                        for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                        {
                            v_float32 x = vx_load(src);
                            v_store(dst + i, v_sub(v_add(vx_load(src - cn), vx_load(src + cn)), v_add(x, x)));
                        }
#endif
                }
                else
                {
                    v_float32 k0 = vx_setall_f32(kx[0]), k1 = vx_setall_f32(kx[1]);
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                        v_store(dst + i, v_muladd(vx_load(src), k0, v_mul(v_add(vx_load(src - cn), vx_load(src + cn)), k1)));
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                {
#if CV_FMA3 || CV_AVX2
                    v_float32 k0 = vx_setall_f32(-2);
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                        v_store(dst + i, v_muladd(vx_load(src), k0, v_add(vx_load(src - 2 * cn), vx_load(src + 2 * cn))));
#else
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                    {
                        v_float32 x = vx_load(src);
                        v_store(dst + i, v_sub(v_add(vx_load(src - 2*cn), vx_load(src + 2*cn)), v_add(x, x)));
                    }
#endif
                }
                else
                {
                    v_float32 k0 = vx_setall_f32(kx[0]), k1 = vx_setall_f32(kx[1]), k2 = vx_setall_f32(kx[2]);
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                        v_store(dst + i, v_muladd(v_add(vx_load(src + 2 * cn), vx_load(src - 2 * cn)), k2, v_muladd(vx_load(src), k0, v_mul(v_add(vx_load(src - cn), vx_load(src + cn)), k1))));
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                        v_store(dst + i, v_sub(vx_load(src + cn), vx_load(src - cn)));
                else
                {
                    v_float32 k1 = vx_setall_f32(kx[1]);
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                        v_store(dst + i, v_mul(v_sub(vx_load(src + cn), vx_load(src - cn)), k1));
                }
            }
            else if( _ksize == 5 )
            {
                v_float32 k1 = vx_setall_f32(kx[1]), k2 = vx_setall_f32(kx[2]);
                for ( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes(), src += VTraits<v_float32>::vlanes() )
                    v_store(dst + i, v_muladd(v_sub(vx_load(src + 2 * cn), vx_load(src - 2 * cn)), k2, v_mul(v_sub(vx_load(src + cn), vx_load(src - cn)), k1)));
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
        delta = 0;
    }
    SymmColumnVec_32f(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        CV_INSTRUMENT_REGION();

        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        float* dst = (float*)_dst;

        if( symmetrical )
        {

#if CV_AVX
            {
                const float *S, *S2;
                const __m256 d8 = _mm256_set1_ps(delta);

                for( ; i <= width - 16; i += 16 )
                {
                    __m256 f = _mm256_set1_ps(ky[0]);
                    __m256 s0, s1;
                    __m256 x0;
                    S = src[0] + i;
                    s0 = _mm256_loadu_ps(S);
#if CV_FMA3
                    s0 = _mm256_fmadd_ps(s0, f, d8);
#else
                    s0 = _mm256_add_ps(_mm256_mul_ps(s0, f), d8);
#endif
                    s1 = _mm256_loadu_ps(S+8);
#if CV_FMA3
                    s1 = _mm256_fmadd_ps(s1, f, d8);
#else
                    s1 = _mm256_add_ps(_mm256_mul_ps(s1, f), d8);
#endif

                    for( k = 1; k <= ksize2; k++ )
                    {
                        S = src[k] + i;
                        S2 = src[-k] + i;
                        f = _mm256_set1_ps(ky[k]);
                        x0 = _mm256_add_ps(_mm256_loadu_ps(S), _mm256_loadu_ps(S2));
#if CV_FMA3
                        s0 = _mm256_fmadd_ps(x0, f, s0);
#else
                        s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
#endif
                        x0 = _mm256_add_ps(_mm256_loadu_ps(S+8), _mm256_loadu_ps(S2+8));
#if CV_FMA3
                        s1 = _mm256_fmadd_ps(x0, f, s1);
#else
                        s1 = _mm256_add_ps(s1, _mm256_mul_ps(x0, f));
#endif
                    }

                    _mm256_storeu_ps(dst + i, s0);
                    _mm256_storeu_ps(dst + i + 8, s1);
                }
            }
#endif
            const v_float32 d4 = vx_setall_f32(delta);
            const v_float32 k0 = vx_setall_f32(ky[0]);
            for( ; i <= width - 4*VTraits<v_float32>::vlanes(); i += 4*VTraits<v_float32>::vlanes() )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                v_float32 s1 = v_muladd(vx_load(src[0] + i + VTraits<v_float32>::vlanes()), k0, d4);
                v_float32 s2 = v_muladd(vx_load(src[0] + i + 2*VTraits<v_float32>::vlanes()), k0, d4);
                v_float32 s3 = v_muladd(vx_load(src[0] + i + 3*VTraits<v_float32>::vlanes()), k0, d4);
                for( k = 1; k <= ksize2; k++ )
                {
                    v_float32 k1 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(v_add(vx_load(src[k] + i), vx_load(src[-k] + i)), k1, s0);
                    s1 = v_muladd(v_add(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + VTraits<v_float32>::vlanes())), k1, s1);
                    s2 = v_muladd(v_add(vx_load(src[k] + i + 2 * VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + 2 * VTraits<v_float32>::vlanes())), k1, s2);
                    s3 = v_muladd(v_add(vx_load(src[k] + i + 3 * VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + 3 * VTraits<v_float32>::vlanes())), k1, s3);
                }
                v_store(dst + i, s0);
                v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
                v_store(dst + i + 2*VTraits<v_float32>::vlanes(), s2);
                v_store(dst + i + 3*VTraits<v_float32>::vlanes(), s3);
            }
            if( i <= width - 2*VTraits<v_float32>::vlanes() )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                v_float32 s1 = v_muladd(vx_load(src[0] + i + VTraits<v_float32>::vlanes()), k0, d4);
                for( k = 1; k <= ksize2; k++ )
                {
                    v_float32 k1 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(v_add(vx_load(src[k] + i), vx_load(src[-k] + i)), k1, s0);
                    s1 = v_muladd(v_add(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + VTraits<v_float32>::vlanes())), k1, s1);
                }
                v_store(dst + i, s0);
                v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
                i += 2*VTraits<v_float32>::vlanes();
            }
            if( i <= width - VTraits<v_float32>::vlanes() )
            {
                v_float32 s0 = v_muladd(vx_load(src[0] + i), k0, d4);
                for( k = 1; k <= ksize2; k++ )
                    s0 = v_muladd(v_add(vx_load(src[k] + i), vx_load(src[-k] + i)), vx_setall_f32(ky[k]), s0);
                v_store(dst + i, s0);
                i += VTraits<v_float32>::vlanes();
            }
        }
        else
        {
            CV_DbgAssert(ksize2 > 0);
#if CV_AVX
            {
                const float *S2;
                const __m256 d8 = _mm256_set1_ps(delta);

                for (; i <= width - 16; i += 16)
                {
                    __m256 f, s0 = d8, s1 = d8;
                    __m256 x0;

                    for (k = 1; k <= ksize2; k++)
                    {
                        const float *S = src[k] + i;
                        S2 = src[-k] + i;
                        f = _mm256_set1_ps(ky[k]);
                        x0 = _mm256_sub_ps(_mm256_loadu_ps(S), _mm256_loadu_ps(S2));
#if CV_FMA3
                        s0 = _mm256_fmadd_ps(x0, f, s0);
#else
                        s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
#endif
                        x0 = _mm256_sub_ps(_mm256_loadu_ps(S + 8), _mm256_loadu_ps(S2 + 8));
#if CV_FMA3
                        s1 = _mm256_fmadd_ps(x0, f, s1);
#else
                        s1 = _mm256_add_ps(s1, _mm256_mul_ps(x0, f));
#endif
                    }

                    _mm256_storeu_ps(dst + i, s0);
                    _mm256_storeu_ps(dst + i + 8, s1);
                }
            }
#endif
            const v_float32 d4 = vx_setall_f32(delta);
            const v_float32 k1 = vx_setall_f32(ky[1]);
            for( ; i <= width - 4*VTraits<v_float32>::vlanes(); i += 4*VTraits<v_float32>::vlanes() )
            {
                v_float32 s0 = v_muladd(v_sub(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, d4);
                v_float32 s1 = v_muladd(v_sub(vx_load(src[1] + i + VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + VTraits<v_float32>::vlanes())), k1, d4);
                v_float32 s2 = v_muladd(v_sub(vx_load(src[1] + i + 2 * VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + 2 * VTraits<v_float32>::vlanes())), k1, d4);
                v_float32 s3 = v_muladd(v_sub(vx_load(src[1] + i + 3 * VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + 3 * VTraits<v_float32>::vlanes())), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(v_sub(vx_load(src[k] + i), vx_load(src[-k] + i)), k2, s0);
                    s1 = v_muladd(v_sub(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + VTraits<v_float32>::vlanes())), k2, s1);
                    s2 = v_muladd(v_sub(vx_load(src[k] + i + 2 * VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + 2 * VTraits<v_float32>::vlanes())), k2, s2);
                    s3 = v_muladd(v_sub(vx_load(src[k] + i + 3 * VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + 3 * VTraits<v_float32>::vlanes())), k2, s3);
                }
                v_store(dst + i, s0);
                v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
                v_store(dst + i + 2*VTraits<v_float32>::vlanes(), s2);
                v_store(dst + i + 3*VTraits<v_float32>::vlanes(), s3);
            }
            if( i <= width - 2*VTraits<v_float32>::vlanes() )
            {
                v_float32 s0 = v_muladd(v_sub(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, d4);
                v_float32 s1 = v_muladd(v_sub(vx_load(src[1] + i + VTraits<v_float32>::vlanes()), vx_load(src[-1] + i + VTraits<v_float32>::vlanes())), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                {
                    v_float32 k2 = vx_setall_f32(ky[k]);
                    s0 = v_muladd(v_sub(vx_load(src[k] + i), vx_load(src[-k] + i)), k2, s0);
                    s1 = v_muladd(v_sub(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), vx_load(src[-k] + i + VTraits<v_float32>::vlanes())), k2, s1);
                }
                v_store(dst + i, s0);
                v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
                i += 2*VTraits<v_float32>::vlanes();
            }
            if( i <= width - VTraits<v_float32>::vlanes() )
            {
                v_float32 s0 = v_muladd(v_sub(vx_load(src[1] + i), vx_load(src[-1] + i)), k1, d4);
                for( k = 2; k <= ksize2; k++ )
                    s0 = v_muladd(v_sub(vx_load(src[k] + i), vx_load(src[-k] + i)), vx_setall_f32(ky[k]), s0);
                v_store(dst + i, s0);
                i += VTraits<v_float32>::vlanes();
            }
        }
        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
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
        CV_INSTRUMENT_REGION();

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
                for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes() )
                    v_store(dst + i, v_muladd(vx_load(S1 + i), k0, v_add(v_add(vx_load(S0 + i), vx_load(S2 + i)), d4)));
#else
                if(ky[0] > 0)
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes() )
                    {
                        v_float32 x = vx_load(S1 + i);
                        v_store(dst + i, v_add(vx_load(S0 + i), vx_load(S2 + i), d4, x, x));
                    }
                else
                    for( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes() )
                    {
                        v_float32 x = vx_load(S1 + i);
                        v_store(dst + i, v_sub(v_add(vx_load(S0 + i), vx_load(S2 + i), d4), v_add(x, x)));
                    }
#endif
            }
            else
            {
                v_float32 k0 = vx_setall_f32(ky[0]), k1 = vx_setall_f32(ky[1]);
                for ( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes() )
                    v_store(dst + i, v_muladd(v_add(vx_load(S0 + i), vx_load(S2 + i)), k1, v_muladd(vx_load(S1 + i), k0, d4)));
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for ( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes() )
                    v_store(dst + i, v_add(v_sub(vx_load(S2 + i), vx_load(S0 + i)), d4));
            }
            else
            {
                v_float32 k1 = vx_setall_f32(ky[1]);
                for ( ; i <= width - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes() )
                    v_store(dst + i, v_muladd(v_sub(vx_load(S2 + i), vx_load(S0 + i)), k1, d4));
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
        CV_INSTRUMENT_REGION();

        CV_DbgAssert(_nz > 0);
        const float* kf = (const float*)&coeffs[0];
        int i = 0, k, nz = _nz;

        v_float32 d4 = vx_setall_f32(delta);
        v_float32 f0 = vx_setall_f32(kf[0]);
        for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes() )
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
        if( i <= width - VTraits<v_uint16>::vlanes() )
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
            i += VTraits<v_uint16>::vlanes();
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
        CV_INSTRUMENT_REGION();

        CV_DbgAssert(_nz > 0);
        const float* kf = (const float*)&coeffs[0];
        short* dst = (short*)_dst;
        int i = 0, k, nz = _nz;

        v_float32 d4 = vx_setall_f32(delta);
        v_float32 f0 = vx_setall_f32(kf[0]);
        for( ; i <= width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes() )
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
            v_store(dst + i + VTraits<v_int16>::vlanes(), v_pack(v_round(s2), v_round(s3)));
        }
        if( i <= width - VTraits<v_uint16>::vlanes() )
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
            i += VTraits<v_uint16>::vlanes();
        }
        if( i <= width - VTraits<v_int32>::vlanes() )
        {
            v_float32 s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(src[0] + i))), f0, d4);
            for( k = 1; k < nz; k++ )
                s0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(src[k] + i))), vx_setall_f32(kf[k]), s0);
            v_pack_store(dst + i, v_round(s0));
            i += VTraits<v_int32>::vlanes();
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
        CV_INSTRUMENT_REGION();

        const float* kf = (const float*)&coeffs[0];
        const float** src = (const float**)_src;
        float* dst = (float*)_dst;
        int i = 0, k, nz = _nz;

        v_float32 d4 = vx_setall_f32(delta);
        v_float32 f0 = vx_setall_f32(kf[0]);
        for( ; i <= width - 4*VTraits<v_float32>::vlanes(); i += 4*VTraits<v_float32>::vlanes() )
        {
            v_float32 s0 = v_muladd(vx_load(src[0] + i), f0, d4);
            v_float32 s1 = v_muladd(vx_load(src[0] + i + VTraits<v_float32>::vlanes()), f0, d4);
            v_float32 s2 = v_muladd(vx_load(src[0] + i + 2*VTraits<v_float32>::vlanes()), f0, d4);
            v_float32 s3 = v_muladd(vx_load(src[0] + i + 3*VTraits<v_float32>::vlanes()), f0, d4);
            for( k = 1; k < nz; k++ )
            {
                v_float32 f1 = vx_setall_f32(kf[k]);
                s0 = v_muladd(vx_load(src[k] + i), f1, s0);
                s1 = v_muladd(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), f1, s1);
                s2 = v_muladd(vx_load(src[k] + i + 2*VTraits<v_float32>::vlanes()), f1, s2);
                s3 = v_muladd(vx_load(src[k] + i + 3*VTraits<v_float32>::vlanes()), f1, s3);
            }
            v_store(dst + i, s0);
            v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
            v_store(dst + i + 2*VTraits<v_float32>::vlanes(), s2);
            v_store(dst + i + 3*VTraits<v_float32>::vlanes(), s3);
        }
        if( i <= width - 2*VTraits<v_float32>::vlanes() )
        {
            v_float32 s0 = v_muladd(vx_load(src[0] + i), f0, d4);
            v_float32 s1 = v_muladd(vx_load(src[0] + i + VTraits<v_float32>::vlanes()), f0, d4);
            for( k = 1; k < nz; k++ )
            {
                v_float32 f1 = vx_setall_f32(kf[k]);
                s0 = v_muladd(vx_load(src[k] + i), f1, s0);
                s1 = v_muladd(vx_load(src[k] + i + VTraits<v_float32>::vlanes()), f1, s1);
            }
            v_store(dst + i, s0);
            v_store(dst + i + VTraits<v_float32>::vlanes(), s1);
            i += 2*VTraits<v_float32>::vlanes();
        }
        if( i <= width - VTraits<v_float32>::vlanes() )
        {
            v_float32 s0 = v_muladd(vx_load(src[0] + i), f0, d4);
            for( k = 1; k < nz; k++ )
                s0 = v_muladd(vx_load(src[k] + i), vx_setall_f32(kf[k]), s0);
            v_store(dst + i, s0);
            i += VTraits<v_float32>::vlanes();
        }
        return i;
    }

    int _nz;
    std::vector<uchar> coeffs;
    float delta;
};

#else

typedef RowNoVec RowVec_8u32s;
typedef RowNoVec RowVec_8u32f;
typedef RowNoVec RowVec_16s32f;
typedef RowNoVec RowVec_32f;
typedef SymmRowSmallNoVec SymmRowSmallVec_8u32s;
typedef SymmRowSmallNoVec SymmRowSmallVec_32f;
typedef ColumnNoVec SymmColumnVec_32s8u;
typedef ColumnNoVec SymmColumnVec_32f8u;
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
        CV_INSTRUMENT_REGION();

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
        CV_INSTRUMENT_REGION();

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
        CV_INSTRUMENT_REGION();

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
        CV_INSTRUMENT_REGION();

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
        CV_INSTRUMENT_REGION();

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


Ptr<BaseRowFilter> getLinearRowFilter(
        int srcType, int bufType,
        const Mat& kernel, int anchor,
        int symmetryType)
{
    CV_INSTRUMENT_REGION();

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
        return makePtr<RowFilter<uchar, float, RowVec_8u32f> >
            (kernel, anchor, RowVec_8u32f(kernel));
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

    CV_Error_( cv::Error::StsNotImplemented,
        ("Unsupported combination of source format (=%d), and buffer format (=%d)",
        srcType, bufType));
}


Ptr<BaseColumnFilter> getLinearColumnFilter(
        int bufType, int dstType,
        const Mat& kernel, int anchor,
        int symmetryType, double delta,
        int bits)
{
    CV_INSTRUMENT_REGION();

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
            return makePtr<SymmColumnFilter<Cast<float, uchar>, SymmColumnVec_32f8u> >
                (kernel, anchor, delta, symmetryType, Cast<float, uchar>(),
                SymmColumnVec_32f8u(kernel, symmetryType, 0, delta));
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

    CV_Error_( cv::Error::StsNotImplemented,
        ("Unsupported combination of buffer format (=%d), and destination format (=%d)",
        bufType, dstType));
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


Ptr<BaseFilter> getLinearFilter(
        int srcType, int dstType,
        const Mat& _kernel, Point anchor,
        double delta, int bits)
{
    CV_INSTRUMENT_REGION();

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

    CV_Error_( cv::Error::StsNotImplemented,
        ("Unsupported combination of source format (=%d), and destination format (=%d)",
        srcType, dstType));
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
