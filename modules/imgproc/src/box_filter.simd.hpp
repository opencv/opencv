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
// Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
Ptr<BaseRowFilter> getRowSumFilter(int srcType, int sumType, int ksize, int anchor);
Ptr<BaseColumnFilter> getColumnSumFilter(int sumType, int dstType, int ksize, int anchor, double scale);
Ptr<FilterEngine> createBoxFilter(int srcType, int dstType, Size ksize,
                                  Point anchor, bool normalize, int borderType);

Ptr<BaseRowFilter> getSqrRowSumFilter(int srcType, int sumType, int ksize, int anchor);


#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
/****************************************************************************************\
                                         Box Filter
\****************************************************************************************/

namespace {
template<typename T, typename ST>
struct RowSum :
        public BaseRowFilter
{
    RowSum( int _ksize, int _anchor ) :
        BaseRowFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
    }

    virtual void operator()(const uchar* src, uchar* dst, int width, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        const T* S = (const T*)src;
        ST* D = (ST*)dst;
        int i = 0, k, ksz_cn = ksize*cn;

        width = (width - 1)*cn;
        if( ksize == 3 )
        {
            for( i = 0; i < width + cn; i++ )
            {
                D[i] = (ST)S[i] + (ST)S[i+cn] + (ST)S[i+cn*2];
            }
        }
        else if( ksize == 5 )
        {
            for( i = 0; i < width + cn; i++ )
            {
                D[i] = (ST)S[i] + (ST)S[i+cn] + (ST)S[i+cn*2] + (ST)S[i + cn*3] + (ST)S[i + cn*4];
            }
        }
        else if( cn == 1 )
        {
            ST s = 0;
            for( i = 0; i < ksz_cn; i++ )
                s += (ST)S[i];
            D[0] = s;
            for( i = 0; i < width; i++ )
            {
                s += (ST)S[i + ksz_cn] - (ST)S[i];
                D[i+1] = s;
            }
        }
        else if( cn == 3 )
        {
            ST s0 = 0, s1 = 0, s2 = 0;
            for( i = 0; i < ksz_cn; i += 3 )
            {
                s0 += (ST)S[i];
                s1 += (ST)S[i+1];
                s2 += (ST)S[i+2];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            for( i = 0; i < width; i += 3 )
            {
                s0 += (ST)S[i + ksz_cn] - (ST)S[i];
                s1 += (ST)S[i + ksz_cn + 1] - (ST)S[i + 1];
                s2 += (ST)S[i + ksz_cn + 2] - (ST)S[i + 2];
                D[i+3] = s0;
                D[i+4] = s1;
                D[i+5] = s2;
            }
        }
        else if( cn == 4 )
        {
            ST s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for( i = 0; i < ksz_cn; i += 4 )
            {
                s0 += (ST)S[i];
                s1 += (ST)S[i+1];
                s2 += (ST)S[i+2];
                s3 += (ST)S[i+3];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            D[3] = s3;
            for( i = 0; i < width; i += 4 )
            {
                s0 += (ST)S[i + ksz_cn] - (ST)S[i];
                s1 += (ST)S[i + ksz_cn + 1] - (ST)S[i + 1];
                s2 += (ST)S[i + ksz_cn + 2] - (ST)S[i + 2];
                s3 += (ST)S[i + ksz_cn + 3] - (ST)S[i + 3];
                D[i+4] = s0;
                D[i+5] = s1;
                D[i+6] = s2;
                D[i+7] = s3;
            }
        }
        else
            for( k = 0; k < cn; k++, S++, D++ )
            {
                ST s = 0;
                for( i = 0; i < ksz_cn; i += cn )
                    s += (ST)S[i];
                D[0] = s;
                for( i = 0; i < width; i += cn )
                {
                    s += (ST)S[i + ksz_cn] - (ST)S[i];
                    D[i+cn] = s;
                }
            }
    }
};


template<typename ST, typename T>
struct ColumnSum :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i;
        ST* SUM;
        bool haveScale = scale != 1;
        double _scale = scale;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(ST));

            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const ST* Sp = (const ST*)src[0];

                for( i = 0; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const ST* Sp = (const ST*)src[0];
            const ST* Sm = (const ST*)src[1-ksize];
            T* D = (T*)dst;
            if( haveScale )
            {
                for( i = 0; i <= width - 2; i += 2 )
                {
                    ST s0 = SUM[i] + Sp[i], s1 = SUM[i+1] + Sp[i+1];
                    D[i] = saturate_cast<T>(s0*_scale);
                    D[i+1] = saturate_cast<T>(s1*_scale);
                    s0 -= Sm[i]; s1 -= Sm[i+1];
                    SUM[i] = s0; SUM[i+1] = s1;
                }

                for( ; i < width; i++ )
                {
                    ST s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<T>(s0*_scale);
                    SUM[i] = s0 - Sm[i];
                }
            }
            else
            {
                for( i = 0; i <= width - 2; i += 2 )
                {
                    ST s0 = SUM[i] + Sp[i], s1 = SUM[i+1] + Sp[i+1];
                    D[i] = saturate_cast<T>(s0);
                    D[i+1] = saturate_cast<T>(s1);
                    s0 -= Sm[i]; s1 -= Sm[i+1];
                    SUM[i] = s0; SUM[i+1] = s1;
                }

                for( ; i < width; i++ )
                {
                    ST s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<T>(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }

    double scale;
    int sumCount;
    std::vector<ST> sum;
};


template<>
struct ColumnSum<int, uchar> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int* SUM;
        bool haveScale = scale != 1;
        double _scale = scale;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(int));
            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const int* Sp = (const int*)src[0];
                int i = 0;
#if CV_SIMD
                for (; i <= width - v_int32::nlanes; i += v_int32::nlanes)
                {
                    v_store(SUM + i, vx_load(SUM + i) + vx_load(Sp + i));
                }
#if CV_SIMD_WIDTH > 16
                for (; i <= width - v_int32x4::nlanes; i += v_int32x4::nlanes)
                {
                    v_store(SUM + i, v_load(SUM + i) + v_load(Sp + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const int* Sp = (const int*)src[0];
            const int* Sm = (const int*)src[1-ksize];
            uchar* D = (uchar*)dst;
            if( haveScale )
            {
                int i = 0;
#if CV_SIMD
                v_float32 _v_scale = vx_setall_f32((float)_scale);
                for( ; i <= width - v_uint16::nlanes; i += v_uint16::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_int32 v_s01 = vx_load(SUM + i + v_int32::nlanes) + vx_load(Sp + i + v_int32::nlanes);

                    v_uint32 v_s0d = v_reinterpret_as_u32(v_round(v_cvt_f32(v_s0) * _v_scale));
                    v_uint32 v_s01d = v_reinterpret_as_u32(v_round(v_cvt_f32(v_s01) * _v_scale));

                    v_uint16 v_dst = v_pack(v_s0d, v_s01d);
                    v_pack_store(D + i, v_dst);

                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                    v_store(SUM + i + v_int32::nlanes, v_s01 - vx_load(Sm + i + v_int32::nlanes));
                }
#if CV_SIMD_WIDTH > 16
                v_float32x4 v_scale = v_setall_f32((float)_scale);
                for( ; i <= width-v_uint16x8::nlanes; i+=v_uint16x8::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_int32x4 v_s01 = v_load(SUM + i + v_int32x4::nlanes) + v_load(Sp + i + v_int32x4::nlanes);

                    v_uint32x4 v_s0d = v_reinterpret_as_u32(v_round(v_cvt_f32(v_s0) * v_scale));
                    v_uint32x4 v_s01d = v_reinterpret_as_u32(v_round(v_cvt_f32(v_s01) * v_scale));

                    v_uint16x8 v_dst = v_pack(v_s0d, v_s01d);
                    v_pack_store(D + i, v_dst);

                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                    v_store(SUM + i + v_int32x4::nlanes, v_s01 - v_load(Sm + i + v_int32x4::nlanes));
            }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<uchar>(s0*_scale);
                    SUM[i] = s0 - Sm[i];
                }
            }
            else
            {
                int i = 0;
#if CV_SIMD
                for( ; i <= width-v_uint16::nlanes; i+=v_uint16::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_int32 v_s01 = vx_load(SUM + i + v_int32::nlanes) + vx_load(Sp + i + v_int32::nlanes);

                    v_uint16 v_dst = v_pack(v_reinterpret_as_u32(v_s0), v_reinterpret_as_u32(v_s01));
                    v_pack_store(D + i, v_dst);

                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                    v_store(SUM + i + v_int32::nlanes, v_s01 - vx_load(Sm + i + v_int32::nlanes));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width-v_uint16x8::nlanes; i+=v_uint16x8::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_int32x4 v_s01 = v_load(SUM + i + v_int32x4::nlanes) + v_load(Sp + i + v_int32x4::nlanes);

                    v_uint16x8 v_dst = v_pack(v_reinterpret_as_u32(v_s0), v_reinterpret_as_u32(v_s01));
                    v_pack_store(D + i, v_dst);

                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                    v_store(SUM + i + v_int32x4::nlanes, v_s01 - v_load(Sm + i + v_int32x4::nlanes));
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<uchar>(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }

    double scale;
    int sumCount;
    std::vector<int> sum;
};


template<>
struct ColumnSum<ushort, uchar> :
public BaseColumnFilter
{
    enum { SHIFT = 23 };

    ColumnSum( int _ksize, int _anchor, double _scale ) :
    BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
        divDelta = 0;
        divScale = 1;
        if( scale != 1 )
        {
            int d = cvRound(1./scale);
            double scalef = ((double)(1 << SHIFT))/d;
            divScale = cvFloor(scalef);
            scalef -= divScale;
            divDelta = d/2;
            if( scalef < 0.5 )
                divDelta++;
            else
                divScale++;
        }
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        const int ds = divScale;
        const int dd = divDelta;
        ushort* SUM;
        const bool haveScale = scale != 1;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(SUM[0]));
            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const ushort* Sp = (const ushort*)src[0];
                int i = 0;
#if CV_SIMD
                for( ; i <= width - v_uint16::nlanes; i += v_uint16::nlanes )
                {
                    v_store(SUM + i, vx_load(SUM + i) + vx_load(Sp + i));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width - v_uint16x8::nlanes; i += v_uint16x8::nlanes )
                {
                    v_store(SUM + i, v_load(SUM + i) + v_load(Sp + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const ushort* Sp = (const ushort*)src[0];
            const ushort* Sm = (const ushort*)src[1-ksize];
            uchar* D = (uchar*)dst;
            if( haveScale )
            {
                int i = 0;
#if CV_SIMD
                v_uint32 _ds4 = vx_setall_u32((unsigned)ds);
                v_uint16 _dd8 = vx_setall_u16((ushort)dd);

                for( ; i <= width-v_uint8::nlanes; i+=v_uint8::nlanes )
                {
                    v_uint16 _sm0 = vx_load(Sm + i);
                    v_uint16 _sm1 = vx_load(Sm + i + v_uint16::nlanes);

                    v_uint16 _s0 = v_add_wrap(vx_load(SUM + i), vx_load(Sp + i));
                    v_uint16 _s1 = v_add_wrap(vx_load(SUM + i + v_uint16::nlanes), vx_load(Sp + i + v_uint16::nlanes));

                    v_uint32 _s00, _s01, _s10, _s11;

                    v_expand(_s0 + _dd8, _s00, _s01);
                    v_expand(_s1 + _dd8, _s10, _s11);

                    _s00 = v_shr<SHIFT>(_s00*_ds4);
                    _s01 = v_shr<SHIFT>(_s01*_ds4);
                    _s10 = v_shr<SHIFT>(_s10*_ds4);
                    _s11 = v_shr<SHIFT>(_s11*_ds4);

                    v_int16 r0 = v_pack(v_reinterpret_as_s32(_s00), v_reinterpret_as_s32(_s01));
                    v_int16 r1 = v_pack(v_reinterpret_as_s32(_s10), v_reinterpret_as_s32(_s11));

                    _s0 = v_sub_wrap(_s0, _sm0);
                    _s1 = v_sub_wrap(_s1, _sm1);

                    v_store(D + i, v_pack_u(r0, r1));
                    v_store(SUM + i, _s0);
                    v_store(SUM + i + v_uint16::nlanes, _s1);
                }
#if CV_SIMD_WIDTH > 16
                v_uint32x4 ds4 = v_setall_u32((unsigned)ds);
                v_uint16x8 dd8 = v_setall_u16((ushort)dd);

                for( ; i <= width-v_uint8x16::nlanes; i+=v_uint8x16::nlanes )
                {
                    v_uint16x8 _sm0 = v_load(Sm + i);
                    v_uint16x8 _sm1 = v_load(Sm + i + v_uint16x8::nlanes);

                    v_uint16x8 _s0 = v_add_wrap(v_load(SUM + i), v_load(Sp + i));
                    v_uint16x8 _s1 = v_add_wrap(v_load(SUM + i + v_uint16x8::nlanes), v_load(Sp + i + v_uint16x8::nlanes));

                    v_uint32x4 _s00, _s01, _s10, _s11;

                    v_expand(_s0 + dd8, _s00, _s01);
                    v_expand(_s1 + dd8, _s10, _s11);

                    _s00 = v_shr<SHIFT>(_s00*ds4);
                    _s01 = v_shr<SHIFT>(_s01*ds4);
                    _s10 = v_shr<SHIFT>(_s10*ds4);
                    _s11 = v_shr<SHIFT>(_s11*ds4);

                    v_int16x8 r0 = v_pack(v_reinterpret_as_s32(_s00), v_reinterpret_as_s32(_s01));
                    v_int16x8 r1 = v_pack(v_reinterpret_as_s32(_s10), v_reinterpret_as_s32(_s11));

                    _s0 = v_sub_wrap(_s0, _sm0);
                    _s1 = v_sub_wrap(_s1, _sm1);

                    v_store(D + i, v_pack_u(r0, r1));
                    v_store(SUM + i, _s0);
                    v_store(SUM + i + v_uint16x8::nlanes, _s1);
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = (uchar)((s0 + dd)*ds >> SHIFT);
                    SUM[i] = (ushort)(s0 - Sm[i]);
                }
            }
            else
            {
                int i = 0;
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<uchar>(s0);
                    SUM[i] = (ushort)(s0 - Sm[i]);
                }
            }
            dst += dststep;
        }
    }

    double scale;
    int sumCount;
    int divDelta;
    int divScale;
    std::vector<ushort> sum;
};


template<>
struct ColumnSum<int, short> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i;
        int* SUM;
        bool haveScale = scale != 1;
        double _scale = scale;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(int));
            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const int* Sp = (const int*)src[0];
                i = 0;
#if CV_SIMD
                for( ; i <= width - v_int32::nlanes; i+=v_int32::nlanes )
                {
                    v_store(SUM + i, vx_load(SUM + i) + vx_load(Sp + i));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width - v_int32x4::nlanes; i+=v_int32x4::nlanes )
                {
                    v_store(SUM + i, v_load(SUM + i) + v_load(Sp + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const int* Sp = (const int*)src[0];
            const int* Sm = (const int*)src[1-ksize];
            short* D = (short*)dst;
            if( haveScale )
            {
                i = 0;
#if CV_SIMD
                v_float32 _v_scale = vx_setall_f32((float)_scale);
                for( ; i <= width-v_int16::nlanes; i+=v_int16::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_int32 v_s01 = vx_load(SUM + i + v_int32::nlanes) + vx_load(Sp + i + v_int32::nlanes);

                    v_int32 v_s0d =  v_round(v_cvt_f32(v_s0) * _v_scale);
                    v_int32 v_s01d = v_round(v_cvt_f32(v_s01) * _v_scale);
                    v_store(D + i, v_pack(v_s0d, v_s01d));

                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                    v_store(SUM + i + v_int32::nlanes, v_s01 - vx_load(Sm + i + v_int32::nlanes));
                }
#if CV_SIMD_WIDTH > 16
                v_float32x4 v_scale = v_setall_f32((float)_scale);
                for( ; i <= width-v_int16x8::nlanes; i+=v_int16x8::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_int32x4 v_s01 = v_load(SUM + i + v_int32x4::nlanes) + v_load(Sp + i + v_int32x4::nlanes);

                    v_int32x4 v_s0d =  v_round(v_cvt_f32(v_s0) * v_scale);
                    v_int32x4 v_s01d = v_round(v_cvt_f32(v_s01) * v_scale);
                    v_store(D + i, v_pack(v_s0d, v_s01d));

                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                    v_store(SUM + i + v_int32x4::nlanes, v_s01 - v_load(Sm + i + v_int32x4::nlanes));
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<short>(s0*_scale);
                    SUM[i] = s0 - Sm[i];
                }
            }
            else
            {
                i = 0;
#if CV_SIMD
                for( ; i <= width-v_int16::nlanes; i+=v_int16::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_int32 v_s01 = vx_load(SUM + i + v_int32::nlanes) + vx_load(Sp + i + v_int32::nlanes);

                    v_store(D + i, v_pack(v_s0, v_s01));

                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                    v_store(SUM + i + v_int32::nlanes, v_s01 - vx_load(Sm + i + v_int32::nlanes));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width-v_int16x8::nlanes; i+=v_int16x8::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_int32x4 v_s01 = v_load(SUM + i + v_int32x4::nlanes) + v_load(Sp + i + v_int32x4::nlanes);

                    v_store(D + i, v_pack(v_s0, v_s01));

                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                    v_store(SUM + i + v_int32x4::nlanes, v_s01 - v_load(Sm + i + v_int32x4::nlanes));
                }
#endif
#endif

                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<short>(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }

    double scale;
    int sumCount;
    std::vector<int> sum;
};


template<>
struct ColumnSum<int, ushort> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int* SUM;
        bool haveScale = scale != 1;
        double _scale = scale;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(int));
            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const int* Sp = (const int*)src[0];
                int i = 0;
#if CV_SIMD
                for (; i <= width - v_int32::nlanes; i += v_int32::nlanes)
                {
                    v_store(SUM + i, vx_load(SUM + i) + vx_load(Sp + i));
                }
#if CV_SIMD_WIDTH > 16
                for (; i <= width - v_int32x4::nlanes; i += v_int32x4::nlanes)
                {
                    v_store(SUM + i, v_load(SUM + i) + v_load(Sp + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const int* Sp = (const int*)src[0];
            const int* Sm = (const int*)src[1-ksize];
            ushort* D = (ushort*)dst;
            if( haveScale )
            {
                int i = 0;
#if CV_SIMD
                v_float32 _v_scale = vx_setall_f32((float)_scale);
                for( ; i <= width-v_uint16::nlanes; i+=v_uint16::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_int32 v_s01 = vx_load(SUM + i + v_int32::nlanes) + vx_load(Sp + i + v_int32::nlanes);

                    v_uint32 v_s0d = v_reinterpret_as_u32(v_round(v_cvt_f32(v_s0) * _v_scale));
                    v_uint32 v_s01d = v_reinterpret_as_u32(v_round(v_cvt_f32(v_s01) * _v_scale));
                    v_store(D + i, v_pack(v_s0d, v_s01d));

                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                    v_store(SUM + i + v_int32::nlanes, v_s01 - vx_load(Sm + i + v_int32::nlanes));
                }
#if CV_SIMD_WIDTH > 16
                v_float32x4 v_scale = v_setall_f32((float)_scale);
                for( ; i <= width-v_uint16x8::nlanes; i+=v_uint16x8::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_int32x4 v_s01 = v_load(SUM + i + v_int32x4::nlanes) + v_load(Sp + i + v_int32x4::nlanes);

                    v_uint32x4 v_s0d = v_reinterpret_as_u32(v_round(v_cvt_f32(v_s0) * v_scale));
                    v_uint32x4 v_s01d = v_reinterpret_as_u32(v_round(v_cvt_f32(v_s01) * v_scale));
                    v_store(D + i, v_pack(v_s0d, v_s01d));

                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                    v_store(SUM + i + v_int32x4::nlanes, v_s01 - v_load(Sm + i + v_int32x4::nlanes));
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<ushort>(s0*_scale);
                    SUM[i] = s0 - Sm[i];
                }
            }
            else
            {
                int i = 0;
#if CV_SIMD
                for( ; i <= width-v_uint16::nlanes; i+=v_uint16::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_int32 v_s01 = vx_load(SUM + i + v_int32::nlanes) + vx_load(Sp + i + v_int32::nlanes);

                    v_store(D + i, v_pack(v_reinterpret_as_u32(v_s0), v_reinterpret_as_u32(v_s01)));

                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                    v_store(SUM + i + v_int32::nlanes, v_s01 - vx_load(Sm + i + v_int32::nlanes));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width-v_uint16x8::nlanes; i+=v_uint16x8::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_int32x4 v_s01 = v_load(SUM + i + v_int32x4::nlanes) + v_load(Sp + i + v_int32x4::nlanes);

                    v_store(D + i, v_pack(v_reinterpret_as_u32(v_s0), v_reinterpret_as_u32(v_s01)));

                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                    v_store(SUM + i + v_int32x4::nlanes, v_s01 - v_load(Sm + i + v_int32x4::nlanes));
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<ushort>(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }

    double scale;
    int sumCount;
    std::vector<int> sum;
};

template<>
struct ColumnSum<int, int> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int* SUM;
        bool haveScale = scale != 1;
        double _scale = scale;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(int));
            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const int* Sp = (const int*)src[0];
                int i = 0;
#if CV_SIMD
                for( ; i <= width - v_int32::nlanes; i+=v_int32::nlanes )
                {
                    v_store(SUM + i, vx_load(SUM + i) + vx_load(Sp + i));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width - v_int32x4::nlanes; i+=v_int32x4::nlanes )
                {
                    v_store(SUM + i, v_load(SUM + i) + v_load(Sp + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const int* Sp = (const int*)src[0];
            const int* Sm = (const int*)src[1-ksize];
            int* D = (int*)dst;
            if( haveScale )
            {
                int i = 0;
#if CV_SIMD
                v_float32 _v_scale = vx_setall_f32((float)_scale);
                for( ; i <= width-v_int32::nlanes; i+=v_int32::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_int32 v_s0d = v_round(v_cvt_f32(v_s0) * _v_scale);

                    v_store(D + i, v_s0d);
                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                }
#if CV_SIMD_WIDTH > 16
                v_float32x4 v_scale = v_setall_f32((float)_scale);
                for( ; i <= width-v_int32x4::nlanes; i+=v_int32x4::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_int32x4 v_s0d = v_round(v_cvt_f32(v_s0) * v_scale);

                    v_store(D + i, v_s0d);
                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<int>(s0*_scale);
                    SUM[i] = s0 - Sm[i];
                }
            }
            else
            {
                int i = 0;
#if CV_SIMD
                for( ; i <= width-v_int32::nlanes; i+=v_int32::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);

                    v_store(D + i, v_s0);
                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width-v_int32x4::nlanes; i+=v_int32x4::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);

                    v_store(D + i, v_s0);
                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = s0;
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }

    double scale;
    int sumCount;
    std::vector<int> sum;
};


template<>
struct ColumnSum<int, float> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int* SUM;
        bool haveScale = scale != 1;
        double _scale = scale;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(int));
            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const int* Sp = (const int*)src[0];
                int i = 0;
#if CV_SIMD
                for( ; i <= width - v_int32::nlanes; i+=v_int32::nlanes )
                {
                    v_store(SUM + i, vx_load(SUM + i) + vx_load(Sp + i));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width - v_int32x4::nlanes; i+=v_int32x4::nlanes )
                {
                    v_store(SUM + i, v_load(SUM + i) + v_load(Sp + i));
                }
#endif
#endif

                for( ; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const int * Sp = (const int*)src[0];
            const int * Sm = (const int*)src[1-ksize];
            float* D = (float*)dst;
            if( haveScale )
            {
                int i = 0;

#if CV_SIMD
                v_float32 _v_scale = vx_setall_f32((float)_scale);
                for (; i <= width - v_int32::nlanes; i += v_int32::nlanes)
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_store(D + i, v_cvt_f32(v_s0) * _v_scale);
                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                }
#if CV_SIMD_WIDTH > 16
                v_float32x4 v_scale = v_setall_f32((float)_scale);
                for (; i <= width - v_int32x4::nlanes; i += v_int32x4::nlanes)
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_store(D + i, v_cvt_f32(v_s0) * v_scale);
                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = (float)(s0*_scale);
                    SUM[i] = s0 - Sm[i];
                }
            }
            else
            {
                int i = 0;

#if CV_SIMD
                for( ; i <= width-v_int32::nlanes; i+=v_int32::nlanes )
                {
                    v_int32 v_s0 = vx_load(SUM + i) + vx_load(Sp + i);
                    v_store(D + i, v_cvt_f32(v_s0));
                    v_store(SUM + i, v_s0 - vx_load(Sm + i));
                }
#if CV_SIMD_WIDTH > 16
                for( ; i <= width-v_int32x4::nlanes; i+=v_int32x4::nlanes )
                {
                    v_int32x4 v_s0 = v_load(SUM + i) + v_load(Sp + i);
                    v_store(D + i, v_cvt_f32(v_s0));
                    v_store(SUM + i, v_s0 - v_load(Sm + i));
                }
#endif
#endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = (float)(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }

    double scale;
    int sumCount;
    std::vector<int> sum;
};

}  // namespace anon


Ptr<BaseRowFilter> getRowSumFilter(int srcType, int sumType, int ksize, int anchor)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(sumType);
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(srcType) );

    if( anchor < 0 )
        anchor = ksize/2;

    if( sdepth == CV_8U && ddepth == CV_32S )
        return makePtr<RowSum<uchar, int> >(ksize, anchor);
    if( sdepth == CV_8U && ddepth == CV_16U )
        return makePtr<RowSum<uchar, ushort> >(ksize, anchor);
    if( sdepth == CV_8U && ddepth == CV_64F )
        return makePtr<RowSum<uchar, double> >(ksize, anchor);
    if( sdepth == CV_16U && ddepth == CV_32S )
        return makePtr<RowSum<ushort, int> >(ksize, anchor);
    if( sdepth == CV_16U && ddepth == CV_64F )
        return makePtr<RowSum<ushort, double> >(ksize, anchor);
    if( sdepth == CV_16S && ddepth == CV_32S )
        return makePtr<RowSum<short, int> >(ksize, anchor);
    if( sdepth == CV_32S && ddepth == CV_32S )
        return makePtr<RowSum<int, int> >(ksize, anchor);
    if( sdepth == CV_16S && ddepth == CV_64F )
        return makePtr<RowSum<short, double> >(ksize, anchor);
    if( sdepth == CV_32F && ddepth == CV_64F )
        return makePtr<RowSum<float, double> >(ksize, anchor);
    if( sdepth == CV_64F && ddepth == CV_64F )
        return makePtr<RowSum<double, double> >(ksize, anchor);

    CV_Error_( CV_StsNotImplemented,
        ("Unsupported combination of source format (=%d), and buffer format (=%d)",
        srcType, sumType));
}


Ptr<BaseColumnFilter> getColumnSumFilter(int sumType, int dstType, int ksize, int anchor, double scale)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(sumType), ddepth = CV_MAT_DEPTH(dstType);
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(dstType) );

    if( anchor < 0 )
        anchor = ksize/2;

    if( ddepth == CV_8U && sdepth == CV_32S )
        return makePtr<ColumnSum<int, uchar> >(ksize, anchor, scale);
    if( ddepth == CV_8U && sdepth == CV_16U )
        return makePtr<ColumnSum<ushort, uchar> >(ksize, anchor, scale);
    if( ddepth == CV_8U && sdepth == CV_64F )
        return makePtr<ColumnSum<double, uchar> >(ksize, anchor, scale);
    if( ddepth == CV_16U && sdepth == CV_32S )
        return makePtr<ColumnSum<int, ushort> >(ksize, anchor, scale);
    if( ddepth == CV_16U && sdepth == CV_64F )
        return makePtr<ColumnSum<double, ushort> >(ksize, anchor, scale);
    if( ddepth == CV_16S && sdepth == CV_32S )
        return makePtr<ColumnSum<int, short> >(ksize, anchor, scale);
    if( ddepth == CV_16S && sdepth == CV_64F )
        return makePtr<ColumnSum<double, short> >(ksize, anchor, scale);
    if( ddepth == CV_32S && sdepth == CV_32S )
        return makePtr<ColumnSum<int, int> >(ksize, anchor, scale);
    if( ddepth == CV_32F && sdepth == CV_32S )
        return makePtr<ColumnSum<int, float> >(ksize, anchor, scale);
    if( ddepth == CV_32F && sdepth == CV_64F )
        return makePtr<ColumnSum<double, float> >(ksize, anchor, scale);
    if( ddepth == CV_64F && sdepth == CV_32S )
        return makePtr<ColumnSum<int, double> >(ksize, anchor, scale);
    if( ddepth == CV_64F && sdepth == CV_64F )
        return makePtr<ColumnSum<double, double> >(ksize, anchor, scale);

    CV_Error_( CV_StsNotImplemented,
        ("Unsupported combination of sum format (=%d), and destination format (=%d)",
        sumType, dstType));
}


Ptr<FilterEngine> createBoxFilter(int srcType, int dstType, Size ksize,
                                  Point anchor, bool normalize, int borderType)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(srcType);
    int cn = CV_MAT_CN(srcType), sumType = CV_64F;
    if( sdepth == CV_8U && CV_MAT_DEPTH(dstType) == CV_8U &&
        ksize.width*ksize.height <= 256 )
        sumType = CV_16U;
    else if( sdepth <= CV_32S && (!normalize ||
        ksize.width*ksize.height <= (sdepth == CV_8U ? (1<<23) :
            sdepth == CV_16U ? (1 << 15) : (1 << 16))) )
        sumType = CV_32S;
    sumType = CV_MAKETYPE( sumType, cn );

    Ptr<BaseRowFilter> rowFilter = getRowSumFilter(srcType, sumType, ksize.width, anchor.x );
    Ptr<BaseColumnFilter> columnFilter = getColumnSumFilter(sumType,
        dstType, ksize.height, anchor.y, normalize ? 1./(ksize.width*ksize.height) : 1);

    return makePtr<FilterEngine>(Ptr<BaseFilter>(), rowFilter, columnFilter,
           srcType, dstType, sumType, borderType );
}



/****************************************************************************************\
                                    Squared Box Filter
\****************************************************************************************/
namespace {

template<typename T, typename ST>
struct SqrRowSum :
        public BaseRowFilter
{
    SqrRowSum( int _ksize, int _anchor ) :
        BaseRowFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
    }

    virtual void operator()(const uchar* src, uchar* dst, int width, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        const T* S = (const T*)src;
        ST* D = (ST*)dst;
        int i = 0, k, ksz_cn = ksize*cn;

        width = (width - 1)*cn;
        for( k = 0; k < cn; k++, S++, D++ )
        {
            ST s = 0;
            for( i = 0; i < ksz_cn; i += cn )
            {
                ST val = (ST)S[i];
                s += val*val;
            }
            D[0] = s;
            for( i = 0; i < width; i += cn )
            {
                ST val0 = (ST)S[i], val1 = (ST)S[i + ksz_cn];
                s += val1*val1 - val0*val0;
                D[i+cn] = s;
            }
        }
    }
};

} // namespace anon

Ptr<BaseRowFilter> getSqrRowSumFilter(int srcType, int sumType, int ksize, int anchor)
{
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(sumType);
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(srcType) );

    if( anchor < 0 )
        anchor = ksize/2;

    if( sdepth == CV_8U && ddepth == CV_32S )
        return makePtr<SqrRowSum<uchar, int> >(ksize, anchor);
    if( sdepth == CV_8U && ddepth == CV_64F )
        return makePtr<SqrRowSum<uchar, double> >(ksize, anchor);
    if( sdepth == CV_16U && ddepth == CV_64F )
        return makePtr<SqrRowSum<ushort, double> >(ksize, anchor);
    if( sdepth == CV_16S && ddepth == CV_64F )
        return makePtr<SqrRowSum<short, double> >(ksize, anchor);
    if( sdepth == CV_32F && ddepth == CV_64F )
        return makePtr<SqrRowSum<float, double> >(ksize, anchor);
    if( sdepth == CV_64F && ddepth == CV_64F )
        return makePtr<SqrRowSum<double, double> >(ksize, anchor);

    CV_Error_( CV_StsNotImplemented,
              ("Unsupported combination of source format (=%d), and buffer format (=%d)",
               srcType, sumType));
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
