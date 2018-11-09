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

#include <vector>

#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_imgproc.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

namespace cv
{

/****************************************************************************************\
                                         Box Filter
\****************************************************************************************/

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
#if CV_SIMD
        vx_cleanup();
#endif
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
#if CV_SIMD
        vx_cleanup();
#endif
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
#if CV_SIMD
        vx_cleanup();
#endif
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
#if CV_SIMD
        vx_cleanup();
#endif
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
#if CV_SIMD
        vx_cleanup();
#endif
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
#if CV_SIMD
        vx_cleanup();
#endif
    }

    double scale;
    int sumCount;
    std::vector<int> sum;
};

#ifdef HAVE_OPENCL

static bool ocl_boxFilter3x3_8UC1( InputArray _src, OutputArray _dst, int ddepth,
                                   Size ksize, Point anchor, int borderType, bool normalize )
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if (ddepth < 0)
        ddepth = sdepth;

    if (anchor.x < 0)
        anchor.x = ksize.width / 2;
    if (anchor.y < 0)
        anchor.y = ksize.height / 2;

    if ( !(dev.isIntel() && (type == CV_8UC1) &&
         (_src.offset() == 0) && (_src.step() % 4 == 0) &&
         (_src.cols() % 16 == 0) && (_src.rows() % 2 == 0) &&
         (anchor.x == 1) && (anchor.y == 1) &&
         (ksize.width == 3) && (ksize.height == 3)) )
        return false;

    float alpha = 1.0f / (ksize.height * ksize.width);
    Size size = _src.size();
    size_t globalsize[2] = { 0, 0 };
    size_t localsize[2] = { 0, 0 };
    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", 0, "BORDER_REFLECT_101" };

    globalsize[0] = size.width / 16;
    globalsize[1] = size.height / 2;

    char build_opts[1024];
    sprintf(build_opts, "-D %s %s", borderMap[borderType], normalize ? "-D NORMALIZE" : "");

    ocl::Kernel kernel("boxFilter3x3_8UC1_cols16_rows2", cv::ocl::imgproc::boxFilter3x3_oclsrc, build_opts);
    if (kernel.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(size, CV_MAKETYPE(ddepth, cn));
    if (!(_dst.offset() == 0 && _dst.step() % 4 == 0))
        return false;
    UMat dst = _dst.getUMat();

    int idxArg = kernel.set(0, ocl::KernelArg::PtrReadOnly(src));
    idxArg = kernel.set(idxArg, (int)src.step);
    idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
    idxArg = kernel.set(idxArg, (int)dst.step);
    idxArg = kernel.set(idxArg, (int)dst.rows);
    idxArg = kernel.set(idxArg, (int)dst.cols);
    if (normalize)
        idxArg = kernel.set(idxArg, (float)alpha);

    return kernel.run(2, globalsize, (localsize[0] == 0) ? NULL : localsize, false);
}

static bool ocl_boxFilter( InputArray _src, OutputArray _dst, int ddepth,
                           Size ksize, Point anchor, int borderType, bool normalize, bool sqr = false )
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type), esz = CV_ELEM_SIZE(type);
    bool doubleSupport = dev.doubleFPConfig() > 0;

    if (ddepth < 0)
        ddepth = sdepth;

    if (cn > 4 || (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F)) ||
        _src.offset() % esz != 0 || _src.step() % esz != 0)
        return false;

    if (anchor.x < 0)
        anchor.x = ksize.width / 2;
    if (anchor.y < 0)
        anchor.y = ksize.height / 2;

    int computeUnits = ocl::Device::getDefault().maxComputeUnits();
    float alpha = 1.0f / (ksize.height * ksize.width);
    Size size = _src.size(), wholeSize;
    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    borderType &= ~BORDER_ISOLATED;
    int wdepth = std::max(CV_32F, std::max(ddepth, sdepth)),
        wtype = CV_MAKE_TYPE(wdepth, cn), dtype = CV_MAKE_TYPE(ddepth, cn);

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", 0, "BORDER_REFLECT_101" };
    size_t globalsize[2] = { (size_t)size.width, (size_t)size.height };
    size_t localsize_general[2] = { 0, 1 }, * localsize = NULL;

    UMat src = _src.getUMat();
    if (!isolated)
    {
        Point ofs;
        src.locateROI(wholeSize, ofs);
    }

    int h = isolated ? size.height : wholeSize.height;
    int w = isolated ? size.width : wholeSize.width;

    size_t maxWorkItemSizes[32];
    ocl::Device::getDefault().maxWorkItemSizes(maxWorkItemSizes);
    int tryWorkItems = (int)maxWorkItemSizes[0];

    ocl::Kernel kernel;

    if (dev.isIntel() && !(dev.type() & ocl::Device::TYPE_CPU) &&
        ((ksize.width < 5 && ksize.height < 5 && esz <= 4) ||
         (ksize.width == 5 && ksize.height == 5 && cn == 1)))
    {
        if (w < ksize.width || h < ksize.height)
            return false;

        // Figure out what vector size to use for loading the pixels.
        int pxLoadNumPixels = cn != 1 || size.width % 4 ? 1 : 4;
        int pxLoadVecSize = cn * pxLoadNumPixels;

        // Figure out how many pixels per work item to compute in X and Y
        // directions.  Too many and we run out of registers.
        int pxPerWorkItemX = 1, pxPerWorkItemY = 1;
        if (cn <= 2 && ksize.width <= 4 && ksize.height <= 4)
        {
            pxPerWorkItemX = size.width % 8 ? size.width % 4 ? size.width % 2 ? 1 : 2 : 4 : 8;
            pxPerWorkItemY = size.height % 2 ? 1 : 2;
        }
        else if (cn < 4 || (ksize.width <= 4 && ksize.height <= 4))
        {
            pxPerWorkItemX = size.width % 2 ? 1 : 2;
            pxPerWorkItemY = size.height % 2 ? 1 : 2;
        }
        globalsize[0] = size.width / pxPerWorkItemX;
        globalsize[1] = size.height / pxPerWorkItemY;

        // Need some padding in the private array for pixels
        int privDataWidth = roundUp(pxPerWorkItemX + ksize.width - 1, pxLoadNumPixels);

        // Make the global size a nice round number so the runtime can pick
        // from reasonable choices for the workgroup size
        const int wgRound = 256;
        globalsize[0] = roundUp(globalsize[0], wgRound);

        char build_options[1024], cvt[2][40];
        sprintf(build_options, "-D cn=%d "
                "-D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d "
                "-D PX_LOAD_VEC_SIZE=%d -D PX_LOAD_NUM_PX=%d "
                "-D PX_PER_WI_X=%d -D PX_PER_WI_Y=%d -D PRIV_DATA_WIDTH=%d -D %s -D %s "
                "-D PX_LOAD_X_ITERATIONS=%d -D PX_LOAD_Y_ITERATIONS=%d "
                "-D srcT=%s -D srcT1=%s -D dstT=%s -D dstT1=%s -D WT=%s -D WT1=%s "
                "-D convertToWT=%s -D convertToDstT=%s%s%s -D PX_LOAD_FLOAT_VEC_CONV=convert_%s -D OP_BOX_FILTER",
                cn, anchor.x, anchor.y, ksize.width, ksize.height,
                pxLoadVecSize, pxLoadNumPixels,
                pxPerWorkItemX, pxPerWorkItemY, privDataWidth, borderMap[borderType],
                isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
                privDataWidth / pxLoadNumPixels, pxPerWorkItemY + ksize.height - 1,
                ocl::typeToStr(type), ocl::typeToStr(sdepth), ocl::typeToStr(dtype),
                ocl::typeToStr(ddepth), ocl::typeToStr(wtype), ocl::typeToStr(wdepth),
                ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
                ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]),
                normalize ? " -D NORMALIZE" : "", sqr ? " -D SQR" : "",
                ocl::typeToStr(CV_MAKE_TYPE(wdepth, pxLoadVecSize)) //PX_LOAD_FLOAT_VEC_CONV
                );


        if (!kernel.create("filterSmall", cv::ocl::imgproc::filterSmall_oclsrc, build_options))
            return false;
    }
    else
    {
        localsize = localsize_general;
        for ( ; ; )
        {
            int BLOCK_SIZE_X = tryWorkItems, BLOCK_SIZE_Y = std::min(ksize.height * 10, size.height);

            while (BLOCK_SIZE_X > 32 && BLOCK_SIZE_X >= ksize.width * 2 && BLOCK_SIZE_X > size.width * 2)
                BLOCK_SIZE_X /= 2;
            while (BLOCK_SIZE_Y < BLOCK_SIZE_X / 8 && BLOCK_SIZE_Y * computeUnits * 32 < size.height)
                BLOCK_SIZE_Y *= 2;

            if (ksize.width > BLOCK_SIZE_X || w < ksize.width || h < ksize.height)
                return false;

            char cvt[2][50];
            String opts = format("-D LOCAL_SIZE_X=%d -D BLOCK_SIZE_Y=%d -D ST=%s -D DT=%s -D WT=%s -D convertToDT=%s -D convertToWT=%s"
                                 " -D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d -D %s%s%s%s%s"
                                 " -D ST1=%s -D DT1=%s -D cn=%d",
                                 BLOCK_SIZE_X, BLOCK_SIZE_Y, ocl::typeToStr(type), ocl::typeToStr(CV_MAKE_TYPE(ddepth, cn)),
                                 ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)),
                                 ocl::convertTypeStr(wdepth, ddepth, cn, cvt[0]),
                                 ocl::convertTypeStr(sdepth, wdepth, cn, cvt[1]),
                                 anchor.x, anchor.y, ksize.width, ksize.height, borderMap[borderType],
                                 isolated ? " -D BORDER_ISOLATED" : "", doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                                 normalize ? " -D NORMALIZE" : "", sqr ? " -D SQR" : "",
                                 ocl::typeToStr(sdepth), ocl::typeToStr(ddepth), cn);

            localsize[0] = BLOCK_SIZE_X;
            globalsize[0] = divUp(size.width, BLOCK_SIZE_X - (ksize.width - 1)) * BLOCK_SIZE_X;
            globalsize[1] = divUp(size.height, BLOCK_SIZE_Y);

            kernel.create("boxFilter", cv::ocl::imgproc::boxFilter_oclsrc, opts);
            if (kernel.empty())
                return false;

            size_t kernelWorkGroupSize = kernel.workGroupSize();
            if (localsize[0] <= kernelWorkGroupSize)
                break;
            if (BLOCK_SIZE_X < (int)kernelWorkGroupSize)
                return false;

            tryWorkItems = (int)kernelWorkGroupSize;
        }
    }

    _dst.create(size, CV_MAKETYPE(ddepth, cn));
    UMat dst = _dst.getUMat();

    int idxArg = kernel.set(0, ocl::KernelArg::PtrReadOnly(src));
    idxArg = kernel.set(idxArg, (int)src.step);
    int srcOffsetX = (int)((src.offset % src.step) / src.elemSize());
    int srcOffsetY = (int)(src.offset / src.step);
    int srcEndX = isolated ? srcOffsetX + size.width : wholeSize.width;
    int srcEndY = isolated ? srcOffsetY + size.height : wholeSize.height;
    idxArg = kernel.set(idxArg, srcOffsetX);
    idxArg = kernel.set(idxArg, srcOffsetY);
    idxArg = kernel.set(idxArg, srcEndX);
    idxArg = kernel.set(idxArg, srcEndY);
    idxArg = kernel.set(idxArg, ocl::KernelArg::WriteOnly(dst));
    if (normalize)
        idxArg = kernel.set(idxArg, (float)alpha);

    return kernel.run(2, globalsize, localsize, false);
}

#endif

}


cv::Ptr<cv::BaseRowFilter> cv::getRowSumFilter(int srcType, int sumType, int ksize, int anchor)
{
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


cv::Ptr<cv::BaseColumnFilter> cv::getColumnSumFilter(int sumType, int dstType, int ksize,
                                                     int anchor, double scale)
{
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


cv::Ptr<cv::FilterEngine> cv::createBoxFilter( int srcType, int dstType, Size ksize,
                    Point anchor, bool normalize, int borderType )
{
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

#ifdef HAVE_OPENVX
namespace cv
{
    namespace ovx {
        template <> inline bool skipSmallImages<VX_KERNEL_BOX_3x3>(int w, int h) { return w*h < 640 * 480; }
    }
    static bool openvx_boxfilter(InputArray _src, OutputArray _dst, int ddepth,
                                 Size ksize, Point anchor,
                                 bool normalize, int borderType)
    {
        if (ddepth < 0)
            ddepth = CV_8UC1;
        if (_src.type() != CV_8UC1 || ddepth != CV_8U || !normalize ||
            _src.cols() < 3 || _src.rows() < 3 ||
            ksize.width != 3 || ksize.height != 3 ||
            (anchor.x >= 0 && anchor.x != 1) ||
            (anchor.y >= 0 && anchor.y != 1) ||
            ovx::skipSmallImages<VX_KERNEL_BOX_3x3>(_src.cols(), _src.rows()))
            return false;

        Mat src = _src.getMat();

        if ((borderType & BORDER_ISOLATED) == 0 && src.isSubmatrix())
            return false; //Process isolated borders only
        vx_enum border;
        switch (borderType & ~BORDER_ISOLATED)
        {
        case BORDER_CONSTANT:
            border = VX_BORDER_CONSTANT;
            break;
        case BORDER_REPLICATE:
            border = VX_BORDER_REPLICATE;
            break;
        default:
            return false;
        }

        _dst.create(src.size(), CV_8UC1);
        Mat dst = _dst.getMat();

        try
        {
            ivx::Context ctx = ovx::getOpenVXContext();

            Mat a;
            if (dst.data != src.data)
                a = src;
            else
                src.copyTo(a);

            ivx::Image
                ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                                                  ivx::Image::createAddressing(a.cols, a.rows, 1, (vx_int32)(a.step)), a.data),
                ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                                                  ivx::Image::createAddressing(dst.cols, dst.rows, 1, (vx_int32)(dst.step)), dst.data);

            //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
            //since OpenVX standard says nothing about thread-safety for now
            ivx::border_t prevBorder = ctx.immediateBorder();
            ctx.setImmediateBorder(border, (vx_uint8)(0));
            ivx::IVX_CHECK_STATUS(vxuBox3x3(ctx, ia, ib));
            ctx.setImmediateBorder(prevBorder);
        }
        catch (const ivx::RuntimeError & e)
        {
            VX_DbgThrow(e.what());
        }
        catch (const ivx::WrapperError & e)
        {
            VX_DbgThrow(e.what());
        }

        return true;
    }
}
#endif

#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_boxfilter(Mat &src, Mat &dst, Size ksize, Point anchor, bool normalize, int borderType)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 < 201801
    // Problem with SSE42 optimization for 16s and some 8u modes
    if(ipp::getIppTopFeatures() == ippCPUID_SSE42 && (((src.depth() == CV_16S || src.depth() == CV_16U) && (src.channels() == 3 || src.channels() == 4)) || (src.depth() == CV_8U && src.channels() == 3 && (ksize.width > 5 || ksize.height > 5))))
        return false;

    // Other optimizations has some degradations too
    if((((src.depth() == CV_16S || src.depth() == CV_16U) && (src.channels() == 4)) || (src.depth() == CV_8U && src.channels() == 1 && (ksize.width > 5 || ksize.height > 5))))
        return false;
#endif

    if(!normalize)
        return false;

    if(!ippiCheckAnchor(anchor, ksize))
        return false;

    try
    {
        ::ipp::IwiImage       iwSrc      = ippiGetImage(src);
        ::ipp::IwiImage       iwDst      = ippiGetImage(dst);
        ::ipp::IwiSize        iwKSize    = ippiGetSize(ksize);
        ::ipp::IwiBorderSize  borderSize(iwKSize);
        ::ipp::IwiBorderType  ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return false;

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBox, iwSrc, iwDst, iwKSize, ::ipp::IwDefault(), ippBorder);
    }
    catch (const ::ipp::IwException &)
    {
        return false;
    }

    return true;
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(ksize); CV_UNUSED(anchor); CV_UNUSED(normalize); CV_UNUSED(borderType);
    return false;
#endif
}
}
#endif


void cv::boxFilter( InputArray _src, OutputArray _dst, int ddepth,
                Size ksize, Point anchor,
                bool normalize, int borderType )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_dst.isUMat() &&
               (borderType == BORDER_REPLICATE || borderType == BORDER_CONSTANT ||
                borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101),
               ocl_boxFilter3x3_8UC1(_src, _dst, ddepth, ksize, anchor, borderType, normalize))

    CV_OCL_RUN(_dst.isUMat(), ocl_boxFilter(_src, _dst, ddepth, ksize, anchor, borderType, normalize))

    Mat src = _src.getMat();
    int stype = src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if( ddepth < 0 )
        ddepth = sdepth;
    _dst.create( src.size(), CV_MAKETYPE(ddepth, cn) );
    Mat dst = _dst.getMat();
    if( borderType != BORDER_CONSTANT && normalize && (borderType & BORDER_ISOLATED) != 0 )
    {
        if( src.rows == 1 )
            ksize.height = 1;
        if( src.cols == 1 )
            ksize.width = 1;
    }

    Point ofs;
    Size wsz(src.cols, src.rows);
    if(!(borderType&BORDER_ISOLATED))
        src.locateROI( wsz, ofs );

    CALL_HAL(boxFilter, cv_hal_boxFilter, src.ptr(), src.step, dst.ptr(), dst.step, src.cols, src.rows, sdepth, ddepth, cn,
             ofs.x, ofs.y, wsz.width - src.cols - ofs.x, wsz.height - src.rows - ofs.y, ksize.width, ksize.height,
             anchor.x, anchor.y, normalize, borderType&~BORDER_ISOLATED);

    CV_OVX_RUN(true,
               openvx_boxfilter(src, dst, ddepth, ksize, anchor, normalize, borderType))

    CV_IPP_RUN_FAST(ipp_boxfilter(src, dst, ksize, anchor, normalize, borderType));

    borderType = (borderType&~BORDER_ISOLATED);

    Ptr<FilterEngine> f = createBoxFilter( src.type(), dst.type(),
                        ksize, anchor, normalize, borderType );

    f->apply( src, dst, wsz, ofs );
}


void cv::blur( InputArray src, OutputArray dst,
           Size ksize, Point anchor, int borderType )
{
    CV_INSTRUMENT_REGION();

    boxFilter( src, dst, -1, ksize, anchor, true, borderType );
}


/****************************************************************************************\
                                    Squared Box Filter
\****************************************************************************************/

namespace cv
{

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

static Ptr<BaseRowFilter> getSqrRowSumFilter(int srcType, int sumType, int ksize, int anchor)
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

}

void cv::sqrBoxFilter( InputArray _src, OutputArray _dst, int ddepth,
                       Size ksize, Point anchor,
                       bool normalize, int borderType )
{
    CV_INSTRUMENT_REGION();

    int srcType = _src.type(), sdepth = CV_MAT_DEPTH(srcType), cn = CV_MAT_CN(srcType);
    Size size = _src.size();

    if( ddepth < 0 )
        ddepth = sdepth < CV_32F ? CV_32F : CV_64F;

    if( borderType != BORDER_CONSTANT && normalize )
    {
        if( size.height == 1 )
            ksize.height = 1;
        if( size.width == 1 )
            ksize.width = 1;
    }

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_boxFilter(_src, _dst, ddepth, ksize, anchor, borderType, normalize, true))

    int sumDepth = CV_64F;
    if( sdepth == CV_8U )
        sumDepth = CV_32S;
    int sumType = CV_MAKETYPE( sumDepth, cn ), dstType = CV_MAKETYPE(ddepth, cn);

    Mat src = _src.getMat();
    _dst.create( size, dstType );
    Mat dst = _dst.getMat();

    Ptr<BaseRowFilter> rowFilter = getSqrRowSumFilter(srcType, sumType, ksize.width, anchor.x );
    Ptr<BaseColumnFilter> columnFilter = getColumnSumFilter(sumType,
                                                            dstType, ksize.height, anchor.y,
                                                            normalize ? 1./(ksize.width*ksize.height) : 1);

    Ptr<FilterEngine> f = makePtr<FilterEngine>(Ptr<BaseFilter>(), rowFilter, columnFilter,
                                                srcType, dstType, sumType, borderType );
    Point ofs;
    Size wsz(src.cols, src.rows);
    src.locateROI( wsz, ofs );

    f->apply( src, dst, wsz, ofs );
}

/* End of file. */
