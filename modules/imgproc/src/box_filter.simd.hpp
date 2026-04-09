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
// Copyright (C) 2025-2026, Advanced Micro Devices, all rights reserved.
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
#include <cstddef>

// Align loops to 64-byte boundaries to prevent micro-op cache line splitting,
// which causes performance variation from code layout changes on
// modern x86 microarchitectures (AMD Zen4, Intel Alder Lake+).
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC push_options
#pragma GCC optimize("align-loops=64")
#endif

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
Ptr<BaseRowFilter> getRowSumFilter(int srcType, int sumType, int ksize, int anchor);
Ptr<BaseColumnFilter> getColumnSumFilter(int sumType, int dstType, int ksize, int anchor, double scale);
Ptr<BaseRowColumnFilter> getRowColumnSumFilter(int srcType, int dstType, int ksize, double scale);
Ptr<FilterEngine> createBoxFilter(int srcType, int dstType, Size ksize,
                                  Point anchor, bool normalize, int borderType);

void blockSum(const Mat& src, Mat& dst, Size ksize, Point anchor,
              const Size &wsz, const Point &ofs, bool normalize, int borderType);

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

    virtual void operator()(const uchar* src, uchar* dst, int width, int cn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        const T* s = (const T*)src;
        ST* D = (ST*)dst;
        int i = 0, k, kcn = ksize*cn;

        int len = width*cn;
        if( ksize == 3 )
        {
            if(processInnerRegion)
            {
                for(i = 0; i < len ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
                }
            }
            else
            {
                for(i = 0; i < cn ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
                }
                for(i=len-cn; i < len ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
                }
            }
        }
        else if( ksize == 5 )
        {
            if(processInnerRegion)
            {
                for(i = 0; i < len ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
                }
            }
            else
            {
                for(i = 0; i < min(len,2*cn) ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
                }
                for(i = max(len-(2*cn),0); i < len ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
                }
            }
        }
        else if( cn == 1 )
        {
            ST sum = 0;
            for( i = 0; i < kcn; i++ )
                sum += (ST)s[i];
            D[0] = sum;
            for( i = 0; i < len-cn; i++ )
            {
                sum += (ST)s[i + kcn] - (ST)s[i];
                D[i+1] = sum;
            }
        }
        else if( cn == 3 )
        {
            ST s0 = 0, s1 = 0, s2 = 0;
            for( i = 0; i < kcn; i += 3 )
            {
                s0 += (ST)s[i];
                s1 += (ST)s[i+1];
                s2 += (ST)s[i+2];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            for( i = 0; i < len-cn; i += 3 )
            {
                s0 += (ST)s[i + kcn] - (ST)s[i];
                s1 += (ST)s[i + kcn + 1] - (ST)s[i + 1];
                s2 += (ST)s[i + kcn + 2] - (ST)s[i + 2];
                D[i+3] = s0;
                D[i+4] = s1;
                D[i+5] = s2;
            }
        }
        else if( cn == 4 )
        {
            ST s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for( i = 0; i < kcn; i += 4 )
            {
                s0 += (ST)s[i];
                s1 += (ST)s[i+1];
                s2 += (ST)s[i+2];
                s3 += (ST)s[i+3];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            D[3] = s3;
            for( i = 0; i < len-cn; i += 4 )
            {
                s0 += (ST)s[i + kcn] - (ST)s[i];
                s1 += (ST)s[i + kcn + 1] - (ST)s[i + 1];
                s2 += (ST)s[i + kcn + 2] - (ST)s[i + 2];
                s3 += (ST)s[i + kcn + 3] - (ST)s[i + 3];
                D[i+4] = s0;
                D[i+5] = s1;
                D[i+6] = s2;
                D[i+7] = s3;
            }
        }
        else
        {
            for( k = 0; k < cn; k++, s++, D++ )
            {
                ST sum = 0;
                for( i = 0; i < kcn; i += cn )
                {
                    sum += (ST)s[i];
                }
                D[0] = sum;

                for( i = 0; i < len-cn; i += cn )
                {
                    sum += (ST)s[i + kcn] - (ST)s[i];
                    D[i+cn] = sum;
                }

            }

        }
    }
};

enum { SKIP_SCALING = 0, APPLY_SCALING = 1 };
#include "box_filter_2d.simd.hpp"

template<int SCALE_T, typename ST, typename T>
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
        processInnerRegionPrevRow = true;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }
    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int kcn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i;
        ST* SUM;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        if( processInnerRegion==true && processInnerRegionPrevRow==false)
        {
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

        bool haveScale = (SCALE_T == APPLY_SCALING);
        double _scale = scale;
        for( ; count--; src++ )
        {
            const ST* Sp = (const ST*)src[0];
            const ST* Sm = (const ST*)src[1-ksize];
            T* D = (T*)dst;

            i = 0;
            for( ; i < kcn; i++ )
            {
                ST s0 = SUM[i] + Sp[i];
                D[i] = haveScale ? saturate_cast<T>(s0*_scale) : saturate_cast<T>(s0);
                SUM[i] = s0 - Sm[i];
            }
            if(processInnerRegion)
            {
                if( haveScale )
                {
                    for( ; i <= width - kcn - 2; i += 2 )
                    {
                        ST s0 = SUM[i] + Sp[i], s1 = SUM[i+1] + Sp[i+1];
                        D[i] = saturate_cast<T>(s0*_scale);
                        D[i+1] = saturate_cast<T>(s1*_scale);
                        s0 -= Sm[i]; s1 -= Sm[i+1];
                        SUM[i] = s0; SUM[i+1] = s1;
                    }
                }
                else
                {
                    for( ; i <= width - kcn - 2; i += 2 )
                    {
                        ST s0 = SUM[i] + Sp[i], s1 = SUM[i+1] + Sp[i+1];
                        D[i] = saturate_cast<T>(s0);
                        D[i+1] = saturate_cast<T>(s1);
                        s0 -= Sm[i]; s1 -= Sm[i+1];
                        SUM[i] = s0; SUM[i+1] = s1;
                    }
                }
                for( ; i < width - kcn; i++ )
                {
                    ST s0 = SUM[i] + Sp[i];
                    D[i] = haveScale ? saturate_cast<T>(s0*_scale) : saturate_cast<T>(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }
            for( i = width - kcn; i < width; i++ )
            {
                ST s0 = SUM[i] + Sp[i];
                D[i] = haveScale ? saturate_cast<T>(s0*_scale) : saturate_cast<T>(s0);
                SUM[i] = s0 - Sm[i];
            }

            dst += dststep;
        }
        processInnerRegionPrevRow = processInnerRegion;
    }

    double scale;
    int sumCount;
    bool processInnerRegionPrevRow;
    std::vector<ST> sum;
};

template<int SCALE_T>
struct ColumnSum<SCALE_T, int, uchar> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
        processInnerRegionPrevRow = true;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int kcn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i;
        int* SUM;
        const int vlane = VTraits<v_int32>::vlanes();
        const int vlane16 = VTraits<v_uint16>::vlanes();

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        if( processInnerRegion==true && processInnerRegionPrevRow==false)
        {
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
                for( ; i <= width - vlane; i += vlane )
                {
                    v_store(SUM + i, v_add(vx_load(SUM + i), vx_load(Sp + i)));
                }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                for( ; i <= width - VTraits<v_int32x4>::vlanes(); i += VTraits<v_int32x4>::vlanes() )
                {
                    v_store(SUM + i, v_add(v_load(SUM + i), v_load(Sp + i)));
                }
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

            i = 0;
            for( ; i < kcn; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                if(SCALE_T == APPLY_SCALING)
                    D[i] = saturate_cast<uchar>(s0*scale);
                else
                    D[i] = saturate_cast<uchar>(s0);
                SUM[i] = s0 - Sm[i];
            }

            if(processInnerRegion)
            {
                if(SCALE_T == APPLY_SCALING)
                {
                    v_float32 _v_scale = vx_setall_f32((float)scale);
                    for( ; i <= width - kcn - vlane16; i += vlane16 )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_int32 v_s01 = v_add(vx_load(SUM + i + vlane), vx_load(Sp + i + vlane));

                        v_uint32 v_s0d = v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_s0), _v_scale)));
                        v_uint32 v_s01d = v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_s01), _v_scale)));
                        v_uint16 v_dst = v_pack(v_s0d, v_s01d);
                        v_pack_store(D + i, v_dst);

                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                        v_store(SUM + i + vlane, v_sub(v_s01, vx_load(Sm + i + vlane)));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    v_float32x4 v_scale4 = v_setall_f32((float)scale);
                    for( ; i <= width - kcn - VTraits<v_uint16x8>::vlanes(); i += VTraits<v_uint16x8>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_int32x4 v_s01 = v_add(v_load(SUM + i + VTraits<v_int32x4>::vlanes()), v_load(Sp + i + VTraits<v_int32x4>::vlanes()));

                        v_uint32x4 v_s0d = v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_s0), v_scale4)));
                        v_uint32x4 v_s01d = v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_s01), v_scale4)));
                        v_uint16x8 v_dst = v_pack(v_s0d, v_s01d);
                        v_pack_store(D + i, v_dst);

                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                        v_store(SUM + i + VTraits<v_int32x4>::vlanes(), v_sub(v_s01, v_load(Sm + i + VTraits<v_int32x4>::vlanes())));
                    }
#endif
                }
                else
                {
                    for( ; i <= width - kcn - vlane16; i += vlane16 )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_int32 v_s01 = v_add(vx_load(SUM + i + vlane), vx_load(Sp + i + vlane));

                        v_uint16 v_dst = v_pack(v_reinterpret_as_u32(v_s0), v_reinterpret_as_u32(v_s01));
                        v_pack_store(D + i, v_dst);

                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                        v_store(SUM + i + vlane, v_sub(v_s01, vx_load(Sm + i + vlane)));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    for( ; i <= width - kcn - VTraits<v_uint16x8>::vlanes(); i += VTraits<v_uint16x8>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_int32x4 v_s01 = v_add(v_load(SUM + i + VTraits<v_int32x4>::vlanes()), v_load(Sp + i + VTraits<v_int32x4>::vlanes()));

                        v_uint16x8 v_dst = v_pack(v_reinterpret_as_u32(v_s0), v_reinterpret_as_u32(v_s01));
                        v_pack_store(D + i, v_dst);

                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                        v_store(SUM + i + VTraits<v_int32x4>::vlanes(), v_sub(v_s01, v_load(Sm + i + VTraits<v_int32x4>::vlanes())));
                    }
#endif
                }
                for( ; i < width - kcn; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    if(SCALE_T == APPLY_SCALING)
                        D[i] = saturate_cast<uchar>(s0*scale);
                    else
                        D[i] = saturate_cast<uchar>(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }

            for( i = width - kcn; i < width; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                if(SCALE_T == APPLY_SCALING)
                    D[i] = saturate_cast<uchar>(s0*scale);
                else
                    D[i] = saturate_cast<uchar>(s0);
                SUM[i] = s0 - Sm[i];
            }
            dst += dststep;
        }
        processInnerRegionPrevRow = processInnerRegion;
    }

    double scale;
    int sumCount;
    bool processInnerRegionPrevRow;
    std::vector<int> sum;
};


template<>
struct ColumnSum<APPLY_SCALING, ushort, uchar> :
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
        processInnerRegionPrevRow = true;
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

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int kcn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        const int ds = divScale;
        const int dd = divDelta;
        ushort* SUM;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vlane = VTraits<v_uint16>::vlanes();
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
        const int vlane16x8 = VTraits<v_uint16x8>::vlanes();
#endif
#endif
        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        if( processInnerRegion==true && processInnerRegionPrevRow==false)
        {
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; i <= width - vlane; i += vlane )
                {
                    v_store(SUM + i, v_add(vx_load(SUM + i), vx_load(Sp + i)));
                }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                for( ; i <= width - vlane16x8; i += vlane16x8 )
                {
                    v_store(SUM + i, v_add(v_load(SUM + i), v_load(Sp + i)));
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

            int i = 0;
            for( ; i < kcn; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                D[i] = (uchar)((s0 + dd)*ds >> SHIFT);
                SUM[i] = (ushort)(s0 - Sm[i]);
            }
            if(processInnerRegion)
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_uint32 _ds4 = vx_setall_u32((unsigned)ds);
                v_uint16 _dd8 = vx_setall_u16((ushort)dd);

                for( ; i <= width-VTraits<v_uint8>::vlanes(); i+=VTraits<v_uint8>::vlanes() )
                {
                    v_uint16 _sm0 = vx_load(Sm + i);
                    v_uint16 _sm1 = vx_load(Sm + i + vlane);

                    v_uint16 _s0 = v_add_wrap(vx_load(SUM + i), vx_load(Sp + i));
                    v_uint16 _s1 = v_add_wrap(vx_load(SUM + i + vlane), vx_load(Sp + i + vlane));

                    v_uint32 _s00, _s01, _s10, _s11;

                    v_expand(v_add(_s0, _dd8), _s00, _s01);
                    v_expand(v_add(_s1, _dd8), _s10, _s11);

                    _s00 = v_shr<SHIFT>(v_mul(_s00, _ds4));
                    _s01 = v_shr<SHIFT>(v_mul(_s01, _ds4));
                    _s10 = v_shr<SHIFT>(v_mul(_s10, _ds4));
                    _s11 = v_shr<SHIFT>(v_mul(_s11, _ds4));

                    v_int16 r0 = v_pack(v_reinterpret_as_s32(_s00), v_reinterpret_as_s32(_s01));
                    v_int16 r1 = v_pack(v_reinterpret_as_s32(_s10), v_reinterpret_as_s32(_s11));

                    _s0 = v_sub_wrap(_s0, _sm0);
                    _s1 = v_sub_wrap(_s1, _sm1);

                    v_store(D + i, v_pack_u(r0, r1));
                    v_store(SUM + i, _s0);
                    v_store(SUM + i + vlane, _s1);
                }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                v_uint32x4 ds4 = v_setall_u32((unsigned)ds);
                v_uint16x8 dd8 = v_setall_u16((ushort)dd);
                for( ; i <= width-VTraits<v_uint8x16>::vlanes(); i+=VTraits<v_uint8x16>::vlanes() )
                {
                    v_uint16x8 _sm0 = v_load(Sm + i);
                    v_uint16x8 _sm1 = v_load(Sm + i + vlane16x8);

                    v_uint16x8 _s0 = v_add_wrap(v_load(SUM + i), v_load(Sp + i));
                    v_uint16x8 _s1 = v_add_wrap(v_load(SUM + i + vlane16x8), v_load(Sp + i + vlane16x8));

                    v_uint32x4 _s00, _s01, _s10, _s11;

                    v_expand(v_add(_s0, dd8), _s00, _s01);
                    v_expand(v_add(_s1, dd8), _s10, _s11);

                    _s00 = v_shr<SHIFT>(v_mul(_s00, ds4));
                    _s01 = v_shr<SHIFT>(v_mul(_s01, ds4));
                    _s10 = v_shr<SHIFT>(v_mul(_s10, ds4));
                    _s11 = v_shr<SHIFT>(v_mul(_s11, ds4));

                    v_int16x8 r0 = v_pack(v_reinterpret_as_s32(_s00), v_reinterpret_as_s32(_s01));
                    v_int16x8 r1 = v_pack(v_reinterpret_as_s32(_s10), v_reinterpret_as_s32(_s11));

                    _s0 = v_sub_wrap(_s0, _sm0);
                    _s1 = v_sub_wrap(_s1, _sm1);

                    v_store(D + i, v_pack_u(r0, r1));
                    v_store(SUM + i, _s0);
                    v_store(SUM + i + vlane16x8, _s1);
                }
#endif
#endif
                for( ; i < width-kcn; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = (uchar)((s0 + dd)*ds >> SHIFT);
                    SUM[i] = (ushort)(s0 - Sm[i]);
                }
            }//processInnerRegion
            for(i = width-kcn ; i < width; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                D[i] = (uchar)((s0 + dd)*ds >> SHIFT);
                SUM[i] = (ushort)(s0 - Sm[i]);
            }
            dst += dststep;
        }
        processInnerRegionPrevRow = processInnerRegion;
    }

    double scale;
    int sumCount;
    int divDelta;
    int divScale;
    bool processInnerRegionPrevRow;
    std::vector<ushort> sum;
};

template<int SCALE_T>
struct ColumnSum<SCALE_T, int, short> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
        scaleShift = 0;
        processInnerRegionPrevRow = true;
        if(SCALE_T == APPLY_SCALING && _scale > 0 && _scale < 1)
        {
            double inv = 1.0 / _scale;
            int intInv = cvRound(inv);
            if(intInv > 1 && (intInv & (intInv - 1)) == 0 && std::abs(inv - intInv) < 1e-10)
            {
                int temp = intInv;
                while(temp > 1) { temp >>= 1; scaleShift++; }
            }
        }
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    short inline cast_scale(int v)
    {
        if(SCALE_T == APPLY_SCALING)
        {
            if(scaleShift > 0)
                return saturate_cast<short>(v >> scaleShift);
            return saturate_cast<short>(v*scale);
        }
        else
            return saturate_cast<short>(v);
    }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int kcn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        int i;
        int* SUM;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        if( processInnerRegion==true && processInnerRegionPrevRow==false)
        {
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; i <= width - VTraits<v_int32>::vlanes(); i+=VTraits<v_int32>::vlanes() )
                {
                    v_store(SUM + i, v_add(vx_load(SUM + i), vx_load(Sp + i)));
                }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                for( ; i <= width - VTraits<v_int32x4>::vlanes(); i+=VTraits<v_int32x4>::vlanes() )
                {
                    v_store(SUM + i, v_add(v_load(SUM + i), v_load(Sp + i)));
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

            i = 0;
            for( ; i < kcn; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                D[i] = cast_scale(s0);
                SUM[i] = s0 - Sm[i];
            }

            if(processInnerRegion)
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if(SCALE_T == APPLY_SCALING)
                {
                    if(scaleShift > 0)
                    {
                        for( ; i <= width-VTraits<v_int16>::vlanes(); i+=VTraits<v_int16>::vlanes() )
                        {
                            v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                            v_int32 v_s01 = v_add(vx_load(SUM + i + VTraits<v_int32>::vlanes()), vx_load(Sp + i + VTraits<v_int32>::vlanes()));

                            v_int32 v_s0d = v_shr(v_s0, scaleShift);
                            v_int32 v_s01d = v_shr(v_s01, scaleShift);
                            v_store(D + i, v_pack(v_s0d, v_s01d));

                            v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                            v_store(SUM + i + VTraits<v_int32>::vlanes(), v_sub(v_s01, vx_load(Sm + i + VTraits<v_int32>::vlanes())));
                        }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                        for( ; i <= width-VTraits<v_int16x8>::vlanes(); i+=VTraits<v_int16x8>::vlanes() )
                        {
                            v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                            v_int32x4 v_s01 = v_add(v_load(SUM + i + VTraits<v_int32x4>::vlanes()), v_load(Sp + i + VTraits<v_int32x4>::vlanes()));

                            v_int32x4 v_s0d = v_shr(v_s0, scaleShift);
                            v_int32x4 v_s01d = v_shr(v_s01, scaleShift);
                            v_store(D + i, v_pack(v_s0d, v_s01d));

                            v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                            v_store(SUM + i + VTraits<v_int32x4>::vlanes(), v_sub(v_s01, v_load(Sm + i + VTraits<v_int32x4>::vlanes())));
                        }
#endif
                    }
                    else
                    {
                        v_float32 _v_scale = vx_setall_f32((float)scale);
                        for( ; i <= width-VTraits<v_int16>::vlanes(); i+=VTraits<v_int16>::vlanes() )
                        {
                            v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                            v_int32 v_s01 = v_add(vx_load(SUM + i + VTraits<v_int32>::vlanes()), vx_load(Sp + i + VTraits<v_int32>::vlanes()));

                            v_int32 v_s0d = v_round(v_mul(v_cvt_f32(v_s0), _v_scale));
                            v_int32 v_s01d = v_round(v_mul(v_cvt_f32(v_s01), _v_scale));
                            v_store(D + i, v_pack(v_s0d, v_s01d));

                            v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                            v_store(SUM + i + VTraits<v_int32>::vlanes(), v_sub(v_s01, vx_load(Sm + i + VTraits<v_int32>::vlanes())));
                        }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                        v_float32x4 v_scale = v_setall_f32((float)scale);
                        for( ; i <= width-VTraits<v_int16x8>::vlanes(); i+=VTraits<v_int16x8>::vlanes() )
                        {
                            v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                            v_int32x4 v_s01 = v_add(v_load(SUM + i + VTraits<v_int32x4>::vlanes()), v_load(Sp + i + VTraits<v_int32x4>::vlanes()));

                            v_int32x4 v_s0d = v_round(v_mul(v_cvt_f32(v_s0), v_scale));
                            v_int32x4 v_s01d = v_round(v_mul(v_cvt_f32(v_s01), v_scale));
                            v_store(D + i, v_pack(v_s0d, v_s01d));

                            v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                            v_store(SUM + i + VTraits<v_int32x4>::vlanes(), v_sub(v_s01, v_load(Sm + i + VTraits<v_int32x4>::vlanes())));
                        }
#endif
                    }
                }
                else
                {
                    for( ; i <= width-VTraits<v_int16>::vlanes(); i+=VTraits<v_int16>::vlanes() )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_int32 v_s01 = v_add(vx_load(SUM + i + VTraits<v_int32>::vlanes()), vx_load(Sp + i + VTraits<v_int32>::vlanes()));

                        v_store(D + i, v_pack(v_s0, v_s01));

                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                        v_store(SUM + i + VTraits<v_int32>::vlanes(), v_sub(v_s01, vx_load(Sm + i + VTraits<v_int32>::vlanes())));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    for( ; i <= width-VTraits<v_int16x8>::vlanes(); i+=VTraits<v_int16x8>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_int32x4 v_s01 = v_add(v_load(SUM + i + VTraits<v_int32x4>::vlanes()), v_load(Sp + i + VTraits<v_int32x4>::vlanes()));

                        v_store(D + i, v_pack(v_s0, v_s01));

                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                        v_store(SUM + i + VTraits<v_int32x4>::vlanes(), v_sub(v_s01, v_load(Sm + i + VTraits<v_int32x4>::vlanes())));
                    }
#endif
                }
#endif
                for( ; i < width-kcn; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = cast_scale(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }//processInnerRegion

            for(i = width-kcn ; i < width; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                D[i] = cast_scale(s0);
                SUM[i] = s0 - Sm[i];
            }
            dst += dststep;
        }
        processInnerRegionPrevRow = processInnerRegion;
    }

private:
    double scale;
    int scaleShift;
    int sumCount;
    bool processInnerRegionPrevRow;
    std::vector<int> sum;
};


template<int SCALE_T>
struct ColumnSum<SCALE_T, int, ushort> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
        processInnerRegionPrevRow = true;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    ushort inline cast_scale(int v)
    {
        if(SCALE_T == APPLY_SCALING)
            return saturate_cast<ushort>(v*scale);
        else
            return saturate_cast<ushort>(v);
    }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int kcn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        int* SUM;

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        if( processInnerRegion==true && processInnerRegionPrevRow==false)
        {
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; i <= width - VTraits<v_int32>::vlanes(); i+=VTraits<v_int32>::vlanes() )
                {
                    v_store(SUM + i, v_add(vx_load(SUM + i), vx_load(Sp + i)));
                }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                for( ; i <= width - VTraits<v_int32x4>::vlanes(); i+=VTraits<v_int32x4>::vlanes() )
                {
                    v_store(SUM + i, v_add(v_load(SUM + i), v_load(Sp + i)));
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

            int i = 0;
            for( ; i < kcn; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                D[i] = cast_scale(s0);
                SUM[i] = s0 - Sm[i];
            }

            if(processInnerRegion)
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if(SCALE_T == APPLY_SCALING)
                {
                    v_float32 _v_scale = vx_setall_f32((float)scale);
                    for( ; i <= width-VTraits<v_uint16>::vlanes(); i+=VTraits<v_uint16>::vlanes() )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_int32 v_s01 = v_add(vx_load(SUM + i + VTraits<v_int32>::vlanes()), vx_load(Sp + i + VTraits<v_int32>::vlanes()));

                        v_uint32 v_s0d = v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_s0), _v_scale)));
                        v_uint32 v_s01d = v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_s01), _v_scale)));
                        v_store(D + i, v_pack(v_s0d, v_s01d));

                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                        v_store(SUM + i + VTraits<v_int32>::vlanes(), v_sub(v_s01, vx_load(Sm + i + VTraits<v_int32>::vlanes())));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    v_float32x4 v_scale = v_setall_f32((float)scale);
                    for( ; i <= width-VTraits<v_uint16x8>::vlanes(); i+=VTraits<v_uint16x8>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_int32x4 v_s01 = v_add(v_load(SUM + i + VTraits<v_int32x4>::vlanes()), v_load(Sp + i + VTraits<v_int32x4>::vlanes()));

                        v_uint32x4 v_s0d = v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_s0), v_scale)));
                        v_uint32x4 v_s01d = v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_s01), v_scale)));
                        v_store(D + i, v_pack(v_s0d, v_s01d));

                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                        v_store(SUM + i + VTraits<v_int32x4>::vlanes(), v_sub(v_s01, v_load(Sm + i + VTraits<v_int32x4>::vlanes())));
                    }
#endif
                }
                else
                {
                    for( ; i <= width-VTraits<v_uint16>::vlanes(); i+=VTraits<v_uint16>::vlanes() )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_int32 v_s01 = v_add(vx_load(SUM + i + VTraits<v_int32>::vlanes()), vx_load(Sp + i + VTraits<v_int32>::vlanes()));

                        v_store(D + i, v_pack(v_reinterpret_as_u32(v_s0), v_reinterpret_as_u32(v_s01)));

                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                        v_store(SUM + i + VTraits<v_int32>::vlanes(), v_sub(v_s01, vx_load(Sm + i + VTraits<v_int32>::vlanes())));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    for( ; i <= width-VTraits<v_uint16x8>::vlanes(); i+=VTraits<v_uint16x8>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_int32x4 v_s01 = v_add(v_load(SUM + i + VTraits<v_int32x4>::vlanes()), v_load(Sp + i + VTraits<v_int32x4>::vlanes()));

                        v_store(D + i, v_pack(v_reinterpret_as_u32(v_s0), v_reinterpret_as_u32(v_s01)));

                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                        v_store(SUM + i + VTraits<v_int32x4>::vlanes(), v_sub(v_s01, v_load(Sm + i + VTraits<v_int32x4>::vlanes())));
                    }
#endif
                }
#endif
                for( ; i < width-kcn; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = cast_scale(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }//processInnerRegion

            for(i = width-kcn ; i < width; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                D[i] = cast_scale(s0);
                SUM[i] = s0 - Sm[i];
            }
            dst += dststep;
        }
        processInnerRegionPrevRow = processInnerRegion;
    }

private:
    double scale;
    int sumCount;
    bool processInnerRegionPrevRow;
    std::vector<int> sum;
};

//struct ColumnSum<int, int> :
template<int SCALE_T>
struct ColumnSum<SCALE_T, int, int> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
        processInnerRegionPrevRow = true;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int kcn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i;
        int* SUM;
        const int vlane = VTraits<v_int32>::vlanes();

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        if( processInnerRegion==true && processInnerRegionPrevRow==false)
        {
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
                for( ; i <= width - vlane; i += vlane )
                {
                    v_store(SUM + i, v_add(vx_load(SUM + i), vx_load(Sp + i)));
                }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                for( ; i <= width - VTraits<v_int32x4>::vlanes(); i += VTraits<v_int32x4>::vlanes() )
                {
                    v_store(SUM + i, v_add(v_load(SUM + i), v_load(Sp + i)));
                }
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

            i = 0;
            for( ; i < kcn; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                if(SCALE_T == APPLY_SCALING)
                    D[i] = saturate_cast<int>(s0*scale);
                else
                    D[i] = s0;
                SUM[i] = s0 - Sm[i];
            }

            if(processInnerRegion)
            {
                if(SCALE_T == APPLY_SCALING)
                {
                    v_float32 _v_scale = vx_setall_f32((float)scale);
                    for( ; i <= width - kcn - vlane; i += vlane )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_store(D + i, v_round(v_mul(v_cvt_f32(v_s0), _v_scale)));
                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    v_float32x4 v_scale4 = v_setall_f32((float)scale);
                    for( ; i <= width - kcn - VTraits<v_int32x4>::vlanes(); i += VTraits<v_int32x4>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_store(D + i, v_round(v_mul(v_cvt_f32(v_s0), v_scale4)));
                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                    }
#endif
                }
                else
                {
                    for( ; i <= width - kcn - vlane; i += vlane )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_store(D + i, v_s0);
                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    for( ; i <= width - kcn - VTraits<v_int32x4>::vlanes(); i += VTraits<v_int32x4>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_store(D + i, v_s0);
                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                    }
#endif
                }
                for( ; i < width - kcn; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    if(SCALE_T == APPLY_SCALING)
                        D[i] = saturate_cast<int>(s0*scale);
                    else
                        D[i] = s0;
                    SUM[i] = s0 - Sm[i];
                }
            }

            for( i = width - kcn; i < width; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                if(SCALE_T == APPLY_SCALING)
                    D[i] = saturate_cast<int>(s0*scale);
                else
                    D[i] = s0;
                SUM[i] = s0 - Sm[i];
            }
            dst += dststep;
        }
        processInnerRegionPrevRow = processInnerRegion;
    }

    double scale;
    int sumCount;
    bool processInnerRegionPrevRow;
    std::vector<int> sum;
};

template<int SCALE_T>
struct ColumnSum<SCALE_T, int, float> :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
        processInnerRegionPrevRow = true;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int kcn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i;
        int* SUM;
        const int vlane = VTraits<v_int32>::vlanes();

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        if( processInnerRegion==true && processInnerRegionPrevRow==false)
        {
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
                for( ; i <= width - vlane; i += vlane )
                {
                    v_store(SUM + i, v_add(vx_load(SUM + i), vx_load(Sp + i)));
                }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                for( ; i <= width - VTraits<v_int32x4>::vlanes(); i += VTraits<v_int32x4>::vlanes() )
                {
                    v_store(SUM + i, v_add(v_load(SUM + i), v_load(Sp + i)));
                }
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
            float* D = (float*)dst;

            i = 0;
            for( ; i < kcn; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                if(SCALE_T == APPLY_SCALING)
                    D[i] = (float)(s0*scale);
                else
                    D[i] = (float)s0;
                SUM[i] = s0 - Sm[i];
            }

            if(processInnerRegion)
            {
                if(SCALE_T == APPLY_SCALING)
                {
                    v_float32 _v_scale = vx_setall_f32((float)scale);
                    for( ; i <= width - kcn - vlane; i += vlane )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_store(D + i, v_mul(v_cvt_f32(v_s0), _v_scale));
                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    v_float32x4 v_scale4 = v_setall_f32((float)scale);
                    for( ; i <= width - kcn - VTraits<v_int32x4>::vlanes(); i += VTraits<v_int32x4>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_store(D + i, v_mul(v_cvt_f32(v_s0), v_scale4));
                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                    }
#endif
                }
                else
                {
                    for( ; i <= width - kcn - vlane; i += vlane )
                    {
                        v_int32 v_s0 = v_add(vx_load(SUM + i), vx_load(Sp + i));
                        v_store(D + i, v_cvt_f32(v_s0));
                        v_store(SUM + i, v_sub(v_s0, vx_load(Sm + i)));
                    }
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
                    for( ; i <= width - kcn - VTraits<v_int32x4>::vlanes(); i += VTraits<v_int32x4>::vlanes() )
                    {
                        v_int32x4 v_s0 = v_add(v_load(SUM + i), v_load(Sp + i));
                        v_store(D + i, v_cvt_f32(v_s0));
                        v_store(SUM + i, v_sub(v_s0, v_load(Sm + i)));
                    }
#endif
                }
                for( ; i < width - kcn; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    if(SCALE_T == APPLY_SCALING)
                        D[i] = (float)(s0*scale);
                    else
                        D[i] = (float)s0;
                    SUM[i] = s0 - Sm[i];
                }
            }

            for( i = width - kcn; i < width; i++ )
            {
                int s0 = SUM[i] + Sp[i];
                if(SCALE_T == APPLY_SCALING)
                    D[i] = (float)(s0*scale);
                else
                    D[i] = (float)s0;
                SUM[i] = s0 - Sm[i];
            }
            dst += dststep;
        }
        processInnerRegionPrevRow = processInnerRegion;
    }

    double scale;
    int sumCount;
    bool processInnerRegionPrevRow;
    std::vector<int> sum;
};


// End align-loops region.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC pop_options
#endif

#define VEC_ALIGN CV_MALLOC_ALIGN

template<typename ST, typename T>
inline void BlockSumBorderInplace(const T* ref, const T* S, T* R,
    const int *btab, int width, int cn, int dx1, int dx2, int borderType)
{
    int j=0, k;
    for( k=0; k < dx1*cn; k++, j++ )
    {
        if( borderType != BORDER_CONSTANT )
            R[j] = ref[btab[k]];
        else
            R[j] = 0;
    }
    for( k=0; k < width*cn; k++, j++ )
    {
        R[j] = S[k];
    }
    for( k=0; k < dx2*cn; k++, j++ )
    {
        if( borderType != BORDER_CONSTANT )
            R[j] = ref[btab[dx1*cn+k]];
        else
            R[j] = 0;
    }
}

template<typename ST, typename T>
inline int BlockSumCoreRow(const T* S, ST* SUM, T* D, ST** buf_ptr, double scale,
     int widthcn, int i, int cn, int kheight, int kwidth)
{
    bool haveScale = scale != 1;
    if( haveScale )
    {
        for( ; i < widthcn; i++ )
        {
            ST Ss = 0;
            for( int k=0; k < kwidth; k++ )
                Ss += (ST)S[i+k*cn];
            buf_ptr[kheight-1][i] = Ss;
            ST s0 = SUM[i] + Ss;
            D[i] = saturate_cast<T>(s0*scale);
            SUM[i] = s0 - buf_ptr[0][i];
        }
    }
    else
    {
        for( ; i < widthcn; i++ )
        {
            ST Ss = 0;
            for( int k=0; k < kwidth; k++ )
                Ss += (ST)S[i+k*cn];
            buf_ptr[kheight-1][i] = Ss;
            ST s0 = SUM[i] + Ss;
            D[i] = saturate_cast<T>(s0);
            SUM[i] = s0 - buf_ptr[0][i];
        }
    }
    return i;
}

template<typename ST, typename T>
inline int BlockSumCore(const T* S, ST* SUM, T* D, ST** buf_ptr, double scale,
     int widthcn, int i, int cn, int kheight, int kwidth)
{
    bool haveScale = scale != 1;
    switch(kwidth)
    {
        case 3:
            if( haveScale )
            {
                for( ; i < widthcn; i++ )
                {
                    ST Sp = (ST)S[i] + (ST)S[i+1*cn] + (ST)S[i+2*cn];
                    buf_ptr[kheight-1][i] = Sp;
                    ST s0 = SUM[i] + Sp;
                    D[i] = saturate_cast<T>(s0*scale);
                    SUM[i] = s0 - buf_ptr[0][i];
                }
            }
            else
            {
                for( ; i < widthcn; i++ )
                {
                    ST Sp = (ST)S[i] + (ST)S[i+1*cn] + (ST)S[i+2*cn];
                    buf_ptr[kheight-1][i] = Sp;
                    ST s0 = SUM[i] + Sp;
                    D[i] = saturate_cast<T>(s0);
                    SUM[i] = s0 - buf_ptr[0][i];
                }
            }
            break;
        case 5:
            if( haveScale )
            {
                for( ; i < widthcn; i++ )
                {
                    ST Sp = (ST)S[i] + (ST)S[i+1*cn] + (ST)S[i+2*cn] + (ST)S[i+3*cn] + (ST)S[i+4*cn];
                    buf_ptr[kheight-1][i] = Sp;
                    ST s0 = SUM[i] + Sp;
                    D[i] = saturate_cast<T>(s0*scale);
                    SUM[i] = s0 - buf_ptr[0][i];
                }
            }
            else
            {
                for( ; i < widthcn; i++ )
                {
                    ST Sp = (ST)S[i] + (ST)S[i+1*cn] + (ST)S[i+2*cn] + (ST)S[i+3*cn] + (ST)S[i+4*cn];
                    buf_ptr[kheight-1][i] = Sp;
                    ST s0 = SUM[i] + Sp;
                    D[i] = saturate_cast<T>(s0);
                    SUM[i] = s0 - buf_ptr[0][i];
                }
            }
            break;
        case 1:
            i = BlockSumCoreRow(S, SUM, D, buf_ptr, scale, widthcn, i, cn, kheight, 1);
            break;
        case 2:
            i = BlockSumCoreRow(S, SUM, D, buf_ptr, scale, widthcn, i, cn, kheight, 2);
            break;
        case 4:
            i = BlockSumCoreRow(S, SUM, D, buf_ptr, scale, widthcn, i, cn, kheight, 4);
            break;
        default:
            CV_Error_( cv::Error::StsNotImplemented,
                ("Unsupported kernel width (=%d)", kwidth));
            break;
    }
    return i;
}

uchar* alignCore(uchar* ptr) // align core region to VEC_ALIGN
{
    return alignPtr((uchar*)ptr, VEC_ALIGN);
}

template<typename ST, typename T>
void blockSum(const Mat& _src, Mat& _dst, Size ksize, Point anchor, const Size &wsz,
     const Point &ofs, double scale, int borderType, int sumType)
{
    int i, j;
    cv::utils::BufferArea area, area2;
    int* borderTab = 0;
    uchar* buf = 0, *constBorder = 0;
    uchar* border = 0, *inplaceSrc = 0, *inplaceDst = 0;
    std::vector<ST*> bufPtr;
    const uchar* src = _src.ptr();
    uchar* dst = _dst.ptr();
    Size wholeSize = wsz;
    Rect roi = Rect(ofs, _src.size());
    CV_Assert( roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 &&
        roi.x + roi.width <= wholeSize.width &&
        roi.y + roi.height <= wholeSize.height );

    size_t srcStep = _src.step;
    size_t dstStep = _dst.step;
    int srcType = _src.type();
    int cn = CV_MAT_CN(srcType);
    int sesz = (int)getElemSize(srcType);
    int besz = (int)getElemSize(sumType);
    int kwidth = ksize.width;
    int kheight = ksize.height;
    int borderLength = std::max(kwidth - 1, 1);
    int width = roi.width;
    int height = roi.height;
    int width1 = width + kwidth - 1;
    int height1 = height + kheight - 1;
    int xofs1 = std::min(roi.x, anchor.x);
    int dx1 = std::max(anchor.x - roi.x, 0);
    int dx2 = std::max(kwidth - anchor.x - 1 + roi.x + roi.width - wholeSize.width, 0);
    int dy2 = std::max(kheight - anchor.y - 1 + roi.y + roi.height - wholeSize.height, 0);
    int borderLeft = min(dx1, width)*cn;
    int bufWidth = (width1+borderLeft+VEC_ALIGN);
    int bufStep = bufWidth*besz;

    src -= xofs1*sesz;
    area.allocate(borderTab, borderLength*cn);
    area.allocate(buf, bufWidth*(kheight+1)*besz);
    area.allocate(constBorder, bufWidth*sesz);
    area.commit();
    area.zeroFill();

    // compute border tables
    if( dx1 > 0 || dx2 > 0 )
    {
        if( borderType != BORDER_CONSTANT )
        {
            int xofs1w = std::min(roi.x, anchor.x) - roi.x;
            int wholeWidth = wholeSize.width;
            int* btabx = borderTab;

            for( i = 0; i < dx1; i++ )
            {
                int p0 = (borderInterpolate(i-dx1, wholeWidth, borderType) + xofs1w)*cn;
                for( j = 0; j < cn; j++ )
                    btabx[i*cn + j] = p0 + j;
            }

            for( i = 0; i < dx2; i++ )
            {
                int p0 = (borderInterpolate(wholeWidth+i, wholeWidth, borderType) + xofs1w)*cn;
                for( j = 0; j < cn; j++ )
                    btabx[(i + dx1)*cn + j] = p0 + j;
            }
        }
    }


    area2.allocate(border, bufWidth*sesz);
    area2.allocate(inplaceDst, bufWidth*sesz);
    if( dy2 )
    {
        area2.allocate(inplaceSrc, dy2*bufWidth*sesz);
        area2.commit();
        if( borderType == BORDER_CONSTANT )
            memset(inplaceSrc, 0, dy2*bufWidth*sesz);
        else
        {
            for( int idx=0; idx < dy2; ++idx )
            {
                uchar* out = (uchar*)alignCore(&inplaceSrc[bufWidth*sesz*idx]);
                int rc = height1-(dy2-idx);
                int srcY = borderInterpolate(rc + roi.y - anchor.y, wholeSize.height, borderType);
                const uchar* inp = (uchar*)(src+(srcY-roi.y)*srcStep);
                memcpy(out, inp, width*sesz);
            }
        }
    }
    else
        area2.commit();

    bufPtr.resize(kheight);

    const T *S;
    const T* ref;
    T* D;
    T* R;
    ST** buf_ptr = &bufPtr[0];
    const int *btab = borderTab;
    T* C = (T*)alignCore(constBorder);
    ST* SUM = (ST*)alignCore(&buf[bufStep*kheight]);
    for( int k=0;k<kheight;k++ )
        buf_ptr[k] = (ST*)alignCore(&buf[bufStep*k]);

    for( int rc=0, bbi=0; rc < height1; rc++ )
    {
        int srcY = borderInterpolate(rc + roi.y - anchor.y, wholeSize.height, borderType);
        if( srcY < 0 ) // can happen only with constant border type
            ref = (const T*)(C);
        else if( rc >= height1-dy2 )
        {
            int idx = rc - (height1-dy2);
            ref = (const T*)alignCore(&inplaceSrc[bufWidth*sesz*idx]);
        }
        else
            ref = (const T*)(src+(srcY-roi.y)*(ptrdiff_t)srcStep);

        S = ref;
        R = (T*)alignPtr(border, VEC_ALIGN);
        if ( rc < (kheight-1) )
            D = (T*)alignCore(inplaceDst);
        else
            D = (T*)dst;
        for( int k=0; k < kheight; k++ )
            buf_ptr[k] = (ST*)alignCore(&buf[((bbi+k)%kheight)*bufStep]);

        int widthcn = width*cn;
        i=0;
        BlockSumBorderInplace<ST,T>(ref, S, R, btab, width + kwidth - 1 - dx1 - dx2, cn, dx1, dx2, borderType);
        BlockSumCore<ST,T>(R, SUM, D, buf_ptr, scale, widthcn, i, cn, kheight, kwidth);


        bbi++; bbi %= kheight;
        if( rc >= (kheight-1) )
            dst += dstStep;
    }
}

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

    CV_Error_( cv::Error::StsNotImplemented,
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

    if(scale==1)//box
    {
        if( ddepth == CV_8U && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_8U && sdepth == CV_16U )
            return makePtr<ColumnSum<SKIP_SCALING, ushort, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_8U && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_16U && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, ushort> >(ksize, anchor, scale);
        if( ddepth == CV_16U && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, ushort> >(ksize, anchor, scale);
        if( ddepth == CV_16S && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, short> >(ksize, anchor, scale);
        if( ddepth == CV_16S && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, short> >(ksize, anchor, scale);
        if( ddepth == CV_32S && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, int> >(ksize, anchor, scale);
        if( ddepth == CV_32F && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, float> >(ksize, anchor, scale);
        if( ddepth == CV_32F && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, float> >(ksize, anchor, scale);
        if( ddepth == CV_64F && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, double> >(ksize, anchor, scale);
        if( ddepth == CV_64F && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, double> >(ksize, anchor, scale);
    }
    else
    {
        if( ddepth == CV_8U && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_8U && sdepth == CV_16U )
            return makePtr<ColumnSum<APPLY_SCALING, ushort, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_8U && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_16U && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, ushort> >(ksize, anchor, scale);
        if( ddepth == CV_16U && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, ushort> >(ksize, anchor, scale);
        if( ddepth == CV_16S && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, short> >(ksize, anchor, scale);
        if( ddepth == CV_16S && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, short> >(ksize, anchor, scale);
        if( ddepth == CV_32S && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, int> >(ksize, anchor, scale);
        if( ddepth == CV_32F && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, float> >(ksize, anchor, scale);
        if( ddepth == CV_32F && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, float> >(ksize, anchor, scale);
        if( ddepth == CV_64F && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, double> >(ksize, anchor, scale);
        if( ddepth == CV_64F && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, double> >(ksize, anchor, scale);
    }

    CV_Error_( cv::Error::StsNotImplemented,
        ("Unsupported combination of sum format (=%d), and destination format (=%d)",
        sumType, dstType));
}

Ptr<BaseRowColumnFilter> getRowColumnSumFilter(int srcType, int dstType, int ksize, double scale)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);
    CV_Assert( CV_MAT_CN(srcType) == CV_MAT_CN(dstType) );

    if(sdepth == ddepth)
    {
        if( ksize == 3 )
        {
            if(scale==1)
            {
#if defined(__x86_64__) || defined(_M_X64)
                if( sdepth == CV_8U  )
                    return makePtr<Sum3x3<SKIP_SCALING, uchar, ushort, v_uint8, v_uint16> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum3x3<SKIP_SCALING, ushort, uint32_t, v_uint16, v_uint32> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum3x3<SKIP_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
#endif
                if( sdepth == CV_32S )
                    return makePtr<Sum3x3sameType<SKIP_SCALING,  int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F && checkHardwareSupport(CV_CPU_LOAD_AGU_GT_2) )
                    return makePtr<Sum3x3_64f<SKIP_SCALING, float, double> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                if( sdepth == CV_64F )
                    return makePtr<Sum3x3sameType<SKIP_SCALING, double,v_float64> >(ksize, scale);
#endif
            }
            else
            {
#if defined(__x86_64__) || defined(_M_X64)
                if( sdepth == CV_8U  )
                    return makePtr<Sum3x3_8u<APPLY_SCALING> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum3x3<APPLY_SCALING, ushort, uint32_t, v_uint16, v_uint32> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum3x3<APPLY_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
#endif
                if( sdepth == CV_32S )
                    return makePtr<Sum3x3sameType<APPLY_SCALING, int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F && checkHardwareSupport(CV_CPU_LOAD_AGU_GT_2) )
                    return makePtr<Sum3x3_64f<APPLY_SCALING, float, double> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                if( sdepth == CV_64F )
                    return makePtr<Sum3x3sameType<APPLY_SCALING, double, v_float64> >(ksize, scale);
#endif
            }
        }
        else if (ksize == 5)
        {
            if(scale==1)
            {
#if defined(__x86_64__) || defined(_M_X64)
                if( sdepth == CV_8U  )
                    return makePtr<Sum5x5<SKIP_SCALING, uchar, ushort, v_uint8, v_uint16, void> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum5x5<SKIP_SCALING, ushort, uint32_t, v_uint16, v_uint32, void> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum5x5<SKIP_SCALING, short, int32_t, v_int16, v_int32, void> >(ksize, scale);
#endif
                if( sdepth == CV_32S )
                    return makePtr<Sum5x5sameType<SKIP_SCALING, int, v_int32> >(ksize, scale); //intermediate stored can be stored in int64_t, but matching reference code to avoid output mismatch.
                if( sdepth == CV_32F && checkHardwareSupport(CV_CPU_LOAD_AGU_GT_2) )
                    return makePtr<Sum5x5_64f<SKIP_SCALING, float, double> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                    if( sdepth == CV_64F )
                    return makePtr<Sum5x5sameType<SKIP_SCALING, double, v_float64> >(ksize, scale);
#endif
            }
            else
            {
#if defined(__x86_64__) || defined(_M_X64)
                if( sdepth == CV_16U  )
                    return makePtr<Sum5x5<APPLY_SCALING, ushort, uint32_t, v_uint16, v_uint32, void> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum5x5<APPLY_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
#endif
                if( sdepth == CV_32S )
                    return makePtr<Sum5x5sameType<APPLY_SCALING, int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F && checkHardwareSupport(CV_CPU_LOAD_AGU_GT_2) )
                    return makePtr<Sum5x5_64f<APPLY_SCALING, float, double> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                if( sdepth == CV_64F )
                    return makePtr<Sum5x5sameType<APPLY_SCALING, double, v_float64> >(ksize, scale);
#endif
            }
        }
    }
    return NULL;
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

    Ptr<BaseRowColumnFilter> rowColumnFilter = nullptr;
    if(ksize.width == ksize.height &&
       (ksize.width ==3 || ksize.width ==5))
    {
        rowColumnFilter = getRowColumnSumFilter(srcType, dstType, ksize.width, normalize ? 1./(ksize.width*ksize.height) : 1);
    }
    return makePtr<FilterEngine>(Ptr<BaseFilter>(), rowFilter, columnFilter, rowColumnFilter,
           srcType, dstType, sumType, borderType );
}

void blockSum(const Mat& src, Mat& dst, Size ksize, Point anchor,
    const Size &wsz, const Point &ofs, bool normalize, int borderType)
{
    CV_INSTRUMENT_REGION();

    int cn = CV_MAT_CN(src.type()), sumType = CV_64F;
    sumType = CV_MAKETYPE( sumType, cn );

    CV_Assert( CV_MAT_CN(src.type()) == CV_MAT_CN(dst.type()) );
    int sdepth = CV_MAT_DEPTH(sumType), ddepth = CV_MAT_DEPTH(dst.type());

    if( anchor.x < 0 )
        anchor.x = ksize.width/2;

    if( anchor.y < 0 )
        anchor.y = ksize.height/2;

    double scale = normalize ? 1./(ksize.width*ksize.height) : 1;

    if( ddepth == CV_32F && sdepth == CV_64F )
        blockSum<double, float> (src, dst, ksize, anchor, wsz, ofs, scale, borderType, sumType);
    else if( ddepth == CV_64F && sdepth == CV_64F )
        blockSum<double, double> (src, dst, ksize, anchor, wsz, ofs, scale, borderType, sumType);
    else
        CV_Error_( cv::Error::StsNotImplemented,
            ("Unsupported combination of sum format (=%d), and destination format (=%d)",
            sumType, dst.type()));
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

    virtual void operator()(const uchar* src, uchar* dst, int width, int cn, bool) CV_OVERRIDE
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

    CV_Error_( cv::Error::StsNotImplemented,
              ("Unsupported combination of source format (=%d), and buffer format (=%d)",
               srcType, sumType));
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
