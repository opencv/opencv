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
// Copyright (C) 2025, Advanced Micro Devices, all rights reserved.
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
Ptr<BaseColumnFilter> getColumnSumFilter(int sumType, int dstType, int ksize, int anchor, double scale, int kernelSizeLog2);
Ptr<BaseRowColumnFilter> getRowColumnSumFilter(int srcType, int dstType, int ksize, double scale);
Ptr<FilterEngine> createBoxFilter(int srcType, int dstType, Size ksize,
                                  Point anchor, bool normalize, int borderType);
void blockSumInPlace(const Mat& src, Mat& dst, Size ksize, Point anchor,
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
            for(i = 0; i < cn ; i++ )
            {
                D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
            }
            if(processInnerRegion)
            {
                for(; i < len-cn ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
                }
            }
            for(i=len-cn; i < len ; i++ )
            {
                D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
            }

        }
        else if( ksize == 5 )
        {
            for(i = 0; i < min(len,2*cn) ; i++ )
            {
                D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
            }
            if(processInnerRegion)
            {
                for(; i < max(len-(2*cn),0) ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
                }
            }
            for(i = max(len-(2*cn),0); i < len ; i++ )
            {
                D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
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

enum { SKIP_SCALING = 0, APPLY_SCALING = 1, APPLY_SCALING_WITH_SHIFT = 2 };

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
        kernelSizeLog2 = (int)_scale; // used only if SCALE_T == APPLY_SCALING_WITH_SHIFT
        sumCount = 0;
        processInnerRegionPrevRow = true;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }
    T inline cast_scale(ST v)
    {
        if(SCALE_T == APPLY_SCALING)
            return saturate_cast<T>(v*scale);
        if(SCALE_T == APPLY_SCALING_WITH_SHIFT)
            return cast_scale_shift(v);
        else
            return saturate_cast<T>(v);
    }

private:
    T inline cast_scale_shift(int v) const { return saturate_cast<T>(v >> kernelSizeLog2); }
    T inline cast_scale_shift(short v) const { return saturate_cast<T>(v >> kernelSizeLog2); }
    T inline cast_scale_shift(ushort v) const { return saturate_cast<T>(v >> kernelSizeLog2); }
    T inline cast_scale_shift(uchar v) const { return saturate_cast<T>(v >> kernelSizeLog2); }
    T inline cast_scale_shift(float ) const { return 0; }  /* dummy */
    T inline cast_scale_shift(double ) const { return 0; } /* dummy */

public:
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

        int m = 0;
        for(; m < count; m++, src++ )
        {
            const ST* Sp = (const ST*)src[0];
            const ST* Sm = (const ST*)src[1-ksize];
            T* D = (T*)dst;

            i = 0;
            for( ; i < kcn; i++ )
            {
                ST s0 = SUM[i] + Sp[i];
                D[i] = cast_scale(s0);
                SUM[i] = s0 - Sm[i];
            }
            if(processInnerRegion)
            {
                for( ; i < width-kcn; i++ )
                {
                    ST s0 = SUM[i] + Sp[i];
                    D[i] = cast_scale(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }
            for(i = width-kcn; i < width; i++ )
            {
                ST s0 = SUM[i] + Sp[i];
                D[i] = cast_scale(s0);
                SUM[i] = s0 - Sm[i];
            }

            dst += dststep;
        }
        processInnerRegionPrevRow = processInnerRegion;
    }

    double scale;
    int sumCount;
    int kernelSizeLog2;
    bool processInnerRegionPrevRow;
    std::vector<ST> sum;
};

#if (CV_SIMD || CV_SIMD_SCALABLE)
v_float64 vx_setall(double value) {
    return vx_setall_f64(value);
}
v_uint32 vx_setall(unsigned value) {
    return vx_setall_u32(value);
}
v_int32 vx_setall(int value) {
    return vx_setall_s32(value);
}
v_uint16 vx_setall(ushort value) {
    return vx_setall_u16(value);
}
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
inline void v_expand(const v_float32 &a, v_float64 &b0, v_float64 &b1)
{
    b0 = v_cvt_f64(a);
    b1 = v_cvt_f64_high(a);
}
inline v_float32 v_pack(const v_float64 &a0, const v_float64 &a1)
{
    return v_cvt_f32(a0, a1);
}
#endif
#endif

template<int SCALE_T, typename ET, typename WET, typename VET, typename VFT>
struct Sum3x3 :
        public BaseRowColumnFilter
{
    Sum3x3( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
    }
    virtual void reset() CV_OVERRIDE { }

#if (CV_SIMD || CV_SIMD_SCALABLE)
    void inline loadRow(const ET* src, int cn, VFT &a0, VFT &a1, VFT &b0, VFT &b1, VFT &c0, VFT &c1)
    {
        const ET* src_ptr = src - cn;
        v_expand(vx_load(src_ptr), a0, a1);
        v_expand(vx_load(src_ptr+cn),      b0, b1);
        v_expand(vx_load(src_ptr+cn*2), c0, c1);
    }
    void inline addRow(const VFT &a0, const VFT &a1, const VFT &b0, const VFT &b1, VFT &r0, VFT &r1)
    {
        r0 = v_add(r0,  v_add(a0, b0));
        r1 = v_add(r1,  v_add(a1, b1));
    }
    void inline loadRowAdd(const ET* src, int VECSZ,
                           int cn, VFT &r0, VFT &r1, VFT &r0v1, VFT &r1v1, VFT &r0v2, VFT &r1v2, VFT &r0v3, VFT &r1v3)
    {
        VFT a0, a1, b0, b1;
        VFT a0v1, a1v1, b0v1, b1v1;
        VFT a0v2, a1v2, b0v2, b1v2;
        VFT a0v3, a1v3, b0v3, b1v3;
        loadRow(src, cn, a0, a1, b0, b1, r0, r1);
        loadRow(src+VECSZ, cn, a0v1, a1v1, b0v1, b1v1, r0v1, r1v1);
        loadRow(src+(VECSZ*2), cn, a0v2, a1v2, b0v2, b1v2, r0v2, r1v2);
        loadRow(src+(VECSZ*3), cn, a0v3, a1v3, b0v3, b1v3, r0v3, r1v3);
        addRow(a0, a1, b0, b1, r0, r1);
        addRow(a0v1, a1v1, b0v1, b1v1, r0v1, r1v1);
        addRow(a0v2, a1v2, b0v2, b1v2, r0v2, r1v2);
        addRow(a0v3, a1v3, b0v3, b1v3, r0v3, r1v3);
    }
    void inline scaleVal3x3(VFT &b0, const VFT &v_32768, const VFT &v_mulFactor)
    {
        if (SCALE_T==APPLY_SCALING)
        {
            if (std::is_floating_point<WET>::value)
                b0 = v_mul(b0, vx_setall((WET)scale));
            else
            {
                if (std::is_same<ET, uchar>::value || std::is_same<ET, char>::value)
                {
                    VFT bsub = v_shr<1>(b0); // 1/2
                    b0 = v_shr<8>(v_sub(v_mul(b0, v_mulFactor), bsub));
                }
                else if (std::is_same<ET, ushort>::value)
                    b0 = v_shr<16>(v_sub(v_mul(b0, v_mulFactor), v_32768));
                else if(std::is_same<ET, short>::value)
                    b0 = v_shr<16>(v_add(v_mul(b0, v_mulFactor), v_32768));
            }
        }
    }
#endif

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;

        int idst = 1;
        int offset = 1;
        int dstOffset = 1;
        int v = idst - dstOffset;
        int len = (width - offset) * cn;
        int x = offset * cn;
        int maxRow = height-2;
        const int ETSZ = sizeof(ET);
        const int src_inc = src_stride / ETSZ;
        int j = v;
        const ET* src_ptr = (const ET*)(src + j * src_stride);

#if (CV_SIMD || CV_SIMD_SCALABLE)
        WET val_32768 = 32768;
        VFT v_32768 = vx_setall(val_32768);
        VFT v_mulFactor;
        if (SCALE_T==APPLY_SCALING)
        {
            if (std::is_floating_point<WET>::value)
                v_mulFactor = vx_setall((WET)scale);
            else
            {
                WET val = 29;
                if (std::is_same<ET, uchar>::value || std::is_same<ET, char>::value)
                    val = 29;
                else if (std::is_same<ET, ushort>::value || std::is_same<ET, short>::value)
                    val = 7282;
                v_mulFactor = vx_setall(val);
            }
        }
        const int VECSZ = VTraits<VET>::vlanes();
        const int VECSZ_2 = VECSZ << 1;
        const int VECSZ_3 = VECSZ * 3;
        const int VECSZ_4 = VECSZ << 2;

        VFT b00, b01, b00v1, b01v1, b00v2, b01v2, b00v3, b01v3;
        VFT b10, b11, b10v1, b11v1, b10v2, b11v2, b10v3, b11v3;
        VFT b20, b21, b20v1, b21v1, b20v2, b21v2, b20v3, b21v3;

        for (; x < len; x += (VECSZ_4))
        {
            if ( x > len - VECSZ_4 )
            {
                if (x == cn*offset || src == dst)
                    break;
                x = len - VECSZ_4;
            }
            int idst_ = idst;
            j = v;
            const ET* src_0 = src_ptr + x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;

            //row 0 and 1
            loadRowAdd(src_0, VECSZ, cn, b00, b01, b00v1, b01v1, b00v2, b01v2, b00v3, b01v3 );
            loadRowAdd(src_1, VECSZ, cn, b10, b11, b10v1, b11v1, b10v2, b11v2, b10v3, b11v3 );

            for (; j < maxRow; j++, idst_++)
            {
                ET* dstx = (ET*)(dst + (idst_ * dst_stride));
                dstx += x;
                b00   = v_add(b00, b10);
                b00v1 = v_add(b00v1, b10v1);
                b00v2 = v_add(b00v2, b10v2);
                b00v3 = v_add(b00v3, b10v3);

                b01   = v_add(b01, b11);
                b01v1 = v_add(b01v1, b11v1);
                b01v2 = v_add(b01v2, b11v2);
                b01v3 = v_add(b01v3, b11v3);

                loadRowAdd(src_2, VECSZ, cn, b20, b21, b20v1, b21v1, b20v2, b21v2, b20v3, b21v3);
                src_2 += src_inc;

                b00   = v_add(b20, b00);
                b00v1 = v_add(b20v1, b00v1);
                b00v2 = v_add(b20v2, b00v2);
                b00v3 = v_add(b20v3, b00v3);
                b01   = v_add(b21, b01);
                b01v1 = v_add(b21v1, b01v1);
                b01v2 = v_add(b21v2, b01v2);
                b01v3 = v_add(b21v3, b01v3);
                if (SCALE_T==APPLY_SCALING)
                {
                    scaleVal3x3(b00, v_32768, v_mulFactor);
                    scaleVal3x3(b00v1, v_32768, v_mulFactor);
                    scaleVal3x3(b00v2, v_32768, v_mulFactor);
                    scaleVal3x3(b00v3, v_32768, v_mulFactor);
                    scaleVal3x3(b01, v_32768, v_mulFactor);
                    scaleVal3x3(b01v1, v_32768, v_mulFactor);
                    scaleVal3x3(b01v2, v_32768, v_mulFactor);
                    scaleVal3x3(b01v3, v_32768, v_mulFactor);
                }
                v_store(dstx, v_pack(b00, b01));
                v_store(dstx + VECSZ, v_pack(b00v1, b01v1));
                v_store(dstx + VECSZ_2, v_pack(b00v2, b01v2));
                v_store(dstx + VECSZ_3, v_pack(b00v3, b01v3));

                b00 = b10; b01 = b11;
                b10 = b20; b11 = b21;

                b00v1 = b10v1; b01v1 = b11v1;
                b10v1 = b20v1; b11v1 = b21v1;

                b00v2 = b10v2; b01v2 = b11v2;
                b10v2 = b20v2; b11v2 = b21v2;

                b00v3 = b10v3; b01v3 = b11v3;
                b10v3 = b20v3; b11v3 = b21v3;
            }
        }
#endif
        for (; x < len; x++)
        {
            int idst_ = idst;
            j = v;
            WET b0, b1, b2;
            const ET* src_0 = src_ptr + x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;
            b0 = src_0[-cn] + src_0[0] + src_0[cn];
            b1 = src_1[-cn] + src_1[0] + src_1[cn];

            for (; j < maxRow; j++, idst_++)
            {
                ET* dstx = (ET*)(dst + (idst_ * dst_stride));
                b2 = src_2[-cn] + src_2[0] + src_2[cn];
                src_2 += src_inc;
                if (SCALE_T==APPLY_SCALING)
                {
                    dstx[x] = saturate_cast<ET>((b1 + b0 + b2 )* _scale);
                }
                else
                {
                    dstx[x] = saturate_cast<ET>(b1 + b0 + b2);
                }
                b0 = b1;
                b1 = b2;
            }
        }
    }

    double scale;
};

//float
template<int SCALE_T, typename ET, typename WET>
struct Sum3x3_64f :
        public BaseRowColumnFilter
{
    Sum3x3_64f( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
    }
    virtual void reset() CV_OVERRIDE { }

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;
        int idst = 1;
        int offset = 1;
        int idstOffset = 1;
        int v = idst - idstOffset;
        int len = (width - offset) * cn;
        int x = offset * cn;
        int maxRow = height-2;
        const int ETSZ = sizeof(ET);
        const int src_inc = src_stride / ETSZ;
        const int dst_inc = dst_stride / ETSZ;
        int j = v;

        if(sum.size() < (size_t)len)
            sum.resize(len);
        if(rowSum.size() < (size_t)(len*2))
            rowSum.resize(len*2);

        int idst_ = idst;
        const ET* src_j0 = (const ET*)(src + j * src_stride);
        const ET* src_j1 = (const ET*)(src + (j+1) * src_stride);
        const ET* src_j2 = (const ET*)(src + (j+2) * src_stride);
        ET* dstx = (ET*)(dst + (idst_ * dst_stride));
        for (; j < min(maxRow,v+1); j++, idst_++) //1st Row
        {
            x = offset * cn;
            int x_st = 0;
            const ET* src_0 = src_j0 + x;
            const ET* src_1 = src_j1 + x;
            const ET* src_2 = src_j2 + x;
            for (; x < len; x++, x_st++)
            {
                WET b0, b1, b2;
                b0 = src_0[-cn] + src_0[0] + src_0[cn];
                b1 = src_1[-cn] + src_1[0] + src_1[cn];
                b2 = src_2[-cn] + src_2[0] + src_2[cn];

                WET bsum = b1 + b2;
                sum[x_st]          = bsum;
                rowSum[x_st]       = b1;
                rowSum[x_st + len] = b2;
                if (SCALE_T==APPLY_SCALING)
                    dstx[x] = saturate_cast<ET>((b0 + bsum)* _scale);
                else
                    dstx[x] = saturate_cast<ET>(b0 + bsum);
                src_0++;
                src_1++;
                src_2++;
            }
            src_j0 += src_inc;
            src_j1 += src_inc;
            src_j2 += src_inc;
            dstx += dst_inc;
        }

        int r0_idx = 0;
        for (; j < maxRow; j++, idst_++)
        {
            x = offset * cn;
            int x_st = 0;
            int idx_offset = r0_idx * len;

            for (; x < len; x++, x_st++)
            {
                WET b0, b2;
                const ET* src_2 = src_j2 + x;

                b0 = rowSum[x_st + idx_offset];
                b2 = src_2[-cn] + src_2[0] + src_2[cn];
                rowSum[x_st + idx_offset] = b2;

                WET bsum = sum[x_st] + b2;
                if (SCALE_T==APPLY_SCALING)
                    dstx[x] = saturate_cast<ET>(bsum* _scale);
                else
                    dstx[x] = saturate_cast<ET>(bsum);
                sum[x_st] = bsum - b0;
            }
            src_j0 += src_inc;
            src_j1 += src_inc;
            src_j2 += src_inc;
            dstx += dst_inc;

            r0_idx++;
            if (r0_idx > 1) r0_idx = 0;
        }
    }
    double scale;
    std::vector<WET>sum;
    std::vector<WET>rowSum;
};

template<int SCALE_T, typename ET, typename VET>
struct Sum3x3sameType :
        public BaseRowColumnFilter
{
    Sum3x3sameType( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
    }
    virtual void reset() CV_OVERRIDE { }

#if (CV_SIMD || CV_SIMD_SCALABLE)
    void inline loadRow(const ET* src, int cn, VET &a,  VET &b,  VET &c)
    {
        a = vx_load(src - cn);
        b = vx_load(src);
        c = vx_load(src + cn);
    }
    void inline addRow(const VET &a, const VET &b, VET &r)
    {
        r = v_add(r, v_add(a, b));
    }
    void inline loadRowAdd(const ET* src_0, const ET* src_0v1, const ET* src_0v2, const ET* src_0v3,
                           int cn, VET &r, VET &rv1, VET &rv2, VET &rv3)
    {
        VET a, b;
        VET av1, bv1;
        VET av2, bv2;
        VET av3, bv3;
        loadRow(src_0, cn, a, b, r);
        loadRow(src_0v1, cn, av1, bv1, rv1);
        loadRow(src_0v2, cn, av2, bv2, rv2);
        loadRow(src_0v3, cn, av3, bv3, rv3);
        addRow(a, b, r);
        addRow(av1, bv1, rv1);
        addRow(av2, bv2, rv2);
        addRow(av3, bv3, rv3);
    }

    void inline scaleVal3x3sameType(VET &b0, const v_float32& _v_scale)
    {
        if (SCALE_T==APPLY_SCALING)
        {
            scaleVal3x3sameTypeImpl(b0, _v_scale);
        }
    }

    template<typename T = ET>
    typename std::enable_if<std::is_floating_point<T>::value, void>::type
    inline scaleVal3x3sameTypeImpl(VET &b0, const v_float32& /*_v_scale*/)
    {
        b0 = v_mul(b0, vx_setall((ET)scale));
    }

    template<typename T = ET>
    typename std::enable_if<!std::is_floating_point<T>::value &&
                           (std::is_same<T, uint>::value || std::is_same<T, int>::value), void>::type
    inline scaleVal3x3sameTypeImpl(VET &b0, const v_float32& _v_scale)
    {
        b0 = v_round(v_mul(v_cvt_f32(b0), _v_scale));
    }

    template<typename T = ET>
    typename std::enable_if<!std::is_floating_point<T>::value &&
                           !(std::is_same<T, uint>::value || std::is_same<T, int>::value), void>::type
    inline scaleVal3x3sameTypeImpl(VET &, const v_float32&)
    {
        // No scaling for other integer types
    }
#endif

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;

        int idst = 1;
        int offset = 1;
        int idstOffset = 1;
        int v = idst - idstOffset;
        int len = (width - offset) * cn;
        int x = offset * cn;
        int maxRow = height-2;
        const int ETSZ = sizeof(ET);
        const int src_inc = src_stride / ETSZ;
        int j = v;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 _v_scale = vx_setall_f32((float)_scale);
        const int VECSZ = VTraits<VET>::vlanes();
        const int VECSZ_2 = VECSZ<<1;
        const int VECSZ_3 = VECSZ*3;
        const int VECSZ_4 = VECSZ<<2;

        VET b0, b1, b2;
        VET b0v1, b1v1, b2v1;
        VET b0v2, b1v2, b2v2;
        VET b0v3, b1v3, b2v3;

        const ET* src_ptr = (const ET*)(src + j * src_stride);
        for (; x < len; x += (VECSZ_4))
        {
            if ( x > len - VECSZ_4 )
            {
                if (x == cn*offset || src == dst)
                    break;
                x = len - VECSZ_4;
            }
            int idst_ = idst;
            j = v;
            const ET* src_0 = src_ptr +x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;
            const ET* src_0v1 = src_0 + VECSZ;
            const ET* src_1v1 = src_1 + VECSZ;
            const ET* src_2v1 = src_2 + VECSZ;
            const ET* src_0v2 = src_0 + VECSZ_2;
            const ET* src_1v2 = src_1 + VECSZ_2;
            const ET* src_2v2 = src_2 + VECSZ_2;
            const ET* src_0v3 = src_0 + VECSZ_3;
            const ET* src_1v3 = src_1 + VECSZ_3;
            const ET* src_2v3 = src_2 + VECSZ_3;

            //row 0 and 1
            loadRowAdd(src_0, src_0v1, src_0v2, src_0v3, cn, b0, b0v1, b0v2, b0v3);
            loadRowAdd(src_1, src_1v1, src_1v2, src_1v3, cn, b1, b1v1, b1v2, b1v3);

            for (; j < maxRow; j++, idst_++)
            {
                ET* dstx = (ET*)(dst + (idst_ * dst_stride));
                loadRowAdd(src_2, src_2v1, src_2v2, src_2v3, cn, b2, b2v1, b2v2, b2v3);
                src_2 += src_inc;
                src_2v1 += src_inc;
                src_2v2 += src_inc;
                src_2v3 += src_inc;

                b0   = v_add(b2, v_add(b0, b1));
                b0v1 = v_add(b2v1, v_add(b0v1, b1v1));
                b0v2 = v_add(b2v2, v_add(b0v2, b1v2));
                b0v3 = v_add(b2v3, v_add(b0v3, b1v3));
                if (SCALE_T==APPLY_SCALING)
                {
                    scaleVal3x3sameType(b0, _v_scale);
                    scaleVal3x3sameType(b0v1, _v_scale);
                    scaleVal3x3sameType(b0v2, _v_scale);
                    scaleVal3x3sameType(b0v3, _v_scale);
                }
                vx_store(dstx + x, b0);
                vx_store(dstx + x + VECSZ, b0v1);
                vx_store(dstx + x + VECSZ_2, b0v2);
                vx_store(dstx + x + VECSZ_3, b0v3);
                b0 = b1; b0v1 = b1v1; b0v2 = b1v2; b0v3 = b1v3;
                b1 = b2; b1v1 = b2v1; b1v2 = b2v2; b1v3 = b2v3;
            }
        }
#endif
        for (; x < len; x++)
        {
            int idst_ = idst;
            j = v;
            ET a0, a1, a2;
            const ET* src_0 = ((const ET*)(src + j * src_stride))+x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;
            a0 = src_0[-cn] + src_0[0] + src_0[cn];
            a1 = src_1[-cn] + src_1[0] + src_1[cn];

            for (; j < maxRow; j++, idst_++)
            {
                ET* dstx = (ET*)(dst + (idst_ * dst_stride));
                a2 = src_2[-cn] + src_2[0] + src_2[cn];
                src_2 += src_inc;
                if (SCALE_T==APPLY_SCALING)
                    dstx[x] = saturate_cast<ET>((a1 + a0 + a2 )* _scale);
                else
                    dstx[x] = saturate_cast<ET>(a1 + a0 + a2);
                a0 = a1;
                a1 = a2;
            }
        }
    }

    double scale;
};


template<int SCALE_T, typename ET, typename WET, typename VET, typename VFT, typename Derived = void>
struct Sum5x5 :
        public BaseRowColumnFilter
{
    Sum5x5( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
    }
    virtual void reset() CV_OVERRIDE { }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    void inline loadRow(const ET* src, int cn, VFT &a0, VFT &a1, VFT &b0, VFT &b1,
                            VFT &c0, VFT &c1, VFT &d0, VFT &d1, VFT &e0, VFT &e1)
    {
        const ET* src_ptr = src - cn*2;
        v_expand(vx_load(src_ptr), a0, a1);
        v_expand(vx_load(src_ptr + cn), b0, b1);
        v_expand(vx_load(src_ptr + cn*2), c0, c1);
        v_expand(vx_load(src_ptr + cn*3), d0, d1);
        v_expand(vx_load(src_ptr + cn*4), e0, e1);
    }
    void inline addRow(const VFT &a0, const VFT &a1, const VFT &b0, const VFT &b1,
                          const VFT &c0, const VFT &c1, const VFT &d0, const VFT &d1, VFT &r0, VFT &r1)
    {
        r0 = v_add(r0, v_add(d0, v_add(c0, v_add(a0, b0))));
        r1 = v_add(r1, v_add(d1, v_add(c1, v_add(a1, b1))));
    }
    void inline loadRowAdd(const ET* src_0, int cn, VFT &r0, VFT &r1)
    {
        VFT a0, a1, b0, b1, c0, c1, d0, d1;
        loadRow(src_0, cn, a0, a1, b0, b1, c0, c1, d0, d1, r0, r1);
        addRow(a0, a1, b0, b1, c0, c1, d0, d1, r0, r1);
    }
    VFT inline scaleVal5x5(VFT &b0, const VFT &v_mulFactor)
    {
        if (std::is_floating_point<WET>::value)
            return v_mul(b0, vx_setall((WET)scale));
        else
        {
            VFT berr = v_shr<2>(b0); // 1/4
            if (std::is_same<ET, uchar>::value || std::is_same<ET, char>::value)
                return v_shr<8>(v_add(v_mul(b0, v_mulFactor), berr));
            else if (std::is_same<ET, ushort>::value || std::is_same<ET, short>::value)
                return v_shr<15>(v_sub(v_mul(b0, v_mulFactor), berr));
        }
    }
#endif

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;
        int idst = 2;
        int offset = 2;
        int idstOffset = 2;
        int v = idst - idstOffset;
        int len = (width - offset) * cn;
        int x = offset * cn;
        int maxRow = height-4;
        const int ETSZ = sizeof(ET);
        const int src_inc = src_stride / ETSZ;
        const int dst_inc = dst_stride / ETSZ;
        int cn2 = cn*2;
        int j = v;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        VFT v_896;
        WET val_896 = 896;
        v_896 = vx_setall(val_896);
        VFT v_mulFactor;
        if (SCALE_T==APPLY_SCALING)
        {
            if (std::is_floating_point<WET>::value)
                v_mulFactor = vx_setall((WET)scale);
            else
            {
                WET val = 10;
                if (std::is_same<ET, uchar>::value || std::is_same<ET, char>::value)
                    val = 10;
                else if (std::is_same<ET, ushort>::value || std::is_same<ET, short>::value)
                    val = 1311;
                v_mulFactor = vx_setall(val);
            }
        }

        const ET* src_ptr = (const ET*)(src + j * src_stride);
        const int VECSZ = VTraits<VET>::vlanes();

        VFT b00, b01, b10, b11, b20, b21, b30, b31, b40, b41;

        for (; x < len; x += VECSZ)
        {
            if (x > len - VECSZ)
            {
                if (x == cn*offset || src == dst)
                    break;
                x = len - VECSZ;
            }
            int idst_ = idst;
            j = v;
            const ET* src_0 = src_ptr + x;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;
            const ET* src_3 = src_2 + src_inc;
            const ET* src_4 = src_3 + src_inc;

            loadRowAdd(src_0, cn, b00, b01);
            loadRowAdd(src_1, cn, b10, b11);
            loadRowAdd(src_2, cn, b20, b21);
            loadRowAdd(src_3, cn, b30, b31);
            ET* dstx = (ET*)(dst + (idst_ * dst_stride));
            dstx += x;

            VFT bsum0, bsum1, b0, b1;
            bsum0   = v_add(v_add(b10, b00), v_add(b20, b30));
            bsum1   = v_add(v_add(b11, b01), v_add(b21, b31));
            for (; j < maxRow; j++, idst_++)
            {
                loadRowAdd(src_4, cn, b40, b41);
                src_4 += src_inc;
                b0 = bsum0 = v_add(bsum0, b40);
                b1 = bsum1 = v_add(bsum1, b41);
                if (SCALE_T==APPLY_SCALING)
                {
                    b0 = scaleVal5x5(bsum0, v_mulFactor);
                    b1 = scaleVal5x5(bsum1, v_mulFactor);
                }
                v_store(dstx, v_pack(b0, b1));
                bsum0 = v_sub(bsum0, b00);
                bsum1 = v_sub(bsum1, b01);

                b00 = b10; b01 = b11;
                b10 = b20; b11 = b21;
                b20 = b30; b21 = b31;
                b30 = b40; b31 = b41;
                dstx += dst_inc;
            }
        }
#endif
        for (; x < len; x++)
        {
            int idst_ = idst;
            j = v;
            const ET* src_0 = ((const ET*)(src + j * src_stride))+x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;
            const ET* src_3 = src_2 + src_inc;
            const ET* src_4 = src_3 + src_inc;
            WET b0, b1, b2, b3, b4;
            b0 = (WET)src_0[-cn2] + (WET)src_0[-cn] + (WET)src_0[0] + (WET)src_0[cn] + (WET)src_0[cn2];
            b1 = (WET)src_1[-cn2] + (WET)src_1[-cn] + (WET)src_1[0] + (WET)src_1[cn] + (WET)src_1[cn2];
            b2 = (WET)src_2[-cn2] + (WET)src_2[-cn] + (WET)src_2[0] + (WET)src_2[cn] + (WET)src_2[cn2];
            b3 = (WET)src_3[-cn2] + (WET)src_3[-cn] + (WET)src_3[0] + (WET)src_3[cn] + (WET)src_3[cn2];
            ET* dstx = (ET*)(dst + (idst_ * dst_stride));
            dstx += x;

            WET bsum = b0 + b1 + b2 + b3;
            for (; j < maxRow; j++, idst_++)
            {
                b4 = (WET)src_4[-cn2] + (WET)src_4[-cn] + (WET)src_4[0] + (WET)src_4[cn] + (WET)src_4[cn2];
                src_4 += src_inc;
                bsum = bsum + b4;
                if (SCALE_T)
                    *dstx = saturate_cast<ET>(bsum * _scale);
                else
                    *dstx = saturate_cast<ET>(bsum);
                bsum = bsum - b0;
                b0 = b1;
                b1 = b2;
                b2 = b3;
                b3 = b4;
                dstx += dst_inc;
            }
        }
    }
    double scale;
};

template<int SCALE_T, typename ET, typename WET>
struct Sum5x5_64f :
        public BaseRowColumnFilter
{
    Sum5x5_64f( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
    }
    virtual void reset() CV_OVERRIDE { }

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;
        int idst = 2;
        int offset = 2;
        int idstOffset = 2;
        int v = idst - idstOffset;
        int len = (width - offset) * cn;
        int x = offset * cn;
        int maxRow = height-4;
        const int ETSZ = sizeof(ET);
        const int src_inc = src_stride / ETSZ;
        const int dst_inc = dst_stride / ETSZ;
        int cn2 = cn*2;
        int j = v;

        if(sum.size() < (size_t)len)
            sum.resize(len);
        if(rowSum.size() < (size_t)(len*4))
            rowSum.resize(len*4);

        int idst_ = idst;
        j = v;
        const uchar * src_j = src + j * src_stride;
        const ET* src_j0 = (const ET*)(src_j);
        const ET* src_j1 = src_j0 + src_inc;
        const ET* src_j2 = src_j1 + src_inc;
        const ET* src_j3 = src_j2 + src_inc;
        const ET* src_j4 = src_j3 + src_inc;
        ET* dstx = (ET*)(dst + (idst_ * dst_stride));
        for (; j < min(maxRow,v+1); j++, idst_++) //1st Row
        {
            x = offset * cn;

            int x_st = 0;
            for (; x < len; x++, x_st++)
            {
                WET b0, b1, b2, b3, b4;
                const ET* src_0 = src_j0 + x;
                const ET* src_1 = src_j1 + x;
                const ET* src_2 = src_j2 + x;
                const ET* src_3 = src_j3 + x;
                const ET* src_4 = src_j4 + x;

                b0 = src_0[-cn2] + src_0[-cn] + src_0[0] + src_0[cn] + src_0[cn2];
                b1 = src_1[-cn2] + src_1[-cn] + src_1[0] + src_1[cn] + src_1[cn2];
                b2 = src_2[-cn2] + src_2[-cn] + src_2[0] + src_2[cn] + src_2[cn2];
                b3 = src_3[-cn2] + src_3[-cn] + src_3[0] + src_3[cn] + src_3[cn2];
                b4 = src_4[-cn2] + src_4[-cn] + src_4[0] + src_4[cn] + src_4[cn2];

                WET bsum = b1 + b2 + b3 + b4;
                sum[x_st] = bsum;
                rowSum[x_st] = b1;
                rowSum[x_st + len] = b2;
                rowSum[x_st + len*2] = b3;
                rowSum[x_st + len*3] = b4;

                if (SCALE_T==APPLY_SCALING)
                    dstx[x] = saturate_cast<ET>((b0 + bsum)* _scale);
                else
                    dstx[x] = saturate_cast<ET>(b0 + bsum);
            }
            src_j0 += src_inc;
            src_j1 += src_inc;
            src_j2 += src_inc;
            src_j3 += src_inc;
            src_j4 += src_inc;
            dstx += dst_inc;
        }

        int r0_idx = 0;
        for (; j < maxRow; j++, idst_++)
        {
            x = offset * cn;
            int x_st = 0;
            int idx_offset = r0_idx * len;

            for (; x < len; x++, x_st++)//vectorization of this code results in lower performance
            {
                WET b0, b4;
                const ET* src_4 = src_j4 + x;

                b0 = rowSum[x_st + idx_offset];
                b4 = src_4[-cn2] + src_4[-cn] + src_4[0] + src_4[cn] + src_4[cn2];
                rowSum[x_st + idx_offset] = b4;

                WET bsum = sum[x_st] + b4;
                if (SCALE_T==APPLY_SCALING)
                    dstx[x] = saturate_cast<ET>(bsum* _scale);
                else
                    dstx[x] = saturate_cast<ET>(bsum);
                sum[x_st] = bsum - b0;
            }

            r0_idx++;
            if (r0_idx > 3) r0_idx = 0;

            src_j0 += src_inc;
            src_j1 += src_inc;
            src_j2 += src_inc;
            src_j3 += src_inc;
            src_j4 += src_inc;
            dstx += dst_inc;
        }
    }
    double scale;
    std::vector<WET>sum;
    std::vector<WET>rowSum;
};

template<int SCALE_T, typename ET,  typename VET>
struct Sum5x5sameType :
        public BaseRowColumnFilter
{
    Sum5x5sameType( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
    }
    virtual void reset() CV_OVERRIDE { }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    void inline loadRow(const ET* src, int cn, VET &a, VET &b, VET &c, VET &d, VET &e)
    {
        a = vx_load(src - cn*2);
        b = vx_load(src - cn);
        c = vx_load(src);
        d = vx_load(src + cn);
        e = vx_load(src + cn*2);
    }

    void inline loadRowAdd(const ET* src, int cn, VET &r)
    {
        VET a, b, c, d, e;
        loadRow(src, cn, a, b, c, d, e);
        r = v_add(v_add(v_add(a, b), v_add(c, d)), e);
    }

    void inline loadRowAdd(const ET* src_0, int VECSZ, int cn, VET &r, VET &r_v1, VET &r_v2, VET &r_v3)
    {
        VET a, b, c, d, e;
        VET a_v1, b_v1, c_v1, d_v1, e_v1;
        VET a_v2, b_v2, c_v2, d_v2, e_v2;
        VET a_v3, b_v3, c_v3, d_v3, e_v3;

        loadRow(src_0, cn, a, b, c, d, e);
        loadRow(src_0 + VECSZ, cn, a_v1, b_v1, c_v1, d_v1, e_v1);
        loadRow(src_0 + VECSZ*2, cn, a_v2, b_v2, c_v2, d_v2, e_v2);
        loadRow(src_0 + VECSZ*3, cn, a_v3, b_v3, c_v3, d_v3, e_v3);

        r = v_add(v_add(v_add(a, b), v_add(c, d)), e);
        r_v1 = v_add(v_add(v_add(a_v1, b_v1), v_add(c_v1, d_v1)), e_v1);
        r_v2 = v_add(v_add(v_add(a_v2, b_v2), v_add(c_v2, d_v2)), e_v2);
        r_v3 = v_add(v_add(v_add(a_v3, b_v3), v_add(c_v3, d_v3)), e_v3);
    }

    void inline scaleValsameType(VET &b0, const v_float32& _v_scale)
    {
        if (SCALE_T==APPLY_SCALING)
        {
            scaleValsameTypeImpl(b0, _v_scale);
        }
    }

    template<typename T = ET>
    typename std::enable_if<std::is_floating_point<T>::value, void>::type
    inline scaleValsameTypeImpl(VET &b0, const v_float32& /*_v_scale*/)
    {
        b0 = v_mul(b0, vx_setall((ET)scale));
    }

    template<typename T = ET>
    typename std::enable_if<!std::is_floating_point<T>::value &&
                           (std::is_same<T, uint>::value || std::is_same<T, int>::value), void>::type
    inline scaleValsameTypeImpl(VET &b0, const v_float32& _v_scale)
    {
        b0 = v_round(v_mul(v_cvt_f32(b0), _v_scale));
    }

    template<typename T = ET>
    typename std::enable_if<!std::is_floating_point<T>::value &&
                           !(std::is_same<T, uint>::value || std::is_same<T, int>::value), void>::type
    inline scaleValsameTypeImpl(VET &, const v_float32& )
    {
        // No scaling for other integer types
    }
#endif
    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;
        int idst = 2;

        int offset = 2;
        int idstOffset = 2;
        int v = idst - idstOffset;
        int j = v;
        int len = (width - offset) * cn;
        int x = offset * cn;
        int maxRow = height-4;
        const int ETSZ = sizeof(ET);
        const int src_inc = src_stride / ETSZ;
        int cn2 = cn*2;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 _v_scale = vx_setall_f32((float)_scale);
        const int VECSZ = VTraits<VET>::vlanes();
        const int VECSZ_2 = VECSZ<<1;
        const int VECSZ_3 = VECSZ*3;
        const int VECSZ_4 = VECSZ<<2;
        for (; x < len; x += VECSZ_4)
        {
            if (x > len - VECSZ_4)
            {
                if (x == cn*offset || src == dst)
                    break;
                x = len - VECSZ_4;
            }
            int idst_ = idst;
            j = v;
            const ET* src_0 = ((const ET*)(src + j * src_stride)) + x;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;
            const ET* src_3 = src_2 + src_inc;
            const ET* src_4 = src_3 + src_inc;

            VET b0, b1, b2, b3;
            VET b0v1, b1v1, b2v1, b3v1;
            VET b0v2, b1v2, b2v2, b3v2;
            VET b0v3, b1v3, b2v3, b3v3;
            VET sum, sumv1, sumv2, sumv3;

            // row 0 to 3
            loadRowAdd(src_0, VECSZ, cn, b0, b0v1, b0v2, b0v3);
            loadRowAdd(src_1, VECSZ, cn, b1, b1v1, b1v2, b1v3);
            loadRowAdd(src_2, VECSZ, cn, b2, b2v1, b2v2, b2v3);
            loadRowAdd(src_3, VECSZ, cn, b3, b3v1, b3v2, b3v3);

            sum = v_add(v_add(b1, b0), v_add(b2, b3));
            sumv1 = v_add(v_add(b1v1, b0v1), v_add(b2v1, b3v1));
            sumv2 = v_add(v_add(b1v2, b0v2), v_add(b2v2, b3v2));
            sumv3 = v_add(v_add(b1v3, b0v3), v_add(b2v3, b3v3));

            for (; j < maxRow; j++, idst_++)
            {
                ET* dstx = (ET*)(dst + (idst_ * dst_stride));
                VET b4, b4v1, b4v2, b4v3;
                // row 4
                loadRowAdd(src_4, VECSZ, cn, b4, b4v1, b4v2, b4v3);
                src_4 += src_inc;

                sum = v_add(sum, b4);
                sumv1 = v_add(sumv1, b4v1);
                sumv2 = v_add(sumv2, b4v2);
                sumv3 = v_add(sumv3, b4v3);
                if (SCALE_T==APPLY_SCALING)
                {
                    scaleValsameType(sum, _v_scale);
                    scaleValsameType(sumv1, _v_scale);
                    scaleValsameType(sumv2, _v_scale);
                    scaleValsameType(sumv3, _v_scale);
                }
                v_store(dstx + x, sum);
                v_store(dstx + x + VECSZ, sumv1);
                v_store(dstx + x + VECSZ_2, sumv2);
                v_store(dstx + x + VECSZ_3, sumv3);

                sum   = v_sub(sum, b0);
                sumv1 = v_sub(sumv1, b0v1);
                sumv2 = v_sub(sumv2, b0v2);
                sumv3 = v_sub(sumv3, b0v3);

                b0 = b1; b0v1 = b1v1; b0v2 = b1v2; b0v3 = b1v3;
                b1 = b2; b1v1 = b2v1; b1v2 = b2v2; b1v3 = b2v3;
                b2 = b3; b2v1 = b3v1; b2v2 = b3v2; b2v3 = b3v3;
                b3 = b4; b3v1 = b4v1; b3v2 = b4v2; b3v3 = b4v3;
            }
        }
#endif
        for (; x < len; x++)
        {
            int idst_ = idst;
            j = v;
            const ET* src_0 = ((const ET*)(src + j * src_stride))+x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;
            const ET* src_3 = src_2 + src_inc;
            const ET* src_4 = src_3 + src_inc;

            ET b0, b1, b2, b3, b4;
            b0 = (ET)src_0[-cn2] + (ET)src_0[-cn] + (ET)src_0[0] + (ET)src_0[cn] + (ET)src_0[cn2];
            b1 = (ET)src_1[-cn2] + (ET)src_1[-cn] + (ET)src_1[0] + (ET)src_1[cn] + (ET)src_1[cn2];
            b2 = (ET)src_2[-cn2] + (ET)src_2[-cn] + (ET)src_2[0] + (ET)src_2[cn] + (ET)src_2[cn2];
            b3 = (ET)src_3[-cn2] + (ET)src_3[-cn] + (ET)src_3[0] + (ET)src_3[cn] + (ET)src_3[cn2];

            for (; j < maxRow; j++, idst_++)
            {
                ET* dstx = (ET*)(dst + (idst_ * dst_stride));
                b4 = (ET)src_4[-cn2] + (ET)src_4[-cn] + (ET)src_4[0] + (ET)src_4[cn] + (ET)src_4[cn2];
                src_4 += src_inc;
                if (SCALE_T)
                {
                    dstx[x] = saturate_cast<ET>((b1 + b0 + b2 + b3 + b4 )* _scale);
                }
                else
                {
                    dstx[x] = saturate_cast<ET>(b1 + b0 + b2 + b3 + b4 );
                }
                b0 = b1;
                b1 = b2;
                b2 = b3;
                b3 = b4;
            }
        }
    }
    double scale;
};

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
void blockSumInPlace(const Mat& _src, Mat& _dst, Size ksize, Point anchor, const Size &wsz,
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
            ref = (const T*)(src+(srcY-roi.y)*srcStep);

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
        BlockSumBorderInplace<ST,T>(ref, S, R, btab, width, cn, dx1, dx2, borderType);
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


Ptr<BaseColumnFilter> getColumnSumFilter(int sumType, int dstType, int ksize, int anchor, double scale, int kernelSizeLog2)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(sumType), ddepth = CV_MAT_DEPTH(dstType);
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(dstType) );

    if( anchor < 0 )
        anchor = ksize/2;

    if(scale==1)
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
        if(kernelSizeLog2 && !(sdepth==CV_64F || sdepth==CV_32F))
        {
            scale = (double)kernelSizeLog2;
            if( ddepth == CV_8U && sdepth == CV_32S )
                return makePtr<ColumnSum<APPLY_SCALING_WITH_SHIFT, int, uchar> >(ksize, anchor, scale);
            if( ddepth == CV_8U && sdepth == CV_16U )
                return makePtr<ColumnSum<APPLY_SCALING_WITH_SHIFT, ushort, uchar> >(ksize, anchor, scale);
            if( ddepth == CV_16U && sdepth == CV_32S )
                return makePtr<ColumnSum<APPLY_SCALING_WITH_SHIFT, int, ushort> >(ksize, anchor, scale);
            if( ddepth == CV_16S && sdepth == CV_32S )
                return makePtr<ColumnSum<APPLY_SCALING_WITH_SHIFT, int, short> >(ksize, anchor, scale);
            if( ddepth == CV_32S && sdepth == CV_32S )
                return makePtr<ColumnSum<APPLY_SCALING_WITH_SHIFT, int, int> >(ksize, anchor, scale);
            if( ddepth == CV_32F && sdepth == CV_32S )
                return makePtr<ColumnSum<APPLY_SCALING_WITH_SHIFT, int, float> >(ksize, anchor, scale);
            if( ddepth == CV_64F && sdepth == CV_32S )
                return makePtr<ColumnSum<APPLY_SCALING_WITH_SHIFT, int, double> >(ksize, anchor, scale);
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
                if( sdepth == CV_8U  )
                    return makePtr<Sum3x3<SKIP_SCALING,  uchar, ushort, v_uint8,  v_uint16> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum3x3<SKIP_SCALING, ushort, uint32_t, v_uint16, v_uint32> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum3x3<SKIP_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
                if( sdepth == CV_32S )
                    return makePtr<Sum3x3sameType<SKIP_SCALING,  int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F )
                    return makePtr<Sum3x3_64f<SKIP_SCALING, float, double> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                if( sdepth == CV_64F )
                    return makePtr<Sum3x3sameType<SKIP_SCALING, double,v_float64> >(ksize, scale);
#endif
            }
            else
            {
                if( sdepth == CV_8U  )
                    return makePtr<Sum3x3<APPLY_SCALING, uchar, ushort, v_uint8,  v_uint16> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum3x3<APPLY_SCALING, ushort, uint32_t, v_uint16, v_uint32> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum3x3<APPLY_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
                if( sdepth == CV_32S )
                    return makePtr<Sum3x3sameType<APPLY_SCALING, int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F )
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
                if( sdepth == CV_8U  )
                    return makePtr<Sum5x5<SKIP_SCALING, uchar, ushort, v_uint8,  v_uint16, void> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum5x5<SKIP_SCALING, ushort, uint32_t, v_uint16, v_uint32, void> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum5x5<SKIP_SCALING, short, int32_t, v_int16, v_int32, void> >(ksize, scale);
                if( sdepth == CV_32S )
                    return makePtr<Sum5x5sameType<SKIP_SCALING, int, v_int32> >(ksize, scale); //intermediate stored can be stored in int64_t, but matching reference code to avoid output mismatch.
                if( sdepth == CV_32F )
                    return makePtr<Sum5x5_64f<SKIP_SCALING, float, double> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                    if( sdepth == CV_64F )
                    return makePtr<Sum5x5sameType<SKIP_SCALING, double, v_float64> >(ksize, scale);
#endif
            }
            else
            {
                if( sdepth == CV_8U  )
                    return makePtr<Sum5x5<APPLY_SCALING, uchar, ushort, v_uint8,  v_uint16, void> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum5x5<APPLY_SCALING, ushort, uint32_t, v_uint16, v_uint32, void> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum5x5<APPLY_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
                if( sdepth == CV_32S )
                    return makePtr<Sum5x5sameType<APPLY_SCALING, int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F )
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

    int kernelSizeLog2 = 0;
    if(ksize.width == ksize.height)
    {
        int size2D = ksize.width * ksize.height;
        if ((size2D & (size2D - 1)) == 0) //check if size is power of 2
        {
            // Compute power or exponent
            int temp = size2D;
            while (temp > 1) {
                temp >>= 1;
                kernelSizeLog2++;
            }
        }
    }
    Ptr<BaseColumnFilter> columnFilter = getColumnSumFilter(sumType,
        dstType, ksize.height, anchor.y, normalize ? 1./(ksize.width*ksize.height) : 1, kernelSizeLog2);

    Ptr<BaseRowColumnFilter> rowColumnFilter = nullptr;
    if(ksize.width == ksize.height &&
       (ksize.width ==3 || ksize.width ==5))
    {
        rowColumnFilter = getRowColumnSumFilter(srcType, dstType, ksize.width, normalize ? 1./(ksize.width*ksize.height) : 1);
    }
    return makePtr<FilterEngine>(Ptr<BaseFilter>(), rowFilter, columnFilter, rowColumnFilter,
           srcType, dstType, sumType, borderType );
}

void blockSumInPlace(const Mat& src, Mat& dst, Size ksize, Point anchor,
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
        blockSumInPlace<double, float> (src, dst, ksize, anchor, wsz, ofs, scale, borderType, sumType);
    else if( ddepth == CV_64F && sdepth == CV_64F )
        blockSumInPlace<double, double> (src, dst, ksize, anchor, wsz, ofs, scale, borderType, sumType);
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
