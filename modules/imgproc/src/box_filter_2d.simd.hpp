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

// 2D (row+column combined) Sum3x3 / Sum5x5 filters for boxFilter / blur.
// These are SIMD-optimized rowColumnFilter implementations that process
// both row and column passes in a single traversal.

#if (CV_SIMD || CV_SIMD_SCALABLE)
// Sum3x3/Sum5x5: 32-bit sums of 16-bit pixels, same rounding as the separable ColumnSum
static inline v_uint32 roundScale(const v_uint32& s, const v_float32& v_scale)
{ return v_reinterpret_as_u32(v_round(v_mul(v_cvt_f32(v_reinterpret_as_s32(s)), v_scale))); }
static inline v_int32 roundScale(const v_int32& s, const v_float32& v_scale)
{ return v_round(v_mul(v_cvt_f32(s), v_scale)); }
template<typename V>
static inline V roundScale(const V& s, const v_float32&) { return s; }

// Sum3x3sameType/Sum5x5sameType
static inline v_int32 scaleSameType(const v_int32& s, double, const v_float32& v_scale)
{ return v_round(v_mul(v_cvt_f32(s), v_scale)); }
static inline v_float32 scaleSameType(const v_float32& s, double, const v_float32& v_scale)
{ return v_mul(s, v_scale); }
#endif
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
static inline v_float64 scaleSameType(const v_float64& s, double scale, const v_float32&)
{ return v_mul(s, vx_setall_f64(scale)); }
#endif

// scalar tails scale floating-point sums in ET, as the vector loops do
template<typename ET>
static inline typename std::enable_if<std::is_floating_point<ET>::value, ET>::type
scaleSum(ET s, double scale) { return s * (ET)scale; }

template<typename ET>
static inline typename std::enable_if<!std::is_floating_point<ET>::value, ET>::type
scaleSum(ET s, double scale) { return saturate_cast<ET>(s * scale); }

template<int SCALE_T, typename ET, typename WET, typename VET, typename VFT>
struct Sum3x3 :
        public BaseRowColumnFilter
{
    Sum3x3( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        minWidth = 2*VTraits<VET>::vlanes() + 8;
#endif
    }
    virtual void reset() CV_OVERRIDE { }

#if (CV_SIMD || CV_SIMD_SCALABLE)
    void inline loadRow(const ET* src, int cn, VFT &a0, VFT &a1, VFT &b0, VFT &b1, VFT &c0, VFT &c1)
    {
        const ET* src_ptr = src - cn;
        const int half = VTraits<VFT>::vlanes();
        a0 = vx_load_expand(src_ptr);
        a1 = vx_load_expand(src_ptr + half);
        b0 = vx_load_expand(src_ptr + cn);
        b1 = vx_load_expand(src_ptr + cn + half);
        c0 = vx_load_expand(src_ptr + cn*2);
        c1 = vx_load_expand(src_ptr + cn*2 + half);
    }
    void inline addRow(const VFT &a0, const VFT &a1, const VFT &b0, const VFT &b1, VFT &r0, VFT &r1)
    {
        r0 = v_add(r0,  v_add(a0, b0));
        r1 = v_add(r1,  v_add(a1, b1));
    }
    void inline loadRowAdd(const ET* src, int VECSZ,
                           int cn, VFT &r0, VFT &r1, VFT &r0v1, VFT &r1v1)
    {
        VFT a0, a1, b0, b1;
        VFT a0v1, a1v1, b0v1, b1v1;
        loadRow(src, cn, a0, a1, b0, b1, r0, r1);
        loadRow(src+VECSZ, cn, a0v1, a1v1, b0v1, b1v1, r0v1, r1v1);
        addRow(a0, a1, b0, b1, r0, r1);
        addRow(a0v1, a1v1, b0v1, b1v1, r0v1, r1v1);
    }
#endif

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;
        int yTopOffset = 1;
        int yBottomOffset = 1;
        if(yidx >0)
            yTopOffset = 0;
        if( (yidx + height) < wholeHeight)
            yBottomOffset = 0;
        int maxRow = height-yTopOffset-yBottomOffset;
        int idst = yTopOffset;

        int xoffset = 1;
        int v = idst - yTopOffset;
        int len = (width - xoffset) * cn;
        int x = xoffset * cn;

        const int ETSZ = sizeof(ET);
        const int src_inc = src_stride / ETSZ;
        int j = v;
        const ET* src_ptr = (const ET*)(src + j * src_stride);

#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 v_scale = vx_setall_f32((float)scale);
        const int VECSZ = VTraits<VET>::vlanes();
        const int VECSZ_2 = VECSZ << 1;

        VFT b00, b01, b00v1, b01v1;
        VFT b10, b11, b10v1, b11v1;
        VFT b20, b21, b20v1, b21v1;

        for (; x < len; x += (VECSZ_2))
        {
            if ( x > len - VECSZ_2 )
            {
                if (x == cn*xoffset || src == dst)
                    break;
                x = len - VECSZ_2;
            }
            int idst_ = idst;
            j = v;
            const ET* src_0 = src_ptr + x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;

            //row 0 and 1
            loadRowAdd(src_0, VECSZ, cn, b00, b01, b00v1, b01v1);
            loadRowAdd(src_1, VECSZ, cn, b10, b11, b10v1, b11v1);

            for (; j < maxRow; j++, idst_++)
            {
                ET* dstx = (ET*)(dst + (idst_ * dst_stride));
                dstx += x;
                b00   = v_add(b00, b10);
                b00v1 = v_add(b00v1, b10v1);
                b01   = v_add(b01, b11);
                b01v1 = v_add(b01v1, b11v1);

                loadRowAdd(src_2, VECSZ, cn, b20, b21, b20v1, b21v1);
                src_2 += src_inc;

                b00   = v_add(b20, b00);
                b00v1 = v_add(b20v1, b00v1);
                b01   = v_add(b21, b01);
                b01v1 = v_add(b21v1, b01v1);

                if (SCALE_T==APPLY_SCALING)
                {
                    b00   = roundScale(b00, v_scale);
                    b00v1 = roundScale(b00v1, v_scale);
                    b01   = roundScale(b01, v_scale);
                    b01v1 = roundScale(b01v1, v_scale);
                }
                v_store(dstx, v_pack(b00, b01));
                v_store(dstx + VECSZ, v_pack(b00v1, b01v1));

                b00 = b10; b01 = b11;
                b10 = b20; b11 = b21;
                b00v1 = b10v1; b01v1 = b11v1;
                b10v1 = b20v1; b11v1 = b21v1;
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

// Normalized 3x3 box filter for uchar -> uchar. Works in one-vector strips with
// 16-bit sums; column-major traversal keeps row sums in registers across rows.
struct Sum3x3_8u :
        public BaseRowColumnFilter
{
    Sum3x3_8u( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        minWidth = VTraits<v_uint8>::vlanes() + 8;
#endif
    }
    virtual void reset() CV_OVERRIDE { }

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;
        int yTopOffset = 1;
        int yBottomOffset = 1;
        if(yidx > 0)
            yTopOffset = 0;
        if( (yidx + height) < wholeHeight)
            yBottomOffset = 0;
        int maxRow = height - yTopOffset - yBottomOffset;
        int idst = yTopOffset;

        int xoffset = 1;
        int v = idst - yTopOffset;
        int len = (width - xoffset) * cn;
        int x = xoffset * cn;

        const int src_inc = src_stride;
        int j = v;
        const uchar* src_ptr = src + j * src_inc;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int VECSZ16 = VTraits<v_uint16>::vlanes();
        const int VECSZ8 = VTraits<v_uint8>::vlanes();

        // mul_hi(s + 4, 7282) == round(s/9) exactly for every s <= 9*255
        v_uint16 v_4 = vx_setall_u16(4);
        v_uint16 v_mulFactor = vx_setall_u16(7282);

        for (; x < len; x += VECSZ8)
        {
            if ( x > len - VECSZ8 )
            {
                if (x == cn * xoffset || src == dst)
                    break;
                x = len - VECSZ8;
            }
            int idst_ = idst;
            j = v;
            const uchar* src_0 = src_ptr + x;
            const uchar* src_1 = src_0 + src_inc;
            const uchar* src_2 = src_1 + src_inc;

            // Row sums for row 0
            v_uint16 r0_lo = v_add(v_add(vx_load_expand(src_0 - cn), vx_load_expand(src_0)), vx_load_expand(src_0 + cn));
            v_uint16 r0_hi = v_add(v_add(vx_load_expand(src_0 - cn + VECSZ16), vx_load_expand(src_0 + VECSZ16)), vx_load_expand(src_0 + cn + VECSZ16));
            // Row sums for row 1
            v_uint16 r1_lo = v_add(v_add(vx_load_expand(src_1 - cn), vx_load_expand(src_1)), vx_load_expand(src_1 + cn));
            v_uint16 r1_hi = v_add(v_add(vx_load_expand(src_1 - cn + VECSZ16), vx_load_expand(src_1 + VECSZ16)), vx_load_expand(src_1 + cn + VECSZ16));

            for (; j < maxRow; j++, idst_++)
            {
                uchar* dstx = dst + idst_ * dst_stride + x;

                // Row sums for row 2
                v_uint16 r2_lo = v_add(v_add(vx_load_expand(src_2 - cn), vx_load_expand(src_2)), vx_load_expand(src_2 + cn));
                v_uint16 r2_hi = v_add(v_add(vx_load_expand(src_2 - cn + VECSZ16), vx_load_expand(src_2 + VECSZ16)), vx_load_expand(src_2 + cn + VECSZ16));
                src_2 += src_inc;

                // Column sums (3x3 box sum)
                v_uint16 s_lo = v_add(v_add(r0_lo, r1_lo), r2_lo);
                v_uint16 s_hi = v_add(v_add(r0_hi, r1_hi), r2_hi);

                s_lo = v_mul_hi(v_add(s_lo, v_4), v_mulFactor);
                s_hi = v_mul_hi(v_add(s_hi, v_4), v_mulFactor);

                v_store(dstx, v_pack(s_lo, s_hi));

                // Slide window
                r0_lo = r1_lo; r0_hi = r1_hi;
                r1_lo = r2_lo; r1_hi = r2_hi;
            }
        }
#endif
        // Scalar tail
        for (; x < len; x++)
        {
            int idst_ = idst;
            j = v;
            ushort b0, b1, b2;
            const uchar* src_0 = src_ptr + x;
            const uchar* src_1 = src_0 + src_inc;
            const uchar* src_2 = src_1 + src_inc;
            b0 = src_0[-cn] + src_0[0] + src_0[cn];
            b1 = src_1[-cn] + src_1[0] + src_1[cn];

            for (; j < maxRow; j++, idst_++)
            {
                uchar* dstx = dst + idst_ * dst_stride;
                b2 = src_2[-cn] + src_2[0] + src_2[cn];
                src_2 += src_inc;
                dstx[x] = saturate_cast<uchar>((b1 + b0 + b2) * _scale);
                b0 = b1;
                b1 = b2;
            }
        }
    }

    double scale;
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
        minWidth = 4*VTraits<VET>::vlanes() + 8;
#endif
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

#endif

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;

        int yTopOffset = 1;
        int yBottomOffset = 1;
        if(yidx >0)
            yTopOffset = 0;
        if( (yidx + height) < wholeHeight)
            yBottomOffset = 0;
        int maxRow = height-yTopOffset-yBottomOffset;
        int idst = yTopOffset;

        int xoffset = 1;
        int v = idst - yTopOffset;
        int len = (width - xoffset) * cn;
        int x = xoffset * cn;

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
                if (x == cn*xoffset || src == dst)
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
                    b0 = scaleSameType(b0, _scale, _v_scale);
                    b0v1 = scaleSameType(b0v1, _scale, _v_scale);
                    b0v2 = scaleSameType(b0v2, _scale, _v_scale);
                    b0v3 = scaleSameType(b0v3, _scale, _v_scale);
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
                    dstx[x] = scaleSum<ET>(a1 + a0 + a2, _scale);
                else
                    dstx[x] = saturate_cast<ET>(a1 + a0 + a2);
                a0 = a1;
                a1 = a2;
            }
        }
    }

    double scale;
};


template<int SCALE_T, typename ET, typename WET, typename VET, typename VFT>
struct Sum5x5 :
        public BaseRowColumnFilter
{
    Sum5x5( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        minWidth = VTraits<VET>::vlanes() + 16;
#endif
    }
    virtual void reset() CV_OVERRIDE { }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    void inline loadRow(const ET* src, int cn, VFT &a0, VFT &a1, VFT &b0, VFT &b1,
                            VFT &c0, VFT &c1, VFT &d0, VFT &d1, VFT &e0, VFT &e1)
    {
        const ET* src_ptr = src - cn*2;
        const int half = VTraits<VFT>::vlanes();
        a0 = vx_load_expand(src_ptr);
        a1 = vx_load_expand(src_ptr + half);
        b0 = vx_load_expand(src_ptr + cn);
        b1 = vx_load_expand(src_ptr + cn + half);
        c0 = vx_load_expand(src_ptr + cn*2);
        c1 = vx_load_expand(src_ptr + cn*2 + half);
        d0 = vx_load_expand(src_ptr + cn*3);
        d1 = vx_load_expand(src_ptr + cn*3 + half);
        e0 = vx_load_expand(src_ptr + cn*4);
        e1 = vx_load_expand(src_ptr + cn*4 + half);
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
#endif

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;
        int yTopOffset = 2;
        int yBottomOffset = 2;
        if(yidx >1)
            yTopOffset = 0;
        if( (yidx + height+1) < wholeHeight)
            yBottomOffset = 0;
        int maxRow = height-yTopOffset-yBottomOffset;
        int idst = yTopOffset;


        int xoffset = 2;
        int v = idst - yTopOffset;
        int len = (width - xoffset) * cn;
        int x = xoffset * cn;

        const int ETSZ = sizeof(ET);
        const int src_inc = src_stride / ETSZ;
        const int dst_inc = dst_stride / ETSZ;
        int cn2 = cn*2;
        int j = v;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 v_scale = vx_setall_f32((float)scale);

        const ET* src_ptr = (const ET*)(src + j * src_stride);
        const int VECSZ = VTraits<VET>::vlanes();

        VFT b00, b01, b10, b11, b20, b21, b30, b31, b40, b41;

        for (; x < len; x += VECSZ)
        {
            if (x > len - VECSZ)
            {
                if (x == cn*xoffset || src == dst)
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
                    b0 = roundScale(bsum0, v_scale);
                    b1 = roundScale(bsum1, v_scale);
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

// Normalized 5x5 box filter for uchar -> uchar, laid out like Sum3x3_8u.
struct Sum5x5_8u :
        public BaseRowColumnFilter
{
    Sum5x5_8u( int _ksize, double _scale ) :
        BaseRowColumnFilter()
    {
        ksize = _ksize;
        scale = _scale;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        minWidth = VTraits<v_uint8>::vlanes() + 16;
#endif
    }
    virtual void reset() CV_OVERRIDE { }

#if (CV_SIMD || CV_SIMD_SCALABLE)
    static inline void rowSum(const uchar* p, int cn, v_uint16& lo, v_uint16& hi)
    {
        const int half = VTraits<v_uint16>::vlanes();
        const uchar* q = p - cn*2;
        lo = v_add(v_add(v_add(vx_load_expand(q), vx_load_expand(q + cn)),
                         v_add(vx_load_expand(q + cn*2), vx_load_expand(q + cn*3))), vx_load_expand(q + cn*4));
        q += half;
        hi = v_add(v_add(v_add(vx_load_expand(q), vx_load_expand(q + cn)),
                         v_add(vx_load_expand(q + cn*2), vx_load_expand(q + cn*3))), vx_load_expand(q + cn*4));
    }
#endif

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        int yTopOffset = 2;
        int yBottomOffset = 2;
        if(yidx > 1)
            yTopOffset = 0;
        if( (yidx + height + 1) < wholeHeight)
            yBottomOffset = 0;
        int maxRow = height - yTopOffset - yBottomOffset;
        int idst = yTopOffset;

        int xoffset = 2;
        int v = idst - yTopOffset;
        int len = (width - xoffset) * cn;
        int x = xoffset * cn;
        int cn2 = cn*2;

        const int src_inc = src_stride;
        int j = v;
        const uchar* src_ptr = src + j * src_inc;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int VECSZ8 = VTraits<v_uint8>::vlanes();

        // mul_hi(s + 12, 5243) >> 1 == round(s/25) exactly for every s <= 25*255;
        // no 16-bit multiplier with a 16-bit shift is exact over that range
        v_uint16 v_12 = vx_setall_u16(12);
        v_uint16 v_mulFactor = vx_setall_u16(5243);

        for (; x < len; x += VECSZ8)
        {
            if ( x > len - VECSZ8 )
            {
                if (x == cn * xoffset || src == dst)
                    break;
                x = len - VECSZ8;
            }
            int idst_ = idst;
            j = v;
            const uchar* src_0 = src_ptr + x;
            const uchar* src_4 = src_0 + src_inc*4;
            uchar* dstx = dst + idst_ * dst_stride + x;

            v_uint16 r0_lo, r0_hi, r1_lo, r1_hi, r2_lo, r2_hi, r3_lo, r3_hi;
            rowSum(src_0, cn, r0_lo, r0_hi);
            rowSum(src_0 + src_inc, cn, r1_lo, r1_hi);
            rowSum(src_0 + src_inc*2, cn, r2_lo, r2_hi);
            rowSum(src_0 + src_inc*3, cn, r3_lo, r3_hi);
            v_uint16 s_lo = v_add(v_add(r0_lo, r1_lo), v_add(r2_lo, r3_lo));
            v_uint16 s_hi = v_add(v_add(r0_hi, r1_hi), v_add(r2_hi, r3_hi));

            for (; j < maxRow; j++, idst_++)
            {
                v_uint16 r4_lo, r4_hi;
                rowSum(src_4, cn, r4_lo, r4_hi);
                src_4 += src_inc;

                s_lo = v_add(s_lo, r4_lo);
                s_hi = v_add(s_hi, r4_hi);
                v_uint16 o_lo = v_shr<1>(v_mul_hi(v_add(s_lo, v_12), v_mulFactor));
                v_uint16 o_hi = v_shr<1>(v_mul_hi(v_add(s_hi, v_12), v_mulFactor));
                v_store(dstx, v_pack(o_lo, o_hi));
                dstx += dst_stride;

                s_lo = v_sub(s_lo, r0_lo);
                s_hi = v_sub(s_hi, r0_hi);
                r0_lo = r1_lo; r0_hi = r1_hi;
                r1_lo = r2_lo; r1_hi = r2_hi;
                r2_lo = r3_lo; r2_hi = r3_hi;
                r3_lo = r4_lo; r3_hi = r4_hi;
            }
        }
#endif
        for (; x < len; x++)
        {
            int idst_ = idst;
            j = v;
            const uchar* src_0 = src_ptr + x;
            const uchar* src_4 = src_0 + src_inc*4;
            int b[5];
            for (int r = 0; r < 4; r++)
            {
                const uchar* p = src_0 + r*src_inc;
                b[r] = p[-cn2] + p[-cn] + p[0] + p[cn] + p[cn2];
            }
            int bsum = b[0] + b[1] + b[2] + b[3];
            for (; j < maxRow; j++, idst_++)
            {
                b[4] = src_4[-cn2] + src_4[-cn] + src_4[0] + src_4[cn] + src_4[cn2];
                src_4 += src_inc;
                bsum += b[4];
                dst[idst_ * dst_stride + x] = (uchar)((bsum + 12) / 25);
                bsum -= b[0];
                b[0] = b[1]; b[1] = b[2]; b[2] = b[3]; b[3] = b[4];
            }
        }
    }

    double scale;
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
        minWidth = 4*VTraits<VET>::vlanes() + 16;
#endif
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

#endif
    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        double _scale = scale;
        int yTopOffset = 2;
        int yBottomOffset = 2;
        if(yidx >1)
            yTopOffset = 0;
        if( (yidx + height+1) < wholeHeight)
            yBottomOffset = 0;
        int maxRow = height-yTopOffset-yBottomOffset;
        int idst = yTopOffset;

        int xoffset = 2;
        int v = idst - yTopOffset;
        int j = v;
        int len = (width - xoffset) * cn;
        int x = xoffset * cn;

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
                if (x == cn*xoffset || src == dst)
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

                if (std::is_floating_point<ET>::value)
                {
                    // a running sum would accumulate rounding error down the column
                    sum   = v_add(v_add(v_add(v_add(b1, b0), b2), b3), b4);
                    sumv1 = v_add(v_add(v_add(v_add(b1v1, b0v1), b2v1), b3v1), b4v1);
                    sumv2 = v_add(v_add(v_add(v_add(b1v2, b0v2), b2v2), b3v2), b4v2);
                    sumv3 = v_add(v_add(v_add(v_add(b1v3, b0v3), b2v3), b3v3), b4v3);
                }
                else
                {
                    sum = v_add(sum, b4);
                    sumv1 = v_add(sumv1, b4v1);
                    sumv2 = v_add(sumv2, b4v2);
                    sumv3 = v_add(sumv3, b4v3);
                }
                VET out = sum, outv1 = sumv1, outv2 = sumv2, outv3 = sumv3;
                if (SCALE_T==APPLY_SCALING)
                {
                    out = scaleSameType(out, _scale, _v_scale);
                    outv1 = scaleSameType(outv1, _scale, _v_scale);
                    outv2 = scaleSameType(outv2, _scale, _v_scale);
                    outv3 = scaleSameType(outv3, _scale, _v_scale);
                }
                v_store(dstx + x, out);
                v_store(dstx + x + VECSZ, outv1);
                v_store(dstx + x + VECSZ_2, outv2);
                v_store(dstx + x + VECSZ_3, outv3);

                if (!std::is_floating_point<ET>::value)
                {
                    sum   = v_sub(sum, b0);
                    sumv1 = v_sub(sumv1, b0v1);
                    sumv2 = v_sub(sumv2, b0v2);
                    sumv3 = v_sub(sumv3, b0v3);
                }

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
            // same association as the vector loadRowAdd, so float results do not depend on x
            b0 = ((src_0[-cn2] + src_0[-cn]) + (src_0[0] + src_0[cn])) + src_0[cn2];
            b1 = ((src_1[-cn2] + src_1[-cn]) + (src_1[0] + src_1[cn])) + src_1[cn2];
            b2 = ((src_2[-cn2] + src_2[-cn]) + (src_2[0] + src_2[cn])) + src_2[cn2];
            b3 = ((src_3[-cn2] + src_3[-cn]) + (src_3[0] + src_3[cn])) + src_3[cn2];

            for (; j < maxRow; j++, idst_++)
            {
                ET* dstx = (ET*)(dst + (idst_ * dst_stride));
                b4 = ((src_4[-cn2] + src_4[-cn]) + (src_4[0] + src_4[cn])) + src_4[cn2];
                src_4 += src_inc;
                if (SCALE_T)
                {
                    dstx[x] = scaleSum<ET>(b1 + b0 + b2 + b3 + b4, _scale);
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
