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
template<typename ET, typename AT>
static inline typename std::enable_if<std::is_floating_point<ET>::value, ET>::type
scaleSum(AT s, double scale) { return s * (ET)scale; }

template<typename ET, typename AT>
static inline typename std::enable_if<!std::is_floating_point<ET>::value, ET>::type
scaleSum(AT s, double scale) { return saturate_cast<ET>(s * scale); }

// Rows and x range of one call for a (2R+1)x(2R+1) kernel: output rows start at dstRow0
// (src row 0 is the first tap row), and the first/last R columns are left to the caller.
template<int R>
struct BoxFilter2DRange
{
    BoxFilter2DRange( int width, int height, int yidx, int wholeHeight, int cn )
    {
        int yTopOffset = yidx > R - 1 ? 0 : R;
        int yBottomOffset = yidx + height + R - 1 < wholeHeight ? 0 : R;
        rows = height - yTopOffset - yBottomOffset;
        dstRow0 = yTopOffset;
        xbegin = R*cn;
        xend = (width - R)*cn;
    }
    int rows, dstRow0, xbegin, xend;
};

// Moves the last vector strip back so it ends at xend. It then overlaps the previous strip,
// which is only allowed when there is one and the filter is not running in place.
template<int R>
static inline bool boxFilter2DStrip( int& x, int step, const BoxFilter2DRange<R>& r, bool inplace )
{
    if( x > r.xend - step )
    {
        if( x == r.xbegin || inplace )
            return false;
        x = r.xend - step;
    }
    return true;
}

// Horizontal sum of the 2R+1 taps around p. Floating-point sums depend on the association,
// so the scalar and vector versions add in the same order.
template<int R> struct BoxTaps;

template<> struct BoxTaps<1>
{
    template<typename WT, typename T> static inline WT sum( const T* p, int cn )
    { return ((WT)p[-cn] + (WT)p[0]) + (WT)p[cn]; }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    template<class L, typename T> static inline auto vsum( const T* p, int cn ) -> decltype(L::ld(p))
    { return v_add(v_add(L::ld(p - cn), L::ld(p)), L::ld(p + cn)); }
#endif
};

template<> struct BoxTaps<2>
{
    template<typename WT, typename T> static inline WT sum( const T* p, int cn )
    { return (((WT)p[-cn*2] + (WT)p[-cn]) + ((WT)p[0] + (WT)p[cn])) + (WT)p[cn*2]; }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    template<class L, typename T> static inline auto vsum( const T* p, int cn ) -> decltype(L::ld(p))
    { return v_add(v_add(v_add(L::ld(p - cn*2), L::ld(p - cn)), v_add(L::ld(p), L::ld(p + cn))), L::ld(p + cn*2)); }
#endif
};

#if (CV_SIMD || CV_SIMD_SCALABLE)
struct BoxLoad
{
    template<typename T> static inline auto ld( const T* p ) -> decltype(vx_load(p)) { return vx_load(p); }
};
struct BoxLoadExpand
{
    template<typename T> static inline auto ld( const T* p ) -> decltype(vx_load_expand(p)) { return vx_load_expand(p); }
};

// Widened row sums of one vector of ET pixels, split into its low and high halves.
template<int R, typename ET, typename VFT>
static inline void boxRowSumExpand( const ET* p, int cn, VFT& lo, VFT& hi )
{
    lo = BoxTaps<R>::template vsum<BoxLoadExpand>(p, cn);
    hi = BoxTaps<R>::template vsum<BoxLoadExpand>(p + VTraits<VFT>::vlanes(), cn);
}
#endif

template<int SCALE_T, typename ET>
struct BoxTailOut
{
    explicit BoxTailOut( double _scale ) : scale(_scale) {}
    template<typename AT> inline ET operator()( AT s ) const
    { return SCALE_T == APPLY_SCALING ? scaleSum<ET>(s, scale) : saturate_cast<ET>(s); }
    double scale;
};

// round(s/25) for the normalized 8u 5x5 kernel
struct BoxTailDiv25
{
    inline uchar operator()( int s ) const { return (uchar)((s + 12) / 25); }
};

// Scalar columns [x, r.xend): row sums in WT, the column sum in WT's arithmetic type.
template<int R, typename WT, typename ET, class Out>
static inline void boxFilter2DTail( const uchar* src, uchar* dst, int src_stride, int dst_stride,
                                    const BoxFilter2DRange<R>& r, int x, int cn, const Out& out )
{
    typedef decltype(WT() + WT()) AT;
    const int src_inc = src_stride / (int)sizeof(ET);
    for( ; x < r.xend; x++ )
    {
        const ET* s = (const ET*)src + x;
        WT b[2*R + 1];
        for( int k = 0; k < 2*R; k++, s += src_inc )
            b[k] = BoxTaps<R>::template sum<WT>(s, cn);
        for( int j = 0; j < r.rows; j++, s += src_inc )
        {
            b[2*R] = BoxTaps<R>::template sum<WT>(s, cn);
            AT sum = b[1] + b[0];
            for( int k = 2; k <= 2*R; k++ )
                sum += b[k];
            ((ET*)(dst + (ptrdiff_t)(r.dstRow0 + j)*dst_stride))[x] = out(sum);
            for( int k = 0; k < 2*R; k++ )
                b[k] = b[k + 1];
        }
    }
}

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

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        const BoxFilter2DRange<1> r(width, height, yidx, wholeHeight, cn);
        int x = r.xbegin;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int src_inc = src_stride / (int)sizeof(ET);
        const ET* src_ptr = (const ET*)src;
        v_float32 v_scale = vx_setall_f32((float)scale);
        const int VECSZ = VTraits<VET>::vlanes();
        const int VECSZ_2 = VECSZ << 1;

        VFT b00, b01, b00v1, b01v1;
        VFT b10, b11, b10v1, b11v1;
        VFT b20, b21, b20v1, b21v1;

        for (; x < r.xend; x += (VECSZ_2))
        {
            if( !boxFilter2DStrip(x, VECSZ_2, r, src == dst) )
                break;
            const ET* src_0 = src_ptr + x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;

            //row 0 and 1
            boxRowSumExpand<1>(src_0, cn, b00, b01);
            boxRowSumExpand<1>(src_0 + VECSZ, cn, b00v1, b01v1);
            boxRowSumExpand<1>(src_1, cn, b10, b11);
            boxRowSumExpand<1>(src_1 + VECSZ, cn, b10v1, b11v1);

            for (int j = 0; j < r.rows; j++)
            {
                ET* dstx = (ET*)(dst + (ptrdiff_t)(r.dstRow0 + j)*dst_stride);
                dstx += x;
                b00   = v_add(b00, b10);
                b00v1 = v_add(b00v1, b10v1);
                b01   = v_add(b01, b11);
                b01v1 = v_add(b01v1, b11v1);

                boxRowSumExpand<1>(src_2, cn, b20, b21);
                boxRowSumExpand<1>(src_2 + VECSZ, cn, b20v1, b21v1);
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
        boxFilter2DTail<1, WET, ET>(src, dst, src_stride, dst_stride, r, x, cn, BoxTailOut<SCALE_T, ET>(scale));
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
        const BoxFilter2DRange<1> r(width, height, yidx, wholeHeight, cn);
        int x = r.xbegin;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int src_inc = src_stride;
        const int VECSZ8 = VTraits<v_uint8>::vlanes();

        // mul_hi(s + 4, 7282) == round(s/9) exactly for every s <= 9*255
        v_uint16 v_4 = vx_setall_u16(4);
        v_uint16 v_mulFactor = vx_setall_u16(7282);

        for (; x < r.xend; x += VECSZ8)
        {
            if( !boxFilter2DStrip(x, VECSZ8, r, src == dst) )
                break;
            const uchar* src_0 = src + x;
            const uchar* src_1 = src_0 + src_inc;
            const uchar* src_2 = src_1 + src_inc;

            v_uint16 r0_lo, r0_hi, r1_lo, r1_hi;
            boxRowSumExpand<1>(src_0, cn, r0_lo, r0_hi);
            boxRowSumExpand<1>(src_1, cn, r1_lo, r1_hi);

            for (int j = 0; j < r.rows; j++)
            {
                uchar* dstx = dst + (ptrdiff_t)(r.dstRow0 + j)*dst_stride + x;

                v_uint16 r2_lo, r2_hi;
                boxRowSumExpand<1>(src_2, cn, r2_lo, r2_hi);
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
        boxFilter2DTail<1, int, uchar>(src, dst, src_stride, dst_stride, r, x, cn, BoxTailOut<APPLY_SCALING, uchar>(scale));
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
    static inline void loadRowAdd(const ET* src, int VECSZ, int cn, VET &r, VET &rv1, VET &rv2, VET &rv3)
    {
        r   = BoxTaps<1>::vsum<BoxLoad>(src, cn);
        rv1 = BoxTaps<1>::vsum<BoxLoad>(src + VECSZ, cn);
        rv2 = BoxTaps<1>::vsum<BoxLoad>(src + VECSZ*2, cn);
        rv3 = BoxTaps<1>::vsum<BoxLoad>(src + VECSZ*3, cn);
    }
#endif

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        const BoxFilter2DRange<1> r(width, height, yidx, wholeHeight, cn);
        int x = r.xbegin;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        double _scale = scale;
        const int src_inc = src_stride / (int)sizeof(ET);
        v_float32 _v_scale = vx_setall_f32((float)_scale);
        const int VECSZ = VTraits<VET>::vlanes();
        const int VECSZ_2 = VECSZ<<1;
        const int VECSZ_3 = VECSZ*3;
        const int VECSZ_4 = VECSZ<<2;

        VET b0, b1, b2;
        VET b0v1, b1v1, b2v1;
        VET b0v2, b1v2, b2v2;
        VET b0v3, b1v3, b2v3;

        const ET* src_ptr = (const ET*)src;
        for (; x < r.xend; x += (VECSZ_4))
        {
            if( !boxFilter2DStrip(x, VECSZ_4, r, src == dst) )
                break;
            const ET* src_0 = src_ptr +x ;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;

            //row 0 and 1
            loadRowAdd(src_0, VECSZ, cn, b0, b0v1, b0v2, b0v3);
            loadRowAdd(src_1, VECSZ, cn, b1, b1v1, b1v2, b1v3);

            for (int j = 0; j < r.rows; j++)
            {
                ET* dstx = (ET*)(dst + (ptrdiff_t)(r.dstRow0 + j)*dst_stride);
                loadRowAdd(src_2, VECSZ, cn, b2, b2v1, b2v2, b2v3);
                src_2 += src_inc;

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
        boxFilter2DTail<1, ET, ET>(src, dst, src_stride, dst_stride, r, x, cn, BoxTailOut<SCALE_T, ET>(scale));
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

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        const BoxFilter2DRange<2> r(width, height, yidx, wholeHeight, cn);
        int x = r.xbegin;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int src_inc = src_stride / (int)sizeof(ET);
        const int dst_inc = dst_stride / (int)sizeof(ET);
        v_float32 v_scale = vx_setall_f32((float)scale);

        const ET* src_ptr = (const ET*)src;
        const int VECSZ = VTraits<VET>::vlanes();

        VFT b00, b01, b10, b11, b20, b21, b30, b31, b40, b41;

        for (; x < r.xend; x += VECSZ)
        {
            if( !boxFilter2DStrip(x, VECSZ, r, src == dst) )
                break;
            const ET* src_0 = src_ptr + x;
            const ET* src_1 = src_0 + src_inc;
            const ET* src_2 = src_1 + src_inc;
            const ET* src_3 = src_2 + src_inc;
            const ET* src_4 = src_3 + src_inc;

            boxRowSumExpand<2>(src_0, cn, b00, b01);
            boxRowSumExpand<2>(src_1, cn, b10, b11);
            boxRowSumExpand<2>(src_2, cn, b20, b21);
            boxRowSumExpand<2>(src_3, cn, b30, b31);
            ET* dstx = (ET*)(dst + (ptrdiff_t)r.dstRow0*dst_stride);
            dstx += x;

            VFT bsum0, bsum1, b0, b1;
            bsum0   = v_add(v_add(b10, b00), v_add(b20, b30));
            bsum1   = v_add(v_add(b11, b01), v_add(b21, b31));
            for (int j = 0; j < r.rows; j++)
            {
                boxRowSumExpand<2>(src_4, cn, b40, b41);
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
        boxFilter2DTail<2, WET, ET>(src, dst, src_stride, dst_stride, r, x, cn, BoxTailOut<SCALE_T, ET>(scale));
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

    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        const BoxFilter2DRange<2> r(width, height, yidx, wholeHeight, cn);
        int x = r.xbegin;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int src_inc = src_stride;
        const int VECSZ8 = VTraits<v_uint8>::vlanes();

        // mul_hi(s + 12, 5243) >> 1 == round(s/25) exactly for every s <= 25*255;
        // no 16-bit multiplier with a 16-bit shift is exact over that range
        v_uint16 v_12 = vx_setall_u16(12);
        v_uint16 v_mulFactor = vx_setall_u16(5243);

        for (; x < r.xend; x += VECSZ8)
        {
            if( !boxFilter2DStrip(x, VECSZ8, r, src == dst) )
                break;
            const uchar* src_0 = src + x;
            const uchar* src_4 = src_0 + src_inc*4;
            uchar* dstx = dst + (ptrdiff_t)r.dstRow0*dst_stride + x;

            v_uint16 r0_lo, r0_hi, r1_lo, r1_hi, r2_lo, r2_hi, r3_lo, r3_hi;
            boxRowSumExpand<2>(src_0, cn, r0_lo, r0_hi);
            boxRowSumExpand<2>(src_0 + src_inc, cn, r1_lo, r1_hi);
            boxRowSumExpand<2>(src_0 + src_inc*2, cn, r2_lo, r2_hi);
            boxRowSumExpand<2>(src_0 + src_inc*3, cn, r3_lo, r3_hi);
            v_uint16 s_lo = v_add(v_add(r0_lo, r1_lo), v_add(r2_lo, r3_lo));
            v_uint16 s_hi = v_add(v_add(r0_hi, r1_hi), v_add(r2_hi, r3_hi));

            for (int j = 0; j < r.rows; j++)
            {
                v_uint16 r4_lo, r4_hi;
                boxRowSumExpand<2>(src_4, cn, r4_lo, r4_hi);
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
        boxFilter2DTail<2, int, uchar>(src, dst, src_stride, dst_stride, r, x, cn, BoxTailDiv25());
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
    static inline void loadRowAdd(const ET* src, int VECSZ, int cn, VET &r, VET &rv1, VET &rv2, VET &rv3)
    {
        r   = BoxTaps<2>::vsum<BoxLoad>(src, cn);
        rv1 = BoxTaps<2>::vsum<BoxLoad>(src + VECSZ, cn);
        rv2 = BoxTaps<2>::vsum<BoxLoad>(src + VECSZ*2, cn);
        rv3 = BoxTaps<2>::vsum<BoxLoad>(src + VECSZ*3, cn);
    }
#endif
    virtual void operator()(const uchar* src, uchar* dst, int src_stride, int dst_stride, int width, int height, int yidx, int wholeHeight, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();
        const BoxFilter2DRange<2> r(width, height, yidx, wholeHeight, cn);
        int x = r.xbegin;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        double _scale = scale;
        const int src_inc = src_stride / (int)sizeof(ET);
        v_float32 _v_scale = vx_setall_f32((float)_scale);
        const int VECSZ = VTraits<VET>::vlanes();
        const int VECSZ_2 = VECSZ<<1;
        const int VECSZ_3 = VECSZ*3;
        const int VECSZ_4 = VECSZ<<2;
        for (; x < r.xend; x += VECSZ_4)
        {
            if( !boxFilter2DStrip(x, VECSZ_4, r, src == dst) )
                break;
            const ET* src_0 = ((const ET*)src) + x;
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

            for (int j = 0; j < r.rows; j++)
            {
                ET* dstx = (ET*)(dst + (ptrdiff_t)(r.dstRow0 + j)*dst_stride);
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
        boxFilter2DTail<2, ET, ET>(src, dst, src_stride, dst_stride, r, x, cn, BoxTailOut<SCALE_T, ET>(scale));
    }
    double scale;
};
