// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "color.hpp"

#define IPP_DISABLE_CVTCOLOR_GRAY2BGR_8UC3 1

namespace cv
{

////////////////// Various 3/4-channel to 3/4-channel RGB transformations /////////////////

template<typename _Tp> struct v_type;

template<>
struct v_type<uchar>{
    typedef v_uint8 t;
};

template<>
struct v_type<ushort>{
    typedef v_uint16 t;
};

template<>
struct v_type<float>{
    typedef v_float32 t;
};

template<typename _Tp> struct v_set;

template<>
struct v_set<uchar>
{
    static inline v_type<uchar>::t set(uchar x)
    {
        return vx_setall_u8(x);
    }
};

template<>
struct v_set<ushort>
{
    static inline v_type<ushort>::t set(ushort x)
    {
        return vx_setall_u16(x);
    }
};

template<>
struct v_set<float>
{
    static inline v_type<float>::t set(float x)
    {
        return vx_setall_f32(x);
    }
};

template<typename _Tp>
struct RGB2RGB
{
    typedef _Tp channel_type;
    typedef typename v_type<_Tp>::t vt;

    RGB2RGB(int _srccn, int _dstcn, int _blueIdx) :
        srccn(_srccn), dstcn(_dstcn), blueIdx(_blueIdx)
    {
        CV_Assert(srccn == 3 || srccn == 4);
        CV_Assert(dstcn == 3 || dstcn == 4);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, dcn = dstcn, bi = blueIdx;
        int i = 0;
        _Tp alphav = ColorChannel<_Tp>::max();

#if CV_SIMD
        const int vsize = vt::nlanes;

        for(; i < n-vsize+1;
            i += vsize, src += vsize*scn, dst += vsize*dcn)
        {
            vt a, b, c, d;
            if(scn == 4)
            {
                v_load_deinterleave(src, a, b, c, d);
            }
            else
            {
                v_load_deinterleave(src, a, b, c);
                d = v_set<_Tp>::set(alphav);
            }
            if(bi == 2)
                swap(a, c);

            if(dcn == 4)
            {
                v_store_interleave(dst, a, b, c, d);
            }
            else
            {
                v_store_interleave(dst, a, b, c);
            }
        }
        vx_cleanup();
#endif
        for ( ; i < n; i++, src += scn, dst += dcn )
        {
            _Tp t0 = src[0], t1 = src[1], t2 = src[2];
            dst[bi  ] = t0;
            dst[1]         = t1;
            dst[bi^2] = t2;
            if(dcn == 4)
            {
                _Tp d = scn == 4 ? src[3] : alphav;
                dst[3] = d;
            }
        }
    }

    int srccn, dstcn, blueIdx;
};


/////////// Transforming 16-bit (565 or 555) RGB to/from 24/32-bit (888[8]) RGB //////////

struct RGB5x52RGB
{
    typedef uchar channel_type;

    RGB5x52RGB(int _dstcn, int _blueIdx, int _greenBits)
        : dstcn(_dstcn), blueIdx(_blueIdx), greenBits(_greenBits)
    {
        #if CV_NEON
        v_n3 = vdupq_n_u16(~3);
        v_n7 = vdupq_n_u16(~7);
        v_255 = vdupq_n_u8(255);
        v_0 = vdupq_n_u8(0);
        v_mask = vdupq_n_u16(0x8000);
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        if( greenBits == 6 )
        {
            #if CV_NEON
            for ( ; i <= n - 16; i += 16, dst += dcn * 16)
            {
                uint16x8_t v_src0 = vld1q_u16((const ushort *)src + i), v_src1 = vld1q_u16((const ushort *)src + i + 8);
                uint8x16_t v_b = vcombine_u8(vmovn_u16(vshlq_n_u16(v_src0, 3)), vmovn_u16(vshlq_n_u16(v_src1, 3)));
                uint8x16_t v_g = vcombine_u8(vmovn_u16(vandq_u16(vshrq_n_u16(v_src0, 3), v_n3)),
                                             vmovn_u16(vandq_u16(vshrq_n_u16(v_src1, 3), v_n3)));
                uint8x16_t v_r = vcombine_u8(vmovn_u16(vandq_u16(vshrq_n_u16(v_src0, 8), v_n7)),
                                             vmovn_u16(vandq_u16(vshrq_n_u16(v_src1, 8), v_n7)));
                if (dcn == 3)
                {
                    uint8x16x3_t v_dst;
                    v_dst.val[bidx] = v_b;
                    v_dst.val[1] = v_g;
                    v_dst.val[bidx^2] = v_r;
                    vst3q_u8(dst, v_dst);
                }
                else
                {
                    uint8x16x4_t v_dst;
                    v_dst.val[bidx] = v_b;
                    v_dst.val[1] = v_g;
                    v_dst.val[bidx^2] = v_r;
                    v_dst.val[3] = v_255;
                    vst4q_u8(dst, v_dst);
                }
            }
            #endif
            for( ; i < n; i++, dst += dcn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[bidx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 3) & ~3);
                dst[bidx ^ 2] = (uchar)((t >> 8) & ~7);
                if( dcn == 4 )
                    dst[3] = 255;
            }
        }
        else
        {
            #if CV_NEON
            for ( ; i <= n - 16; i += 16, dst += dcn * 16)
            {
                uint16x8_t v_src0 = vld1q_u16((const ushort *)src + i), v_src1 = vld1q_u16((const ushort *)src + i + 8);
                uint8x16_t v_b = vcombine_u8(vmovn_u16(vshlq_n_u16(v_src0, 3)), vmovn_u16(vshlq_n_u16(v_src1, 3)));
                uint8x16_t v_g = vcombine_u8(vmovn_u16(vandq_u16(vshrq_n_u16(v_src0, 2), v_n7)),
                                             vmovn_u16(vandq_u16(vshrq_n_u16(v_src1, 2), v_n7)));
                uint8x16_t v_r = vcombine_u8(vmovn_u16(vandq_u16(vshrq_n_u16(v_src0, 7), v_n7)),
                                             vmovn_u16(vandq_u16(vshrq_n_u16(v_src1, 7), v_n7)));
                if (dcn == 3)
                {
                    uint8x16x3_t v_dst;
                    v_dst.val[bidx] = v_b;
                    v_dst.val[1] = v_g;
                    v_dst.val[bidx^2] = v_r;
                    vst3q_u8(dst, v_dst);
                }
                else
                {
                    uint8x16x4_t v_dst;
                    v_dst.val[bidx] = v_b;
                    v_dst.val[1] = v_g;
                    v_dst.val[bidx^2] = v_r;
                    v_dst.val[3] = vbslq_u8(vcombine_u8(vqmovn_u16(vandq_u16(v_src0, v_mask)),
                                                        vqmovn_u16(vandq_u16(v_src1, v_mask))), v_255, v_0);
                    vst4q_u8(dst, v_dst);
                }
            }
            #endif
            for( ; i < n; i++, dst += dcn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[bidx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 2) & ~7);
                dst[bidx ^ 2] = (uchar)((t >> 7) & ~7);
                if( dcn == 4 )
                    dst[3] = t & 0x8000 ? 255 : 0;
            }
        }
    }

    int dstcn, blueIdx, greenBits;
    #if CV_NEON
    uint16x8_t v_n3, v_n7, v_mask;
    uint8x16_t v_255, v_0;
    #endif
};


struct RGB2RGB5x5
{
    typedef uchar channel_type;

    RGB2RGB5x5(int _srccn, int _blueIdx, int _greenBits)
        : srccn(_srccn), blueIdx(_blueIdx), greenBits(_greenBits)
    {
        #if CV_NEON
        v_n3 = vdup_n_u8(~3);
        v_n7 = vdup_n_u8(~7);
        v_mask = vdupq_n_u16(0x8000);
        v_0 = vdupq_n_u16(0);
        v_full = vdupq_n_u16(0xffff);
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        if (greenBits == 6)
        {
            if (scn == 3)
            {
                #if CV_NEON
                for ( ; i <= n - 8; i += 8, src += 24 )
                {
                    uint8x8x3_t v_src = vld3_u8(src);
                    uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src.val[bidx], 3));
                    v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[1], v_n3)), 3));
                    v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[bidx^2], v_n7)), 8));
                    vst1q_u16((ushort *)dst + i, v_dst);
                }
                #endif
                for ( ; i < n; i++, src += 3 )
                    ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~3) << 3)|((src[bidx^2]&~7) << 8));
            }
            else
            {
                #if CV_NEON
                for ( ; i <= n - 8; i += 8, src += 32 )
                {
                    uint8x8x4_t v_src = vld4_u8(src);
                    uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src.val[bidx], 3));
                    v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[1], v_n3)), 3));
                    v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[bidx^2], v_n7)), 8));
                    vst1q_u16((ushort *)dst + i, v_dst);
                }
                #endif
                for ( ; i < n; i++, src += 4 )
                    ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~3) << 3)|((src[bidx^2]&~7) << 8));
            }
        }
        else if (scn == 3)
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8, src += 24 )
            {
                uint8x8x3_t v_src = vld3_u8(src);
                uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src.val[bidx], 3));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[1], v_n7)), 2));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[bidx^2], v_n7)), 7));
                vst1q_u16((ushort *)dst + i, v_dst);
            }
            #endif
            for ( ; i < n; i++, src += 3 )
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~7) << 2)|((src[bidx^2]&~7) << 7));
        }
        else
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8, src += 32 )
            {
                uint8x8x4_t v_src = vld4_u8(src);
                uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src.val[bidx], 3));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[1], v_n7)), 2));
                v_dst = vorrq_u16(v_dst, vorrq_u16(vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[bidx^2], v_n7)), 7),
                                                   vbslq_u16(veorq_u16(vceqq_u16(vmovl_u8(v_src.val[3]), v_0), v_full), v_mask, v_0)));
                vst1q_u16((ushort *)dst + i, v_dst);
            }
            #endif
            for ( ; i < n; i++, src += 4 )
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~7) << 2)|
                    ((src[bidx^2]&~7) << 7)|(src[3] ? 0x8000 : 0));
        }
    }

    int srccn, blueIdx, greenBits;
    #if CV_NEON
    uint8x8_t v_n3, v_n7;
    uint16x8_t v_mask, v_0, v_full;
    #endif
};

///////////////////////////////// Color to/from Grayscale ////////////////////////////////

template<typename _Tp>
struct Gray2RGB
{
    typedef _Tp channel_type;

    Gray2RGB(int _dstcn) : dstcn(_dstcn) {}
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        if( dstcn == 3 )
            for( int i = 0; i < n; i++, dst += 3 )
            {
                dst[0] = dst[1] = dst[2] = src[i];
            }
        else
        {
            _Tp alpha = ColorChannel<_Tp>::max();
            for( int i = 0; i < n; i++, dst += 4 )
            {
                dst[0] = dst[1] = dst[2] = src[i];
                dst[3] = alpha;
            }
        }
    }

    int dstcn;
};


struct Gray2RGB5x5
{
    typedef uchar channel_type;

    Gray2RGB5x5(int _greenBits) : greenBits(_greenBits)
    {
        #if CV_NEON
        v_n7 = vdup_n_u8(~7);
        v_n3 = vdup_n_u8(~3);
        #elif CV_SSE2
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        v_n7 = _mm_set1_epi16(~7);
        v_n3 = _mm_set1_epi16(~3);
        v_zero = _mm_setzero_si128();
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i = 0;
        if( greenBits == 6 )
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8 )
            {
                uint8x8_t v_src = vld1_u8(src + i);
                uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src, 3));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src, v_n3)), 3));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src, v_n7)), 8));
                vst1q_u16((ushort *)dst + i, v_dst);
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; i <= n - 16; i += 16 )
                {
                    __m128i v_src = _mm_loadu_si128((__m128i const *)(src + i));

                    __m128i v_src_p = _mm_unpacklo_epi8(v_src, v_zero);
                    __m128i v_dst = _mm_or_si128(_mm_srli_epi16(v_src_p, 3),
                                    _mm_or_si128(_mm_slli_epi16(_mm_and_si128(v_src_p, v_n3), 3),
                                                 _mm_slli_epi16(_mm_and_si128(v_src_p, v_n7), 8)));
                    _mm_storeu_si128((__m128i *)((ushort *)dst + i), v_dst);

                    v_src_p = _mm_unpackhi_epi8(v_src, v_zero);
                    v_dst = _mm_or_si128(_mm_srli_epi16(v_src_p, 3),
                            _mm_or_si128(_mm_slli_epi16(_mm_and_si128(v_src_p, v_n3), 3),
                                         _mm_slli_epi16(_mm_and_si128(v_src_p, v_n7), 8)));
                    _mm_storeu_si128((__m128i *)((ushort *)dst + i + 8), v_dst);
                }
            }
            #endif
            for ( ; i < n; i++ )
            {
                int t = src[i];
                ((ushort*)dst)[i] = (ushort)((t >> 3)|((t & ~3) << 3)|((t & ~7) << 8));
            }
        }
        else
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8 )
            {
                uint16x8_t v_src = vmovl_u8(vshr_n_u8(vld1_u8(src + i), 3));
                uint16x8_t v_dst = vorrq_u16(vorrq_u16(v_src, vshlq_n_u16(v_src, 5)), vshlq_n_u16(v_src, 10));
                vst1q_u16((ushort *)dst + i, v_dst);
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; i <= n - 16; i += 8 )
                {
                    __m128i v_src = _mm_loadu_si128((__m128i const *)(src + i));

                    __m128i v_src_p = _mm_srli_epi16(_mm_unpacklo_epi8(v_src, v_zero), 3);
                    __m128i v_dst = _mm_or_si128(v_src_p,
                                    _mm_or_si128(_mm_slli_epi32(v_src_p, 5),
                                                 _mm_slli_epi16(v_src_p, 10)));
                    _mm_storeu_si128((__m128i *)((ushort *)dst + i), v_dst);

                    v_src_p = _mm_srli_epi16(_mm_unpackhi_epi8(v_src, v_zero), 3);
                    v_dst = _mm_or_si128(v_src_p,
                            _mm_or_si128(_mm_slli_epi16(v_src_p, 5),
                                         _mm_slli_epi16(v_src_p, 10)));
                    _mm_storeu_si128((__m128i *)((ushort *)dst + i + 8), v_dst);
                }
            }
            #endif
            for( ; i < n; i++ )
            {
                int t = src[i] >> 3;
                ((ushort*)dst)[i] = (ushort)(t|(t << 5)|(t << 10));
            }
        }
    }
    int greenBits;

    #if CV_NEON
    uint8x8_t v_n7, v_n3;
    #elif CV_SSE2
    __m128i v_n7, v_n3, v_zero;
    bool haveSIMD;
    #endif
};


struct RGB5x52Gray
{
    typedef uchar channel_type;

    RGB5x52Gray(int _greenBits) : greenBits(_greenBits)
    {
        #if CV_NEON
        v_b2y = vdup_n_u16(B2Y);
        v_g2y = vdup_n_u16(G2Y);
        v_r2y = vdup_n_u16(R2Y);
        v_delta = vdupq_n_u32(1 << (yuv_shift - 1));
        v_f8 = vdupq_n_u16(0xf8);
        v_fc = vdupq_n_u16(0xfc);
        #elif CV_SSE2
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        const __m128i v_b2y = _mm_set1_epi16(B2Y);
        const __m128i v_g2y = _mm_set1_epi16(G2Y);
        v_bg2y = _mm_unpacklo_epi16(v_b2y, v_g2y);
        const __m128i v_r2y = _mm_set1_epi16(R2Y);
        const __m128i v_one = _mm_set1_epi16(1);
        v_rd2y = _mm_unpacklo_epi16(v_r2y, v_one);
        v_delta = _mm_slli_epi16(v_one, yuv_shift - 1);
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i = 0;
        if( greenBits == 6 )
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8)
            {
                uint16x8_t v_src = vld1q_u16((ushort *)src + i);
                uint16x8_t v_t0 = vandq_u16(vshlq_n_u16(v_src, 3), v_f8),
                           v_t1 = vandq_u16(vshrq_n_u16(v_src, 3), v_fc),
                           v_t2 = vandq_u16(vshrq_n_u16(v_src, 8), v_f8);

                uint32x4_t v_dst0 = vmlal_u16(vmlal_u16(vmull_u16(vget_low_u16(v_t0), v_b2y),
                                              vget_low_u16(v_t1), v_g2y), vget_low_u16(v_t2), v_r2y);
                uint32x4_t v_dst1 = vmlal_u16(vmlal_u16(vmull_u16(vget_high_u16(v_t0), v_b2y),
                                              vget_high_u16(v_t1), v_g2y), vget_high_u16(v_t2), v_r2y);
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_delta), yuv_shift);
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_delta), yuv_shift);

                vst1_u8(dst + i, vmovn_u16(vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1))));
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; i <= n - 8; i += 8)
                {
                    __m128i v_src = _mm_loadu_si128((__m128i const *)((ushort *)src + i));
                    __m128i v_b = _mm_srli_epi16(_mm_slli_epi16(v_src, 11), 8),
                            v_g = _mm_srli_epi16(_mm_slli_epi16(_mm_srli_epi16(v_src, 5), 10),8),
                            v_r = _mm_slli_epi16(_mm_srli_epi16(v_src, 11), 3);

                    __m128i v_bg_lo = _mm_unpacklo_epi16(v_b, v_g);
                    __m128i v_rd_lo = _mm_unpacklo_epi16(v_r, v_delta);
                    __m128i v_bg_hi = _mm_unpackhi_epi16(v_b, v_g);
                    __m128i v_rd_hi = _mm_unpackhi_epi16(v_r, v_delta);
                    v_bg_lo = _mm_madd_epi16(v_bg_lo, v_bg2y);
                    v_rd_lo = _mm_madd_epi16(v_rd_lo, v_rd2y);
                    v_bg_hi = _mm_madd_epi16(v_bg_hi, v_bg2y);
                    v_rd_hi = _mm_madd_epi16(v_rd_hi, v_rd2y);

                    __m128i v_bgr_lo = _mm_add_epi32(v_bg_lo, v_rd_lo);
                    __m128i v_bgr_hi = _mm_add_epi32(v_bg_hi, v_rd_hi);
                    v_bgr_lo = _mm_srli_epi32(v_bgr_lo, yuv_shift);
                    v_bgr_hi = _mm_srli_epi32(v_bgr_hi, yuv_shift);

                    __m128i v_dst = _mm_packs_epi32(v_bgr_lo, v_bgr_hi);
                    v_dst = _mm_packus_epi16(v_dst, v_dst);
                    _mm_storel_epi64((__m128i *)(dst + i), v_dst);
                }
            }
            #endif
            for ( ; i < n; i++)
            {
                int t = ((ushort*)src)[i];
                dst[i] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                           ((t >> 3) & 0xfc)*G2Y +
                                           ((t >> 8) & 0xf8)*R2Y, yuv_shift);
            }
        }
        else
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8)
            {
                uint16x8_t v_src = vld1q_u16((ushort *)src + i);
                uint16x8_t v_t0 = vandq_u16(vshlq_n_u16(v_src, 3), v_f8),
                           v_t1 = vandq_u16(vshrq_n_u16(v_src, 2), v_f8),
                           v_t2 = vandq_u16(vshrq_n_u16(v_src, 7), v_f8);

                uint32x4_t v_dst0 = vmlal_u16(vmlal_u16(vmull_u16(vget_low_u16(v_t0), v_b2y),
                                              vget_low_u16(v_t1), v_g2y), vget_low_u16(v_t2), v_r2y);
                uint32x4_t v_dst1 = vmlal_u16(vmlal_u16(vmull_u16(vget_high_u16(v_t0), v_b2y),
                                              vget_high_u16(v_t1), v_g2y), vget_high_u16(v_t2), v_r2y);
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_delta), yuv_shift);
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_delta), yuv_shift);

                vst1_u8(dst + i, vmovn_u16(vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1))));
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; i <= n - 8; i += 8)
                {
                    __m128i v_src = _mm_loadu_si128((__m128i const *)((ushort *)src + i));
                    __m128i v_b = _mm_srli_epi16(_mm_slli_epi16(v_src, 11), 8),
                            v_g = _mm_srli_epi16(_mm_slli_epi16(_mm_srli_epi16(v_src, 5), 11),8),
                            v_r = _mm_srli_epi16(_mm_slli_epi16(_mm_srli_epi16(v_src, 10), 11),8);

                    __m128i v_bg_lo = _mm_unpacklo_epi16(v_b, v_g);
                    __m128i v_rd_lo = _mm_unpacklo_epi16(v_r, v_delta);
                    __m128i v_bg_hi = _mm_unpackhi_epi16(v_b, v_g);
                    __m128i v_rd_hi = _mm_unpackhi_epi16(v_r, v_delta);
                    v_bg_lo = _mm_madd_epi16(v_bg_lo, v_bg2y);
                    v_rd_lo = _mm_madd_epi16(v_rd_lo, v_rd2y);
                    v_bg_hi = _mm_madd_epi16(v_bg_hi, v_bg2y);
                    v_rd_hi = _mm_madd_epi16(v_rd_hi, v_rd2y);

                    __m128i v_bgr_lo = _mm_add_epi32(v_bg_lo, v_rd_lo);
                    __m128i v_bgr_hi = _mm_add_epi32(v_bg_hi, v_rd_hi);
                    v_bgr_lo = _mm_srli_epi32(v_bgr_lo, yuv_shift);
                    v_bgr_hi = _mm_srli_epi32(v_bgr_hi, yuv_shift);

                    __m128i v_dst = _mm_packs_epi32(v_bgr_lo, v_bgr_hi);
                    v_dst = _mm_packus_epi16(v_dst, v_dst);
                    _mm_storel_epi64((__m128i *)(dst + i), v_dst);
                }
            }
            #endif
            for ( ; i < n; i++)
            {
                int t = ((ushort*)src)[i];
                dst[i] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                           ((t >> 2) & 0xf8)*G2Y +
                                           ((t >> 7) & 0xf8)*R2Y, yuv_shift);
            }
        }
    }
    int greenBits;

    #if CV_NEON
    uint16x4_t v_b2y, v_g2y, v_r2y;
    uint32x4_t v_delta;
    uint16x8_t v_f8, v_fc;
    #elif CV_SSE2
    bool haveSIMD;
    __m128i v_bg2y, v_rd2y;
    __m128i v_delta;
    #endif
};


template<typename _Tp> struct RGB2Gray
{
    typedef _Tp channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { R2YF, G2YF, B2YF };
        memcpy( coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]) );
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = saturate_cast<_Tp>(src[0]*cb + src[1]*cg + src[2]*cr);
    }
    int srccn;
    float coeffs[3];
};

template<> struct RGB2Gray<uchar>
{
    typedef uchar channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* coeffs) : srccn(_srccn)
    {
        const int coeffs0[] = { R2Y, G2Y, B2Y };
        if(!coeffs) coeffs = coeffs0;

        int b = 0, g = 0, r = (1 << (yuv_shift-1));
        int db = coeffs[blueIdx^2], dg = coeffs[1], dr = coeffs[blueIdx];

        for( int i = 0; i < 256; i++, b += db, g += dg, r += dr )
        {
            tab[i] = b;
            tab[i+256] = g;
            tab[i+512] = r;
        }
    }
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn;
        const int* _tab = tab;
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = (uchar)((_tab[src[0]] + _tab[src[1]+256] + _tab[src[2]+512]) >> yuv_shift);
    }
    int srccn;
    int tab[256*3];
};

#if CV_NEON

template <>
struct RGB2Gray<ushort>
{
    typedef ushort channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) :
        srccn(_srccn)
    {
        static const int coeffs0[] = { R2Y, G2Y, B2Y };
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]));
        if( blueIdx == 0 )
            std::swap(coeffs[0], coeffs[2]);

        v_cb = vdup_n_u16(coeffs[0]);
        v_cg = vdup_n_u16(coeffs[1]);
        v_cr = vdup_n_u16(coeffs[2]);
        v_delta = vdupq_n_u32(1 << (yuv_shift - 1));
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn, cb = coeffs[0], cg = coeffs[1], cr = coeffs[2], i = 0;

        for ( ; i <= n - 8; i += 8, src += scn * 8)
        {
            uint16x8_t v_b, v_r, v_g;
            if (scn == 3)
            {
                uint16x8x3_t v_src = vld3q_u16(src);
                v_b = v_src.val[0];
                v_g = v_src.val[1];
                v_r = v_src.val[2];
            }
            else
            {
                uint16x8x4_t v_src = vld4q_u16(src);
                v_b = v_src.val[0];
                v_g = v_src.val[1];
                v_r = v_src.val[2];
            }

            uint32x4_t v_dst0_ = vmlal_u16(vmlal_u16(
                                           vmull_u16(vget_low_u16(v_b), v_cb),
                                                     vget_low_u16(v_g), v_cg),
                                                     vget_low_u16(v_r), v_cr);
            uint32x4_t v_dst1_ = vmlal_u16(vmlal_u16(
                                           vmull_u16(vget_high_u16(v_b), v_cb),
                                                     vget_high_u16(v_g), v_cg),
                                                     vget_high_u16(v_r), v_cr);

            uint16x4_t v_dst0 = vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst0_, v_delta), yuv_shift));
            uint16x4_t v_dst1 = vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst1_, v_delta), yuv_shift));

            vst1q_u16(dst + i, vcombine_u16(v_dst0, v_dst1));
        }

        for ( ; i <= n - 4; i += 4, src += scn * 4)
        {
            uint16x4_t v_b, v_r, v_g;
            if (scn == 3)
            {
                uint16x4x3_t v_src = vld3_u16(src);
                v_b = v_src.val[0];
                v_g = v_src.val[1];
                v_r = v_src.val[2];
            }
            else
            {
                uint16x4x4_t v_src = vld4_u16(src);
                v_b = v_src.val[0];
                v_g = v_src.val[1];
                v_r = v_src.val[2];
            }

            uint32x4_t v_dst = vmlal_u16(vmlal_u16(
                                         vmull_u16(v_b, v_cb),
                                                   v_g, v_cg),
                                                   v_r, v_cr);

            vst1_u16(dst + i, vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst, v_delta), yuv_shift)));
        }

        for( ; i < n; i++, src += scn)
            dst[i] = (ushort)CV_DESCALE((unsigned)(src[0]*cb + src[1]*cg + src[2]*cr), yuv_shift);
    }

    int srccn, coeffs[3];
    uint16x4_t v_cb, v_cg, v_cr;
    uint32x4_t v_delta;
};

template <>
struct RGB2Gray<float>
{
    typedef float channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { R2YF, G2YF, B2YF };
        memcpy( coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]) );
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);

        v_cb = vdupq_n_f32(coeffs[0]);
        v_cg = vdupq_n_f32(coeffs[1]);
        v_cr = vdupq_n_f32(coeffs[2]);
    }

    void operator()(const float * src, float * dst, int n) const
    {
        int scn = srccn, i = 0;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];

        if (scn == 3)
        {
            for ( ; i <= n - 8; i += 8, src += scn * 8)
            {
                float32x4x3_t v_src = vld3q_f32(src);
                vst1q_f32(dst + i, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));

                v_src = vld3q_f32(src + scn * 4);
                vst1q_f32(dst + i + 4, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));
            }

            for ( ; i <= n - 4; i += 4, src += scn * 4)
            {
                float32x4x3_t v_src = vld3q_f32(src);
                vst1q_f32(dst + i, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));
            }
        }
        else
        {
            for ( ; i <= n - 8; i += 8, src += scn * 8)
            {
                float32x4x4_t v_src = vld4q_f32(src);
                vst1q_f32(dst + i, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));

                v_src = vld4q_f32(src + scn * 4);
                vst1q_f32(dst + i + 4, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));
            }

            for ( ; i <= n - 4; i += 4, src += scn * 4)
            {
                float32x4x4_t v_src = vld4q_f32(src);
                vst1q_f32(dst + i, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));
            }
        }

        for ( ; i < n; i++, src += scn)
            dst[i] = src[0]*cb + src[1]*cg + src[2]*cr;
    }

    int srccn;
    float coeffs[3];
    float32x4_t v_cb, v_cg, v_cr;
};

#elif CV_SSE2

#if CV_SSE4_1

template <>
struct RGB2Gray<ushort>
{
    typedef ushort channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) :
        srccn(_srccn)
    {
        static const int coeffs0[] = { R2Y, G2Y, B2Y };
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]));
        if( blueIdx == 0 )
            std::swap(coeffs[0], coeffs[2]);

        v_delta = _mm_set1_epi32(1 << (yuv_shift - 1));
        v_zero = _mm_setzero_si128();

        haveSIMD = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    // 16s x 8
    void process(__m128i* v_rgb, __m128i* v_coeffs,
                 __m128i & v_gray) const
    {
        __m128i v_rgb_hi[4];
        v_rgb_hi[0] = _mm_cmplt_epi16(v_rgb[0], v_zero);
        v_rgb_hi[1] = _mm_cmplt_epi16(v_rgb[1], v_zero);
        v_rgb_hi[2] = _mm_cmplt_epi16(v_rgb[2], v_zero);
        v_rgb_hi[3] = _mm_cmplt_epi16(v_rgb[3], v_zero);

        v_rgb_hi[0] = _mm_and_si128(v_rgb_hi[0], v_coeffs[1]);
        v_rgb_hi[1] = _mm_and_si128(v_rgb_hi[1], v_coeffs[1]);
        v_rgb_hi[2] = _mm_and_si128(v_rgb_hi[2], v_coeffs[1]);
        v_rgb_hi[3] = _mm_and_si128(v_rgb_hi[3], v_coeffs[1]);

        v_rgb_hi[0] = _mm_hadd_epi16(v_rgb_hi[0], v_rgb_hi[1]);
        v_rgb_hi[2] = _mm_hadd_epi16(v_rgb_hi[2], v_rgb_hi[3]);
        v_rgb_hi[0] = _mm_hadd_epi16(v_rgb_hi[0], v_rgb_hi[2]);

        v_rgb[0] = _mm_madd_epi16(v_rgb[0], v_coeffs[0]);
        v_rgb[1] = _mm_madd_epi16(v_rgb[1], v_coeffs[0]);
        v_rgb[2] = _mm_madd_epi16(v_rgb[2], v_coeffs[0]);
        v_rgb[3] = _mm_madd_epi16(v_rgb[3], v_coeffs[0]);

        v_rgb[0] = _mm_hadd_epi32(v_rgb[0], v_rgb[1]);
        v_rgb[2] = _mm_hadd_epi32(v_rgb[2], v_rgb[3]);

        v_rgb[0] = _mm_add_epi32(v_rgb[0], v_delta);
        v_rgb[2] = _mm_add_epi32(v_rgb[2], v_delta);

        v_rgb[0] = _mm_srai_epi32(v_rgb[0], yuv_shift);
        v_rgb[2] = _mm_srai_epi32(v_rgb[2], yuv_shift);

        v_gray = _mm_packs_epi32(v_rgb[0], v_rgb[2]);
        v_gray = _mm_add_epi16(v_gray, v_rgb_hi[0]);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn, cb = coeffs[0], cg = coeffs[1], cr = coeffs[2], i = 0;

        if (scn == 3 && haveSIMD)
        {
            __m128i v_coeffs[2];
            v_coeffs[0] = _mm_set_epi16(0, (short)coeffs[2], (short)coeffs[1], (short)coeffs[0], (short)coeffs[2], (short)coeffs[1], (short)coeffs[0], 0);
            v_coeffs[1] = _mm_slli_epi16(v_coeffs[0], 2);

            for ( ; i <= n - 8; i += 8, src += scn * 8)
            {
                __m128i v_src[3];
                v_src[0] = _mm_loadu_si128((__m128i const *)(src));
                v_src[1] = _mm_loadu_si128((__m128i const *)(src + 8));
                v_src[2] = _mm_loadu_si128((__m128i const *)(src + 16));

                __m128i v_rgb[4];
                v_rgb[0] = _mm_slli_si128(v_src[0], 2);
                v_rgb[1] = _mm_alignr_epi8(v_src[1], v_src[0], 10);
                v_rgb[2] = _mm_alignr_epi8(v_src[2], v_src[1], 6);
                v_rgb[3] = _mm_srli_si128(v_src[2], 2);

                __m128i v_gray;
                process(v_rgb, v_coeffs,
                        v_gray);

                _mm_storeu_si128((__m128i *)(dst + i), v_gray);
            }
        }
        else if (scn == 4 && haveSIMD)
        {
            __m128i v_coeffs[2];
            v_coeffs[0] = _mm_set_epi16(0, (short)coeffs[2], (short)coeffs[1], (short)coeffs[0], 0, (short)coeffs[2], (short)coeffs[1], (short)coeffs[0]);
            v_coeffs[1] = _mm_slli_epi16(v_coeffs[0], 2);

            for ( ; i <= n - 8; i += 8, src += scn * 8)
            {
                __m128i v_rgb[4];
                v_rgb[0] = _mm_loadu_si128((__m128i const *)(src));
                v_rgb[1] = _mm_loadu_si128((__m128i const *)(src + 8));
                v_rgb[2] = _mm_loadu_si128((__m128i const *)(src + 16));
                v_rgb[3] = _mm_loadu_si128((__m128i const *)(src + 24));

                __m128i v_gray;
                process(v_rgb, v_coeffs,
                        v_gray);

                _mm_storeu_si128((__m128i *)(dst + i), v_gray);
            }
        }

        for( ; i < n; i++, src += scn)
            dst[i] = (ushort)CV_DESCALE((unsigned)(src[0]*cb + src[1]*cg + src[2]*cr), yuv_shift);
    }

    int srccn, coeffs[3];
    __m128i v_delta;
    __m128i v_zero;
    bool haveSIMD;
};

#endif // CV_SSE4_1

template <>
struct RGB2Gray<float>
{
    typedef float channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { R2YF, G2YF, B2YF };
        memcpy( coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]) );
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);

        v_cb = _mm_set1_ps(coeffs[0]);
        v_cg = _mm_set1_ps(coeffs[1]);
        v_cr = _mm_set1_ps(coeffs[2]);

        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
    }

    void process(__m128 v_b, __m128 v_g, __m128 v_r,
                 __m128 & v_gray) const
    {
        v_gray = _mm_mul_ps(v_r, v_cr);
        v_gray = _mm_add_ps(v_gray, _mm_mul_ps(v_g, v_cg));
        v_gray = _mm_add_ps(v_gray, _mm_mul_ps(v_b, v_cb));
    }

    void operator()(const float * src, float * dst, int n) const
    {
        int scn = srccn, i = 0;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];

        if (scn == 3 && haveSIMD)
        {
            for ( ; i <= n - 8; i += 8, src += scn * 8)
            {
                __m128 v_r0 = _mm_loadu_ps(src);
                __m128 v_r1 = _mm_loadu_ps(src + 4);
                __m128 v_g0 = _mm_loadu_ps(src + 8);
                __m128 v_g1 = _mm_loadu_ps(src + 12);
                __m128 v_b0 = _mm_loadu_ps(src + 16);
                __m128 v_b1 = _mm_loadu_ps(src + 20);

                _mm_deinterleave_ps(v_r0, v_r1, v_g0, v_g1, v_b0, v_b1);

                __m128 v_gray0;
                process(v_r0, v_g0, v_b0,
                        v_gray0);

                __m128 v_gray1;
                process(v_r1, v_g1, v_b1,
                        v_gray1);

                _mm_storeu_ps(dst + i, v_gray0);
                _mm_storeu_ps(dst + i + 4, v_gray1);
            }
        }
        else if (scn == 4 && haveSIMD)
        {
            for ( ; i <= n - 8; i += 8, src += scn * 8)
            {
                __m128 v_r0 = _mm_loadu_ps(src);
                __m128 v_r1 = _mm_loadu_ps(src + 4);
                __m128 v_g0 = _mm_loadu_ps(src + 8);
                __m128 v_g1 = _mm_loadu_ps(src + 12);
                __m128 v_b0 = _mm_loadu_ps(src + 16);
                __m128 v_b1 = _mm_loadu_ps(src + 20);
                __m128 v_a0 = _mm_loadu_ps(src + 24);
                __m128 v_a1 = _mm_loadu_ps(src + 28);

                _mm_deinterleave_ps(v_r0, v_r1, v_g0, v_g1, v_b0, v_b1, v_a0, v_a1);

                __m128 v_gray0;
                process(v_r0, v_g0, v_b0,
                        v_gray0);

                __m128 v_gray1;
                process(v_r1, v_g1, v_b1,
                        v_gray1);

                _mm_storeu_ps(dst + i, v_gray0);
                _mm_storeu_ps(dst + i + 4, v_gray1);
            }
        }

        for ( ; i < n; i++, src += scn)
            dst[i] = src[0]*cb + src[1]*cg + src[2]*cr;
    }

    int srccn;
    float coeffs[3];
    __m128 v_cb, v_cg, v_cr;
    bool haveSIMD;
};

#endif // CV_SSE2

#if !CV_NEON && !CV_SSE4_1

template<> struct RGB2Gray<ushort>
{
    typedef ushort channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] = { R2Y, G2Y, B2Y };
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]));
        if( blueIdx == 0 )
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn, cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = (ushort)CV_DESCALE((unsigned)(src[0]*cb + src[1]*cg + src[2]*cr), yuv_shift);
    }
    int srccn;
    int coeffs[3];
};

#endif // !CV_NEON && !CV_SSE4_1


/////////////////////////// RGBA <-> mRGBA (alpha premultiplied) //////////////

template<typename _Tp>
struct RGBA2mRGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val  = ColorChannel<_Tp>::max();
        _Tp half_val = ColorChannel<_Tp>::half();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;

            *dst++ = (v0 * v3 + half_val) / max_val;
            *dst++ = (v1 * v3 + half_val) / max_val;
            *dst++ = (v2 * v3 + half_val) / max_val;
            *dst++ = v3;
        }
    }
};


template<typename _Tp>
struct mRGBA2RGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val = ColorChannel<_Tp>::max();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;
            _Tp v3_half = v3 / 2;

            *dst++ = (v3==0)? 0 : (v0 * max_val + v3_half) / v3;
            *dst++ = (v3==0)? 0 : (v1 * max_val + v3_half) / v3;
            *dst++ = (v3==0)? 0 : (v2 * max_val + v3_half) / v3;
            *dst++ = v3;
        }
    }
};

//
// IPP functions
//

#if NEED_IPP

static ippiColor2GrayFunc ippiColor2GrayC3Tab[] =
{
    (ippiColor2GrayFunc)ippiColorToGray_8u_C3C1R, 0, (ippiColor2GrayFunc)ippiColorToGray_16u_C3C1R, 0,
    0, (ippiColor2GrayFunc)ippiColorToGray_32f_C3C1R, 0, 0
};

static ippiColor2GrayFunc ippiColor2GrayC4Tab[] =
{
    (ippiColor2GrayFunc)ippiColorToGray_8u_AC4C1R, 0, (ippiColor2GrayFunc)ippiColorToGray_16u_AC4C1R, 0,
    0, (ippiColor2GrayFunc)ippiColorToGray_32f_AC4C1R, 0, 0
};

static ippiGeneralFunc ippiRGB2GrayC3Tab[] =
{
    (ippiGeneralFunc)ippiRGBToGray_8u_C3C1R, 0, (ippiGeneralFunc)ippiRGBToGray_16u_C3C1R, 0,
    0, (ippiGeneralFunc)ippiRGBToGray_32f_C3C1R, 0, 0
};

static ippiGeneralFunc ippiRGB2GrayC4Tab[] =
{
    (ippiGeneralFunc)ippiRGBToGray_8u_AC4C1R, 0, (ippiGeneralFunc)ippiRGBToGray_16u_AC4C1R, 0,
    0, (ippiGeneralFunc)ippiRGBToGray_32f_AC4C1R, 0, 0
};


#if !IPP_DISABLE_CVTCOLOR_GRAY2BGR_8UC3
static IppStatus ippiGrayToRGB_C1C3R(const Ipp8u*  pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize roiSize)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_8u_C1C3R, pSrc, srcStep, pDst, dstStep, roiSize);
}
#endif
static IppStatus ippiGrayToRGB_C1C3R(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_16u_C1C3R, pSrc, srcStep, pDst, dstStep, roiSize);
}
static IppStatus ippiGrayToRGB_C1C3R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_32f_C1C3R, pSrc, srcStep, pDst, dstStep, roiSize);
}

static IppStatus ippiGrayToRGB_C1C4R(const Ipp8u*  pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize roiSize, Ipp8u  aval)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_8u_C1C4R, pSrc, srcStep, pDst, dstStep, roiSize, aval);
}
static IppStatus ippiGrayToRGB_C1C4R(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize, Ipp16u aval)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_16u_C1C4R, pSrc, srcStep, pDst, dstStep, roiSize, aval);
}
static IppStatus ippiGrayToRGB_C1C4R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f aval)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_32f_C1C4R, pSrc, srcStep, pDst, dstStep, roiSize, aval);
}

struct IPPColor2GrayFunctor
{
    IPPColor2GrayFunctor(ippiColor2GrayFunc _func) :
        ippiColorToGray(_func)
    {
        coeffs[0] = B2YF;
        coeffs[1] = G2YF;
        coeffs[2] = R2YF;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiColorToGray ? CV_INSTRUMENT_FUN_IPP(ippiColorToGray, src, srcStep, dst, dstStep, ippiSize(cols, rows), coeffs) >= 0 : false;
    }
private:
    ippiColor2GrayFunc ippiColorToGray;
    Ipp32f coeffs[3];
};

template <typename T>
struct IPPGray2BGRFunctor
{
    IPPGray2BGRFunctor(){}

    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiGrayToRGB_C1C3R((T*)src, srcStep, (T*)dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
};

template <typename T>
struct IPPGray2BGRAFunctor
{
    IPPGray2BGRAFunctor()
    {
        alpha = ColorChannel<T>::max();
    }

    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiGrayToRGB_C1C4R((T*)src, srcStep, (T*)dst, dstStep, ippiSize(cols, rows), alpha) >= 0;
    }

    T alpha;
};

static IppStatus CV_STDCALL ippiSwapChannels_8u_C3C4Rf(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_8u_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP8u);
}

static IppStatus CV_STDCALL ippiSwapChannels_16u_C3C4Rf(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_16u_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP16u);
}

static IppStatus CV_STDCALL ippiSwapChannels_32f_C3C4Rf(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP32f);
}

// shared
ippiReorderFunc ippiSwapChannelsC3C4RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3C4Rf, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3C4Rf, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3C4Rf, 0, 0
};

static ippiGeneralFunc ippiCopyAC4C3RTab[] =
{
    (ippiGeneralFunc)ippiCopy_8u_AC4C3R, 0, (ippiGeneralFunc)ippiCopy_16u_AC4C3R, 0,
    0, (ippiGeneralFunc)ippiCopy_32f_AC4C3R, 0, 0
};

// shared
ippiReorderFunc ippiSwapChannelsC4C3RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C4C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C4C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C4C3R, 0, 0
};

// shared
ippiReorderFunc ippiSwapChannelsC3RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3R, 0, 0
};

#if IPP_VERSION_X100 >= 810
static ippiReorderFunc ippiSwapChannelsC4RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C4R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C4R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C4R, 0, 0
};
#endif

#endif

//
// HAL functions
//

namespace hal
{

// 8u, 16u, 32f
void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, int dcn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoBGR, cv_hal_cvtBGRtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, scn, dcn, swapBlue);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
    CV_IPP_CHECK()
    {
    if(scn == 3 && dcn == 4 && !swapBlue)
    {
        if ( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                             IPPReorderFunctor(ippiSwapChannelsC3C4RTab[depth], 0, 1, 2)) )
            return;
    }
    else if(scn == 4 && dcn == 3 && !swapBlue)
    {
        if ( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                             IPPGeneralFunctor(ippiCopyAC4C3RTab[depth])) )
            return;
    }
    else if(scn == 3 && dcn == 4 && swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderFunctor(ippiSwapChannelsC3C4RTab[depth], 2, 1, 0)) )
            return;
    }
    else if(scn == 4 && dcn == 3 && swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderFunctor(ippiSwapChannelsC4C3RTab[depth], 2, 1, 0)) )
            return;
    }
    else if(scn == 3 && dcn == 3 && swapBlue)
    {
        if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                IPPReorderFunctor(ippiSwapChannelsC3RTab[depth], 2, 1, 0)) )
            return;
    }
#if IPP_VERSION_X100 >= 810
    else if(scn == 4 && dcn == 4 && swapBlue)
    {
        if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                IPPReorderFunctor(ippiSwapChannelsC4RTab[depth], 2, 1, 0)) )
            return;
    }
    }
#endif
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<uchar>(scn, dcn, blueIdx));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<ushort>(scn, dcn, blueIdx));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<float>(scn, dcn, blueIdx));
}

// only 8u
void cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int scn, bool swapBlue, int greenBits)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoBGR5x5, cv_hal_cvtBGRtoBGR5x5, src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, greenBits);

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB5x5(scn, swapBlue ? 2 : 0, greenBits));
}

// only 8u
void cvtBGR5x5toBGR(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int dcn, bool swapBlue, int greenBits)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGR5x5toBGR, cv_hal_cvtBGR5x5toBGR, src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, greenBits);

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB5x52RGB(dcn, swapBlue ? 2 : 0, greenBits));
}

// 8u, 16u, 32f
void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int scn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoGray, cv_hal_cvtBGRtoGray, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
    CV_IPP_CHECK()
    {
        if(depth == CV_32F && scn == 3 && !swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPColor2GrayFunctor(ippiColor2GrayC3Tab[depth])) )
                return;
        }
        else if(depth == CV_32F && scn == 3 && swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralFunctor(ippiRGB2GrayC3Tab[depth])) )
                return;
        }
        else if(depth == CV_32F && scn == 4 && !swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPColor2GrayFunctor(ippiColor2GrayC4Tab[depth])) )
                return;
        }
        else if(depth == CV_32F && scn == 4 && swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralFunctor(ippiRGB2GrayC4Tab[depth])) )
                return;
        }
    }
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<uchar>(scn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<ushort>(scn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<float>(scn, blueIdx, 0));
}

// 8u, 16u, 32f
void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int dcn)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtGraytoBGR, cv_hal_cvtGraytoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
    CV_IPP_CHECK()
    {
        bool ippres = false;
        if(dcn == 3)
        {
            if( depth == CV_8U )
            {
#if !IPP_DISABLE_CVTCOLOR_GRAY2BGR_8UC3
                ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRFunctor<Ipp8u>());
#endif
            }
            else if( depth == CV_16U )
                ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRFunctor<Ipp16u>());
            else
                ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRFunctor<Ipp32f>());
        }
        else if(dcn == 4)
        {
            if( depth == CV_8U )
                ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRAFunctor<Ipp8u>());
            else if( depth == CV_16U )
                ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRAFunctor<Ipp16u>());
            else
                ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRAFunctor<Ipp32f>());
        }
        if(ippres)
            return;
    }
#endif

    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<uchar>(dcn));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<ushort>(dcn));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<float>(dcn));
}

// only 8u
void cvtBGR5x5toGray(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGR5x5toGray, cv_hal_cvtBGR5x5toGray, src_data, src_step, dst_data, dst_step, width, height, greenBits);
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB5x52Gray(greenBits));
}

// only 8u
void cvtGraytoBGR5x5(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtGraytoBGR5x5, cv_hal_cvtGraytoBGR5x5, src_data, src_step, dst_data, dst_step, width, height, greenBits);
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB5x5(greenBits));
}

void cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtRGBAtoMultipliedRGBA, cv_hal_cvtRGBAtoMultipliedRGBA, src_data, src_step, dst_data, dst_step, width, height);

#ifdef HAVE_IPP
    CV_IPP_CHECK()
    {
    if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                        IPPGeneralFunctor((ippiGeneralFunc)ippiAlphaPremul_8u_AC4R)))
        return;
    }
#endif

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGBA2mRGBA<uchar>());
}

void cvtMultipliedRGBAtoRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtMultipliedRGBAtoRGBA, cv_hal_cvtMultipliedRGBAtoRGBA, src_data, src_step, dst_data, dst_step, width, height);
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, mRGBA2RGBA<uchar>());
}

} // namespace hal

//
// OCL calls
//

#ifdef HAVE_OPENCL

bool oclCvtColorBGR2BGR( InputArray _src, OutputArray _dst, int dcn, bool reverse )
{
    OclHelper< Set<3, 4>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("RGB", ocl::imgproc::color_rgb_oclsrc,
                       format("-D dcn=%d -D bidx=0 -D %s", dcn, reverse ? "REVERSE" : "ORDER")))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR25x5( InputArray _src, OutputArray _dst, int bidx, int gbits )
{
    OclHelper< Set<3, 4>, Set<2>, Set<CV_8U> > h(_src, _dst, 2);

    if(!h.createKernel("RGB2RGB5x5", ocl::imgproc::color_rgb_oclsrc,
                       format("-D dcn=2 -D bidx=%d -D greenbits=%d", bidx, gbits)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColor5x52BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int gbits)
{
    OclHelper< Set<2>, Set<3, 4>, Set<CV_8U> > h(_src, _dst, dcn);

    if(!h.createKernel("RGB5x52RGB", ocl::imgproc::color_rgb_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D greenbits=%d", dcn, bidx, gbits)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColor5x52Gray( InputArray _src, OutputArray _dst, int gbits)
{
    OclHelper< Set<2>, Set<1>, Set<CV_8U> > h(_src, _dst, 1);

    if(!h.createKernel("BGR5x52Gray", ocl::imgproc::color_rgb_oclsrc,
                       format("-D dcn=1 -D bidx=0 -D greenbits=%d", gbits)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorGray25x5( InputArray _src, OutputArray _dst, int gbits)
{
    OclHelper< Set<1>, Set<2>, Set<CV_8U> > h(_src, _dst, 2);

    if(!h.createKernel("Gray2BGR5x5", ocl::imgproc::color_rgb_oclsrc,
                        format("-D dcn=2 -D bidx=0 -D greenbits=%d", gbits)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2Gray( InputArray _src, OutputArray _dst, int bidx)
{
    OclHelper< Set<3, 4>, Set<1>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 1);

    int stripeSize = 1;
    if(!h.createKernel("RGB2Gray", ocl::imgproc::color_rgb_oclsrc,
                       format("-D dcn=1 -D bidx=%d -D STRIPE_SIZE=%d", bidx, stripeSize)))
    {
        return false;
    }

    h.globalSize[0] = (h.src.cols + stripeSize - 1)/stripeSize;
    return h.run();
}

bool oclCvtColorGray2BGR( InputArray _src, OutputArray _dst, int dcn)
{
    OclHelper< Set<1>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);
    if(!h.createKernel("Gray2RGB", ocl::imgproc::color_rgb_oclsrc,
                       format("-D bidx=0 -D dcn=%d", dcn)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorRGBA2mRGBA( InputArray _src, OutputArray _dst)
{
    OclHelper< Set<4>, Set<4>, Set<CV_8U> > h(_src, _dst, 4);

    if(!h.createKernel("RGBA2mRGBA", ocl::imgproc::color_rgb_oclsrc,
                       "-D dcn=4 -D bidx=3"))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColormRGBA2RGBA( InputArray _src, OutputArray _dst)
{
    OclHelper< Set<4>, Set<4>, Set<CV_8U> > h(_src, _dst, 4);

    if(!h.createKernel("mRGBA2RGBA", ocl::imgproc::color_rgb_oclsrc,
                       "-D dcn=4 -D bidx=3"))
    {
        return false;
    }

    return h.run();
}

#endif

//
// HAL calls
//

void cvtColorBGR2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb)
{
    CvtHelper< Set<3, 4>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtBGRtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, dcn, swapb);
}

void cvtColorBGR25x5( InputArray _src, OutputArray _dst, bool swapb, int gbits)
{
    CvtHelper< Set<3, 4>, Set<2>, Set<CV_8U> > h(_src, _dst, 2);

    hal::cvtBGRtoBGR5x5(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                        h.scn, swapb, gbits);
}

void cvtColor5x52BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int gbits)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<2>, Set<3, 4>, Set<CV_8U> > h(_src, _dst, dcn);

    hal::cvtBGR5x5toBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                        dcn, swapb, gbits);
}

void cvtColorBGR2Gray( InputArray _src, OutputArray _dst, bool swapb)
{
    CvtHelper< Set<3, 4>, Set<1>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 1);

    hal::cvtBGRtoGray(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                      h.depth, h.scn, swapb);
}

void cvtColorGray2BGR( InputArray _src, OutputArray _dst, int dcn)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<1>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtGraytoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, h.depth, dcn);
}

void cvtColor5x52Gray( InputArray _src, OutputArray _dst, int gbits)
{
    CvtHelper< Set<2>, Set<1>, Set<CV_8U> > h(_src, _dst, 1);

    hal::cvtBGR5x5toGray(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, gbits);
}

void cvtColorGray25x5( InputArray _src, OutputArray _dst, int gbits)
{
    CvtHelper< Set<1>, Set<2>, Set<CV_8U> > h(_src, _dst, 2);

    hal::cvtGraytoBGR5x5(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, gbits);
}

void cvtColorRGBA2mRGBA( InputArray _src, OutputArray _dst)
{
    CvtHelper< Set<4>, Set<4>, Set<CV_8U> > h(_src, _dst, 4);

    hal::cvtRGBAtoMultipliedRGBA(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows);
}

void cvtColormRGBA2RGBA( InputArray _src, OutputArray _dst)
{
    CvtHelper< Set<4>, Set<4>, Set<CV_8U> > h(_src, _dst, 4);

    hal::cvtMultipliedRGBAtoRGBA(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows);
}

} // namespace cv
