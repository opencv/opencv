// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "color.hpp"

namespace cv
{

////////////////////////////////////// RGB <-> HSV ///////////////////////////////////////


struct RGB2HSV_b
{
    typedef uchar channel_type;

    RGB2HSV_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange)
    {
        CV_Assert( hrange == 180 || hrange == 256 );
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        const int hsv_shift = 12;

        static int sdiv_table[256];
        static int hdiv_table180[256];
        static int hdiv_table256[256];
        static volatile bool initialized = false;

        int hr = hrange;
        const int* hdiv_table = hr == 180 ? hdiv_table180 : hdiv_table256;
        n *= 3;

        if( !initialized )
        {
            sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;
            for( i = 1; i < 256; i++ )
            {
                sdiv_table[i] = saturate_cast<int>((255 << hsv_shift)/(1.*i));
                hdiv_table180[i] = saturate_cast<int>((180 << hsv_shift)/(6.*i));
                hdiv_table256[i] = saturate_cast<int>((256 << hsv_shift)/(6.*i));
            }
            initialized = true;
        }

        for( i = 0; i < n; i += 3, src += scn )
        {
            int b = src[bidx], g = src[1], r = src[bidx^2];
            int h, s, v = b;
            int vmin = b;
            int vr, vg;

            CV_CALC_MAX_8U( v, g );
            CV_CALC_MAX_8U( v, r );
            CV_CALC_MIN_8U( vmin, g );
            CV_CALC_MIN_8U( vmin, r );

            uchar diff = saturate_cast<uchar>(v - vmin);
            vr = v == r ? -1 : 0;
            vg = v == g ? -1 : 0;

            s = (diff * sdiv_table[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            h = (vr & (g - b)) +
                (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            h += h < 0 ? hr : 0;

            dst[i] = saturate_cast<uchar>(h);
            dst[i+1] = (uchar)s;
            dst[i+2] = (uchar)v;
        }
    }

    int srccn, blueIdx, hrange;
};


struct RGB2HSV_f
{
    typedef float channel_type;

    RGB2HSV_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h, s, v;

            float vmin, diff;

            v = vmin = r;
            if( v < g ) v = g;
            if( v < b ) v = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = v - vmin;
            s = diff/(float)(fabs(v) + FLT_EPSILON);
            diff = (float)(60./(diff + FLT_EPSILON));
            if( v == r )
                h = (g - b)*diff;
            else if( v == g )
                h = (b - r)*diff + 120.f;
            else
                h = (r - g)*diff + 240.f;

            if( h < 0 ) h += 360.f;

            dst[i] = h*hscale;
            dst[i+1] = s;
            dst[i+2] = v;
        }
    }

    int srccn, blueIdx;
    float hrange;
};


struct HSV2RGB_f
{
    typedef float channel_type;

    HSV2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {
        #if CV_SSE2
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(__m128& v_h0, __m128& v_h1, __m128& v_s0,
                 __m128& v_s1, __m128& v_v0, __m128& v_v1) const
    {
        v_h0 = _mm_mul_ps(v_h0, _mm_set1_ps(hscale));
        v_h1 = _mm_mul_ps(v_h1, _mm_set1_ps(hscale));

        __m128 v_pre_sector0 = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_h0));
        __m128 v_pre_sector1 = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_h1));

        v_h0 = _mm_sub_ps(v_h0, v_pre_sector0);
        v_h1 = _mm_sub_ps(v_h1, v_pre_sector1);

        __m128 v_tab00 = v_v0;
        __m128 v_tab01 = v_v1;
        __m128 v_tab10 = _mm_mul_ps(v_v0, _mm_sub_ps(_mm_set1_ps(1.0f), v_s0));
        __m128 v_tab11 = _mm_mul_ps(v_v1, _mm_sub_ps(_mm_set1_ps(1.0f), v_s1));
        __m128 v_tab20 = _mm_mul_ps(v_v0, _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(v_s0, v_h0)));
        __m128 v_tab21 = _mm_mul_ps(v_v1, _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(v_s1, v_h1)));
        __m128 v_tab30 = _mm_mul_ps(v_v0, _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(v_s0, _mm_sub_ps(_mm_set1_ps(1.0f), v_h0))));
        __m128 v_tab31 = _mm_mul_ps(v_v1, _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(v_s1, _mm_sub_ps(_mm_set1_ps(1.0f), v_h1))));

        __m128 v_sector0 = _mm_div_ps(v_pre_sector0, _mm_set1_ps(6.0f));
        __m128 v_sector1 = _mm_div_ps(v_pre_sector1, _mm_set1_ps(6.0f));
        v_sector0 = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_sector0));
        v_sector1 = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_sector1));
        v_sector0 = _mm_mul_ps(v_sector0, _mm_set1_ps(6.0f));
        v_sector1 = _mm_mul_ps(v_sector1, _mm_set1_ps(6.0f));
        v_sector0 = _mm_sub_ps(v_pre_sector0, v_sector0);
        v_sector1 = _mm_sub_ps(v_pre_sector1, v_sector1);

        v_h0 = _mm_and_ps(v_tab10, _mm_cmplt_ps(v_sector0, _mm_set1_ps(2.0f)));
        v_h1 = _mm_and_ps(v_tab11, _mm_cmplt_ps(v_sector1, _mm_set1_ps(2.0f)));
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(v_tab30, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(2.0f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(v_tab31, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(2.0f))));
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(v_tab00, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(3.0f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(v_tab01, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(3.0f))));
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(v_tab00, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(4.0f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(v_tab01, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(4.0f))));
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(v_tab20, _mm_cmpgt_ps(v_sector0, _mm_set1_ps(4.0f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(v_tab21, _mm_cmpgt_ps(v_sector1, _mm_set1_ps(4.0f))));
        v_s0 = _mm_and_ps(v_tab30, _mm_cmplt_ps(v_sector0, _mm_set1_ps(1.0f)));
        v_s1 = _mm_and_ps(v_tab31, _mm_cmplt_ps(v_sector1, _mm_set1_ps(1.0f)));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_tab00, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(1.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_tab01, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(1.0f))));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_tab00, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(2.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_tab01, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(2.0f))));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_tab20, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(3.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_tab21, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(3.0f))));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_tab10, _mm_cmpgt_ps(v_sector0, _mm_set1_ps(3.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_tab11, _mm_cmpgt_ps(v_sector1, _mm_set1_ps(3.0f))));
        v_v0 = _mm_and_ps(v_tab00, _mm_cmplt_ps(v_sector0, _mm_set1_ps(1.0f)));
        v_v1 = _mm_and_ps(v_tab01, _mm_cmplt_ps(v_sector1, _mm_set1_ps(1.0f)));
        v_v0 = _mm_or_ps(v_v0, _mm_and_ps(v_tab20, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(1.0f))));
        v_v1 = _mm_or_ps(v_v1, _mm_and_ps(v_tab21, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(1.0f))));
        v_v0 = _mm_or_ps(v_v0, _mm_and_ps(v_tab10, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(2.0f))));
        v_v1 = _mm_or_ps(v_v1, _mm_and_ps(v_tab11, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(2.0f))));
        v_v0 = _mm_or_ps(v_v0, _mm_and_ps(v_tab10, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(3.0f))));
        v_v1 = _mm_or_ps(v_v1, _mm_and_ps(v_tab11, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(3.0f))));
        v_v0 = _mm_or_ps(v_v0, _mm_and_ps(v_tab30, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(4.0f))));
        v_v1 = _mm_or_ps(v_v1, _mm_and_ps(v_tab31, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(4.0f))));
        v_v0 = _mm_or_ps(v_v0, _mm_and_ps(v_tab00, _mm_cmpgt_ps(v_sector0, _mm_set1_ps(4.0f))));
        v_v1 = _mm_or_ps(v_v1, _mm_and_ps(v_tab01, _mm_cmpgt_ps(v_sector1, _mm_set1_ps(4.0f))));
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, dcn = dstcn;
        float _hscale = hscale;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        #if CV_SSE2
        if (haveSIMD)
        {
            for( ; i <= n - 24; i += 24, dst += dcn * 8 )
            {
                __m128 v_h0 = _mm_loadu_ps(src + i +  0);
                __m128 v_h1 = _mm_loadu_ps(src + i +  4);
                __m128 v_s0 = _mm_loadu_ps(src + i +  8);
                __m128 v_s1 = _mm_loadu_ps(src + i + 12);
                __m128 v_v0 = _mm_loadu_ps(src + i + 16);
                __m128 v_v1 = _mm_loadu_ps(src + i + 20);

                _mm_deinterleave_ps(v_h0, v_h1, v_s0, v_s1, v_v0, v_v1);

                process(v_h0, v_h1, v_s0, v_s1, v_v0, v_v1);

                if (dcn == 3)
                {
                    if (bidx)
                    {
                        _mm_interleave_ps(v_v0, v_v1, v_s0, v_s1, v_h0, v_h1);

                        _mm_storeu_ps(dst +  0, v_v0);
                        _mm_storeu_ps(dst +  4, v_v1);
                        _mm_storeu_ps(dst +  8, v_s0);
                        _mm_storeu_ps(dst + 12, v_s1);
                        _mm_storeu_ps(dst + 16, v_h0);
                        _mm_storeu_ps(dst + 20, v_h1);
                    }
                    else
                    {
                        _mm_interleave_ps(v_h0, v_h1, v_s0, v_s1, v_v0, v_v1);

                        _mm_storeu_ps(dst +  0, v_h0);
                        _mm_storeu_ps(dst +  4, v_h1);
                        _mm_storeu_ps(dst +  8, v_s0);
                        _mm_storeu_ps(dst + 12, v_s1);
                        _mm_storeu_ps(dst + 16, v_v0);
                        _mm_storeu_ps(dst + 20, v_v1);
                    }
                }
                else
                {
                    __m128 v_a0 = _mm_set1_ps(alpha);
                    __m128 v_a1 = _mm_set1_ps(alpha);
                    if (bidx)
                    {
                        _mm_interleave_ps(v_v0, v_v1, v_s0, v_s1, v_h0, v_h1, v_a0, v_a1);

                        _mm_storeu_ps(dst +  0, v_v0);
                        _mm_storeu_ps(dst +  4, v_v1);
                        _mm_storeu_ps(dst +  8, v_s0);
                        _mm_storeu_ps(dst + 12, v_s1);
                        _mm_storeu_ps(dst + 16, v_h0);
                        _mm_storeu_ps(dst + 20, v_h1);
                        _mm_storeu_ps(dst + 24, v_a0);
                        _mm_storeu_ps(dst + 28, v_a1);
                    }
                    else
                    {
                        _mm_interleave_ps(v_h0, v_h1, v_s0, v_s1, v_v0, v_v1, v_a0, v_a1);

                        _mm_storeu_ps(dst +  0, v_h0);
                        _mm_storeu_ps(dst +  4, v_h1);
                        _mm_storeu_ps(dst +  8, v_s0);
                        _mm_storeu_ps(dst + 12, v_s1);
                        _mm_storeu_ps(dst + 16, v_v0);
                        _mm_storeu_ps(dst + 20, v_v1);
                        _mm_storeu_ps(dst + 24, v_a0);
                        _mm_storeu_ps(dst + 28, v_a1);
                    }
                }
            }
        }
        #endif
        for( ; i < n; i += 3, dst += dcn )
        {
            float h = src[i], s = src[i+1], v = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = v;
            else
            {
                static const int sector_data[][3]=
                    {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;
                h *= _hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );
                sector = cvFloor(h);
                h -= sector;
                if( (unsigned)sector >= 6u )
                {
                    sector = 0;
                    h = 0.f;
                }

                tab[0] = v;
                tab[1] = v*(1.f - s);
                tab[2] = v*(1.f - s*h);
                tab[3] = v*(1.f - s*(1.f - h));

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
    #if CV_SSE2
    bool haveSIMD;
    #endif
};


struct HSV2RGB_b
{
    typedef uchar channel_type;

    HSV2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(3, _blueIdx, (float)_hrange)
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(1.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        #elif CV_SSE2
        v_scale = _mm_set1_ps(255.0f);
        v_alpha = _mm_set1_ps(ColorChannel<uchar>::max());
        v_zero = _mm_setzero_si128();
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(__m128i v_r, __m128i v_g, __m128i v_b,
                 const __m128& v_coeffs_,
                 float * buf) const
    {
        __m128 v_r0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_r, v_zero));
        __m128 v_g0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_g, v_zero));
        __m128 v_b0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_b, v_zero));

        __m128 v_r1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_r, v_zero));
        __m128 v_g1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_g, v_zero));
        __m128 v_b1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_b, v_zero));

        __m128 v_coeffs = v_coeffs_;

        v_r0 = _mm_mul_ps(v_r0, v_coeffs);
        v_g1 = _mm_mul_ps(v_g1, v_coeffs);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));

        v_r1 = _mm_mul_ps(v_r1, v_coeffs);
        v_b0 = _mm_mul_ps(v_b0, v_coeffs);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));

        v_g0 = _mm_mul_ps(v_g0, v_coeffs);
        v_b1 = _mm_mul_ps(v_b1, v_coeffs);

        _mm_store_ps(buf, v_r0);
        _mm_store_ps(buf + 4, v_r1);
        _mm_store_ps(buf + 8, v_g0);
        _mm_store_ps(buf + 12, v_g1);
        _mm_store_ps(buf + 16, v_b0);
        _mm_store_ps(buf + 20, v_b1);
    }
    #endif

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
        #if CV_SSE2
        __m128 v_coeffs = _mm_set_ps(1.f, 1.f/255.f, 1.f/255.f, 1.f);
        #endif

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0)));
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0)));
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; j <= (dn - 8) * 3; j += 24)
                {
                    __m128i v_src0 = _mm_loadu_si128((__m128i const *)(src + j));
                    __m128i v_src1 = _mm_loadl_epi64((__m128i const *)(src + j + 16));

                    process(_mm_unpacklo_epi8(v_src0, v_zero),
                            _mm_unpackhi_epi8(v_src0, v_zero),
                            _mm_unpacklo_epi8(v_src1, v_zero),
                            v_coeffs,
                            buf + j);
                }
            }
            #endif

            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            j = 0;
            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #elif CV_SSE2
            if (dcn == 3 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 16); j += 16, dst += 16)
                {
                    __m128 v_src0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_src1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_src2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);
                    __m128 v_src3 = _mm_mul_ps(_mm_load_ps(buf + j + 12), v_scale);

                    __m128i v_dst0 = _mm_packs_epi32(_mm_cvtps_epi32(v_src0),
                                                     _mm_cvtps_epi32(v_src1));
                    __m128i v_dst1 = _mm_packs_epi32(_mm_cvtps_epi32(v_src2),
                                                     _mm_cvtps_epi32(v_src3));

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            else if (dcn == 4 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 12); j += 12, dst += 16)
                {
                    __m128 v_buf0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_buf1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_buf2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);

                    __m128 v_ba0 = _mm_unpackhi_ps(v_buf0, v_alpha);
                    __m128 v_ba1 = _mm_unpacklo_ps(v_buf2, v_alpha);

                    __m128i v_src0 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf0, v_ba0, 0x44));
                    __m128i v_src1 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba0, v_buf1, 0x4e)), 0x78);
                    __m128i v_src2 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf1, v_ba1, 0x4e));
                    __m128i v_src3 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba1, v_buf2, 0xee)), 0x78);

                    __m128i v_dst0 = _mm_packs_epi32(v_src0, v_src1);
                    __m128i v_dst1 = _mm_packs_epi32(v_src2, v_src3);

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    HSV2RGB_f cvt;
    #if CV_NEON
    float32x4_t v_scale, v_scale_inv;
    uint8x8_t v_alpha;
    #elif CV_SSE2
    __m128 v_scale;
    __m128 v_alpha;
    __m128i v_zero;
    bool haveSIMD;
    #endif
};


///////////////////////////////////// RGB <-> HLS ////////////////////////////////////////

struct RGB2HLS_f
{
    typedef float channel_type;

    RGB2HLS_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hscale(_hrange/360.f) {
        #if CV_SSE2
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(__m128& v_b0, __m128& v_b1, __m128& v_g0,
                 __m128& v_g1, __m128& v_r0, __m128& v_r1) const
    {
        __m128 v_max0 = _mm_max_ps(_mm_max_ps(v_b0, v_g0), v_r0);
        __m128 v_max1 = _mm_max_ps(_mm_max_ps(v_b1, v_g1), v_r1);
        __m128 v_min0 = _mm_min_ps(_mm_min_ps(v_b0, v_g0), v_r0);
        __m128 v_min1 = _mm_min_ps(_mm_min_ps(v_b1, v_g1), v_r1);
        __m128 v_diff0 = _mm_sub_ps(v_max0, v_min0);
        __m128 v_diff1 = _mm_sub_ps(v_max1, v_min1);
        __m128 v_sum0 = _mm_add_ps(v_max0, v_min0);
        __m128 v_sum1 = _mm_add_ps(v_max1, v_min1);
        __m128 v_l0 = _mm_mul_ps(v_sum0, _mm_set1_ps(0.5f));
        __m128 v_l1 = _mm_mul_ps(v_sum1, _mm_set1_ps(0.5f));

        __m128 v_gel0 = _mm_cmpge_ps(v_l0, _mm_set1_ps(0.5f));
        __m128 v_gel1 = _mm_cmpge_ps(v_l1, _mm_set1_ps(0.5f));
        __m128 v_s0 = _mm_and_ps(v_gel0, _mm_sub_ps(_mm_set1_ps(2.0f), v_sum0));
        __m128 v_s1 = _mm_and_ps(v_gel1, _mm_sub_ps(_mm_set1_ps(2.0f), v_sum1));
        v_s0 = _mm_or_ps(v_s0, _mm_andnot_ps(v_gel0, v_sum0));
        v_s1 = _mm_or_ps(v_s1, _mm_andnot_ps(v_gel1, v_sum1));
        v_s0 = _mm_div_ps(v_diff0, v_s0);
        v_s1 = _mm_div_ps(v_diff1, v_s1);

        __m128 v_gteps0 = _mm_cmpgt_ps(v_diff0, _mm_set1_ps(FLT_EPSILON));
        __m128 v_gteps1 = _mm_cmpgt_ps(v_diff1, _mm_set1_ps(FLT_EPSILON));

        v_diff0 = _mm_div_ps(_mm_set1_ps(60.f), v_diff0);
        v_diff1 = _mm_div_ps(_mm_set1_ps(60.f), v_diff1);

        __m128 v_eqr0 = _mm_cmpeq_ps(v_max0, v_r0);
        __m128 v_eqr1 = _mm_cmpeq_ps(v_max1, v_r1);
        __m128 v_h0 = _mm_and_ps(v_eqr0, _mm_mul_ps(_mm_sub_ps(v_g0, v_b0), v_diff0));
        __m128 v_h1 = _mm_and_ps(v_eqr1, _mm_mul_ps(_mm_sub_ps(v_g1, v_b1), v_diff1));
        __m128 v_eqg0 = _mm_cmpeq_ps(v_max0, v_g0);
        __m128 v_eqg1 = _mm_cmpeq_ps(v_max1, v_g1);
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(_mm_andnot_ps(v_eqr0, v_eqg0), _mm_add_ps(_mm_mul_ps(_mm_sub_ps(v_b0, v_r0), v_diff0), _mm_set1_ps(120.f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(_mm_andnot_ps(v_eqr1, v_eqg1), _mm_add_ps(_mm_mul_ps(_mm_sub_ps(v_b1, v_r1), v_diff1), _mm_set1_ps(120.f))));
        v_h0 = _mm_or_ps(v_h0, _mm_andnot_ps(_mm_or_ps(v_eqr0, v_eqg0), _mm_add_ps(_mm_mul_ps(_mm_sub_ps(v_r0, v_g0), v_diff0), _mm_set1_ps(240.f))));
        v_h1 = _mm_or_ps(v_h1, _mm_andnot_ps(_mm_or_ps(v_eqr1, v_eqg1), _mm_add_ps(_mm_mul_ps(_mm_sub_ps(v_r1, v_g1), v_diff1), _mm_set1_ps(240.f))));
        v_h0 = _mm_add_ps(v_h0, _mm_and_ps(_mm_cmplt_ps(v_h0, _mm_setzero_ps()), _mm_set1_ps(360.f)));
        v_h1 = _mm_add_ps(v_h1, _mm_and_ps(_mm_cmplt_ps(v_h1, _mm_setzero_ps()), _mm_set1_ps(360.f)));
        v_h0 = _mm_mul_ps(v_h0, _mm_set1_ps(hscale));
        v_h1 = _mm_mul_ps(v_h1, _mm_set1_ps(hscale));

        v_b0 = _mm_and_ps(v_gteps0, v_h0);
        v_b1 = _mm_and_ps(v_gteps1, v_h1);
        v_g0 = v_l0;
        v_g1 = v_l1;
        v_r0 = _mm_and_ps(v_gteps0, v_s0);
        v_r1 = _mm_and_ps(v_gteps1, v_s1);
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, scn = srccn;
        n *= 3;

        #if CV_SSE2
        if (haveSIMD)
        {
            for( ; i <= n - 24; i += 24, src += scn * 8 )
            {
                __m128 v_b0 = _mm_loadu_ps(src +  0);
                __m128 v_b1 = _mm_loadu_ps(src +  4);
                __m128 v_g0 = _mm_loadu_ps(src +  8);
                __m128 v_g1 = _mm_loadu_ps(src + 12);
                __m128 v_r0 = _mm_loadu_ps(src + 16);
                __m128 v_r1 = _mm_loadu_ps(src + 20);

                if (scn == 3)
                {
                    _mm_deinterleave_ps(v_b0, v_b1, v_g0, v_g1, v_r0, v_r1);
                }
                else
                {
                    __m128 v_a0 = _mm_loadu_ps(src + 24);
                    __m128 v_a1 = _mm_loadu_ps(src + 28);
                    _mm_deinterleave_ps(v_b0, v_b1, v_g0, v_g1, v_r0, v_r1, v_a0, v_a1);
                }

                if (bidx)
                {
                    __m128 v_tmp0 = v_b0;
                    __m128 v_tmp1 = v_b1;
                    v_b0 = v_r0;
                    v_b1 = v_r1;
                    v_r0 = v_tmp0;
                    v_r1 = v_tmp1;
                }

                process(v_b0, v_b1, v_g0, v_g1, v_r0, v_r1);

                _mm_interleave_ps(v_b0, v_b1, v_g0, v_g1, v_r0, v_r1);

                _mm_storeu_ps(dst + i +  0, v_b0);
                _mm_storeu_ps(dst + i +  4, v_b1);
                _mm_storeu_ps(dst + i +  8, v_g0);
                _mm_storeu_ps(dst + i + 12, v_g1);
                _mm_storeu_ps(dst + i + 16, v_r0);
                _mm_storeu_ps(dst + i + 20, v_r1);
            }
        }
        #endif

        for( ; i < n; i += 3, src += scn )
        {
            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h = 0.f, s = 0.f, l;
            float vmin, vmax, diff;

            vmax = vmin = r;
            if( vmax < g ) vmax = g;
            if( vmax < b ) vmax = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = vmax - vmin;
            l = (vmax + vmin)*0.5f;

            if( diff > FLT_EPSILON )
            {
                s = l < 0.5f ? diff/(vmax + vmin) : diff/(2 - vmax - vmin);
                diff = 60.f/diff;

                if( vmax == r )
                    h = (g - b)*diff;
                else if( vmax == g )
                    h = (b - r)*diff + 120.f;
                else
                    h = (r - g)*diff + 240.f;

                if( h < 0.f ) h += 360.f;
            }

            dst[i] = h*hscale;
            dst[i+1] = l;
            dst[i+2] = s;
        }
    }

    int srccn, blueIdx;
    float hscale;
    #if CV_SSE2
    bool haveSIMD;
    #endif
};


struct RGB2HLS_b
{
    typedef uchar channel_type;

    RGB2HLS_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), cvt(3, _blueIdx, (float)_hrange)
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(1.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        #elif CV_SSE2
        v_scale_inv = _mm_set1_ps(1.f/255.f);
        v_zero = _mm_setzero_si128();
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(const float * buf,
                 __m128 & v_coeffs, uchar * dst) const
    {
        __m128 v_l0f = _mm_load_ps(buf);
        __m128 v_l1f = _mm_load_ps(buf + 4);
        __m128 v_u0f = _mm_load_ps(buf + 8);
        __m128 v_u1f = _mm_load_ps(buf + 12);

        v_l0f = _mm_mul_ps(v_l0f, v_coeffs);
        v_u1f = _mm_mul_ps(v_u1f, v_coeffs);
        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x92));
        v_u0f = _mm_mul_ps(v_u0f, v_coeffs);
        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x92));
        v_l1f = _mm_mul_ps(v_l1f, v_coeffs);

        __m128i v_l = _mm_packs_epi32(_mm_cvtps_epi32(v_l0f), _mm_cvtps_epi32(v_l1f));
        __m128i v_u = _mm_packs_epi32(_mm_cvtps_epi32(v_u0f), _mm_cvtps_epi32(v_u1f));
        __m128i v_l0 = _mm_packus_epi16(v_l, v_u);

        _mm_storeu_si128((__m128i *)(dst), v_l0);
    }
    #endif

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, scn = srccn;
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
        #if CV_SSE2
        __m128 v_coeffs = _mm_set_ps(1.f, 255.f, 255.f, 1.f);
        #endif

        for( i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, src += 8 * scn)
            {
                uint16x8_t v_t0, v_t1, v_t2;

                if (scn == 3)
                {
                    uint8x8x3_t v_src = vld3_u8(src);
                    v_t0 = vmovl_u8(v_src.val[0]);
                    v_t1 = vmovl_u8(v_src.val[1]);
                    v_t2 = vmovl_u8(v_src.val[2]);
                }
                else
                {
                    uint8x8x4_t v_src = vld4_u8(src);
                    v_t0 = vmovl_u8(v_src.val[0]);
                    v_t1 = vmovl_u8(v_src.val[1]);
                    v_t2 = vmovl_u8(v_src.val[2]);
                }

                float32x4x3_t v_dst;
                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #elif CV_SSE2
            if (scn == 3 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 16); j += 16, src += 16)
                {
                    __m128i v_src = _mm_loadu_si128((__m128i const *)src);

                    __m128i v_src_p = _mm_unpacklo_epi8(v_src, v_zero);
                    _mm_store_ps(buf + j, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src_p, v_zero)), v_scale_inv));
                    _mm_store_ps(buf + j + 4, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src_p, v_zero)), v_scale_inv));

                    v_src_p = _mm_unpackhi_epi8(v_src, v_zero);
                    _mm_store_ps(buf + j + 8, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src_p, v_zero)), v_scale_inv));
                    _mm_store_ps(buf + j + 12, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src_p, v_zero)), v_scale_inv));
                }

                int jr = j % 3;
                if (jr)
                    src -= jr, j -= jr;
            }
            else if (scn == 4 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 12); j += 12, src += 16)
                {
                    __m128i v_src = _mm_loadu_si128((__m128i const *)src);

                    __m128i v_src_lo = _mm_unpacklo_epi8(v_src, v_zero);
                    __m128i v_src_hi = _mm_unpackhi_epi8(v_src, v_zero);
                    _mm_storeu_ps(buf + j, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src_lo, v_zero)), v_scale_inv));
                    _mm_storeu_ps(buf + j + 3, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src_lo, v_zero)), v_scale_inv));
                    _mm_storeu_ps(buf + j + 6, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src_hi, v_zero)), v_scale_inv));
                    float tmp = buf[j + 8];
                    _mm_storeu_ps(buf + j + 8, _mm_mul_ps(_mm_cvtepi32_ps(_mm_shuffle_epi32(_mm_unpackhi_epi16(v_src_hi, v_zero), 0x90)), v_scale_inv));
                    buf[j + 8] = tmp;
                }

                int jr = j % 3;
                if (jr)
                    src -= jr, j -= jr;
            }
            #endif
            for( ; j < dn*3; j += 3, src += scn )
            {
                buf[j] = src[0]*(1.f/255.f);
                buf[j+1] = src[1]*(1.f/255.f);
                buf[j+2] = src[2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            j = 0;
            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);

                uint8x8x3_t v_dst;
                v_dst.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_src0.val[0])),
                                                       vqmovn_u32(cv_vrndq_u32_f32(v_src1.val[0]))));
                v_dst.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                       vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                v_dst.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                       vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));
                vst3_u8(dst + j, v_dst);
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; j <= (dn - 16) * 3; j += 48)
                {
                    process(buf + j,
                            v_coeffs, dst + j);

                    process(buf + j + 16,
                            v_coeffs, dst + j + 16);

                    process(buf + j + 32,
                            v_coeffs, dst + j + 32);
                }
            }
            #endif
            for( ; j < dn*3; j += 3 )
            {
                dst[j] = saturate_cast<uchar>(buf[j]);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*255.f);
            }
        }
    }

    int srccn;
    RGB2HLS_f cvt;
    #if CV_NEON
    float32x4_t v_scale, v_scale_inv;
    uint8x8_t v_alpha;
    #elif CV_SSE2
    __m128 v_scale_inv;
    __m128i v_zero;
    bool haveSIMD;
    #endif
};


struct HLS2RGB_f
{
    typedef float channel_type;

    HLS2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {
        #if CV_SSE2
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(__m128& v_h0, __m128& v_h1, __m128& v_l0,
                 __m128& v_l1, __m128& v_s0, __m128& v_s1) const
    {
        __m128 v_lel0 = _mm_cmple_ps(v_l0, _mm_set1_ps(0.5f));
        __m128 v_lel1 = _mm_cmple_ps(v_l1, _mm_set1_ps(0.5f));
        __m128 v_p20 = _mm_andnot_ps(v_lel0, _mm_sub_ps(_mm_add_ps(v_l0, v_s0), _mm_mul_ps(v_l0, v_s0)));
        __m128 v_p21 = _mm_andnot_ps(v_lel1, _mm_sub_ps(_mm_add_ps(v_l1, v_s1), _mm_mul_ps(v_l1, v_s1)));
        v_p20 = _mm_or_ps(v_p20, _mm_and_ps(v_lel0, _mm_mul_ps(v_l0, _mm_add_ps(_mm_set1_ps(1.0f), v_s0))));
        v_p21 = _mm_or_ps(v_p21, _mm_and_ps(v_lel1, _mm_mul_ps(v_l1, _mm_add_ps(_mm_set1_ps(1.0f), v_s1))));

        __m128 v_p10 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), v_l0), v_p20);
        __m128 v_p11 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), v_l1), v_p21);

        v_h0 = _mm_mul_ps(v_h0, _mm_set1_ps(hscale));
        v_h1 = _mm_mul_ps(v_h1, _mm_set1_ps(hscale));

        __m128 v_pre_sector0 = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_h0));
        __m128 v_pre_sector1 = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_h1));

        v_h0 = _mm_sub_ps(v_h0, v_pre_sector0);
        v_h1 = _mm_sub_ps(v_h1, v_pre_sector1);

        __m128 v_p2_p10 = _mm_sub_ps(v_p20, v_p10);
        __m128 v_p2_p11 = _mm_sub_ps(v_p21, v_p11);
        __m128 v_tab20 = _mm_add_ps(v_p10, _mm_mul_ps(v_p2_p10, _mm_sub_ps(_mm_set1_ps(1.0f), v_h0)));
        __m128 v_tab21 = _mm_add_ps(v_p11, _mm_mul_ps(v_p2_p11, _mm_sub_ps(_mm_set1_ps(1.0f), v_h1)));
        __m128 v_tab30 = _mm_add_ps(v_p10, _mm_mul_ps(v_p2_p10, v_h0));
        __m128 v_tab31 = _mm_add_ps(v_p11, _mm_mul_ps(v_p2_p11, v_h1));

        __m128 v_sector0 = _mm_div_ps(v_pre_sector0, _mm_set1_ps(6.0f));
        __m128 v_sector1 = _mm_div_ps(v_pre_sector1, _mm_set1_ps(6.0f));
        v_sector0 = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_sector0));
        v_sector1 = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_sector1));
        v_sector0 = _mm_mul_ps(v_sector0, _mm_set1_ps(6.0f));
        v_sector1 = _mm_mul_ps(v_sector1, _mm_set1_ps(6.0f));
        v_sector0 = _mm_sub_ps(v_pre_sector0, v_sector0);
        v_sector1 = _mm_sub_ps(v_pre_sector1, v_sector1);

        v_h0 = _mm_and_ps(v_p10, _mm_cmplt_ps(v_sector0, _mm_set1_ps(2.0f)));
        v_h1 = _mm_and_ps(v_p11, _mm_cmplt_ps(v_sector1, _mm_set1_ps(2.0f)));
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(v_tab30, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(2.0f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(v_tab31, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(2.0f))));
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(v_p20, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(3.0f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(v_p21, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(3.0f))));
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(v_p20, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(4.0f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(v_p21, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(4.0f))));
        v_h0 = _mm_or_ps(v_h0, _mm_and_ps(v_tab20, _mm_cmpgt_ps(v_sector0, _mm_set1_ps(4.0f))));
        v_h1 = _mm_or_ps(v_h1, _mm_and_ps(v_tab21, _mm_cmpgt_ps(v_sector1, _mm_set1_ps(4.0f))));
        v_l0 = _mm_and_ps(v_tab30, _mm_cmplt_ps(v_sector0, _mm_set1_ps(1.0f)));
        v_l1 = _mm_and_ps(v_tab31, _mm_cmplt_ps(v_sector1, _mm_set1_ps(1.0f)));
        v_l0 = _mm_or_ps(v_l0, _mm_and_ps(v_p20, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(1.0f))));
        v_l1 = _mm_or_ps(v_l1, _mm_and_ps(v_p21, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(1.0f))));
        v_l0 = _mm_or_ps(v_l0, _mm_and_ps(v_p20, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(2.0f))));
        v_l1 = _mm_or_ps(v_l1, _mm_and_ps(v_p21, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(2.0f))));
        v_l0 = _mm_or_ps(v_l0, _mm_and_ps(v_tab20, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(3.0f))));
        v_l1 = _mm_or_ps(v_l1, _mm_and_ps(v_tab21, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(3.0f))));
        v_l0 = _mm_or_ps(v_l0, _mm_and_ps(v_p10, _mm_cmpgt_ps(v_sector0, _mm_set1_ps(3.0f))));
        v_l1 = _mm_or_ps(v_l1, _mm_and_ps(v_p11, _mm_cmpgt_ps(v_sector1, _mm_set1_ps(3.0f))));
        v_s0 = _mm_and_ps(v_p20, _mm_cmplt_ps(v_sector0, _mm_set1_ps(1.0f)));
        v_s1 = _mm_and_ps(v_p21, _mm_cmplt_ps(v_sector1, _mm_set1_ps(1.0f)));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_tab20, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(1.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_tab21, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(1.0f))));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_p10, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(2.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_p11, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(2.0f))));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_p10, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(3.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_p11, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(3.0f))));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_tab30, _mm_cmpeq_ps(v_sector0, _mm_set1_ps(4.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_tab31, _mm_cmpeq_ps(v_sector1, _mm_set1_ps(4.0f))));
        v_s0 = _mm_or_ps(v_s0, _mm_and_ps(v_p20, _mm_cmpgt_ps(v_sector0, _mm_set1_ps(4.0f))));
        v_s1 = _mm_or_ps(v_s1, _mm_and_ps(v_p21, _mm_cmpgt_ps(v_sector1, _mm_set1_ps(4.0f))));
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, dcn = dstcn;
        float _hscale = hscale;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        #if CV_SSE2
        if (haveSIMD)
        {
            for( ; i <= n - 24; i += 24, dst += dcn * 8 )
            {
                __m128 v_h0 = _mm_loadu_ps(src + i +  0);
                __m128 v_h1 = _mm_loadu_ps(src + i +  4);
                __m128 v_l0 = _mm_loadu_ps(src + i +  8);
                __m128 v_l1 = _mm_loadu_ps(src + i + 12);
                __m128 v_s0 = _mm_loadu_ps(src + i + 16);
                __m128 v_s1 = _mm_loadu_ps(src + i + 20);

                _mm_deinterleave_ps(v_h0, v_h1, v_l0, v_l1, v_s0, v_s1);

                process(v_h0, v_h1, v_l0, v_l1, v_s0, v_s1);

                if (dcn == 3)
                {
                    if (bidx)
                    {
                        _mm_interleave_ps(v_s0, v_s1, v_l0, v_l1, v_h0, v_h1);

                        _mm_storeu_ps(dst +  0, v_s0);
                        _mm_storeu_ps(dst +  4, v_s1);
                        _mm_storeu_ps(dst +  8, v_l0);
                        _mm_storeu_ps(dst + 12, v_l1);
                        _mm_storeu_ps(dst + 16, v_h0);
                        _mm_storeu_ps(dst + 20, v_h1);
                    }
                    else
                    {
                        _mm_interleave_ps(v_h0, v_h1, v_l0, v_l1, v_s0, v_s1);

                        _mm_storeu_ps(dst +  0, v_h0);
                        _mm_storeu_ps(dst +  4, v_h1);
                        _mm_storeu_ps(dst +  8, v_l0);
                        _mm_storeu_ps(dst + 12, v_l1);
                        _mm_storeu_ps(dst + 16, v_s0);
                        _mm_storeu_ps(dst + 20, v_s1);
                    }
                }
                else
                {
                    __m128 v_a0 = _mm_set1_ps(alpha);
                    __m128 v_a1 = _mm_set1_ps(alpha);
                    if (bidx)
                    {
                        _mm_interleave_ps(v_s0, v_s1, v_l0, v_l1, v_h0, v_h1, v_a0, v_a1);

                        _mm_storeu_ps(dst +  0, v_s0);
                        _mm_storeu_ps(dst +  4, v_s1);
                        _mm_storeu_ps(dst +  8, v_l0);
                        _mm_storeu_ps(dst + 12, v_l1);
                        _mm_storeu_ps(dst + 16, v_h0);
                        _mm_storeu_ps(dst + 20, v_h1);
                        _mm_storeu_ps(dst + 24, v_a0);
                        _mm_storeu_ps(dst + 28, v_a1);
                    }
                    else
                    {
                        _mm_interleave_ps(v_h0, v_h1, v_l0, v_l1, v_s0, v_s1, v_a0, v_a1);

                        _mm_storeu_ps(dst +  0, v_h0);
                        _mm_storeu_ps(dst +  4, v_h1);
                        _mm_storeu_ps(dst +  8, v_l0);
                        _mm_storeu_ps(dst + 12, v_l1);
                        _mm_storeu_ps(dst + 16, v_s0);
                        _mm_storeu_ps(dst + 20, v_s1);
                        _mm_storeu_ps(dst + 24, v_a0);
                        _mm_storeu_ps(dst + 28, v_a1);
                    }
                }
            }
        }
        #endif
        for( ; i < n; i += 3, dst += dcn )
        {
            float h = src[i], l = src[i+1], s = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = l;
            else
            {
                static const int sector_data[][3]=
                {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;

                float p2 = l <= 0.5f ? l*(1 + s) : l + s - l*s;
                float p1 = 2*l - p2;

                h *= _hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );

                assert( 0 <= h && h < 6 );
                sector = cvFloor(h);
                h -= sector;

                tab[0] = p2;
                tab[1] = p1;
                tab[2] = p1 + (p2 - p1)*(1-h);
                tab[3] = p1 + (p2 - p1)*h;

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
    #if CV_SSE2
    bool haveSIMD;
    #endif
};


struct HLS2RGB_b
{
    typedef uchar channel_type;

    HLS2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(3, _blueIdx, (float)_hrange)
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(1.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        #elif CV_SSE2
        v_scale = _mm_set1_ps(255.f);
        v_alpha = _mm_set1_ps(ColorChannel<uchar>::max());
        v_zero = _mm_setzero_si128();
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(__m128i v_r, __m128i v_g, __m128i v_b,
                 const __m128& v_coeffs_,
                 float * buf) const
    {
        __m128 v_r0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_r, v_zero));
        __m128 v_g0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_g, v_zero));
        __m128 v_b0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_b, v_zero));

        __m128 v_r1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_r, v_zero));
        __m128 v_g1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_g, v_zero));
        __m128 v_b1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_b, v_zero));

        __m128 v_coeffs = v_coeffs_;

        v_r0 = _mm_mul_ps(v_r0, v_coeffs);
        v_g1 = _mm_mul_ps(v_g1, v_coeffs);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));

        v_r1 = _mm_mul_ps(v_r1, v_coeffs);
        v_b0 = _mm_mul_ps(v_b0, v_coeffs);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));

        v_g0 = _mm_mul_ps(v_g0, v_coeffs);
        v_b1 = _mm_mul_ps(v_b1, v_coeffs);

        _mm_store_ps(buf, v_r0);
        _mm_store_ps(buf + 4, v_r1);
        _mm_store_ps(buf + 8, v_g0);
        _mm_store_ps(buf + 12, v_g1);
        _mm_store_ps(buf + 16, v_b0);
        _mm_store_ps(buf + 20, v_b1);
    }
    #endif

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
        #if CV_SSE2
        __m128 v_coeffs = _mm_set_ps(1.f, 1.f/255.f, 1.f/255.f, 1.f);
        #endif

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0)));
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0)));
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; j <= (dn - 8) * 3; j += 24)
                {
                    __m128i v_src0 = _mm_loadu_si128((__m128i const *)(src + j));
                    __m128i v_src1 = _mm_loadl_epi64((__m128i const *)(src + j + 16));

                    process(_mm_unpacklo_epi8(v_src0, v_zero),
                            _mm_unpackhi_epi8(v_src0, v_zero),
                            _mm_unpacklo_epi8(v_src1, v_zero),
                            v_coeffs,
                            buf + j);
                }
            }
            #endif
            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            j = 0;
            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #elif CV_SSE2
            if (dcn == 3 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 16); j += 16, dst += 16)
                {
                    __m128 v_src0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_src1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_src2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);
                    __m128 v_src3 = _mm_mul_ps(_mm_load_ps(buf + j + 12), v_scale);

                    __m128i v_dst0 = _mm_packs_epi32(_mm_cvtps_epi32(v_src0),
                                                     _mm_cvtps_epi32(v_src1));
                    __m128i v_dst1 = _mm_packs_epi32(_mm_cvtps_epi32(v_src2),
                                                     _mm_cvtps_epi32(v_src3));

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            else if (dcn == 4 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 12); j += 12, dst += 16)
                {
                    __m128 v_buf0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_buf1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_buf2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);

                    __m128 v_ba0 = _mm_unpackhi_ps(v_buf0, v_alpha);
                    __m128 v_ba1 = _mm_unpacklo_ps(v_buf2, v_alpha);

                    __m128i v_src0 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf0, v_ba0, 0x44));
                    __m128i v_src1 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba0, v_buf1, 0x4e)), 0x78);
                    __m128i v_src2 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf1, v_ba1, 0x4e));
                    __m128i v_src3 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba1, v_buf2, 0xee)), 0x78);

                    __m128i v_dst0 = _mm_packs_epi32(v_src0, v_src1);
                    __m128i v_dst1 = _mm_packs_epi32(v_src2, v_src3);

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    HLS2RGB_f cvt;
    #if CV_NEON
    float32x4_t v_scale, v_scale_inv;
    uint8x8_t v_alpha;
    #elif CV_SSE2
    __m128 v_scale;
    __m128 v_alpha;
    __m128i v_zero;
    bool haveSIMD;
    #endif
};

//
// IPP functions
//

#if NEED_IPP

#if !IPP_DISABLE_RGB_HSV
static ippiGeneralFunc ippiRGB2HSVTab[] =
{
    (ippiGeneralFunc)ippiRGBToHSV_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHSV_16u_C3R, 0,
    0, 0, 0, 0
};
#endif

static ippiGeneralFunc ippiHSV2RGBTab[] =
{
    (ippiGeneralFunc)ippiHSVToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHSVToRGB_16u_C3R, 0,
    0, 0, 0, 0
};

static ippiGeneralFunc ippiRGB2HLSTab[] =
{
    (ippiGeneralFunc)ippiRGBToHLS_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHLS_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToHLS_32f_C3R, 0, 0
};

static ippiGeneralFunc ippiHLS2RGBTab[] =
{
    (ippiGeneralFunc)ippiHLSToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHLSToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiHLSToRGB_32f_C3R, 0, 0
};

#endif

//
// HAL functions
//

namespace hal
{

// 8u, 32f
void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoHSV, cv_hal_cvtBGRtoHSV, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isFullRange, isHSV);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
    CV_IPP_CHECK()
    {
        if(depth == CV_8U && isFullRange)
        {
            if (isHSV)
            {
#if !IPP_DISABLE_RGB_HSV // breaks OCL accuracy tests
                if(scn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                            IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(scn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(scn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 0, 1, 2, depth)) )
                        return;
                }
#endif
            }
            else
            {
                if(scn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                            IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(scn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(scn == 3 && swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                            IPPGeneralFunctor(ippiRGB2HLSTab[depth])) )
                        return;
                }
                else if(scn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
        }
    }
#endif

    int hrange = depth == CV_32F ? 360 : isFullRange ? 256 : 180;
    int blueIdx = swapBlue ? 2 : 0;
    if(isHSV)
    {
        if(depth == CV_8U)
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2HSV_b(scn, blueIdx, hrange));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2HSV_f(scn, blueIdx, static_cast<float>(hrange)));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2HLS_b(scn, blueIdx, hrange));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2HLS_f(scn, blueIdx, static_cast<float>(hrange)));
    }
}

// 8u, 32f
void cvtHSVtoBGR(const uchar * src_data, size_t src_step,
                        uchar * dst_data, size_t dst_step,
                        int width, int height,
                        int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtHSVtoBGR, cv_hal_cvtHSVtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isFullRange, isHSV);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
    CV_IPP_CHECK()
    {
        if (depth == CV_8U && isFullRange)
        {
            if (isHSV)
            {
                if(dcn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                            IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(dcn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(dcn == 3 && swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                            IPPGeneralFunctor(ippiHSV2RGBTab[depth])) )
                        return;
                }
                else if(dcn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
            else
            {
                if(dcn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                            IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(dcn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(dcn == 3 && swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                            IPPGeneralFunctor(ippiHLS2RGBTab[depth])) )
                        return;
                }
                else if(dcn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
        }
    }
#endif

    int hrange = depth == CV_32F ? 360 : isFullRange ? 255 : 180;
    int blueIdx = swapBlue ? 2 : 0;
    if(isHSV)
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, HSV2RGB_b(dcn, blueIdx, hrange));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, HSV2RGB_f(dcn, blueIdx, static_cast<float>(hrange)));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, HLS2RGB_b(dcn, blueIdx, hrange));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, HLS2RGB_f(dcn, blueIdx, static_cast<float>(hrange)));
    }
}

} // namespace hal

//
// OCL calls
//

#ifdef HAVE_OPENCL

bool oclCvtColorHSV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool full )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    int hrange = _src.depth() == CV_32F ? 360 : (!full ? 180 : 255);

    if(!h.createKernel("HSV2RGB", ocl::imgproc::color_hsv_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D hrange=%d -D hscale=%ff", dcn, bidx, hrange, 6.f/hrange)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorHLS2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool full )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    int hrange = _src.depth() == CV_32F ? 360 : (!full ? 180 : 255);

    if(!h.createKernel("HLS2RGB", ocl::imgproc::color_hsv_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D hrange=%d -D hscale=%ff", dcn, bidx, hrange, 6.f/hrange)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2HLS( InputArray _src, OutputArray _dst, int bidx, bool full )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    float hscale = (_src.depth() == CV_32F ? 360.f : (!full ? 180.f : 256.f))/360.f;

    if(!h.createKernel("RGB2HLS", ocl::imgproc::color_hsv_oclsrc,
                       format("-D hscale=%ff -D bidx=%d -D dcn=3", hscale, bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2HSV( InputArray _src, OutputArray _dst, int bidx, bool full )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    int hrange = _src.depth() == CV_32F ? 360 : (!full ? 180 : 256);

    cv::String options = (_src.depth() == CV_8U ?
                          format("-D hrange=%d -D bidx=%d -D dcn=3", hrange, bidx) :
                          format("-D hscale=%ff -D bidx=%d -D dcn=3", hrange*(1.f/360.f), bidx));

    if(!h.createKernel("RGB2HSV", ocl::imgproc::color_hsv_oclsrc, options))
    {
        return false;
    }

    if(_src.depth() == CV_8U)
    {
        static UMat sdiv_data;
        static UMat hdiv_data180;
        static UMat hdiv_data256;
        static int sdiv_table[256];
        static int hdiv_table180[256];
        static int hdiv_table256[256];
        static volatile bool initialized180 = false, initialized256 = false;
        volatile bool & initialized = hrange == 180 ? initialized180 : initialized256;

        if (!initialized)
        {
            int * const hdiv_table = hrange == 180 ? hdiv_table180 : hdiv_table256, hsv_shift = 12;
            UMat & hdiv_data = hrange == 180 ? hdiv_data180 : hdiv_data256;

            sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;

            int v = 255 << hsv_shift;
            if (!initialized180 && !initialized256)
            {
                for(int i = 1; i < 256; i++ )
                    sdiv_table[i] = saturate_cast<int>(v/(1.*i));
                Mat(1, 256, CV_32SC1, sdiv_table).copyTo(sdiv_data);
            }

            v = hrange << hsv_shift;
            for (int i = 1; i < 256; i++ )
                hdiv_table[i] = saturate_cast<int>(v/(6.*i));

            Mat(1, 256, CV_32SC1, hdiv_table).copyTo(hdiv_data);
            initialized = true;
        }

        h.setArg(ocl::KernelArg::PtrReadOnly(sdiv_data));
        h.setArg(hrange == 256 ? ocl::KernelArg::PtrReadOnly(hdiv_data256) :
                                 ocl::KernelArg::PtrReadOnly(hdiv_data180));
    }

    return h.run();
}

#endif

//
// HAL calls
//

void cvtColorBGR2HLS( InputArray _src, OutputArray _dst, bool swapb, bool fullRange )
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoHSV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, fullRange, false);
}

void cvtColorBGR2HSV( InputArray _src, OutputArray _dst, bool swapb, bool fullRange )
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoHSV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, fullRange, true);
}

void cvtColorHLS2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool fullRange)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtHSVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, fullRange, false);
}

void cvtColorHSV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool fullRange)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtHSVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, fullRange, true);
}


} // namespace cv
