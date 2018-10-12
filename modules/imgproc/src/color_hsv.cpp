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
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange) {
        #if CV_SIMD128
        hasSIMD = hasSIMD128();
        #endif
    }

    #if CV_SIMD128
    inline void process(v_float32x4& v_r, v_float32x4& v_g,
                        v_float32x4& v_b, float hscale) const
    {
        v_float32x4 v_min_rgb = v_min(v_min(v_r, v_g), v_b);
        v_float32x4 v_max_rgb = v_max(v_max(v_r, v_g), v_b);

        v_float32x4 v_eps = v_setall_f32(FLT_EPSILON);
        v_float32x4 v_diff = v_max_rgb - v_min_rgb;
        v_float32x4 v_s = v_diff / (v_abs(v_max_rgb) + v_eps);

        v_float32x4 v_r_eq_max = v_r == v_max_rgb;
        v_float32x4 v_g_eq_max = v_g == v_max_rgb;
        v_float32x4 v_h = v_select(v_r_eq_max, v_g - v_b,
                          v_select(v_g_eq_max, v_b - v_r, v_r - v_g));
        v_float32x4 v_res = v_select(v_r_eq_max, (v_g < v_b) & v_setall_f32(360.0f),
                            v_select(v_g_eq_max, v_setall_f32(120.0f), v_setall_f32(240.0f)));
        v_float32x4 v_rev_diff = v_setall_f32(60.0f) / (v_diff + v_eps);
        v_r = v_muladd(v_h, v_rev_diff, v_res) * v_setall_f32(hscale);

        v_g = v_s;
        v_b = v_max_rgb;
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

        #if CV_SIMD128
        if (hasSIMD)
        {
            if (scn == 3) {
                if (bidx) {
                    for ( ; i <= n - 12; i += 12, src += scn * 4)
                    {
                        v_float32x4 v_r;
                        v_float32x4 v_g;
                        v_float32x4 v_b;
                        v_load_deinterleave(src, v_r, v_g, v_b);
                        process(v_r, v_g, v_b, hscale);
                        v_store_interleave(dst + i, v_r, v_g, v_b);
                    }
                } else {
                    for ( ; i <= n - 12; i += 12, src += scn * 4)
                    {
                        v_float32x4 v_r;
                        v_float32x4 v_g;
                        v_float32x4 v_b;
                        v_load_deinterleave(src, v_r, v_g, v_b);
                        process(v_b, v_g, v_r, hscale);
                        v_store_interleave(dst + i, v_b, v_g, v_r);
                    }
                }
            } else { // scn == 4
                if (bidx) {
                    for ( ; i <= n - 12; i += 12, src += scn * 4)
                    {
                        v_float32x4 v_r;
                        v_float32x4 v_g;
                        v_float32x4 v_b;
                        v_float32x4 v_a;
                        v_load_deinterleave(src, v_r, v_g, v_b, v_a);
                        process(v_r, v_g, v_b, hscale);
                        v_store_interleave(dst + i, v_r, v_g, v_b);
                    }
                } else {
                    for ( ; i <= n - 12; i += 12, src += scn * 4)
                    {
                        v_float32x4 v_r;
                        v_float32x4 v_g;
                        v_float32x4 v_b;
                        v_float32x4 v_a;
                        v_load_deinterleave(src, v_r, v_g, v_b, v_a);
                        process(v_b, v_g, v_r, hscale);
                        v_store_interleave(dst + i, v_b, v_g, v_r);
                    }
                }
            }
        }
        #endif

        for( ; i < n; i += 3, src += scn )
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
    #if CV_SIMD128
    bool hasSIMD;
    #endif
};


#if CV_SIMD128
inline void HSV2RGB_simd(v_float32x4& v_h, v_float32x4& v_s, v_float32x4& v_v, float hscale)
{
    v_h = v_h * v_setall_f32(hscale);
    v_float32x4 v_pre_sector = v_cvt_f32(v_trunc(v_h));
    v_h = v_h - v_pre_sector;
    v_float32x4 v_tab0 = v_v;
    v_float32x4 v_one = v_setall_f32(1.0f);
    v_float32x4 v_tab1 = v_v * (v_one - v_s);
    v_float32x4 v_tab2 = v_v * (v_one - (v_s * v_h));
    v_float32x4 v_tab3 = v_v * (v_one - (v_s * (v_one - v_h)));

    v_float32x4 v_one_sixth = v_setall_f32(1.0f / 6.0f);
    v_float32x4 v_sector = v_pre_sector * v_one_sixth;
    v_sector = v_cvt_f32(v_trunc(v_sector));
    v_float32x4 v_six = v_setall_f32(6.0f);
    v_sector = v_pre_sector - (v_sector * v_six);

    v_float32x4 v_two = v_setall_f32(2.0f);
    v_h = v_tab1 & (v_sector < v_two);
    v_h = v_h | (v_tab3 & (v_sector == v_two));
    v_float32x4 v_three = v_setall_f32(3.0f);
    v_h = v_h | (v_tab0 & (v_sector == v_three));
    v_float32x4 v_four = v_setall_f32(4.0f);
    v_h = v_h | (v_tab0 & (v_sector == v_four));
    v_h = v_h | (v_tab2 & (v_sector > v_four));

    v_s = v_tab3 & (v_sector < v_one);
    v_s = v_s | (v_tab0 & (v_sector == v_one));
    v_s = v_s | (v_tab0 & (v_sector == v_two));
    v_s = v_s | (v_tab2 & (v_sector == v_three));
    v_s = v_s | (v_tab1 & (v_sector > v_three));

    v_v = v_tab0 & (v_sector < v_one);
    v_v = v_v | (v_tab2 & (v_sector == v_one));
    v_v = v_v | (v_tab1 & (v_sector == v_two));
    v_v = v_v | (v_tab1 & (v_sector == v_three));
    v_v = v_v | (v_tab3 & (v_sector == v_four));
    v_v = v_v | (v_tab0 & (v_sector > v_four));
}
#endif


inline void HSV2RGB_native(const float* src, float* dst, const float hscale, const int bidx)
{
    float h = src[0], s = src[1], v = src[2];
    float b, g, r;

    if( s == 0 )
        b = g = r = v;
    else
    {
        static const int sector_data[][3]=
            {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
        float tab[4];
        int sector;
        h *= hscale;
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
}


struct HSV2RGB_f
{
    typedef float channel_type;

    HSV2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {
        #if CV_SIMD128
        hasSIMD = hasSIMD128();
        #endif
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, dcn = dstcn;
        n *= 3;

        if (dcn == 3)
        {
            #if CV_SIMD128
            if (hasSIMD)
            {
                for (; i <= n - 12; i += 12, dst += dcn * 4)
                {
                    v_float32x4 v_src[3];
                    v_load_deinterleave(src + i, v_src[0], v_src[1], v_src[2]);
                    HSV2RGB_simd(v_src[0], v_src[1], v_src[2], hscale);
                    v_store_interleave(dst, v_src[bidx], v_src[1], v_src[bidx^2]);
                }
            }
            #endif
            for( ; i < n; i += 3, dst += dcn )
            {
                HSV2RGB_native(src + i, dst, hscale, bidx);
            }
        } else { // dcn == 4
            float alpha = ColorChannel<float>::max();
            #if CV_SIMD128
            if (hasSIMD)
            {
                for (; i <= n - 12; i += 12, dst += dcn * 4)
                {
                    v_float32x4 v_src[3];
                    v_load_deinterleave(src + i, v_src[0], v_src[1], v_src[2]);
                    HSV2RGB_simd(v_src[0], v_src[1], v_src[2], hscale);
                    v_float32x4 v_a = v_setall_f32(alpha);
                    v_store_interleave(dst, v_src[bidx], v_src[1], v_src[bidx^2], v_a);
                }
            }
            #endif
            for( ; i < n; i += 3, dst += dcn )
            {
                HSV2RGB_native(src + i, dst, hscale, bidx);
                dst[3] = alpha;
            }
        }
    }

    int dstcn, blueIdx;
    float hscale;
    #if CV_SIMD128
    bool hasSIMD;
    #endif
};


struct HSV2RGB_b
{
    typedef uchar channel_type;

    HSV2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.0f / _hrange)
    {
        #if CV_SIMD128
        hasSIMD = hasSIMD128();
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int j = 0, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();

        #if CV_SIMD128
        if (hasSIMD)
        {
            for (j = 0; j <= (n - 16) * 3; j += 48, dst += dcn * 16)
            {
                v_uint8x16 h_b, s_b, v_b;
                v_uint16x8 h_w[2], s_w[2], v_w[2];
                v_uint32x4 h_u[4], s_u[4], v_u[4];
                v_load_deinterleave(src + j, h_b, s_b, v_b);
                v_expand(h_b, h_w[0], h_w[1]);
                v_expand(s_b, s_w[0], s_w[1]);
                v_expand(v_b, v_w[0], v_w[1]);
                v_expand(h_w[0], h_u[0], h_u[1]);
                v_expand(h_w[1], h_u[2], h_u[3]);
                v_expand(s_w[0], s_u[0], s_u[1]);
                v_expand(s_w[1], s_u[2], s_u[3]);
                v_expand(v_w[0], v_u[0], v_u[1]);
                v_expand(v_w[1], v_u[2], v_u[3]);

                v_int32x4 b_i[4], g_i[4], r_i[4];
                v_float32x4 v_coeff0 = v_setall_f32(1.0f / 255.0f);
                v_float32x4 v_coeff1 = v_setall_f32(255.0f);

                for( int k = 0; k < 4; k++ )
                {
                    v_float32x4 v_src[3];
                    v_src[0] = v_cvt_f32(v_reinterpret_as_s32(h_u[k]));
                    v_src[1] = v_cvt_f32(v_reinterpret_as_s32(s_u[k]));
                    v_src[2] = v_cvt_f32(v_reinterpret_as_s32(v_u[k]));

                    v_src[1] *= v_coeff0;
                    v_src[2] *= v_coeff0;
                    HSV2RGB_simd(v_src[0], v_src[1], v_src[2], hscale);

                    v_src[0] *= v_coeff1;
                    v_src[1] *= v_coeff1;
                    v_src[2] *= v_coeff1;
                    b_i[k] = v_trunc(v_src[0]);
                    g_i[k] = v_trunc(v_src[1]);
                    r_i[k] = v_trunc(v_src[2]);
                }

                v_uint16x8 r_w[2], g_w[2], b_w[2];
                v_uint8x16 r_b, g_b, b_b;

                r_w[0] = v_pack_u(r_i[0], r_i[1]);
                r_w[1] = v_pack_u(r_i[2], r_i[3]);
                r_b = v_pack(r_w[0], r_w[1]);
                g_w[0] = v_pack_u(g_i[0], g_i[1]);
                g_w[1] = v_pack_u(g_i[2], g_i[3]);
                g_b = v_pack(g_w[0], g_w[1]);
                b_w[0] = v_pack_u(b_i[0], b_i[1]);
                b_w[1] = v_pack_u(b_i[2], b_i[3]);
                b_b = v_pack(b_w[0], b_w[1]);

                if( dcn == 3 )
                {
                    if( blueIdx == 0 )
                        v_store_interleave(dst, b_b, g_b, r_b);
                    else
                        v_store_interleave(dst, r_b, g_b, b_b);
                }
                else
                {
                    v_uint8x16 alpha_b = v_setall_u8(alpha);
                    if( blueIdx == 0 )
                        v_store_interleave(dst, b_b, g_b, r_b, alpha_b);
                    else
                        v_store_interleave(dst, r_b, g_b, b_b, alpha_b);
                }
            }
        }
        #endif
        for( ; j < n * 3; j += 3, dst += dcn )
        {
            float buf[6];
            buf[0] = src[j];
            buf[1] = src[j+1] * (1.0f / 255.0f);
            buf[2] = src[j+2] * (1.0f / 255.0f);
            HSV2RGB_native(buf, buf + 3, hscale, blueIdx);
            dst[0] = saturate_cast<uchar>(buf[3] * 255.0f);
            dst[1] = saturate_cast<uchar>(buf[4] * 255.0f);
            dst[2] = saturate_cast<uchar>(buf[5] * 255.0f);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    int blueIdx;
    float hscale;
    #if CV_SIMD128
    bool hasSIMD;
    #endif
};


///////////////////////////////////// RGB <-> HLS ////////////////////////////////////////

struct RGB2HLS_f
{
    typedef float channel_type;

    RGB2HLS_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hscale(_hrange/360.f) {
        #if CV_SIMD128
        hasSIMD = hasSIMD128();
        #endif
    }

    #if CV_SIMD128
    inline void process(v_float32x4& v_r, v_float32x4& v_g,
                        v_float32x4& v_b, v_float32x4& v_hscale) const
    {
        v_float32x4 v_max_rgb = v_max(v_max(v_r, v_g), v_b);
        v_float32x4 v_min_rgb = v_min(v_min(v_r, v_g), v_b);

        v_float32x4 v_diff = v_max_rgb - v_min_rgb;
        v_float32x4 v_sum = v_max_rgb + v_min_rgb;
        v_float32x4 v_half = v_setall_f32(0.5f);
        v_float32x4 v_l = v_sum * v_half;

        v_float32x4 v_s = v_diff / v_select(v_l < v_half, v_sum, v_setall_f32(2.0f) - v_sum);

        v_float32x4 v_r_eq_max = v_max_rgb == v_r;
        v_float32x4 v_g_eq_max = v_max_rgb == v_g;
        v_float32x4 v_h = v_select(v_r_eq_max, v_g - v_b,
                          v_select(v_g_eq_max, v_b - v_r, v_r - v_g));
        v_float32x4 v_res = v_select(v_r_eq_max, (v_g < v_b) & v_setall_f32(360.0f),
                            v_select(v_g_eq_max, v_setall_f32(120.0f), v_setall_f32(240.0f)));
        v_float32x4 v_rev_diff = v_setall_f32(60.0f) / v_diff;
        v_h = v_muladd(v_h, v_rev_diff, v_res) * v_hscale;

        v_float32x4 v_diff_gt_eps = v_diff > v_setall_f32(FLT_EPSILON);
        v_r = v_diff_gt_eps & v_h;
        v_g = v_l;
        v_b = v_diff_gt_eps & v_s;
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, scn = srccn;
        n *= 3;

        #if CV_SIMD128
        if (hasSIMD)
        {
            v_float32x4 v_hscale = v_setall_f32(hscale);
            if (scn == 3) {
                if (bidx) {
                    for ( ; i <= n - 12; i += 12, src += scn * 4)
                    {
                        v_float32x4 v_r;
                        v_float32x4 v_g;
                        v_float32x4 v_b;
                        v_load_deinterleave(src, v_r, v_g, v_b);
                        process(v_r, v_g, v_b, v_hscale);
                        v_store_interleave(dst + i, v_r, v_g, v_b);
                    }
                } else {
                    for ( ; i <= n - 12; i += 12, src += scn * 4)
                    {
                        v_float32x4 v_r;
                        v_float32x4 v_g;
                        v_float32x4 v_b;
                        v_load_deinterleave(src, v_r, v_g, v_b);
                        process(v_b, v_g, v_r, v_hscale);
                        v_store_interleave(dst + i, v_b, v_g, v_r);
                    }
                }
            } else { // scn == 4
                if (bidx) {
                    for ( ; i <= n - 12; i += 12, src += scn * 4)
                    {
                        v_float32x4 v_r;
                        v_float32x4 v_g;
                        v_float32x4 v_b;
                        v_float32x4 v_a;
                        v_load_deinterleave(src, v_r, v_g, v_b, v_a);
                        process(v_r, v_g, v_b, v_hscale);
                        v_store_interleave(dst + i, v_r, v_g, v_b);
                    }
                } else {
                    for ( ; i <= n - 12; i += 12, src += scn * 4)
                    {
                        v_float32x4 v_r;
                        v_float32x4 v_g;
                        v_float32x4 v_b;
                        v_float32x4 v_a;
                        v_load_deinterleave(src, v_r, v_g, v_b, v_a);
                        process(v_b, v_g, v_r, v_hscale);
                        v_store_interleave(dst + i, v_b, v_g, v_r);
                    }
                }
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
    #if CV_SIMD128
    bool hasSIMD;
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
        #if CV_SIMD128
        hasSIMD = hasSIMD128();
        #endif
    }

    #if CV_SIMD128
    inline void process(v_float32x4& v_h, v_float32x4& v_l, v_float32x4& v_s) const
    {
        v_float32x4 v_one = v_setall_f32(1.0f);

        v_float32x4 v_l_le_half = v_l <= v_setall_f32(0.5f);
        v_float32x4 v_ls = v_l * v_s;
        v_float32x4 v_elem0 = v_select(v_l_le_half, v_ls, v_s - v_ls);

        v_float32x4 v_hs_raw = v_h * v_setall_f32(hscale);
        v_float32x4 v_pre_hs = v_cvt_f32(v_trunc(v_hs_raw));
        v_float32x4 v_hs = v_hs_raw - v_pre_hs;
        v_float32x4 v_sector = v_pre_hs - v_setall_f32(6.0f) * v_cvt_f32(v_trunc(v_hs_raw * v_setall_f32(1.0f / 6.0f)));
        v_float32x4 v_elem1 = v_hs + v_hs;

        v_float32x4 v_tab0 = v_l + v_elem0;
        v_float32x4 v_tab1 = v_l - v_elem0;
        v_float32x4 v_tab2 = v_l + v_elem0 - v_elem0 * v_elem1;
        v_float32x4 v_tab3 = v_l - v_elem0 + v_elem0 * v_elem1;

        v_float32x4 v_two  = v_setall_f32(2.0f);
        v_float32x4 v_four = v_setall_f32(4.0f);

        v_h = v_select(v_sector <  v_two , v_tab1,
              v_select(v_sector <= v_two , v_tab3,
              v_select(v_sector <= v_four, v_tab0, v_tab2)));

        v_l = v_select(v_sector <  v_one , v_tab3,
              v_select(v_sector <= v_two , v_tab0,
              v_select(v_sector <  v_four, v_tab2, v_tab1)));

        v_s = v_select(v_sector <  v_one , v_tab0,
              v_select(v_sector <  v_two , v_tab2,
              v_select(v_sector <  v_four, v_tab1,
              v_select(v_sector <= v_four, v_tab3, v_tab0))));
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, dcn = dstcn;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        #if CV_SIMD128
        if (hasSIMD)
        {
            if (dcn == 3)
            {
                if (bidx)
                {
                    for (; i <= n - 12; i += 12, dst += dcn * 4)
                    {
                        v_float32x4 v_h;
                        v_float32x4 v_l;
                        v_float32x4 v_s;
                        v_load_deinterleave(src + i, v_h, v_l, v_s);
                        process(v_h, v_l, v_s);
                        v_store_interleave(dst, v_s, v_l, v_h);
                    }
                } else {
                    for (; i <= n - 12; i += 12, dst += dcn * 4)
                    {
                        v_float32x4 v_h;
                        v_float32x4 v_l;
                        v_float32x4 v_s;
                        v_load_deinterleave(src + i, v_h, v_l, v_s);
                        process(v_h, v_l, v_s);
                        v_store_interleave(dst, v_h, v_l, v_s);
                    }
                }
            } else { // dcn == 4
                if (bidx)
                {
                    for (; i <= n - 12; i += 12, dst += dcn * 4)
                    {
                        v_float32x4 v_h;
                        v_float32x4 v_l;
                        v_float32x4 v_s;
                        v_load_deinterleave(src + i, v_h, v_l, v_s);
                        process(v_h, v_l, v_s);
                        v_float32x4 v_a = v_setall_f32(alpha);
                        v_store_interleave(dst, v_s, v_l, v_h, v_a);
                    }
                } else {
                    for (; i <= n - 12; i += 12, dst += dcn * 4)
                    {
                        v_float32x4 v_h;
                        v_float32x4 v_l;
                        v_float32x4 v_s;
                        v_load_deinterleave(src + i, v_h, v_l, v_s);
                        process(v_h, v_l, v_s);
                        v_float32x4 v_a = v_setall_f32(alpha);
                        v_store_interleave(dst, v_h, v_l, v_s, v_a);
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

                h *= hscale;
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
    #if CV_SIMD128
    bool hasSIMD;
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
    CV_INSTRUMENT_REGION();

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
    CV_INSTRUMENT_REGION();

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
