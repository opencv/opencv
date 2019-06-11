// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations

void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV);
void cvtHSVtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if defined(CV_CPU_BASELINE_MODE)
// included in color.hpp
#else
#include "color.simd_helpers.hpp"
#endif

namespace {
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
        CV_INSTRUMENT_REGION();

        int i, bidx = blueIdx, scn = srccn;
        const int hsv_shift = 12;

        static int sdiv_table[256];
        static int hdiv_table180[256];
        static int hdiv_table256[256];
        static volatile bool initialized = false;

        int hr = hrange;
        const int* hdiv_table = hr == 180 ? hdiv_table180 : hdiv_table256;

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

        i = 0;

#if CV_SIMD
        const int vsize = v_uint8::nlanes;
        for ( ; i <= n - vsize;
              i += vsize, src += scn*vsize, dst += 3*vsize)
        {
            v_uint8 b, g, r;
            if(scn == 4)
            {
                v_uint8 a;
                v_load_deinterleave(src, b, g, r, a);
            }
            else
            {
                v_load_deinterleave(src, b, g, r);
            }

            if(bidx)
                swap(b, r);

            v_uint8 h, s, v;
            v_uint8 vmin;
            v = v_max(b, v_max(g, r));
            vmin = v_min(b, v_min(g, r));

            v_uint8 diff, vr, vg;
            diff = v - vmin;
            v_uint8 v255 = vx_setall_u8(0xff), vz = vx_setzero_u8();
            vr = v_select(v == r, v255, vz);
            vg = v_select(v == g, v255, vz);

            // sdiv = sdiv_table[v]
            v_int32 sdiv[4];
            v_uint16 vd[2];
            v_expand(v, vd[0], vd[1]);
            v_int32 vq[4];
            v_expand(v_reinterpret_as_s16(vd[0]), vq[0], vq[1]);
            v_expand(v_reinterpret_as_s16(vd[1]), vq[2], vq[3]);
            {
                int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) storevq[vsize];
                for (int k = 0; k < 4; k++)
                {
                    v_store_aligned(storevq + k*vsize/4, vq[k]);
                }

                for(int k = 0; k < 4; k++)
                {
                    sdiv[k] = vx_lut(sdiv_table, storevq + k*vsize/4);
                }
            }

            // hdiv = hdiv_table[diff]
            v_int32 hdiv[4];
            v_uint16 diffd[2];
            v_expand(diff, diffd[0], diffd[1]);
            v_int32 diffq[4];
            v_expand(v_reinterpret_as_s16(diffd[0]), diffq[0], diffq[1]);
            v_expand(v_reinterpret_as_s16(diffd[1]), diffq[2], diffq[3]);
            {
                int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) storediffq[vsize];
                for (int k = 0; k < 4; k++)
                {
                    v_store_aligned(storediffq + k*vsize/4, diffq[k]);
                }

                for (int k = 0; k < 4; k++)
                {
                    hdiv[k] = vx_lut((int32_t*)hdiv_table, storediffq + k*vsize/4);
                }
            }

            // s = (diff * sdiv + (1 << (hsv_shift-1))) >> hsv_shift;
            v_int32 sq[4];
            v_int32 vdescale = vx_setall_s32(1 << (hsv_shift-1));
            for (int k = 0; k < 4; k++)
            {
                sq[k] = (diffq[k]*sdiv[k] + vdescale) >> hsv_shift;
            }
            v_int16 sd[2];
            sd[0] = v_pack(sq[0], sq[1]);
            sd[1] = v_pack(sq[2], sq[3]);
            s = v_pack_u(sd[0], sd[1]);

            // expand all to 16 bits
            v_uint16 bdu[2], gdu[2], rdu[2];
            v_expand(b, bdu[0], bdu[1]);
            v_expand(g, gdu[0], gdu[1]);
            v_expand(r, rdu[0], rdu[1]);
            v_int16 bd[2], gd[2], rd[2];
            bd[0] = v_reinterpret_as_s16(bdu[0]);
            bd[1] = v_reinterpret_as_s16(bdu[1]);
            gd[0] = v_reinterpret_as_s16(gdu[0]);
            gd[1] = v_reinterpret_as_s16(gdu[1]);
            rd[0] = v_reinterpret_as_s16(rdu[0]);
            rd[1] = v_reinterpret_as_s16(rdu[1]);

            v_int16 vrd[2], vgd[2];
            v_expand(v_reinterpret_as_s8(vr), vrd[0], vrd[1]);
            v_expand(v_reinterpret_as_s8(vg), vgd[0], vgd[1]);
            v_int16 diffsd[2];
            diffsd[0] = v_reinterpret_as_s16(diffd[0]);
            diffsd[1] = v_reinterpret_as_s16(diffd[1]);

            v_int16 hd[2];
            // h before division
            for (int k = 0; k < 2; k++)
            {
                v_int16 gb = gd[k] - bd[k];
                v_int16 br = bd[k] - rd[k] + (diffsd[k] << 1);
                v_int16 rg = rd[k] - gd[k] + (diffsd[k] << 2);
                hd[k] = (vrd[k] & gb) + ((~vrd[k]) & ((vgd[k] & br) + ((~vgd[k]) & rg)));
            }

            // h div and fix
            v_int32 hq[4];
            v_expand(hd[0], hq[0], hq[1]);
            v_expand(hd[1], hq[2], hq[3]);
            for(int k = 0; k < 4; k++)
            {
                hq[k] = (hq[k]*hdiv[k] + vdescale) >> hsv_shift;
            }
            hd[0] = v_pack(hq[0], hq[1]);
            hd[1] = v_pack(hq[2], hq[3]);
            v_int16 vhr = vx_setall_s16((short)hr);
            v_int16 vzd = vx_setzero_s16();
            hd[0] += v_select(hd[0] < vzd, vhr, vzd);
            hd[1] += v_select(hd[1] < vzd, vhr, vzd);
            h = v_pack_u(hd[0], hd[1]);

            v_store_interleave(dst, h, s, v);
        }
#endif

        for( ; i < n; i++, src += scn, dst += 3 )
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

            dst[0] = saturate_cast<uchar>(h);
            dst[1] = (uchar)s;
            dst[2] = (uchar)v;
        }
    }

    int srccn, blueIdx, hrange;
};


struct RGB2HSV_f
{
    typedef float channel_type;

    RGB2HSV_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange)
    { }

    #if CV_SIMD
    inline void process(const v_float32& v_r, const v_float32& v_g, const v_float32& v_b,
                        v_float32& v_h, v_float32& v_s, v_float32& v_v,
                        float hscale) const
    {
        v_float32 v_min_rgb = v_min(v_min(v_r, v_g), v_b);
        v_float32 v_max_rgb = v_max(v_max(v_r, v_g), v_b);

        v_float32 v_eps = vx_setall_f32(FLT_EPSILON);
        v_float32 v_diff = v_max_rgb - v_min_rgb;
        v_s = v_diff / (v_abs(v_max_rgb) + v_eps);

        v_float32 v_r_eq_max = v_r == v_max_rgb;
        v_float32 v_g_eq_max = v_g == v_max_rgb;
        v_h = v_select(v_r_eq_max, v_g - v_b,
              v_select(v_g_eq_max, v_b - v_r, v_r - v_g));
        v_float32 v_res = v_select(v_r_eq_max, (v_g < v_b) & vx_setall_f32(360.0f),
                          v_select(v_g_eq_max, vx_setall_f32(120.0f), vx_setall_f32(240.0f)));
        v_float32 v_rev_diff = vx_setall_f32(60.0f) / (v_diff + v_eps);
        v_h = v_muladd(v_h, v_rev_diff, v_res) * vx_setall_f32(hscale);

        v_v = v_max_rgb;
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

#if CV_SIMD
        const int vsize = v_float32::nlanes;
        for ( ; i <= n - 3*vsize; i += 3*vsize, src += scn * vsize)
        {
            v_float32 r, g, b, a;
            if(scn == 4)
            {
                v_load_deinterleave(src, r, g, b, a);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, r, g, b);
            }

            if(bidx)
                swap(b, r);

            v_float32 h, s, v;
            process(b, g, r, h, s, v, hscale);

            v_store_interleave(dst + i, h, s, v);
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
};


#if CV_SIMD
inline void HSV2RGB_simd(const v_float32& h, const v_float32& s, const v_float32& v,
                         v_float32& b, v_float32& g, v_float32& r, float hscale)
{
    v_float32 v_h = h;
    v_float32 v_s = s;
    v_float32 v_v = v;

    v_h = v_h * vx_setall_f32(hscale);

    v_float32 v_pre_sector = v_cvt_f32(v_trunc(v_h));
    v_h = v_h - v_pre_sector;
    v_float32 v_tab0 = v_v;
    v_float32 v_one = vx_setall_f32(1.0f);
    v_float32 v_tab1 = v_v * (v_one - v_s);
    v_float32 v_tab2 = v_v * (v_one - (v_s * v_h));
    v_float32 v_tab3 = v_v * (v_one - (v_s * (v_one - v_h)));

    v_float32 v_one_sixth = vx_setall_f32(1.0f / 6.0f);
    v_float32 v_sector = v_pre_sector * v_one_sixth;
    v_sector = v_cvt_f32(v_trunc(v_sector));
    v_float32 v_six = vx_setall_f32(6.0f);
    v_sector = v_pre_sector - (v_sector * v_six);

    v_float32 v_two = vx_setall_f32(2.0f);
    v_h = v_tab1 & (v_sector < v_two);
    v_h = v_h | (v_tab3 & (v_sector == v_two));
    v_float32 v_three = vx_setall_f32(3.0f);
    v_h = v_h | (v_tab0 & (v_sector == v_three));
    v_float32 v_four = vx_setall_f32(4.0f);
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

    b = v_h;
    g = v_s;
    r = v_v;
}
#endif


inline void HSV2RGB_native(float h, float s, float v,
                           float& b, float& g, float& r,
                           const float hscale)
{
    if( s == 0 )
        b = g = r = v;
    else
    {
        static const int sector_data[][3]=
            {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
        float tab[4];
        int sector;
        h *= hscale;
        h = fmod(h, 6.f);
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
}


struct HSV2RGB_f
{
    typedef float channel_type;

    HSV2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange)
    { }

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, bidx = blueIdx, dcn = dstcn;
        float alpha = ColorChannel<float>::max();
        float hs = hscale;
        n *= 3;

#if CV_SIMD
        const int vsize = v_float32::nlanes;
        v_float32 valpha = vx_setall_f32(alpha);
        for (; i <= n - vsize*3; i += vsize*3, dst += dcn * vsize)
        {
            v_float32 h, s, v, b, g, r;
            v_load_deinterleave(src + i, h, s, v);

            HSV2RGB_simd(h, s, v, b, g, r, hs);

            if(bidx)
                swap(b, r);

            if(dcn == 4)
            {
                v_store_interleave(dst, b, g, r, valpha);
            }
            else // dcn == 3
            {
                v_store_interleave(dst, b, g, r);
            }
        }
#endif
        for( ; i < n; i += 3, dst += dcn )
        {
            float h = src[i + 0], s = src[i + 1], v = src[i + 2];
            float b, g, r;
            HSV2RGB_native(h, s, v, b, g, r, hs);

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if(dcn == 4)
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
};


struct HSV2RGB_b
{
    typedef uchar channel_type;

    HSV2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.0f / _hrange)
    { }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int j = 0, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();

#if CV_SIMD
        const int vsize = v_float32::nlanes;

        for (j = 0; j <= (n - vsize*4) * 3; j += 3 * 4 * vsize, dst += dcn * 4 * vsize)
        {
            v_uint8 h_b, s_b, v_b;
            v_uint16 h_w[2], s_w[2], v_w[2];
            v_uint32 h_u[4], s_u[4], v_u[4];
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

            v_int32 b_i[4], g_i[4], r_i[4];
            v_float32 v_coeff0 = vx_setall_f32(1.0f / 255.0f);
            v_float32 v_coeff1 = vx_setall_f32(255.0f);

            for( int k = 0; k < 4; k++ )
            {
                v_float32 h = v_cvt_f32(v_reinterpret_as_s32(h_u[k]));
                v_float32 s = v_cvt_f32(v_reinterpret_as_s32(s_u[k]));
                v_float32 v = v_cvt_f32(v_reinterpret_as_s32(v_u[k]));

                s *= v_coeff0;
                v *= v_coeff0;
                v_float32 b, g, r;
                HSV2RGB_simd(h, s, v, b, g, r, hscale);

                b *= v_coeff1;
                g *= v_coeff1;
                r *= v_coeff1;
                b_i[k] = v_trunc(b);
                g_i[k] = v_trunc(g);
                r_i[k] = v_trunc(r);
            }

            v_uint16 r_w[2], g_w[2], b_w[2];
            v_uint8 r_b, g_b, b_b;

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
                v_uint8 alpha_b = vx_setall_u8(alpha);
                if( blueIdx == 0 )
                    v_store_interleave(dst, b_b, g_b, r_b, alpha_b);
                else
                    v_store_interleave(dst, r_b, g_b, b_b, alpha_b);
            }
        }
#endif

        for( ; j < n * 3; j += 3, dst += dcn )
        {
            float h, s, v, b, g, r;
            h = src[j];
            s = src[j+1] * (1.0f / 255.0f);
            v = src[j+2] * (1.0f / 255.0f);
            HSV2RGB_native(h, s, v, b, g, r, hscale);

            dst[blueIdx]   = saturate_cast<uchar>(b * 255.0f);
            dst[1]         = saturate_cast<uchar>(g * 255.0f);
            dst[blueIdx^2] = saturate_cast<uchar>(r * 255.0f);

            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    int blueIdx;
    float hscale;
};


///////////////////////////////////// RGB <-> HLS ////////////////////////////////////////

struct RGB2HLS_f
{
    typedef float channel_type;

    RGB2HLS_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hscale(_hrange/360.f)
    {
    }

#if CV_SIMD
    inline void process(const v_float32& r, const v_float32& g, const v_float32& b,
                        const v_float32& vhscale,
                        v_float32& h, v_float32& l, v_float32& s) const
    {
        v_float32 maxRgb = v_max(v_max(r, g), b);
        v_float32 minRgb = v_min(v_min(r, g), b);

        v_float32 diff = maxRgb - minRgb;
        v_float32 msum = maxRgb + minRgb;
        v_float32 vhalf = vx_setall_f32(0.5f);
        l = msum * vhalf;

        s = diff / v_select(l < vhalf, msum, vx_setall_f32(2.0f) - msum);

        v_float32 rMaxMask = maxRgb == r;
        v_float32 gMaxMask = maxRgb == g;

        h = v_select(rMaxMask, g - b, v_select(gMaxMask, b - r, r - g));
        v_float32 hpart = v_select(rMaxMask, (g < b) & vx_setall_f32(360.0f),
                          v_select(gMaxMask, vx_setall_f32(120.0f), vx_setall_f32(240.0f)));

        v_float32 invDiff = vx_setall_f32(60.0f) / diff;
        h = v_muladd(h, invDiff, hpart) * vhscale;

        v_float32 diffEpsMask = diff > vx_setall_f32(FLT_EPSILON);

        h = diffEpsMask & h;
        // l = l;
        s = diffEpsMask & s;
    }
#endif

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, bidx = blueIdx, scn = srccn;

#if CV_SIMD
        const int vsize = v_float32::nlanes;
        v_float32 vhscale = vx_setall_f32(hscale);

        for ( ; i <= n - vsize;
              i += vsize, src += scn * vsize, dst += 3 * vsize)
        {
            v_float32 r, g, b, h, l, s;

            if(scn == 4)
            {
                v_float32 a;
                v_load_deinterleave(src, b, g, r, a);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, b, g, r);
            }

            if(bidx)
                swap(r, b);

            process(r, g, b, vhscale, h, l, s);

            v_store_interleave(dst, h, l, s);
        }
#endif

        for( ; i < n; i++, src += scn, dst += 3 )
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

            dst[0] = h*hscale;
            dst[1] = l;
            dst[2] = s;
        }
    }

    int srccn, blueIdx;
    float hscale;
};


struct RGB2HLS_b
{
    typedef uchar channel_type;
    static const int bufChannels = 3;

    RGB2HLS_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), cvt(bufChannels, _blueIdx, (float)_hrange)
    { }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int scn = srccn;

#if CV_SIMD
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[bufChannels*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[bufChannels*BLOCK_SIZE];
#endif

#if CV_SIMD
        static const int fsize = v_float32::nlanes;
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) interTmpM[fsize*3];
        v_store_interleave(interTmpM, vx_setall_f32(1.f), vx_setall_f32(255.f), vx_setall_f32(255.f));
        v_float32 mhls[3];
        for(int k = 0; k < 3; k++)
        {
            mhls[k] = vx_load_aligned(interTmpM + k*fsize);
        }
#endif

        for(int i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

#if CV_SIMD
            v_float32 v255inv = vx_setall_f32(1.f/255.f);
            if (scn == 3)
            {
                int j = 0;
                static const int nBlock = fsize*2;
                for ( ; j <= (dn * bufChannels - nBlock);
                      j += nBlock, src += nBlock)
                {
                    v_uint16 drgb = vx_load_expand(src);
                    v_int32 qrgb0, qrgb1;
                    v_expand(v_reinterpret_as_s16(drgb), qrgb0, qrgb1);
                    v_store_aligned(buf + j + 0*fsize, v_cvt_f32(qrgb0)*v255inv);
                    v_store_aligned(buf + j + 1*fsize, v_cvt_f32(qrgb1)*v255inv);
                }
                for( ; j < dn*3; j++, src++ )
                {
                    buf[j] = src[0]*(1.f/255.f);
                }
            }
            else // if (scn == 4)
            {
                int j = 0;
                static const int nBlock = fsize*4;
                for ( ; j <= dn*bufChannels - nBlock*bufChannels;
                      j += nBlock*bufChannels, src += nBlock*4)
                {
                    v_uint8 rgb[3], dummy;
                    v_load_deinterleave(src, rgb[0], rgb[1], rgb[2], dummy);

                    v_uint16 d[3*2];
                    for(int k = 0; k < 3; k++)
                    {
                        v_expand(rgb[k], d[k*2+0], d[k*2+1]);
                    }
                    v_int32 q[3*4];
                    for(int k = 0; k < 3*2; k++)
                    {
                        v_expand(v_reinterpret_as_s16(d[k]), q[k*2+0], q[k*2+1]);
                    }

                    v_float32 f[3*4];
                    for(int k = 0; k < 3*4; k++)
                    {
                        f[k] = v_cvt_f32(q[k])*v255inv;
                    }

                    for(int k = 0; k < 4; k++)
                    {
                        v_store_interleave(buf + j + k*bufChannels*fsize, f[0*4+k], f[1*4+k], f[2*4+k]);
                    }
                }
                for( ; j < dn*3; j += 3, src += 4 )
                {
                    buf[j+0] = src[0]*(1.f/255.f);
                    buf[j+1] = src[1]*(1.f/255.f);
                    buf[j+2] = src[2]*(1.f/255.f);
                }
            }
#else
            for(int j = 0; j < dn*3; j += 3, src += scn )
            {
                buf[j+0] = src[0]*(1.f/255.f);
                buf[j+1] = src[1]*(1.f/255.f);
                buf[j+2] = src[2]*(1.f/255.f);
            }
#endif
            cvt(buf, buf, dn);

            int j = 0;
#if CV_SIMD
            for( ; j <= dn*3 - fsize*3*4; j += fsize*3*4)
            {
                v_float32 f[3*4];
                for(int k = 0; k < 3*4; k++)
                {
                    f[k] = vx_load_aligned(buf + j + k*fsize);
                }

                for(int k = 0; k < 4; k++)
                {
                    for(int l = 0; l < 3; l++)
                    {
                        f[k*3+l] = f[k*3+l] * mhls[l];
                    }
                }

                v_int32 q[3*4];
                for(int k = 0; k < 3*4; k++)
                {
                    q[k] = v_round(f[k]);
                }

                for(int k = 0; k < 3; k++)
                {
                    v_store(dst + j + k*fsize*4, v_pack_u(v_pack(q[k*4+0], q[k*4+1]),
                                                          v_pack(q[k*4+2], q[k*4+3])));
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
};


struct HLS2RGB_f
{
    typedef float channel_type;

    HLS2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange)
    { }

#if CV_SIMD
    inline void process(const v_float32& h, const v_float32& l, const v_float32& s,
                        v_float32& b, v_float32& g, v_float32& r) const
    {
        v_float32 v1 = vx_setall_f32(1.0f), v2 = vx_setall_f32(2.0f), v4 = vx_setall_f32(4.0f);

        v_float32 lBelowHalfMask = l <= vx_setall_f32(0.5f);
        v_float32 ls = l * s;
        v_float32 elem0 = v_select(lBelowHalfMask, ls, s - ls);

        v_float32 hsRaw = h * vx_setall_f32(hscale);
        v_float32 preHs = v_cvt_f32(v_trunc(hsRaw));
        v_float32 hs = hsRaw - preHs;
        v_float32 sector = preHs - vx_setall_f32(6.0f) * v_cvt_f32(v_trunc(hsRaw * vx_setall_f32(1.0f / 6.0f)));
        v_float32 elem1 = hs + hs;

        v_float32 tab0 = l + elem0;
        v_float32 tab1 = l - elem0;
        v_float32 tab2 = l + elem0 - elem0 * elem1;
        v_float32 tab3 = l - elem0 + elem0 * elem1;

        b = v_select(sector <  v2, tab1,
            v_select(sector <= v2, tab3,
            v_select(sector <= v4, tab0, tab2)));

        g = v_select(sector <  v1, tab3,
            v_select(sector <= v2, tab0,
            v_select(sector <  v4, tab2, tab1)));

        r = v_select(sector <  v1, tab0,
            v_select(sector <  v2, tab2,
            v_select(sector <  v4, tab1,
            v_select(sector <= v4, tab3, tab0))));
    }
#endif

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, bidx = blueIdx, dcn = dstcn;
        float alpha = ColorChannel<float>::max();

#if CV_SIMD
        static const int vsize = v_float32::nlanes;
        for (; i <= n - vsize; i += vsize, src += 3*vsize, dst += dcn*vsize)
        {
            v_float32 h, l, s, r, g, b;
            v_load_deinterleave(src, h, l, s);

            process(h, l, s, b, g, r);

            if(bidx)
                swap(b, r);

            if(dcn == 3)
            {
                v_store_interleave(dst, b, g, r);
            }
            else
            {
                v_float32 a = vx_setall_f32(alpha);
                v_store_interleave(dst, b, g, r, a);
            }
        }
#endif

        for( ; i < n; i++, src += 3, dst += dcn )
        {
            float h = src[0], l = src[1], s = src[2];
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
};


struct HLS2RGB_b
{
    typedef uchar channel_type;
    static const int bufChannels = 3;

    HLS2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(bufChannels, _blueIdx, (float)_hrange)
    { }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();

#if CV_SIMD
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[bufChannels*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[bufChannels*BLOCK_SIZE];
#endif

#if CV_SIMD
        static const int fsize = v_float32::nlanes;
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) interTmpM[fsize*3];
        v_float32 v255inv = vx_setall_f32(1.f/255.f);
        v_store_interleave(interTmpM, vx_setall_f32(1.f), v255inv, v255inv);
        v_float32 mhls[3];
        for(int k = 0; k < 3; k++)
        {
            mhls[k] = vx_load_aligned(interTmpM + k*fsize);
        }
#endif

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

#if CV_SIMD
            for( ; j <= dn*3 - 3*4*fsize; j += 3*4*fsize)
            {
                // 3x uchar -> 3*4 float
                v_uint8 u[3];
                for(int k = 0; k < 3; k++)
                {
                    u[k] = vx_load(src + j + k*4*fsize);
                }
                v_uint16 d[3*2];
                for(int k = 0; k < 3; k++)
                {
                    v_expand(u[k], d[k*2+0], d[k*2+1]);
                }
                v_int32 q[3*4];
                for(int k = 0; k < 3*2; k++)
                {
                    v_expand(v_reinterpret_as_s16(d[k]), q[k*2+0], q[k*2+1]);
                }

                v_float32 f[3*4];
                for(int k = 0; k < 4; k++)
                {
                    for(int l = 0; l < 3; l++)
                    {
                        f[k*3+l] = v_cvt_f32(q[k*3+l])*mhls[l];
                    }
                }

                for (int k = 0; k < 4*3; k++)
                {
                    v_store_aligned(buf + j + k*fsize, f[k]);
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

#if CV_SIMD
            v_float32 v255 = vx_setall_f32(255.f);
            if(dcn == 3)
            {
                int x = 0;
                float* pbuf = buf;
                for( ; x <= dn - 4*fsize; x += 4*fsize, dst += 4*fsize, pbuf += 4*fsize)
                {
                    v_float32 vf[4];
                    vf[0] = vx_load_aligned(pbuf + 0*fsize);
                    vf[1] = vx_load_aligned(pbuf + 1*fsize);
                    vf[2] = vx_load_aligned(pbuf + 2*fsize);
                    vf[3] = vx_load_aligned(pbuf + 3*fsize);
                    v_int32 vi[4];
                    vi[0] = v_round(vf[0]*v255);
                    vi[1] = v_round(vf[1]*v255);
                    vi[2] = v_round(vf[2]*v255);
                    vi[3] = v_round(vf[3]*v255);
                    v_store(dst, v_pack_u(v_pack(vi[0], vi[1]),
                                          v_pack(vi[2], vi[3])));
                }
                for( ; x < dn*3; x++, dst++, pbuf++)
                {
                    dst[0] = saturate_cast<uchar>(pbuf[0]*255.f);
                }
            }
            else // dcn == 4
            {
                int x = 0;
                float* pbuf = buf;
                for ( ; x <= dn - 4*fsize; x += fsize, dst += 4*fsize, pbuf += bufChannels*fsize)
                {
                    v_float32 r[4], g[4], b[4];
                    v_int32 ir[4], ig[4], ib[4];
                    for(int k = 0; k < 4; k++)
                    {
                        v_load_deinterleave(pbuf, r[k], g[k], b[k]);
                        ir[k] = v_round(r[k]*v255);
                        ig[k] = v_round(g[k]*v255);
                        ib[k] = v_round(b[k]*v255);
                    }
                    v_uint8 ur, ug, ub;
                    ur = v_pack_u(v_pack(ir[0], ir[1]), v_pack(ir[2], ir[3]));
                    ug = v_pack_u(v_pack(ig[0], ig[1]), v_pack(ig[2], ig[3]));
                    ub = v_pack_u(v_pack(ib[0], ib[1]), v_pack(ib[2], ib[3]));

                    v_uint8 valpha = vx_setall_u8(alpha);
                    v_store_interleave(dst, ur, ug, ub, valpha);
                }

                for( ; x < dn; x++, dst += dcn, pbuf += bufChannels)
                {
                    dst[0] = saturate_cast<uchar>(pbuf[0]*255.f);
                    dst[1] = saturate_cast<uchar>(pbuf[1]*255.f);
                    dst[2] = saturate_cast<uchar>(pbuf[2]*255.f);
                    dst[3] = alpha;
                }
            }
#else
            for(int x = 0; x < dn*3; x += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[x+0]*255.f);
                dst[1] = saturate_cast<uchar>(buf[x+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[x+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
#endif
        }
    }

    int dstcn;
    HLS2RGB_f cvt;
};

} // namespace anon

// 8u, 32f
void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_INSTRUMENT_REGION();

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

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
