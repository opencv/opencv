// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

#if CV_SIMD_SCALABLE
/* FIX IT:
// std::swap(a, b) is not available for RVV vector types,
// and CV_SWAP needs another "t" as input,
// For compatibility, we swap RVV vector manually by using this macro.

// If others scalable types (e.g. type in ARM SVE) can use std::swap,
// then replace CV_SIMD_SCALABLE with CV_RVV.
// If std::swap is available for RVV vector types in future, remove this macro.
*/
#define swap(a, b) {auto t = a; a = b; b = t;}
#endif

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

        const TablesSingleton& global_tables = TablesSingleton::getInstance();
        hdiv_table_ = hrange == 180 ? global_tables.hdiv_table180 : global_tables.hdiv_table256;
        sdiv_table_ = global_tables.sdiv_table;
    }

    struct TablesSingleton
    {
        int sdiv_table[256];
        int hdiv_table180[256];
        int hdiv_table256[256];

    protected:
        TablesSingleton()
        {
            const int hsv_shift = 12;

            sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;
            for (int i = 1; i < 256; i++)
            {
                sdiv_table[i] = saturate_cast<int>((255 << hsv_shift)/(1.*i));
                hdiv_table180[i] = saturate_cast<int>((180 << hsv_shift)/(6.*i));
                hdiv_table256[i] = saturate_cast<int>((256 << hsv_shift)/(6.*i));
            }
        }
    public:
        static TablesSingleton& getInstance()
        {
            static TablesSingleton g_tables;
            return g_tables;
        }
    };

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int bidx = blueIdx, scn = srccn;
        const int hsv_shift = 12;

        int hr = hrange;
        const int* hdiv_table/*[256]*/ = hdiv_table_;
        const int* sdiv_table/*[256]*/ = sdiv_table_;

        int i = 0;

#if CV_SIMD || CV_SIMD_SCALABLE
        const int vsize = VTraits<v_uint8>::vlanes();
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
            diff = v_sub(v, vmin);
            v_uint8 v255 = vx_setall_u8(0xff), vz = vx_setzero_u8();
            vr = v_select(v_eq(v, r), v255, vz);
            vg = v_select(v_eq(v, g), v255, vz);

            // sdiv = sdiv_table[v]
            v_int32 sdiv0, sdiv1, sdiv2, sdiv3;;
            v_uint16 vd0, vd1, vd2;
            v_expand(v, vd0, vd1);
            v_int32 vq0, vq1, vq2, vq3;
            v_expand(v_reinterpret_as_s16(vd0), vq0, vq1);
            v_expand(v_reinterpret_as_s16(vd1), vq2, vq3);
            {
                int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) storevq[VTraits<v_uint8>::max_nlanes];
                v_store_aligned(storevq, vq0);
                v_store_aligned(storevq + vsize/4, vq1);
                v_store_aligned(storevq + 2*vsize/4, vq2);
                v_store_aligned(storevq + 3*vsize/4, vq3);

                sdiv0 = vx_lut(sdiv_table, storevq);
                sdiv1 = vx_lut(sdiv_table, storevq + vsize/4);
                sdiv2 = vx_lut(sdiv_table, storevq + 2*vsize/4);
                sdiv3 = vx_lut(sdiv_table, storevq + 3*vsize/4);
            }

            // hdiv = hdiv_table[diff]
            v_int32 hdiv0, hdiv1, hdiv2, hdiv3;
            v_uint16 diffd0, diffd1, diffd2;
            v_expand(diff, diffd0, diffd1);
            v_int32 diffq0, diffq1, diffq2, diffq3;
            v_expand(v_reinterpret_as_s16(diffd0), diffq0, diffq1);
            v_expand(v_reinterpret_as_s16(diffd1), diffq2, diffq3);
            {
                int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) storediffq[VTraits<v_uint8>::max_nlanes];
                v_store_aligned(storediffq, diffq0);
                v_store_aligned(storediffq + vsize/4, diffq1);
                v_store_aligned(storediffq + 2*vsize/4, diffq2);
                v_store_aligned(storediffq + 3*vsize/4, diffq3);
                hdiv0 = vx_lut((int32_t*)hdiv_table, storediffq + 0*vsize/4);
                hdiv1 = vx_lut((int32_t*)hdiv_table, storediffq + 1*vsize/4);
                hdiv2 = vx_lut((int32_t*)hdiv_table, storediffq + 2*vsize/4);
                hdiv3 = vx_lut((int32_t*)hdiv_table, storediffq + 3*vsize/4);
            }

            // s = (diff * sdiv + (1 << (hsv_shift-1))) >> hsv_shift;
            v_int32 sq0, sq1, sq2, sq3;
            v_int32 vdescale = vx_setall_s32(1 << (hsv_shift-1));
            sq0 = v_shr<hsv_shift>(v_add(v_mul(diffq0, sdiv0), vdescale));
            sq1 = v_shr<hsv_shift>(v_add(v_mul(diffq1, sdiv1), vdescale));
            sq2 = v_shr<hsv_shift>(v_add(v_mul(diffq2, sdiv2), vdescale));
            sq3 = v_shr<hsv_shift>(v_add(v_mul(diffq3, sdiv3), vdescale));
            v_int16 sd0, sd1;
            sd0 = v_pack(sq0, sq1);
            sd1 = v_pack(sq2, sq3);
            s = v_pack_u(sd0, sd1);

            // expand all to 16 bits
            v_uint16 bdu0, bdu1, gdu0, gdu1, rdu0, rdu1;
            v_expand(b, bdu0, bdu1);
            v_expand(g, gdu0, gdu1);
            v_expand(r, rdu0, rdu1);
            v_int16 bd0, bd1, gd0, gd1, rd0, rd1;
            bd0 = v_reinterpret_as_s16(bdu0);
            bd1 = v_reinterpret_as_s16(bdu1);
            gd0 = v_reinterpret_as_s16(gdu0);
            gd1 = v_reinterpret_as_s16(gdu1);
            rd0 = v_reinterpret_as_s16(rdu0);
            rd1 = v_reinterpret_as_s16(rdu1);

            v_int16 vrd0, vrd1, vgd0, vgd1;
            v_expand(v_reinterpret_as_s8(vr), vrd0, vrd1);
            v_expand(v_reinterpret_as_s8(vg), vgd0, vgd1);
            v_int16 diffsd0, diffsd1;
            diffsd0 = v_reinterpret_as_s16(diffd0);
            diffsd1 = v_reinterpret_as_s16(diffd1);

            v_int16 hd0, hd1;
            // h before division
            v_int16 gb = v_sub(gd0 ,bd0);
            v_int16 br = v_add(v_sub(bd0 ,rd0), v_shl<1>(diffsd0));
            v_int16 rg = v_add(v_sub(rd0 ,gd0), v_shl<2>(diffsd0));
            hd0 = v_add(v_and(vrd0, gb), v_and(v_not(vrd0), v_add(v_and(vgd0, br), v_and(v_not(vgd0), rg))));
            gb = v_sub(gd1, bd1);
            br = v_add(v_sub(bd1, rd1), v_shl<1>(diffsd1));
            rg = v_add(v_sub(rd1, gd1), v_shl<2>(diffsd1));
            hd1 = v_add(v_and(vrd1, gb), v_and(v_not(vrd1), v_add(v_and(vgd1, br), v_and(v_not(vgd1), rg))));

            // h div and fix
            v_int32 hq0, hq1, hq2, hq3;
            v_expand(hd0, hq0, hq1);
            v_expand(hd1, hq2, hq3);
            hq0 = v_shr<hsv_shift>(v_add(v_mul(hq0, hdiv0), vdescale));
            hq1 = v_shr<hsv_shift>(v_add(v_mul(hq1, hdiv1), vdescale));
            hq2 = v_shr<hsv_shift>(v_add(v_mul(hq2, hdiv2), vdescale));
            hq3 = v_shr<hsv_shift>(v_add(v_mul(hq3, hdiv3), vdescale));

            hd0 = v_pack(hq0, hq1);
            hd1 = v_pack(hq2, hq3);
            v_int16 vhr = vx_setall_s16((short)hr);
            v_int16 vzd = vx_setzero_s16();
            hd0 = v_add(hd0 ,v_select(v_lt(hd0, vzd), vhr, vzd));
            hd1 = v_add(hd1 ,v_select(v_lt(hd1, vzd), vhr, vzd));
            h = v_pack_u(hd0, hd1);

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

    const int* hdiv_table_/*[256]*/;
    const int* sdiv_table_/*[256]*/;
};


struct RGB2HSV_f
{
    typedef float channel_type;

    RGB2HSV_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange)
    { }

    #if CV_SIMD || CV_SIMD_SCALABLE
    inline void process(const v_float32& v_r, const v_float32& v_g, const v_float32& v_b,
                        v_float32& v_h, v_float32& v_s, v_float32& v_v,
                        float hscale) const
    {
        v_float32 v_min_rgb = v_min(v_min(v_r, v_g), v_b);
        v_float32 v_max_rgb = v_max(v_max(v_r, v_g), v_b);

        v_float32 v_eps = vx_setall_f32(FLT_EPSILON);
        v_float32 v_diff = v_sub(v_max_rgb, v_min_rgb);
        v_s = v_div(v_diff, v_add(v_abs(v_max_rgb), v_eps));

        v_float32 v_r_eq_max = v_eq(v_r, v_max_rgb);
        v_float32 v_g_eq_max = v_eq(v_g, v_max_rgb);
        v_h = v_select(v_r_eq_max, v_sub(v_g, v_b),
              v_select(v_g_eq_max, v_sub(v_b, v_r), v_sub(v_r, v_g)));
        v_float32 v_res = v_select(v_r_eq_max,
                            v_select(v_lt(v_g, v_b), vx_setall_f32(360.0f), vx_setall_f32(0.0f)),
                            v_select(v_g_eq_max, vx_setall_f32(120.0f), vx_setall_f32(240.0f)));
        v_float32 v_rev_diff = v_div(vx_setall_f32(60.0f), v_add(v_diff, v_eps));
        v_h = v_mul(v_muladd(v_h, v_rev_diff, v_res), vx_setall_f32(hscale));

        v_v = v_max_rgb;
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

#if CV_SIMD || CV_SIMD_SCALABLE
        const int vsize = VTraits<v_float32>::vlanes();
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


#if CV_SIMD || CV_SIMD_SCALABLE
inline void HSV2RGB_simd(const v_float32& h, const v_float32& s, const v_float32& v,
                         v_float32& b, v_float32& g, v_float32& r, float hscale)
{
    v_float32 v_h = h;
    v_float32 v_s = s;
    v_float32 v_v = v;

    v_h = v_mul(v_h, vx_setall_f32(hscale));

    v_float32 v_pre_sector = v_cvt_f32(v_trunc(v_h));
    v_h = v_sub(v_h, v_pre_sector);
    v_float32 v_tab0 = v_v;
    v_float32 v_one = vx_setall_f32(1.0f);
    v_float32 v_tab1 = v_mul(v_v, v_sub(v_one, v_s));
    v_float32 v_tab2 = v_mul(v_v, v_sub(v_one, v_mul(v_s, v_h)));
    v_float32 v_tab3 = v_mul(v_v, v_sub(v_one, v_mul(v_s, v_sub(v_one, v_h))));

    v_float32 v_one_sixth = vx_setall_f32(1.0f / 6.0f);
    v_float32 v_sector = v_mul(v_pre_sector, v_one_sixth);
    v_sector = v_cvt_f32(v_trunc(v_sector));
    v_float32 v_six = vx_setall_f32(6.0f);
    v_sector = v_sub(v_pre_sector, v_mul(v_sector, v_six));

    v_float32 v_two = vx_setall_f32(2.0f);
    v_h = v_select(v_lt(v_sector, v_two), v_tab1, vx_setall_f32(0.0f));
    v_h = v_select(v_eq(v_sector, v_two), v_tab3, v_h);
    v_float32 v_three = vx_setall_f32(3.0f);
    v_h = v_select(v_eq(v_sector, v_three), v_tab0, v_h);
    v_float32 v_four = vx_setall_f32(4.0f);
    v_h = v_select(v_eq(v_sector, v_four), v_tab0, v_h);
    v_h = v_select(v_gt(v_sector, v_four), v_tab2, v_h);

    v_s = v_select(v_lt(v_sector, v_one), v_tab3, v_s);
    v_s = v_select(v_eq(v_sector, v_one), v_tab0, v_s);
    v_s = v_select(v_eq(v_sector, v_two), v_tab0, v_s);
    v_s = v_select(v_eq(v_sector, v_three), v_tab2, v_s);
    v_s = v_select(v_gt(v_sector, v_three), v_tab1, v_s);

    v_v = v_select(v_lt(v_sector, v_one), v_tab0, v_v);
    v_v = v_select(v_eq(v_sector, v_one), v_tab2, v_v);
    v_v = v_select(v_eq(v_sector, v_two), v_tab1, v_v);
    v_v = v_select(v_eq(v_sector, v_three), v_tab1, v_v);
    v_v = v_select(v_eq(v_sector, v_four), v_tab3, v_v);
    v_v = v_select(v_gt(v_sector, v_four), v_tab0, v_v);

    b = v_h;
    g = v_s;
    r = v_v;
}
#endif

// Compute the sector and the new H for HSV and HLS 2 RGB conversions.
inline void ComputeSectorAndClampedH(float& h, int &sector) {
    sector = cvFloor(h);
    h -= sector;
    sector %= 6;
    sector += sector < 0 ? 6 : 0;
}


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
        ComputeSectorAndClampedH(h, sector);

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

#if CV_SIMD || CV_SIMD_SCALABLE
        const int vsize = VTraits<v_float32>::vlanes();
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

#if CV_SIMD || CV_SIMD_SCALABLE
        const int vsize = VTraits<v_float32>::vlanes();

        for (j = 0; j <= (n - vsize*4) * 3; j += 3 * 4 * vsize, dst += dcn * 4 * vsize)
        {
            v_uint8 h_b, s_b, v_b;
            v_uint16 h_w0, h_w1, s_w0, s_w1, v_w0, v_w1;
            v_uint32 h_u0, h_u1, h_u2, h_u3, s_u0, s_u1, s_u2, s_u3, v_u0, v_u1, v_u2, v_u3;
            v_load_deinterleave(src + j, h_b, s_b, v_b);
            v_expand(h_b, h_w0, h_w1);
            v_expand(s_b, s_w0, s_w1);
            v_expand(v_b, v_w0, v_w1);
            v_expand(h_w0, h_u0, h_u1);
            v_expand(h_w1, h_u2, h_u3);
            v_expand(s_w0, s_u0, s_u1);
            v_expand(s_w1, s_u2, s_u3);
            v_expand(v_w0, v_u0, v_u1);
            v_expand(v_w1, v_u2, v_u3);

            v_int32 b_i0, b_i1, b_i2, b_i3, g_i0, g_i1, g_i2, g_i3, r_i0, r_i1, r_i2, r_i3;
            v_float32 v_coeff0 = vx_setall_f32(1.0f / 255.0f);
            v_float32 v_coeff1 = vx_setall_f32(255.0f);

            v_float32 h = v_cvt_f32(v_reinterpret_as_s32(h_u0));
            v_float32 s = v_cvt_f32(v_reinterpret_as_s32(s_u0));
            v_float32 v = v_cvt_f32(v_reinterpret_as_s32(v_u0));

            s = v_mul(s, v_coeff0);
            v = v_mul(v, v_coeff0);
            v_float32 b, g, r;
            HSV2RGB_simd(h, s, v, b, g, r, hscale);

            b = v_mul(b, v_coeff1);
            g = v_mul(g, v_coeff1);
            r = v_mul(r, v_coeff1);
            b_i0 = v_trunc(b);
            g_i0 = v_trunc(g);
            r_i0 = v_trunc(r);

            h = v_cvt_f32(v_reinterpret_as_s32(h_u1));
            s = v_cvt_f32(v_reinterpret_as_s32(s_u1));
            v = v_cvt_f32(v_reinterpret_as_s32(v_u1));

            s = v_mul(s, v_coeff0);
            v = v_mul(v, v_coeff0);
            HSV2RGB_simd(h, s, v, b, g, r, hscale);

            b = v_mul(b, v_coeff1);
            g = v_mul(g, v_coeff1);
            r = v_mul(r, v_coeff1);
            b_i1 = v_trunc(b);
            g_i1 = v_trunc(g);
            r_i1 = v_trunc(r);

            h = v_cvt_f32(v_reinterpret_as_s32(h_u2));
            s = v_cvt_f32(v_reinterpret_as_s32(s_u2));
            v = v_cvt_f32(v_reinterpret_as_s32(v_u2));

            s = v_mul(s, v_coeff0);
            v = v_mul(v, v_coeff0);
            HSV2RGB_simd(h, s, v, b, g, r, hscale);

            b = v_mul(b, v_coeff1);
            g = v_mul(g, v_coeff1);
            r = v_mul(r, v_coeff1);
            b_i2 = v_trunc(b);
            g_i2 = v_trunc(g);
            r_i2 = v_trunc(r);

            h = v_cvt_f32(v_reinterpret_as_s32(h_u3));
            s = v_cvt_f32(v_reinterpret_as_s32(s_u3));
            v = v_cvt_f32(v_reinterpret_as_s32(v_u3));

            s = v_mul(s, v_coeff0);
            v = v_mul(v, v_coeff0);
            HSV2RGB_simd(h, s, v, b, g, r, hscale);

            b = v_mul(b, v_coeff1);
            g = v_mul(g, v_coeff1);
            r = v_mul(r, v_coeff1);
            b_i3 = v_trunc(b);
            g_i3 = v_trunc(g);
            r_i3 = v_trunc(r);

            v_uint16 r_w0, r_w1, g_w0, g_w1, b_w0, b_w1;
            v_uint8 r_b, g_b, b_b;

            r_w0 = v_pack_u(r_i0, r_i1);
            r_w1 = v_pack_u(r_i2, r_i3);
            r_b = v_pack(r_w0, r_w1);
            g_w0 = v_pack_u(g_i0, g_i1);
            g_w1 = v_pack_u(g_i2, g_i3);
            g_b = v_pack(g_w0, g_w1);
            b_w0 = v_pack_u(b_i0, b_i1);
            b_w1 = v_pack_u(b_i2, b_i3);
            b_b = v_pack(b_w0, b_w1);

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

#if CV_SIMD || CV_SIMD_SCALABLE
    inline void process(const v_float32& r, const v_float32& g, const v_float32& b,
                        const v_float32& vhscale,
                        v_float32& h, v_float32& l, v_float32& s) const
    {
        v_float32 maxRgb = v_max(v_max(r, g), b);
        v_float32 minRgb = v_min(v_min(r, g), b);

        v_float32 diff = v_sub(maxRgb, minRgb);
        v_float32 msum = v_add(maxRgb, minRgb);
        v_float32 vhalf = vx_setall_f32(0.5f);
        l = v_mul(msum, vhalf);

        s = v_div(diff, v_select(v_lt(l, vhalf), msum, v_sub(vx_setall_f32(2.0f), msum)));

        v_float32 rMaxMask = v_eq(maxRgb, r);
        v_float32 gMaxMask = v_eq(maxRgb, g);

        h = v_select(rMaxMask, v_sub(g, b), v_select(gMaxMask, v_sub(b, r), v_sub(r, g)));
        v_float32 hpart = v_select(rMaxMask, v_select(v_lt(g, b), vx_setall_f32(360.0f), vx_setall_f32(0.0f)),
                          v_select(gMaxMask, vx_setall_f32(120.0f), vx_setall_f32(240.0f)));

        v_float32 invDiff = v_div(vx_setall_f32(60.0f), diff);
        h = v_mul(v_muladd(h, invDiff, hpart), vhscale);

        v_float32 diffEpsMask = v_gt(diff, vx_setall_f32(FLT_EPSILON));

        h = v_select(diffEpsMask, h, vx_setall_f32(0.0f));
        // l = l;
        s = v_select(diffEpsMask, s, vx_setall_f32(0.0f));
    }
#endif

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, bidx = blueIdx, scn = srccn;

#if CV_SIMD || CV_SIMD_SCALABLE
        const int vsize = VTraits<v_float32>::vlanes();
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

#if CV_SIMD || CV_SIMD_SCALABLE
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[bufChannels*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[bufChannels*BLOCK_SIZE];
#endif

#if CV_SIMD || CV_SIMD_SCALABLE
        static const int fsize = VTraits<v_float32>::vlanes();
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) interTmpM[VTraits<v_float32>::max_nlanes*3];
        v_store_interleave(interTmpM, vx_setall_f32(1.f), vx_setall_f32(255.f), vx_setall_f32(255.f));
        v_float32 mhls0, mhls1, mhls2;
        mhls0 = vx_load_aligned(interTmpM);
        mhls1 = vx_load_aligned(interTmpM + fsize);
        mhls2 = vx_load_aligned(interTmpM + 2*fsize);
#endif

        for(int i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

#if CV_SIMD || CV_SIMD_SCALABLE
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
                    v_store_aligned(buf + j + 0*fsize, v_mul(v_cvt_f32(qrgb0),v255inv));
                    v_store_aligned(buf + j + 1*fsize, v_mul(v_cvt_f32(qrgb1),v255inv));
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
                    v_uint8 rgb0, rgb1, rgb2, rgb3, dummy;
                    v_load_deinterleave(src, rgb0, rgb1, rgb2, dummy);

                    v_uint16 d0,d1,d2,d3,d4,d5;
                    v_expand(rgb0, d0, d1);
                    v_expand(rgb1, d2, d3);
                    v_expand(rgb2, d4, d5);
                    v_int32 q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11;

                    v_expand(v_reinterpret_as_s16(d0), q0, q1);
                    v_expand(v_reinterpret_as_s16(d1), q2, q3);
                    v_expand(v_reinterpret_as_s16(d2), q4, q5);
                    v_expand(v_reinterpret_as_s16(d3), q6, q7);
                    v_expand(v_reinterpret_as_s16(d4), q8, q9);
                    v_expand(v_reinterpret_as_s16(d5), q10, q11);
                    v_float32 f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11;
                    f0 = v_mul(v_cvt_f32(q0), v255inv);
                    f1 = v_mul(v_cvt_f32(q1), v255inv);
                    f2 = v_mul(v_cvt_f32(q2), v255inv);
                    f3 = v_mul(v_cvt_f32(q3), v255inv);
                    f4 = v_mul(v_cvt_f32(q4), v255inv);
                    f5 = v_mul(v_cvt_f32(q5), v255inv);
                    f6 = v_mul(v_cvt_f32(q6), v255inv);
                    f7 = v_mul(v_cvt_f32(q7), v255inv);
                    f8 = v_mul(v_cvt_f32(q8), v255inv);
                    f9 = v_mul(v_cvt_f32(q9), v255inv);
                    f10 = v_mul(v_cvt_f32(q10), v255inv);
                    f11 = v_mul(v_cvt_f32(q11), v255inv);

                    v_store_interleave(buf + j, f0, f4, f8);
                    v_store_interleave(buf + j + bufChannels*fsize, f1, f5, f9);
                    v_store_interleave(buf + j + 2*bufChannels*fsize, f2, f6, f10);
                    v_store_interleave(buf + j + 3*bufChannels*fsize, f3, f7, f11);
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
#if CV_SIMD || CV_SIMD_SCALABLE
            for( ; j <= dn*3 - fsize*3*4; j += fsize*3*4)
            {
                v_float32 f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11;
                f0 = vx_load_aligned(buf + j + 0*fsize);
                f1 = vx_load_aligned(buf + j + 1*fsize);
                f2 = vx_load_aligned(buf + j + 2*fsize);
                f3 = vx_load_aligned(buf + j + 3*fsize);
                f4 = vx_load_aligned(buf + j + 4*fsize);
                f5 = vx_load_aligned(buf + j + 5*fsize);
                f6 = vx_load_aligned(buf + j + 6*fsize);
                f7 = vx_load_aligned(buf + j + 7*fsize);
                f8 = vx_load_aligned(buf + j + 8*fsize);
                f9 = vx_load_aligned(buf + j + 9*fsize);
                f10 = vx_load_aligned(buf + j + 10*fsize);
                f11 = vx_load_aligned(buf + j + 11*fsize);

                f0 = v_mul(f0, mhls0);
                f1 = v_mul(f1, mhls1);
                f2 = v_mul(f2, mhls2);
                f3 = v_mul(f3, mhls0);
                f4 = v_mul(f4, mhls1);
                f5 = v_mul(f5, mhls2);
                f6 = v_mul(f6, mhls0);
                f7 = v_mul(f7, mhls1);
                f8 = v_mul(f8, mhls2);
                f9 = v_mul(f9, mhls0);
                f10 = v_mul(f10, mhls1);
                f11 = v_mul(f11, mhls2);

                v_int32 q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11;
                q0 = v_round(f0);
                q1 = v_round(f1);
                q2 = v_round(f2);
                q3 = v_round(f3);
                q4 = v_round(f4);
                q5 = v_round(f5);
                q6 = v_round(f6);
                q7 = v_round(f7);
                q8 = v_round(f8);
                q9 = v_round(f9);
                q10 = v_round(f10);
                q11 = v_round(f11);

                v_store(dst + j + 0*fsize*4, v_pack_u(v_pack(q0, q1),v_pack(q2, q3)));
                v_store(dst + j + 1*fsize*4, v_pack_u(v_pack(q4, q5),v_pack(q6, q7)));
                v_store(dst + j + 2*fsize*4, v_pack_u(v_pack(q8, q9),v_pack(q10, q11)));
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

#if CV_SIMD || CV_SIMD_SCALABLE
    inline void process(const v_float32& h, const v_float32& l, const v_float32& s,
                        v_float32& b, v_float32& g, v_float32& r) const
    {
        v_float32 v1 = vx_setall_f32(1.0f), v2 = vx_setall_f32(2.0f), v4 = vx_setall_f32(4.0f);

        v_float32 lBelowHalfMask = v_le(l, vx_setall_f32(0.5f));
        v_float32 ls = v_mul(l, s);
        v_float32 elem0 = v_select(lBelowHalfMask, ls, v_sub(s, ls));

        v_float32 hsRaw = v_mul(h, vx_setall_f32(hscale));
        v_float32 preHs = v_cvt_f32(v_trunc(hsRaw));
        v_float32 hs = v_sub(hsRaw, preHs);
        v_float32 sector = v_sub(preHs, v_mul(vx_setall_f32(6.0f), v_cvt_f32(v_trunc(v_mul(hsRaw, vx_setall_f32(1.0f / 6.0f))))));
        v_float32 elem1 = v_add(hs, hs);

        v_float32 tab0 = v_add(l, elem0);
        v_float32 tab1 = v_sub(l, elem0);
        v_float32 tab2 = v_sub(v_add(l, elem0), v_mul(elem0, elem1));
        v_float32 tab3 = v_add(v_sub(l, elem0), v_mul(elem0, elem1));

        b = v_select(v_lt(sector, v2), tab1,
            v_select(v_le(sector, v2), tab3,
            v_select(v_le(sector, v4), tab0, tab2)));

        g = v_select(v_lt(sector, v1), tab3,
            v_select(v_le(sector, v2), tab0,
            v_select(v_lt(sector, v4), tab2, tab1)));

        r = v_select(v_lt(sector, v1), tab0,
            v_select(v_lt(sector, v2), tab2,
            v_select(v_lt(sector, v4), tab1,
            v_select(v_le(sector, v4), tab3, tab0))));
    }
#endif

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, bidx = blueIdx, dcn = dstcn;
        float alpha = ColorChannel<float>::max();

#if CV_SIMD || CV_SIMD_SCALABLE
        static const int vsize = VTraits<v_float32>::vlanes();
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
                ComputeSectorAndClampedH(h, sector);

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

#if CV_SIMD || CV_SIMD_SCALABLE
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[bufChannels*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[bufChannels*BLOCK_SIZE];
#endif

#if CV_SIMD || CV_SIMD_SCALABLE
        static const int fsize = VTraits<v_float32>::vlanes();
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) interTmpM[VTraits<v_float32>::max_nlanes*3];
        v_float32 v255inv = vx_setall_f32(1.f/255.f);
        v_store_interleave(interTmpM, vx_setall_f32(1.f), v255inv, v255inv);
        v_float32 mhls0, mhls1, mhls2;
        mhls0 = vx_load_aligned(interTmpM + 0*fsize);
        mhls1 = vx_load_aligned(interTmpM + 1*fsize);
        mhls2 = vx_load_aligned(interTmpM + 2*fsize);
#endif

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

#if CV_SIMD || CV_SIMD_SCALABLE
            for( ; j <= dn*3 - 3*4*fsize; j += 3*4*fsize)
            {
                // 3x uchar -> 3*4 float
                v_uint8 u0, u1, u2;
                u0 = vx_load(src + j + 0*4*fsize);
                u1 = vx_load(src + j + 1*4*fsize);
                u2 = vx_load(src + j + 2*4*fsize);
                v_uint16 d0, d1, d2, d3, d4, d5;
                v_expand(u0, d0, d1);
                v_expand(u1, d2, d3);
                v_expand(u2, d4, d5);

                v_int32 q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11;
                v_expand(v_reinterpret_as_s16(d0), q0, q1);
                v_expand(v_reinterpret_as_s16(d1), q2, q3);
                v_expand(v_reinterpret_as_s16(d2), q4, q5);
                v_expand(v_reinterpret_as_s16(d3), q6, q7);
                v_expand(v_reinterpret_as_s16(d4), q8, q9);
                v_expand(v_reinterpret_as_s16(d5), q10, q11);

                v_float32 f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11;
                f0 = v_mul(v_cvt_f32(q0),mhls0);
                f1 = v_mul(v_cvt_f32(q1),mhls1);
                f2 = v_mul(v_cvt_f32(q2),mhls2);
                f3 = v_mul(v_cvt_f32(q3),mhls0);
                f4 = v_mul(v_cvt_f32(q4),mhls1);
                f5 = v_mul(v_cvt_f32(q5),mhls2);
                f6 = v_mul(v_cvt_f32(q6),mhls0);
                f7 = v_mul(v_cvt_f32(q7),mhls1);
                f8 = v_mul(v_cvt_f32(q8),mhls2);
                f9 = v_mul(v_cvt_f32(q9),mhls0);
                f10 = v_mul(v_cvt_f32(q10),mhls1);
                f11 = v_mul(v_cvt_f32(q11),mhls2);

                v_store_aligned(buf + j + 0*fsize, f0);
                v_store_aligned(buf + j + 1*fsize, f1);
                v_store_aligned(buf + j + 2*fsize, f2);
                v_store_aligned(buf + j + 3*fsize, f3);
                v_store_aligned(buf + j + 4*fsize, f4);
                v_store_aligned(buf + j + 5*fsize, f5);
                v_store_aligned(buf + j + 6*fsize, f6);
                v_store_aligned(buf + j + 7*fsize, f7);
                v_store_aligned(buf + j + 8*fsize, f8);
                v_store_aligned(buf + j + 9*fsize, f9);
                v_store_aligned(buf + j + 10*fsize, f10);
                v_store_aligned(buf + j + 11*fsize, f11);
            }
#endif
            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

#if CV_SIMD || CV_SIMD_SCALABLE
            v_float32 v255 = vx_setall_f32(255.f);
            if(dcn == 3)
            {
                int x = 0;
                float* pbuf = buf;
                for( ; x <= dn - 4*fsize; x += 4*fsize, dst += 4*fsize, pbuf += 4*fsize)
                {
                    v_float32 vf0, vf1, vf2, vf3;
                    vf0 = vx_load_aligned(pbuf + 0*fsize);
                    vf1 = vx_load_aligned(pbuf + 1*fsize);
                    vf2 = vx_load_aligned(pbuf + 2*fsize);
                    vf3 = vx_load_aligned(pbuf + 3*fsize);
                    v_int32 vi0, vi1, vi2, vi3;
                    vi0 = v_round(v_mul(vf0,v255));
                    vi1 = v_round(v_mul(vf1,v255));
                    vi2 = v_round(v_mul(vf2,v255));
                    vi3 = v_round(v_mul(vf3,v255));
                    v_store(dst, v_pack_u(v_pack(vi0, vi1),
                                          v_pack(vi2, vi3)));
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
                    v_float32 r0, r1, r2, r3, g0, g1, g2, g3, b0, b1, b2, b3;
                    v_int32 ir0, ir1, ir2, ir3, ig0, ig1, ig2, ig3, ib0, ib1, ib2, ib3;
                    v_load_deinterleave(pbuf, r0, g0, b0);
                    ir0 = v_round(v_mul(r0, v255));
                    ig0 = v_round(v_mul(g0, v255));
                    ib0 = v_round(v_mul(b0, v255));
                    v_load_deinterleave(pbuf, r1, g1, b1);
                    ir1 = v_round(v_mul(r1, v255));
                    ig1 = v_round(v_mul(g1, v255));
                    ib1 = v_round(v_mul(b1, v255));
                    v_load_deinterleave(pbuf, r2, g2, b2);
                    ir2 = v_round(v_mul(r2, v255));
                    ig2 = v_round(v_mul(g2, v255));
                    ib2 = v_round(v_mul(b2, v255));
                    v_load_deinterleave(pbuf, r3, g3, b3);
                    ir3 = v_round(v_mul(r3, v255));
                    ig3 = v_round(v_mul(g3, v255));
                    ib3 = v_round(v_mul(b3, v255));
                    v_uint8 ur, ug, ub;
                    ur = v_pack_u(v_pack(ir0, ir1), v_pack(ir2, ir3));
                    ug = v_pack_u(v_pack(ig0, ig1), v_pack(ig2, ig3));
                    ub = v_pack_u(v_pack(ib0, ib1), v_pack(ib2, ib3));

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
