// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isCbCr);
void cvtYUVtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isCbCr);
void cvtTwoPlaneYUVtoBGR(const uchar * y_data, size_t y_step, const uchar * uv_data, size_t uv_step,
                         uchar * dst_data, size_t dst_step,
                         int dst_width, int dst_height,
                         int dcn, bool swapBlue, int uIdx);
void cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                           uchar * dst_data, size_t dst_step,
                           int dst_width, int dst_height,
                           int dcn, bool swapBlue, int uIdx);
void cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step,
                           uchar * dst_data, size_t dst_step,
                           int width, int height,
                           int scn, bool swapBlue, int uIdx);
void cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step,
                         uchar * y_data, uchar * uv_data, size_t dst_step,
                         int width, int height,
                         int scn, bool swapBlue, int uIdx);
void cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int dcn, bool swapBlue, int uIdx, int ycn);
void cvtOnePlaneBGRtoYUV(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int scn, bool swapBlue, int uIdx, int ycn);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if defined(CV_CPU_BASELINE_MODE)
// included in color.hpp
#else
#include "color.simd_helpers.hpp"
#endif

namespace {
//constants for conversion from/to RGB and YUV, YCrCb according to BT.601

#if CV_SIMD_SCALABLE
template <class T>
static void swap(T&a, T&b) {
    T t = a;
    a = b;
    b = t;
}
#endif

//to YCbCr
static const float YCBF = 0.564f; // == 1/2/(1-B2YF)
static const float YCRF = 0.713f; // == 1/2/(1-R2YF)
static const int YCBI = 9241;  // == YCBF*16384
static const int YCRI = 11682; // == YCRF*16384
//to YUV
static const float B2UF = 0.492f;
static const float R2VF = 0.877f;
static const int B2UI = 8061;  // == B2UF*16384
static const int R2VI = 14369; // == R2VF*16384
//from YUV
static const float U2BF = 2.032f;
static const float U2GF = -0.395f;
static const float V2GF = -0.581f;
static const float V2RF = 1.140f;
static const int U2BI = 33292;
static const int U2GI = -6472;
static const int V2GI = -9519;
static const int V2RI = 18678;
//from YCrCb
static const float CB2BF = 1.773f;
static const float CB2GF = -0.344f;
static const float CR2GF = -0.714f;
static const float CR2RF = 1.403f;
static const int CB2BI = 29049;
static const int CB2GI = -5636;
static const int CR2GI = -11698;
static const int CR2RI = 22987;

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

template<typename _Tp> struct RGB2YCrCb_f
{
    typedef _Tp channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, bool _isCrCb) :
        srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF };
        static const float coeffs_yuv[] = { R2YF, G2YF, B2YF, R2VF, B2UF };
        for(int i = 0; i < 5; i++)
        {
            coeffs[i] = isCrCb ? coeffs_crb[i] : coeffs_yuv[i];
        }
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const _Tp delta = ColorChannel<_Tp>::half();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp Y = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Cr = saturate_cast<_Tp>((src[bidx^2] - Y)*C3 + delta);
            _Tp Cb = saturate_cast<_Tp>((src[bidx] - Y)*C4 + delta);
            dst[i] = Y; dst[i+1+yuvOrder] = Cr; dst[i+2-yuvOrder] = Cb;
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    float coeffs[5];
};

template <>
struct RGB2YCrCb_f<float>
{
    typedef float channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, bool _isCrCb) :
        srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF };
        static const float coeffs_yuv[] = { R2YF, G2YF, B2YF, R2VF, B2UF };
        for(int i = 0; i < 5; i++)
        {
            coeffs[i] = isCrCb ? coeffs_crb[i] : coeffs_yuv[i];
        }
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const float * src, float * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const float delta = ColorChannel<float>::half();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];

        int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
        v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4);
        v_float32 vdelta = vx_setall_f32(delta);
        const int vsize = VTraits<v_float32>::vlanes();
        for( ; i <= n-vsize;
             i += vsize, src += vsize*scn, dst += vsize*3)
        {
            v_float32 b, g, r, dummy;
            if(scn == 3)
            {
                v_load_deinterleave(src, b, g, r);
            }
            else
            {
                v_load_deinterleave(src, b, g, r, dummy);
            }

            v_float32 y, cr, cb;
            y = v_fma(b, vc0, v_fma(g, vc1, v_mul(r, vc2)));

            if(bidx)
                swap(r, b);

            cr = v_fma(v_sub(r, y), vc3, vdelta);
            cb = v_fma(v_sub(b, y), vc4, vdelta);

            if(yuvOrder)
            {
                v_store_interleave(dst, y, cb, cr);
            }
            else
            {
                v_store_interleave(dst, y, cr, cb);
            }
        }
        vx_cleanup();
#endif
        for ( ; i < n; i ++, src += scn, dst += 3)
        {
            float Y = src[0]*C0 + src[1]*C1 + src[2]*C2;
            float Cr = (src[bidx^2] - Y)*C3 + delta;
            float Cb = (src[bidx] - Y)*C4 + delta;
            dst[0         ] = Y;
            dst[1+yuvOrder] = Cr;
            dst[2-yuvOrder] = Cb;
        }
    }

    int srccn, blueIdx;
    bool isCrCb;
    float coeffs[5];
};


template<typename _Tp> struct RGB2YCrCb_i
{
    typedef _Tp channel_type;
    static const int shift = yuv_shift;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };

        for(int i = 0; i < 5; i++)
        {
            coeffs[i] = isCrCb ? coeffs_crb[i] : coeffs_yuv[i];
        }
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<_Tp>::half()*(1 << shift);
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, shift);
            dst[i] = saturate_cast<_Tp>(Y);
            dst[i+1+yuvOrder] = saturate_cast<_Tp>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<_Tp>(Cb);
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    int coeffs[5];
};


template<>
struct RGB2YCrCb_i<ushort>
{
    typedef ushort channel_type;
    static const int shift = yuv_shift;
    static const int fix_shift = (int)(sizeof(short)*8 - shift);

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };

        for(int i = 0; i < 5; i++)
        {
            coeffs[i] = isCrCb ? coeffs_crb[i] : coeffs_yuv[i];
        }
        if(blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int sdelta = ColorChannel<ushort>::half()*(1 << shift);
        int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint16>::vlanes();
        const int descale = 1 << (shift-1);

        v_int16 b2y = vx_setall_s16((short)C0);
        v_int16 g2y = vx_setall_s16((short)C1);
        v_int16 r2y = vx_setall_s16((short)C2);
        v_int16 one = vx_setall_s16(1);
        v_int16 z = vx_setzero_s16();

        v_int16 bg2y, r12y;
        v_int16 dummy;
        v_zip(b2y, g2y, bg2y, dummy);
        v_zip(r2y, one, r12y, dummy);

        v_int16 vdescale = vx_setall_s16(1 << (shift-1));
        v_int32 vc3 = vx_setall_s32(C3);
        v_int32 vc4 = vx_setall_s32(C4);
        v_int32 vdd = vx_setall_s32(sdelta + descale);

        for(; i <= n-vsize;
            i += vsize, src += vsize*scn, dst += vsize*3)
        {
            v_uint16 r, g, b, a;
            if(scn == 3)
            {
                v_load_deinterleave(src, b, g, r);
            }
            else
            {
                v_load_deinterleave(src, b, g, r, a);
            }

            v_uint16 y, cr, cb;

            v_int16 sb = v_reinterpret_as_s16(b);
            v_int16 sr = v_reinterpret_as_s16(r);
            v_int16 sg = v_reinterpret_as_s16(g);

            v_int16 bg0, bg1;
            v_int16 rd0, rd1;
            v_zip(sb, sg, bg0, bg1);
            v_zip(sr, vdescale, rd0, rd1);

            // fixing 16bit signed multiplication
            v_int16 mr, mg, mb;
            mr = v_and(v_lt(sr, z), r2y);
            mg = v_and(v_lt(sg, z), g2y);
            mb = v_and(v_lt(sb, z), b2y);
            v_int16 fixmul = v_shl(v_add_wrap(mr, v_add_wrap(mg, mb)), fix_shift);

            v_int32 ssy0 = v_shr(v_add(v_dotprod(bg0, bg2y), v_dotprod(rd0, r12y)), shift);
            v_int32 ssy1 = v_shr(v_add(v_dotprod(bg1, bg2y), v_dotprod(rd1, r12y)), shift);

            y = v_reinterpret_as_u16(v_add_wrap(v_pack(ssy0, ssy1), fixmul));

            if(bidx)
                swap(r, b);

            // (r-Y) and (b-Y) don't fit into int16 or uint16 range
            v_uint32 r0, r1, b0, b1;
            v_expand(r, r0, r1);
            v_expand(b, b0, b1);

            v_uint32 uy0, uy1;
            v_expand(y, uy0, uy1);

            v_int32 sr0 = v_reinterpret_as_s32(r0);
            v_int32 sr1 = v_reinterpret_as_s32(r1);
            v_int32 sb0 = v_reinterpret_as_s32(b0);
            v_int32 sb1 = v_reinterpret_as_s32(b1);
            v_int32 sy0 = v_reinterpret_as_s32(uy0);
            v_int32 sy1 = v_reinterpret_as_s32(uy1);

            sr0 = v_sub(sr0, sy0); sr1 = v_sub(sr1, sy1);
            sb0 = v_sub(sb0, sy0); sb1 = v_sub(sb1, sy1);

            v_int32 v_scr0, v_scr1, v_scb0, v_scb1;

            v_scr0 = v_shr(v_add(v_mul(sr0, vc3), vdd), shift);
            v_scr1 = v_shr(v_add(v_mul(sr1, vc3), vdd), shift);
            v_scb0 = v_shr(v_add(v_mul(sb0, vc4), vdd), shift);
            v_scb1 = v_shr(v_add(v_mul(sb1, vc4), vdd), shift);

            // saturate and pack
            cr = v_pack_u(v_scr0, v_scr1);
            cb = v_pack_u(v_scb0, v_scb1);

            if(yuvOrder)
            {
                v_store_interleave(dst, y, cb, cr);
            }
            else
            {
                v_store_interleave(dst, y, cr, cb);
            }
        }
        vx_cleanup();
#endif
        for( ; i < n; i++, src += scn, dst += 3)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + sdelta, shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + sdelta, shift);
            dst[0]          = saturate_cast<ushort>(Y);
            dst[1+yuvOrder] = saturate_cast<ushort>(Cr);
            dst[2-yuvOrder] = saturate_cast<ushort>(Cb);
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    int coeffs[5];
};


template <>
struct RGB2YCrCb_i<uchar>
{
    typedef uchar channel_type;
    static const int shift = yuv_shift;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        for(int i = 0; i < 5; i++)
        {
            coeffs[i] = isCrCb ? coeffs_crb[i] : coeffs_yuv[i];
        }
        if (blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const uchar * src, uchar * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<uchar>::half()*(1 << shift);

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint8>::vlanes();
        const int descaleShift = 1 << (shift-1);
        v_int16 bg2y;
        v_int16 r12y;
        v_int16 dummy;
        v_zip(vx_setall_s16((short)C0), vx_setall_s16((short)C1), bg2y, dummy);
        v_zip(vx_setall_s16((short)C2), vx_setall_s16( 1), r12y, dummy);

        // delta + descaleShift == descaleShift*(half*2+1)
        v_int16 c3h, c4h;
        const short h21 = (short)(ColorChannel<uchar>::half()*2+1);
        v_zip(vx_setall_s16((short)C3), vx_setall_s16(h21), c3h, dummy);
        v_zip(vx_setall_s16((short)C4), vx_setall_s16(h21), c4h, dummy);

        v_int16 vdescale = vx_setall_s16(descaleShift);

        for( ; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += 3*vsize)
        {
            v_uint8 r, g, b, a;
            if(scn == 3)
            {
                v_load_deinterleave(src, b, g, r);
            }
            else
            {
                v_load_deinterleave(src, b, g, r, a);
            }

            v_uint8 y;

            v_uint16 r0, r1, g0, g1, b0, b1;
            v_expand(r, r0, r1);
            v_expand(g, g0, g1);
            v_expand(b, b0, b1);

            v_int16 sr0, sr1, sg0, sg1, sb0, sb1;
            sr0 = v_reinterpret_as_s16(r0); sr1 = v_reinterpret_as_s16(r1);
            sg0 = v_reinterpret_as_s16(g0); sg1 = v_reinterpret_as_s16(g1);
            sb0 = v_reinterpret_as_s16(b0); sb1 = v_reinterpret_as_s16(b1);

            v_uint32 y00, y01, y10, y11;
            {
                v_int16 bg00, bg01, bg10, bg11;
                v_int16 rd00, rd01, rd10, rd11;
                v_zip(sb0, sg0, bg00, bg01);
                v_zip(sb1, sg1, bg10, bg11);
                v_zip(sr0, vdescale, rd00, rd01);
                v_zip(sr1, vdescale, rd10, rd11);

                y00 = v_shr(v_reinterpret_as_u32(v_add(v_dotprod(bg00, bg2y), v_dotprod(rd00, r12y))), shift);
                y01 = v_shr(v_reinterpret_as_u32(v_add(v_dotprod(bg01, bg2y), v_dotprod(rd01, r12y))), shift);
                y10 = v_shr(v_reinterpret_as_u32(v_add(v_dotprod(bg10, bg2y), v_dotprod(rd10, r12y))), shift);
                y11 = v_shr(v_reinterpret_as_u32(v_add(v_dotprod(bg11, bg2y), v_dotprod(rd11, r12y))), shift);
            }

            v_uint16 y0, y1;
            y0 = v_pack(y00, y01);
            y1 = v_pack(y10, y11);

            y = v_pack(y0, y1);

            v_int16 sy0, sy1;
            sy0 = v_reinterpret_as_s16(y0);
            sy1 = v_reinterpret_as_s16(y1);

            // (r-Y) and (b-Y) don't fit into 8 bit, use 16 bits instead
            sr0 = v_sub_wrap(sr0, sy0);
            sr1 = v_sub_wrap(sr1, sy1);
            sb0 = v_sub_wrap(sb0, sy0);
            sb1 = v_sub_wrap(sb1, sy1);

            if(bidx)
            {
                swap(sr0, sb0); swap(sr1, sb1);
            }

            v_int32 cr00, cr01, cr10, cr11;
            v_int32 cb00, cb01, cb10, cb11;

            // delta + descaleShift == descaleShift*(half*2+1)
            {
                v_int16 rd00, rd01, rd10, rd11;
                v_int16 bd00, bd01, bd10, bd11;

                v_zip(sr0, vdescale, rd00, rd01);
                v_zip(sr1, vdescale, rd10, rd11);

                v_zip(sb0, vdescale, bd00, bd01);
                v_zip(sb1, vdescale, bd10, bd11);

                cr00 = v_dotprod(rd00, c3h);
                cr01 = v_dotprod(rd01, c3h);
                cr10 = v_dotprod(rd10, c3h);
                cr11 = v_dotprod(rd11, c3h);

                cb00 = v_dotprod(bd00, c4h);
                cb01 = v_dotprod(bd01, c4h);
                cb10 = v_dotprod(bd10, c4h);
                cb11 = v_dotprod(bd11, c4h);
            }

            v_uint8 cr, cb;

            cr00 = v_shr(cr00, shift);
            cr01 = v_shr(cr01, shift);
            cr10 = v_shr(cr10, shift);
            cr11 = v_shr(cr11, shift);

            cb00 = v_shr(cb00, shift);
            cb01 = v_shr(cb01, shift);
            cb10 = v_shr(cb10, shift);
            cb11 = v_shr(cb11, shift);

            v_int16 cr0, cr1, cb0, cb1;
            cr0 = v_pack(cr00, cr01); cr1 = v_pack(cr10, cr11);
            cb0 = v_pack(cb00, cb01); cb1 = v_pack(cb10, cb11);

            cr = v_pack_u(cr0, cr1);
            cb = v_pack_u(cb0, cb1);

            if(yuvOrder)
            {
                v_store_interleave(dst, y, cb, cr);
            }
            else
            {
                v_store_interleave(dst, y, cr, cb);
            }
        }
        vx_cleanup();
#endif

        for ( ; i < n; i++, src += scn, dst += 3)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, shift);
            dst[0] = saturate_cast<uchar>(Y);
            dst[1+yuvOrder] = saturate_cast<uchar>(Cr);
            dst[2-yuvOrder] = saturate_cast<uchar>(Cb);
        }
    }

    int srccn, blueIdx, coeffs[5];
    bool isCrCb;
};


template<typename _Tp> struct YCrCb2RGB_f
{
    typedef _Tp channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_cbr[] = {CR2RF, CR2GF, CB2GF, CB2BF};
        static const float coeffs_yuv[] = { V2RF,  V2GF,  U2GF,  U2BF};
        for(int i = 0; i < 4; i++)
        {
            coeffs[i] = isCrCb ? coeffs_cbr[i] : coeffs_yuv[i];
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const _Tp delta = ColorChannel<_Tp>::half(), alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp Y = src[i];
            _Tp Cr = src[i+1+yuvOrder];
            _Tp Cb = src[i+2-yuvOrder];

            _Tp b = saturate_cast<_Tp>(Y + (Cb - delta)*C3);
            _Tp g = saturate_cast<_Tp>(Y + (Cb - delta)*C2 + (Cr - delta)*C1);
            _Tp r = saturate_cast<_Tp>(Y + (Cr - delta)*C0);

            dst[bidx] = b; dst[1] = g; dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    float coeffs[4];
};


template<>
struct YCrCb2RGB_f<float>
{
    typedef float channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_cbr[] = {CR2RF, CR2GF, CB2GF, CB2BF};
        static const float coeffs_yuv[] = { V2RF,  V2GF,  U2GF,  U2BF};
        for(int i = 0; i < 4; i++)
        {
            coeffs[i] = isCrCb ? coeffs_cbr[i] : coeffs_yuv[i];
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const float delta = ColorChannel<float>::half(), alpha = ColorChannel<float>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];

        int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1);
        v_float32 vc2 = vx_setall_f32(C2), vc3 = vx_setall_f32(C3);
        v_float32 vdelta = vx_setall_f32(delta);
        v_float32 valpha = vx_setall_f32(alpha);
        const int vsize = VTraits<v_float32>::vlanes();
        for( ; i <= n-vsize;
             i += vsize, src += vsize*3, dst += vsize*dcn)
        {
            v_float32 y, cr, cb;
            if(yuvOrder)
                v_load_deinterleave(src, y, cb, cr);
            else
                v_load_deinterleave(src, y, cr, cb);

            v_float32 b, g, r;

            cb = v_sub(cb, vdelta); cr = v_sub(cr, vdelta);
            b = v_fma(cb, vc3, y);
            g = v_fma(cr, vc1, v_fma(cb, vc2, y));
            r = v_fma(cr, vc0, y);

            if(bidx)
                swap(r, b);

            if(dcn == 3)
                v_store_interleave(dst, b, g, r);
            else
                v_store_interleave(dst, b, g, r, valpha);
        }
        vx_cleanup();
#endif
        for(; i < n; i++, src += 3, dst += dcn)
        {
            float Y  = src[0];
            float Cr = src[1+yuvOrder];
            float Cb = src[2-yuvOrder];

            float b = Y + (Cb - delta)*C3;
            float g = Y + (Cb - delta)*C2 + (Cr - delta)*C1;
            float r = Y + (Cr - delta)*C0;

            dst[bidx] = b; dst[1] = g; dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    float coeffs[4];
};


template<typename _Tp> struct YCrCb2RGB_i
{
    typedef _Tp channel_type;
    static const int shift = yuv_shift;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        for(int i = 0; i < 4; i++)
        {
            coeffs[i] = isCrCb ? coeffs_crb[i] : coeffs_yuv[i];
        }
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const _Tp delta = ColorChannel<_Tp>::half(), alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp Y = src[i];
            _Tp Cr = src[i+1+yuvOrder];
            _Tp Cb = src[i+2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, shift);

            dst[bidx] = saturate_cast<_Tp>(b);
            dst[1] = saturate_cast<_Tp>(g);
            dst[bidx^2] = saturate_cast<_Tp>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    int coeffs[4];
};


template <>
struct YCrCb2RGB_i<uchar>
{
    typedef uchar channel_type;
    static const int shift = yuv_shift;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        for(int i = 0; i < 4; i++)
        {
            coeffs[i] = isCrCb ? coeffs_crb[i] : coeffs_yuv[i];
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const uchar delta = ColorChannel<uchar>::half(), alpha = ColorChannel<uchar>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint8>::vlanes();
        v_uint8 valpha = vx_setall_u8(alpha);
        v_uint8 vdelta = vx_setall_u8(delta);
        const int descaleShift = 1 << (shift - 1);
        v_int32 vdescale = vx_setall_s32(descaleShift);

        v_int16 vc0 = vx_setall_s16((short)C0), vc1 = vx_setall_s16((short)C1), vc2 = vx_setall_s16((short)C2);
        // if YUV then C3 > 2^15, need to subtract it
        // to fit in short by short multiplication
        v_int16 vc3 = vx_setall_s16(yuvOrder ? (short)(C3-(1 << 15)) : (short)C3);

        for( ; i <= n-vsize;
             i += vsize, src += 3*vsize, dst += dcn*vsize)
        {
            v_uint8 y, cr, cb;
            if(yuvOrder)
            {
                v_load_deinterleave(src, y, cb, cr);
            }
            else
            {
                v_load_deinterleave(src, y, cr, cb);
            }

            cr = v_sub_wrap(cr, vdelta);
            cb = v_sub_wrap(cb, vdelta);

            v_int8 scr = v_reinterpret_as_s8(cr);
            v_int8 scb = v_reinterpret_as_s8(cb);

            v_int16 v_scr0, v_scr1, v_scb0, v_scb1;
            v_expand(scr, v_scr0, v_scr1);
            v_expand(scb, v_scb0, v_scb1);

            v_int32 b00, b01, b10, b11;
            v_int32 g00, g01, g10, g11;
            v_int32 r00, r01, r10, r11;

            v_mul_expand(v_scb0, vc3, b00, b01);
            v_mul_expand(v_scb1, vc3, b10, b11);
            if(yuvOrder)
            {
                // if YUV then C3 > 2^15
                // so we fix the multiplication
                v_int32 cb00, cb01, cb10, cb11;
                v_expand(v_scb0, cb00, cb01);
                v_expand(v_scb1, cb10, cb11);
                b00 = v_add(b00, v_shl<15>(cb00)); b01 = v_add(b01, v_shl<15>(cb01));
                b10 = v_add(b10, v_shl<15>(cb10)); b11 = v_add(b11, v_shl<15>(cb11));
            }

            v_int32 t00, t01, t10, t11;
            v_mul_expand(v_scb0, vc2, t00, t01);
            v_mul_expand(v_scb1, vc2, t10, t11);
            v_mul_expand(v_scr0, vc1, g00, g01);
            v_mul_expand(v_scr1, vc1, g10, g11);
            g00 = v_add(g00, t00); g01 = v_add(g01, t01);
            g10 = v_add(g10, t10); g11 = v_add(g11, t11);
            v_mul_expand(v_scr0, vc0, r00, r01);
            v_mul_expand(v_scr1, vc0, r10, r11);

            b00 = v_shr(v_add(b00, vdescale), shift); b01 = v_shr(v_add(b01, vdescale), shift);
            b10 = v_shr(v_add(b10, vdescale), shift); b11 = v_shr(v_add(b11, vdescale), shift);
            g00 = v_shr(v_add(g00, vdescale), shift); g01 = v_shr(v_add(g01, vdescale), shift);
            g10 = v_shr(v_add(g10, vdescale), shift); g11 = v_shr(v_add(g11, vdescale), shift);
            r00 = v_shr(v_add(r00, vdescale), shift); r01 = v_shr(v_add(r01, vdescale), shift);
            r10 = v_shr(v_add(r10, vdescale), shift); r11 = v_shr(v_add(r11, vdescale), shift);

            v_int16 b0, b1, g0, g1, r0, r1;
            b0 = v_pack(b00, b01); b1 = v_pack(b10, b11);
            g0 = v_pack(g00, g01); g1 = v_pack(g10, g11);
            r0 = v_pack(r00, r01); r1 = v_pack(r10, r11);

            v_uint16 y0, y1;
            v_expand(y, y0, y1);
            v_int16 sy0, sy1;
            sy0 = v_reinterpret_as_s16(y0);
            sy1 = v_reinterpret_as_s16(y1);

            b0 = v_add_wrap(b0, sy0); b1 = v_add_wrap(b1, sy1);
            g0 = v_add_wrap(g0, sy0); g1 = v_add_wrap(g1, sy1);
            r0 = v_add_wrap(r0, sy0); r1 = v_add_wrap(r1, sy1);

            v_uint8 b, g, r;
            b = v_pack_u(b0, b1);
            g = v_pack_u(g0, g1);
            r = v_pack_u(r0, r1);

            if(bidx)
                swap(r, b);

            if(dcn == 3)
            {
                v_store_interleave(dst, b, g, r);
            }
            else
            {
                v_store_interleave(dst, b, g, r, valpha);
            }
        }
        vx_cleanup();
#endif

        for ( ; i < n; i++, src += 3, dst += dcn)
        {
            uchar Y  = src[0];
            uchar Cr = src[1+yuvOrder];
            uchar Cb = src[2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, shift);

            dst[bidx] = saturate_cast<uchar>(b);
            dst[1] = saturate_cast<uchar>(g);
            dst[bidx^2] = saturate_cast<uchar>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    int coeffs[4];
};


template <>
struct YCrCb2RGB_i<ushort>
{
    typedef ushort channel_type;
    static const int shift = yuv_shift;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        for(int i = 0; i < 4; i++)
        {
            coeffs[i] = isCrCb ? coeffs_crb[i] : coeffs_yuv[i];
        }
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const ushort delta = ColorChannel<ushort>::half(), alpha = ColorChannel<ushort>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint16>::vlanes();
        const int descaleShift = 1 << (shift-1);
        v_uint16 valpha = vx_setall_u16(alpha);
        v_uint16 vdelta = vx_setall_u16(delta);
        v_int16 vc0 = vx_setall_s16((short)C0), vc1 = vx_setall_s16((short)C1), vc2 = vx_setall_s16((short)C2);
        // if YUV then C3 > 2^15, need to subtract it
        // to fit in short by short multiplication
        v_int16 vc3 = vx_setall_s16(yuvOrder ? (short)(C3-(1 << 15)) : (short)C3);
        v_int32 vdescale = vx_setall_s32(descaleShift);
        for(; i <= n-vsize;
            i += vsize, src += vsize*3, dst += vsize*dcn)
        {
            v_uint16 y, cr, cb;
            if(yuvOrder)
            {
                v_load_deinterleave(src, y, cb, cr);
            }
            else
            {
                v_load_deinterleave(src, y, cr, cb);
            }

            v_uint32 uy0, uy1;
            v_expand(y, uy0, uy1);
            v_int32 y0 = v_reinterpret_as_s32(uy0);
            v_int32 y1 = v_reinterpret_as_s32(uy1);

            cr = v_sub_wrap(cr, vdelta);
            cb = v_sub_wrap(cb, vdelta);

            v_int32 b0, b1, g0, g1, r0, r1;

            v_int16 scb = v_reinterpret_as_s16(cb);
            v_int16 scr = v_reinterpret_as_s16(cr);
            v_mul_expand(scb, vc3, b0, b1);
            if(yuvOrder)
            {
                // if YUV then C3 > 2^15
                // so we fix the multiplication
                v_int32 cb0, cb1;
                v_expand(scb, cb0, cb1);
                b0 = v_add(b0, v_shl<15>(cb0));
                b1 = v_add(b1, v_shl<15>(cb1));
            }
            v_int32 t0, t1;
            v_mul_expand(scb, vc2, t0, t1);
            v_mul_expand(scr, vc1, g0, g1);
            g0 = v_add(g0, t0); g1 = v_add(g1, t1);
            v_mul_expand(scr, vc0, r0, r1);

            // shifted term doesn't fit into 16 bits, addition is to be done in 32 bits
            b0 = v_add(v_shr(v_add(b0, vdescale), shift), y0);
            b1 = v_add(v_shr(v_add(b1, vdescale), shift), y1);
            g0 = v_add(v_shr(v_add(g0, vdescale), shift), y0);
            g1 = v_add(v_shr(v_add(g1, vdescale), shift), y1);
            r0 = v_add(v_shr(v_add(r0, vdescale), shift), y0);
            r1 = v_add(v_shr(v_add(r1, vdescale), shift), y1);

            // saturate and pack
            v_uint16 b, g, r;
            b = v_pack_u(b0, b1);
            g = v_pack_u(g0, g1);
            r = v_pack_u(r0, r1);

            if(bidx)
                swap(r, b);

            if(dcn == 3)
            {
                v_store_interleave(dst, b, g, r);
            }
            else
            {
                v_store_interleave(dst, b, g, r, valpha);
            }
        }
        vx_cleanup();
#endif

        for ( ; i < n; i++, src += 3, dst += dcn)
        {
            ushort Y  = src[0];
            ushort Cr = src[1+yuvOrder];
            ushort Cb = src[2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, shift);

            dst[bidx]   = saturate_cast<ushort>(b);
            dst[1]      = saturate_cast<ushort>(g);
            dst[bidx^2] = saturate_cast<ushort>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    int coeffs[4];
};


///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////

static const int ITUR_BT_601_CY  = 1220542;
static const int ITUR_BT_601_CUB = 2116026;
static const int ITUR_BT_601_CUG = -409993;
static const int ITUR_BT_601_CVG = -852492;
static const int ITUR_BT_601_CVR = 1673527;
static const int ITUR_BT_601_SHIFT = 20;

// Coefficients for RGB to YUV420p conversion
static const int ITUR_BT_601_CRY =  269484; // 0.299055 * (236-16)/256 * (1 << ITUR_BT_601_SHIFT)
static const int ITUR_BT_601_CGY =  528482; // 0.586472 * (236-16)/256 * (1 << ITUR_BT_601_SHIFT)
static const int ITUR_BT_601_CBY =  102760; // 0.114035 * (236-16)/256 * (1 << ITUR_BT_601_SHIFT)
static const int ITUR_BT_601_CRU = -155188; // -0.148 * (1 << (ITUR_BT_601_SHIFT-1))
static const int ITUR_BT_601_CGU = -305135; // -0.291 * (1 << (ITUR_BT_601_SHIFT-1))
static const int ITUR_BT_601_CBU =  460324; //  0.439 * (1 << (ITUR_BT_601_SHIFT-1))
static const int ITUR_BT_601_CGV = -385875; // -0.368 * (1 << (ITUR_BT_601_SHIFT-1))
static const int ITUR_BT_601_CBV =  -74448; // -0.071 * (1 << (ITUR_BT_601_SHIFT-1))

//R = 1.164(Y - 16) + 1.596(V - 128)
//G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
//B = 1.164(Y - 16)                  + 2.018(U - 128)

//R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
//G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
//B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

static inline void uvToRGBuv(const uchar u, const uchar v, int& ruv, int& guv, int& buv)
{
    int uu, vv;
    uu = int(u) - 128;
    vv = int(v) - 128;

    ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * vv;
    guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * vv + ITUR_BT_601_CUG * uu;
    buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * uu;
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline void uvToRGBuv(const v_uint8& u, const v_uint8& v,
                             v_int32 &ruv0, v_int32 &ruv1, v_int32 &ruv2, v_int32 &ruv3,
                             v_int32 &guv0, v_int32 &guv1, v_int32 &guv2, v_int32 &guv3,
                             v_int32 &buv0, v_int32 &buv1, v_int32 &buv2, v_int32 &buv3)
{
    v_uint8 v128 = vx_setall_u8(128);
    v_int8 su = v_reinterpret_as_s8(v_sub_wrap(u, v128));
    v_int8 sv = v_reinterpret_as_s8(v_sub_wrap(v, v128));

    v_int16 uu0, uu1, vv0, vv1;
    v_expand(su, uu0, uu1);
    v_expand(sv, vv0, vv1);
    v_int32 uuu0, uuu1, uuu2, uuu3;
    v_int32 vvv0, vvv1, vvv2, vvv3;
    v_expand(uu0, uuu0, uuu1); v_expand(uu1, uuu2, uuu3);
    v_expand(vv0, vvv0, vvv1); v_expand(vv1, vvv2, vvv3);

    v_int32 vshift = vx_setall_s32(1 << (ITUR_BT_601_SHIFT - 1));
    v_int32 vr = vx_setall_s32(ITUR_BT_601_CVR);
    v_int32 vg = vx_setall_s32(ITUR_BT_601_CVG);
    v_int32 ug = vx_setall_s32(ITUR_BT_601_CUG);
    v_int32 ub = vx_setall_s32(ITUR_BT_601_CUB);

    auto process_uv = [&](v_int32& ruv, v_int32& guv, v_int32& buv, const v_int32& vv, const v_int32& uu) {
        ruv = v_add(vshift, v_mul(vr, vv));
        guv = v_add(v_add(vshift, v_mul(vg, vv)), v_mul(ug, uu));
        buv = v_add(vshift, v_mul(ub, uu));
    };
    process_uv(ruv0, guv0, buv0, vvv0, uuu0);
    process_uv(ruv1, guv1, buv1, vvv1, uuu1);
    process_uv(ruv2, guv2, buv2, vvv2, uuu2);
    process_uv(ruv3, guv3, buv3, vvv3, uuu3);
}
#endif

static inline void yRGBuvToRGBA(const uchar vy, const int ruv, const int guv, const int buv,
                                uchar& r, uchar& g, uchar& b, uchar& a)
{
    int yy = int(vy);
    int y = std::max(0, yy - 16) * ITUR_BT_601_CY;
    r = saturate_cast<uchar>((y + ruv) >> ITUR_BT_601_SHIFT);
    g = saturate_cast<uchar>((y + guv) >> ITUR_BT_601_SHIFT);
    b = saturate_cast<uchar>((y + buv) >> ITUR_BT_601_SHIFT);
    a = uchar(0xff);
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline void yRGBuvToRGBA(const v_uint8& vy,
                                const v_int32 &ruv0, const v_int32 &ruv1, const v_int32 &ruv2, const v_int32 &ruv3,
                                const v_int32 &guv0, const v_int32 &guv1, const v_int32 &guv2, const v_int32 &guv3,
                                const v_int32 &buv0, const v_int32 &buv1, const v_int32 &buv2, const v_int32 &buv3,
                                v_uint8& rr, v_uint8& gg, v_uint8& bb)
{
    v_uint8 v16 = vx_setall_u8(16);
    v_uint8 posY = v_sub(vy, v16);
    v_uint16 yy0, yy1;
    v_expand(posY, yy0, yy1);
    v_int32 yyy0, yyy1, yyy2, yyy3;
    v_expand(v_reinterpret_as_s16(yy0), yyy0, yyy1);
    v_expand(v_reinterpret_as_s16(yy1), yyy2, yyy3);

    v_int32 vcy = vx_setall_s32(ITUR_BT_601_CY);

    v_int32 y0, y1, y2, y3, r0, r1, r2, r3, g0, g1, g2, g3, b0, b1, b2, b3;

    auto process_yrgb = [&](const v_int32& yy, v_int32& y, v_int32& r, v_int32& g, v_int32& b,
                            const v_int32& ruv, const v_int32& guv, const v_int32& buv) {
        y = v_mul(yy, vcy);
        r = v_shr(v_add(y, ruv), ITUR_BT_601_SHIFT);
        g = v_shr(v_add(y, guv), ITUR_BT_601_SHIFT);
        b = v_shr(v_add(y, buv), ITUR_BT_601_SHIFT);
    };
    process_yrgb(yyy0, y0, r0, g0, b0, ruv0, guv0, buv0);
    process_yrgb(yyy1, y1, r1, g1, b1, ruv1, guv1, buv1);
    process_yrgb(yyy2, y2, r2, g2, b2, ruv2, guv2, buv2);
    process_yrgb(yyy3, y3, r3, g3, b3, ruv3, guv3, buv3);

    v_int16 _r0, _r1, _g0, _g1, _b0, _b1;
    _r0 = v_pack(r0, r1);
    _r1 = v_pack(r2, r3);
    _g0 = v_pack(g0, g1);
    _g1 = v_pack(g2, g3);
    _b0 = v_pack(b0, b1);
    _b1 = v_pack(b2, b3);

    rr = v_pack_u(_r0, _r1);
    gg = v_pack_u(_g0, _g1);
    bb = v_pack_u(_b0, _b1);
}
#endif

template<int bIdx, int dcn, bool is420>
static inline void cvtYuv42xxp2RGB8(const uchar u, const uchar v,
                                    const uchar vy01, const uchar vy11, const uchar vy02, const uchar vy12,
                                    uchar* row1, uchar* row2)
{
    int ruv, guv, buv;
    uvToRGBuv(u, v, ruv, guv, buv);

    uchar r00, g00, b00, a00;
    uchar r01, g01, b01, a01;

    yRGBuvToRGBA(vy01, ruv, guv, buv, r00, g00, b00, a00);
    yRGBuvToRGBA(vy11, ruv, guv, buv, r01, g01, b01, a01);

    row1[2-bIdx] = r00;
    row1[1]      = g00;
    row1[bIdx]   = b00;
    if(dcn == 4)
        row1[3] = a00;

    row1[dcn+2-bIdx] = r01;
    row1[dcn+1]      = g01;
    row1[dcn+0+bIdx] = b01;
    if(dcn == 4)
        row1[7] = a01;

    if(is420)
    {
        uchar r10, g10, b10, a10;
        uchar r11, g11, b11, a11;

        yRGBuvToRGBA(vy02, ruv, guv, buv, r10, g10, b10, a10);
        yRGBuvToRGBA(vy12, ruv, guv, buv, r11, g11, b11, a11);

        row2[2-bIdx] = r10;
        row2[1]      = g10;
        row2[bIdx]   = b10;
        if(dcn == 4)
            row2[3] = a10;

        row2[dcn+2-bIdx] = r11;
        row2[dcn+1]      = g11;
        row2[dcn+0+bIdx] = b11;
        if(dcn == 4)
            row2[7] = a11;
    }
}

// bIdx is 0 or 2, uIdx is 0 or 1, dcn is 3 or 4
template<int bIdx, int uIdx, int dcn>
struct YUV420sp2RGB8Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1;
    size_t my1_step;
    const uchar* muv;
    size_t muv_step;

    YUV420sp2RGB8Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width,
                         const uchar* _y1, size_t _y1_step, const uchar* _uv, size_t _uv_step) :
            dst_data(_dst_data), dst_step(_dst_step), width(_dst_width),
            my1(_y1), my1_step(_y1_step), muv(_uv), muv_step(_uv_step) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        const int rangeBegin = range.start * 2;
        const int rangeEnd   = range.end   * 2;

        const uchar* y1 = my1 + rangeBegin * my1_step, *uv = muv + rangeBegin * muv_step / 2;

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += my1_step * 2, uv += muv_step)
        {
            uchar* row1 = dst_data + dst_step * j;
            uchar* row2 = dst_data + dst_step * (j + 1);
            const uchar* y2 = y1 + my1_step;

            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vsize = VTraits<v_uint8>::vlanes();
            v_uint8 a = vx_setall_u8(uchar(0xff));
            for( ; i <= width - 2*vsize;
                 i += 2*vsize, row1 += vsize*dcn*2, row2 += vsize*dcn*2)
            {
                v_uint8 u, v;
                v_load_deinterleave(uv + i, u, v);

                if(uIdx)
                {
                    swap(u, v);
                }

                v_uint8 vy0, vy1, vy2, vy3;
                v_load_deinterleave(y1 + i, vy0, vy1);
                v_load_deinterleave(y2 + i, vy2, vy3);

                v_int32 ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3;
                uvToRGBuv(u, v,
                        ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3);

                v_uint8 r0, r1, r2, r3, g0, g1, g2, g3, b0, b1, b2, b3;

                auto call_yRGBuvToRGBA = [&](const v_uint8& vy, v_uint8& r, v_uint8& g, v_uint8& b) {
                    yRGBuvToRGBA(vy,
                        ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3,
                        r, g, b);
                };
                call_yRGBuvToRGBA(vy0, r0, g0, b0);
                call_yRGBuvToRGBA(vy1, r1, g1, b1);
                call_yRGBuvToRGBA(vy2, r2, g2, b2);
                call_yRGBuvToRGBA(vy3, r3, g3, b3);

                if(bIdx)
                {
                    swap(r0, b0);
                    swap(r1, b1);
                    swap(r2, b2);
                    swap(r3, b3);
                }

                // [r0...], [r1...] => [r0, r1, r0, r1...], [r0, r1, r0, r1...]
                v_uint8 r0_0, r0_1, r1_0, r1_1;
                v_zip(r0, r1, r0_0, r0_1);
                v_zip(r2, r3, r1_0, r1_1);
                v_uint8 g0_0, g0_1, g1_0, g1_1;
                v_zip(g0, g1, g0_0, g0_1);
                v_zip(g2, g3, g1_0, g1_1);
                v_uint8 b0_0, b0_1, b1_0, b1_1;
                v_zip(b0, b1, b0_0, b0_1);
                v_zip(b2, b3, b1_0, b1_1);

                if(dcn == 4)
                {
                    v_store_interleave(row1 + 0*vsize, b0_0, g0_0, r0_0, a);
                    v_store_interleave(row1 + 4*vsize, b0_1, g0_1, r0_1, a);

                    v_store_interleave(row2 + 0*vsize, b1_0, g1_0, r1_0, a);
                    v_store_interleave(row2 + 4*vsize, b1_1, g1_1, r1_1, a);
                }
                else //dcn == 3
                {
                    v_store_interleave(row1 + 0*vsize, b0_0, g0_0, r0_0);
                    v_store_interleave(row1 + 3*vsize, b0_1, g0_1, r0_1);

                    v_store_interleave(row2 + 0*vsize, b1_0, g1_0, r1_0);
                    v_store_interleave(row2 + 3*vsize, b1_1, g1_1, r1_1);
                }
            }
            vx_cleanup();
#endif
            for ( ; i < width; i += 2, row1 += dcn*2, row2 += dcn*2)
            {
                uchar u = uv[i + 0 + uIdx];
                uchar v = uv[i + 1 - uIdx];

                uchar vy01 = y1[i];
                uchar vy11 = y1[i + 1];
                uchar vy02 = y2[i];
                uchar vy12 = y2[i + 1];

                cvtYuv42xxp2RGB8<bIdx, dcn, true>(u, v, vy01, vy11, vy02, vy12, row1, row2);
            }
        }
    }
};

template<int bIdx, int dcn>
struct YUV420p2RGB8Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *mu, *mv;
    size_t stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGB8Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
        : dst_data(_dst_data), dst_step(_dst_step), width(_dst_width), my1(_y1), mu(_u), mv(_v), stride(_stride), ustepIdx(_ustepIdx), vstepIdx(_vstepIdx) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        const int rangeBegin = range.start * 2;
        const int rangeEnd = range.end * 2;

        int uvsteps[2] = {width/2, static_cast<int>(stride) - width/2};
        int usIdx = ustepIdx, vsIdx = vstepIdx;

        const uchar* y1 = my1 + rangeBegin * stride;
        const uchar* u1 = mu + (range.start / 2) * stride;
        const uchar* v1 = mv + (range.start / 2) * stride;

        if(range.start % 2 == 1)
        {
            u1 += uvsteps[(usIdx++) & 1];
            v1 += uvsteps[(vsIdx++) & 1];
        }

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1])
        {
            uchar* row1 = dst_data + dst_step * j;
            uchar* row2 = dst_data + dst_step * (j + 1);
            const uchar* y2 = y1 + stride;
            int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vsize = VTraits<v_uint8>::vlanes();
            v_uint8 a = vx_setall_u8(uchar(0xff));
            for( ; i <= width/2 - vsize;
                 i += vsize, row1 += vsize*dcn*2, row2 += vsize*dcn*2)
            {
                v_uint8 u, v;
                u = vx_load(u1 + i);
                v = vx_load(v1 + i);

                v_uint8 vy0, vy1, vy2, vy3;
                v_load_deinterleave(y1 + 2*i, vy0, vy1);
                v_load_deinterleave(y2 + 2*i, vy2, vy3);

                v_int32 ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3;
                uvToRGBuv(u, v,
                        ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3);

                v_uint8 r0, r1, r2, r3, g0, g1, g2, g3, b0, b1, b2, b3;

                auto call_yRGBuvToRGBA = [&](const v_uint8& vy, v_uint8& r, v_uint8& g, v_uint8& b) {
                    yRGBuvToRGBA(vy,
                        ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3,
                        r, g, b);
                };
                call_yRGBuvToRGBA(vy0, r0, g0, b0);
                call_yRGBuvToRGBA(vy1, r1, g1, b1);
                call_yRGBuvToRGBA(vy2, r2, g2, b2);
                call_yRGBuvToRGBA(vy3, r3, g3, b3);

                if(bIdx)
                {
                    swap(r0, b0);
                    swap(r1, b1);
                    swap(r2, b2);
                    swap(r3, b3);
                }

                // [r0...], [r1...] => [r0, r1, r0, r1...], [r0, r1, r0, r1...]
                v_uint8 r0_0, r0_1, r1_0, r1_1;
                v_zip(r0, r1, r0_0, r0_1);
                v_zip(r2, r3, r1_0, r1_1);
                v_uint8 g0_0, g0_1, g1_0, g1_1;
                v_zip(g0, g1, g0_0, g0_1);
                v_zip(g2, g3, g1_0, g1_1);
                v_uint8 b0_0, b0_1, b1_0, b1_1;
                v_zip(b0, b1, b0_0, b0_1);
                v_zip(b2, b3, b1_0, b1_1);

                if(dcn == 4)
                {
                    v_store_interleave(row1 + 0*vsize, b0_0, g0_0, r0_0, a);
                    v_store_interleave(row1 + 4*vsize, b0_1, g0_1, r0_1, a);

                    v_store_interleave(row2 + 0*vsize, b1_0, g1_0, r1_0, a);
                    v_store_interleave(row2 + 4*vsize, b1_1, g1_1, r1_1, a);
                }
                else //dcn == 3
                {
                    v_store_interleave(row1 + 0*vsize, b0_0, g0_0, r0_0);
                    v_store_interleave(row1 + 3*vsize, b0_1, g0_1, r0_1);

                    v_store_interleave(row2 + 0*vsize, b1_0, g1_0, r1_0);
                    v_store_interleave(row2 + 3*vsize, b1_1, g1_1, r1_1);
                }
            }
            vx_cleanup();
#endif
            for (; i < width / 2; i += 1, row1 += dcn*2, row2 += dcn*2)
            {
                uchar u = u1[i];
                uchar v = v1[i];

                uchar vy01 = y1[2 * i];
                uchar vy11 = y1[2 * i + 1];
                uchar vy02 = y2[2 * i];
                uchar vy12 = y2[2 * i + 1];

                cvtYuv42xxp2RGB8<bIdx, dcn, true>(u, v, vy01, vy11, vy02, vy12, row1, row2);
            }
        }
    }
};


#define MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION (320*240)

template<int bIdx, int uIdx, int dcn>
inline void cvtYUV420sp2RGB(uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                            const uchar* _y1, size_t _y1_step, const uchar* _uv, size_t _uv_step)
{
    YUV420sp2RGB8Invoker<bIdx, uIdx, dcn> converter(dst_data, dst_step, dst_width, _y1, _y1_step, _uv, _uv_step);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

template<int bIdx, int dcn>
inline void cvtYUV420p2RGB(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGB8Invoker<bIdx, dcn> converter(dst_data, dst_step, dst_width, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

///////////////////////////////////// RGB -> YUV420p /////////////////////////////////////

static inline uchar rgbToY42x(uchar r, uchar g, uchar b)
{
    const int shifted16 = (16 << ITUR_BT_601_SHIFT);
    const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
    int yy = ITUR_BT_601_CRY * r + ITUR_BT_601_CGY * g + ITUR_BT_601_CBY * b + halfShift + shifted16;

    return saturate_cast<uchar>(yy >> ITUR_BT_601_SHIFT);
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline v_uint8 rgbToY42x(const v_uint8& r, const v_uint8& g, const v_uint8& b)
{
    const int shifted16 = (16 << ITUR_BT_601_SHIFT);
    const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
    v_uint16 r0, r1, g0, g1, b0, b1;
    v_expand(r, r0, r1);
    v_expand(g, g0, g1);
    v_expand(b, b0, b1);

    v_uint32 rq0, rq1, rq2, rq3, gq0, gq1, gq2, gq3, bq0, bq1, bq2, bq3;
    v_expand(r0, rq0, rq1); v_expand(r1, rq2, rq3);
    v_expand(g0, gq0, gq1); v_expand(g1, gq2, gq3);
    v_expand(b0, bq0, bq1); v_expand(b1, bq2, bq3);

    v_uint32 ry = vx_setall_u32(ITUR_BT_601_CRY), gy = vx_setall_u32(ITUR_BT_601_CGY);
    v_uint32 by = vx_setall_u32(ITUR_BT_601_CBY), shift = vx_setall_u32(halfShift + shifted16);

    v_uint32 y0, y1, y2, y3;
    y0 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(rq0, ry), v_mul(gq0, gy)), v_mul(bq0, by)), shift));
    y1 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(rq1, ry), v_mul(gq1, gy)), v_mul(bq1, by)), shift));
    y2 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(rq2, ry), v_mul(gq2, gy)), v_mul(bq2, by)), shift));
    y3 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(rq3, ry), v_mul(gq3, gy)), v_mul(bq3, by)), shift));

    v_uint16 _y0, _y1;
    _y0 = v_pack(y0, y1);
    _y1 = v_pack(y2, y3);

    return v_pack(_y0, _y1);
}
#endif

static inline void rgbToUV42x(uchar r, uchar g, uchar b, uchar& u, uchar& v)
{
    const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
    const int shifted128 = (128 << ITUR_BT_601_SHIFT);
    int uu = ITUR_BT_601_CRU * r + ITUR_BT_601_CGU * g + ITUR_BT_601_CBU * b + halfShift + shifted128;
    int vv = ITUR_BT_601_CBU * r + ITUR_BT_601_CGV * g + ITUR_BT_601_CBV * b + halfShift + shifted128;

    u = saturate_cast<uchar>(uu >> ITUR_BT_601_SHIFT);
    v = saturate_cast<uchar>(vv >> ITUR_BT_601_SHIFT);
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline void rgbToUV42x(const v_uint8& r0, const v_uint8& r1, const v_uint8& g0, const v_uint8& g1,
                              const v_uint8& b0, const v_uint8& b1, v_uint8& u, v_uint8& v)
{
    // [r0, r1, r2, r3,..] => [r0, 0, r2, 0,..]
    v_int16 vlowByte = vx_setall_s16(0x00ff);
    v_int16 rd0, rd1, gd0, gd1, bd0, bd1;
    rd0 = v_and(v_reinterpret_as_s16(r0), vlowByte);
    rd1 = v_and(v_reinterpret_as_s16(r1), vlowByte);
    gd0 = v_and(v_reinterpret_as_s16(g0), vlowByte);
    gd1 = v_and(v_reinterpret_as_s16(g1), vlowByte);
    bd0 = v_and(v_reinterpret_as_s16(b0), vlowByte);
    bd1 = v_and(v_reinterpret_as_s16(b1), vlowByte);

    v_int32 rq0, rq1, rq2, rq3, gq0, gq1, gq2, gq3, bq0, bq1, bq2, bq3;
    v_expand(rd0, rq0, rq1);
    v_expand(rd1, rq2, rq3);
    v_expand(gd0, gq0, gq1);
    v_expand(gd1, gq2, gq3);
    v_expand(bd0, bq0, bq1);
    v_expand(bd1, bq2, bq3);

    const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
    const int shifted128 = (128 << ITUR_BT_601_SHIFT);
    v_int32 shift = vx_setall_s32(halfShift + shifted128);
    v_int32 ru, gu, bu, gv, bv;
    ru = vx_setall_s32(ITUR_BT_601_CRU);
    gu = vx_setall_s32(ITUR_BT_601_CGU);
    gv = vx_setall_s32(ITUR_BT_601_CGV);
    bu = vx_setall_s32(ITUR_BT_601_CBU);
    bv = vx_setall_s32(ITUR_BT_601_CBV);

    v_int32 uq0, uq1, uq2, uq3, vq0, vq1, vq2, vq3;
    uq0 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(ru, rq0), v_mul(gu, gq0)), v_mul(bu, bq0)), shift));
    vq0 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(bu, rq0), v_mul(gv, gq0)), v_mul(bv, bq0)), shift));
    uq1 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(ru, rq1), v_mul(gu, gq1)), v_mul(bu, bq1)), shift));
    vq1 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(bu, rq1), v_mul(gv, gq1)), v_mul(bv, bq1)), shift));
    uq2 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(ru, rq2), v_mul(gu, gq2)), v_mul(bu, bq2)), shift));
    vq2 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(bu, rq2), v_mul(gv, gq2)), v_mul(bv, bq2)), shift));
    uq3 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(ru, rq3), v_mul(gu, gq3)), v_mul(bu, bq3)), shift));
    vq3 = v_shr<ITUR_BT_601_SHIFT>(v_add(v_add(v_add(v_mul(bu, rq3), v_mul(gv, gq3)), v_mul(bv, bq3)), shift));

    v_int16 u0, u1, v0, v1;
    u0 = v_pack(uq0, uq1);
    u1 = v_pack(uq2, uq3);
    v0 = v_pack(vq0, vq1);
    v1 = v_pack(vq2, vq3);

    u = v_pack_u(u0, u1);
    v = v_pack_u(v0, v1);
}
#endif


struct RGB8toYUV420pInvoker: public ParallelLoopBody
{
    RGB8toYUV420pInvoker(const uchar * _srcData, size_t _srcStep,
                           uchar * _yData, uchar * _uvData, size_t _dstStep,
                           int _srcWidth, int _srcHeight, int _scn, bool _swapBlue, bool _swapUV, bool _interleave)
        : srcData(_srcData), srcStep(_srcStep),
          yData(_yData), uvData(_uvData), dstStep(_dstStep),
          srcWidth(_srcWidth), srcHeight(_srcHeight),
          srcCn(_scn), swapBlue(_swapBlue), swapUV(_swapUV), interleave(_interleave) { }

    void operator()(const Range& rowRange) const CV_OVERRIDE
    {
        const int w = srcWidth;
        const int h = srcHeight;
        const int scn = srcCn;
        const uchar* srcRow = (uchar*)0;
        uchar* yRow = (uchar*)0, *uRow = (uchar*)0, *vRow = (uchar*)0, *uvRow = (uchar*)0;
        for( int sRow = rowRange.start*2; sRow < rowRange.end*2; sRow++)
        {
            srcRow = srcData + srcStep*sRow;
            yRow = yData + dstStep * sRow;
            bool evenRow = (sRow % 2) == 0;
            if(evenRow)
            {
                if (interleave)
                {
                    uvRow = uvData + dstStep*(sRow/2);
                }
                else
                {
                    uRow = uvData + dstStep * (sRow/4) + ((sRow/2) % 2) * (w/2);
                    vRow = uvData + dstStep * ((sRow + h)/4) + (((sRow + h)/2) % 2) * (w/2);
                }
            }
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vsize = VTraits<v_uint8>::vlanes();

            for( ; i <= w/2 - vsize;
                 i += vsize)
            {
                // processing (2*vsize) pixels at once
                v_uint8 b0, b1, g0, g1, r0, r1, a0, a1;
                if(scn == 4)
                {
                    v_load_deinterleave(srcRow + 2*4*i + 0*vsize, b0, g0, r0, a0);
                    v_load_deinterleave(srcRow + 2*4*i + 4*vsize, b1, g1, r1, a1);
                }
                else // scn == 3
                {
                    v_load_deinterleave(srcRow + 2*3*i + 0*vsize, b0, g0, r0);
                    v_load_deinterleave(srcRow + 2*3*i + 3*vsize, b1, g1, r1);
                }

                if(swapBlue)
                {
                    swap(b0, r0); swap(b1, r1);
                }

                v_uint8 y0, y1;

                y0 = rgbToY42x(r0, g0, b0);
                y1 = rgbToY42x(r1, g1, b1);

                v_store(yRow + 2*i + 0*vsize, y0);
                v_store(yRow + 2*i + 1*vsize, y1);

                if(evenRow)
                {
                    v_uint8 u, v;
                    rgbToUV42x(r0, r1, g0, g1, b0, b1, u, v);

                    if(swapUV)
                    {
                        swap(u, v);
                    }

                    if(interleave)
                    {
                        v_store_interleave(uvRow + 2*i, u, v);
                    }
                    else
                    {
                        v_store(uRow + i, u);
                        v_store(vRow + i, v);
                    }
                }
            }
            vx_cleanup();
#endif
            // processing two pixels at once
            for( ; i < w/2; i++)
            {
                uchar b0, g0, r0;
                uchar b1, g1, r1;
                b0 = srcRow[(2*i+0)*scn + 0];
                g0 = srcRow[(2*i+0)*scn + 1];
                r0 = srcRow[(2*i+0)*scn + 2];
                b1 = srcRow[(2*i+1)*scn + 0];
                g1 = srcRow[(2*i+1)*scn + 1];
                r1 = srcRow[(2*i+1)*scn + 2];

                if(swapBlue)
                {
                    swap(b0, r0); swap(b1, r1);
                }

                uchar y0 = rgbToY42x(r0, g0, b0);
                uchar y1 = rgbToY42x(r1, g1, b1);

                yRow[2*i+0] = y0;
                yRow[2*i+1] = y1;

                if(evenRow)
                {
                    uchar uu, vv;
                    rgbToUV42x(r0, g0, b0, uu, vv);
                    if(swapUV)
                    {
                        swap(uu, vv);
                    }

                    if(interleave)
                    {
                        uvRow[2*i+0] = uu;
                        uvRow[2*i+1] = vv;
                    }
                    else
                    {
                        uRow[i] = uu;
                        vRow[i] = vv;
                    }
                }
            }
        }
    }

    const uchar * srcData;
    size_t srcStep;
    uchar *yData, *uvData;
    size_t dstStep;
    int srcWidth;
    int srcHeight;
    const int srcCn;
    bool swapBlue;
    bool swapUV;
    bool interleave;
};


///////////////////////////////////// YUV422 -> RGB /////////////////////////////////////

// bIdx is 0 or 2; [uIdx, yIdx] is [0, 0], [0, 1], [1, 0]; dcn is 3 or 4
template<int bIdx, int uIdx, int yIdx, int dcn>
struct YUV422toRGB8Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    const uchar * src_data;
    size_t src_step;
    int width;

    YUV422toRGB8Invoker(uchar * _dst_data, size_t _dst_step,
                        const uchar * _src_data, size_t _src_step,
                        int _width)
        : dst_data(_dst_data), dst_step(_dst_step), src_data(_src_data), src_step(_src_step), width(_width) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        // [yIdx, uIdx] | [uidx, vidx]:
        //     0, 0     |     1, 3
        //     0, 1     |     3, 1
        //     1, 0     |     0, 2
        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src_data + rangeBegin * src_step;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += src_step)
        {
            uchar* row = dst_data + dst_step * j;
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vsize = VTraits<v_uint8>::vlanes();
            v_uint8 a = vx_setall_u8(uchar(0xff));
            for(; i <= 2*width - 4*vsize;
                i += 4*vsize, row += vsize*dcn*2)
            {
                v_uint8 u, v, vy0, vy1;
                if(yIdx == 1) // UYVY
                {
                    v_load_deinterleave(yuv_src + i, u, vy0, v, vy1);
                }
                else // YUYV or YVYU
                {
                    v_load_deinterleave(yuv_src + i, vy0, u, vy1, v);
                    if(uIdx == 1) // YVYU
                    {
                        swap(u, v);
                    }
                }

                v_int32 ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3;
                uvToRGBuv(u, v,
                        ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3);

                v_uint8 r0, r1, g0, g1, b0, b1;


                yRGBuvToRGBA(vy0,
                        ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3,
                        r0, g0, b0);
                yRGBuvToRGBA(vy1,
                        ruv0, ruv1, ruv2, ruv3,
                        guv0, guv1, guv2, guv3,
                        buv0, buv1, buv2, buv3,
                        r1, g1, b1);

                if(bIdx)
                {
                    swap(r0, b0);
                    swap(r1, b1);
                }

                // [r0...], [r1...] => [r0, r1, r0, r1...], [r0, r1, r0, r1...]
                v_uint8 r0_0, r0_1;
                v_zip(r0, r1, r0_0, r0_1);
                v_uint8 g0_0, g0_1;
                v_zip(g0, g1, g0_0, g0_1);
                v_uint8 b0_0, b0_1;
                v_zip(b0, b1, b0_0, b0_1);

                if(dcn == 4)
                {
                    v_store_interleave(row + 0*vsize, b0_0, g0_0, r0_0, a);
                    v_store_interleave(row + 4*vsize, b0_1, g0_1, r0_1, a);
                }
                else //dcn == 3
                {
                    v_store_interleave(row + 0*vsize, b0_0, g0_0, r0_0);
                    v_store_interleave(row + 3*vsize, b0_1, g0_1, r0_1);
                }
            }
            vx_cleanup();
#endif
            for (; i < 2 * width; i += 4, row += dcn*2)
            {
                uchar u = yuv_src[i + uidx];
                uchar v = yuv_src[i + vidx];

                uchar vy0 = yuv_src[i + yIdx];
                uchar vy1 = yuv_src[i + yIdx + 2];

                cvtYuv42xxp2RGB8<bIdx, dcn, false>(u, v, vy0, vy1, 0, 0, row, (uchar*)(0));
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION (320*240)

template<int bIdx, int uIdx, int yIdx, int dcn>
inline void cvtYUV422toRGB(uchar * dst_data, size_t dst_step, const uchar * src_data, size_t src_step,
                           int width, int height)
{
    YUV422toRGB8Invoker<bIdx, uIdx, yIdx, dcn> converter(dst_data, dst_step, src_data, src_step, width);
    if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}


///////////////////////////////////// RGB -> YUV422 /////////////////////////////////////

static const int RGB2YUV422_SHIFT = 14;

// Coefficients based on ITU.BT-601, ISBN 1-878707-09-4 (https://fourcc.org/fccyvrgb.php)
// The conversion coefficients for RGB to YUV422 are based on the ones for RGB to YUV.
// For both Y components, the coefficients are applied as given in the link to each input RGB pixel
// separately. For U and V, they are reduced by half to account for two RGB pixels contributing
// to the same U and V values. In other words, the U and V contributions from the two RGB pixels
// are averaged. The integer versions are obtained by multiplying the float versions by 16384
// and rounding to the nearest integer so that resulting values are in these bounds:
// Y : [16, 235]; Cb, Cr: [16, 240] centered at 128

static const int R2Y422 =  4211; // 0.299077 * (236 - 16) / 256 * 16384
static const int G2Y422 =  8258; // 0.586506 * (236 - 16) / 256 * 16384
static const int B2Y422 =  1606; // 0.114062 * (236 - 16) / 256 * 16384

static const int R2U422 = -1212; // -0.148 * 8192
static const int G2U422 = -2384; // -0.291 * 8192
static const int B2U422 =  3596; //  0.439 * 8192
static const int G2V422 = -3015; // -0.368 * 8192
static const int B2V422 =  -582; // -0.071 * 8192

static inline void RGB2Y(const uchar r, const uchar g, const uchar b, uchar& y)
{
    int y_ = r * R2Y422 + g * G2Y422 + b * B2Y422 + (1 << RGB2YUV422_SHIFT) * 16;
    y = saturate_cast<uchar>(((1 << (RGB2YUV422_SHIFT-1)) + y_) >> RGB2YUV422_SHIFT);
}

static inline void RGB2UV(const uchar r1, const uchar g1, const uchar b1,
                          const uchar r2, const uchar g2, const uchar b2,
                          uchar& u, uchar& v)
{
    int sr = r1 + r2, sg = g1 + g2, sb = b1 + b2;

    int u_ = sr * R2U422 + sg * G2U422 + sb * B2U422 + (1 << (RGB2YUV422_SHIFT-1)) * 256;
    u = saturate_cast<uchar>(((1 << (RGB2YUV422_SHIFT-1)) + u_) >> RGB2YUV422_SHIFT);

    int v_ = sr * B2U422 + sg * G2V422 + sb * B2V422 + (1 << (RGB2YUV422_SHIFT-1)) * 256;
    v = saturate_cast<uchar>(((1 << (RGB2YUV422_SHIFT-1)) + v_) >> RGB2YUV422_SHIFT);
}

template<int yidx, int uidx, int vidx>
static inline void cvtRGB82Yuv422(const uchar r1, const uchar g1, const uchar b1,
                                    const uchar r2, const uchar g2, const uchar b2,
                                    uchar* row)
{
    uchar &u = row[uidx], &v = row[vidx], &y1 = row[yidx], &y2 = row[yidx+2];

    RGB2Y(r1, g1, b1, y1);
    RGB2Y(r2, g2, b2, y2);

    RGB2UV(r1, g1, b1, r2, g2, b2, u, v);
}

// bIdx is 0 or 2; [uIdx, yIdx] is [0, 0], [0, 1], [1, 0]; scn is 3 or 4
template<int bIdx, int uIdx, int yIdx, int scn>
struct RGB8toYUV422Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    const uchar * src_data;
    size_t src_step;
    int width;

    RGB8toYUV422Invoker(uchar * _dst_data, size_t _dst_step,
                        const uchar * _src_data, size_t _src_step,
                        int _width)
        : dst_data(_dst_data), dst_step(_dst_step), src_data(_src_data), src_step(_src_step), width(_width) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        // [yIdx, uIdx] | [uidx, vidx]:
        //     0, 0     |     1, 3
        //     0, 1     |     3, 1
        //     1, 0     |     0, 2
        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const int ridx = (2-bIdx);
        const uchar* rgb_src = src_data + rangeBegin * (src_step);
        const uchar* rgb_src2 = rgb_src+scn;

        for (int j = rangeBegin; j < rangeEnd; j++, rgb_src += src_step, rgb_src2 = rgb_src+scn)
        {
            uchar* row = dst_data + (dst_step) * j;
            int i = 0;
            for (; i < scn * width; i += (scn << 1), row += 4)
            {
                const uchar r1 = rgb_src[i+ridx], g1 = rgb_src[i+1], b1 = rgb_src[i+bIdx];
                const uchar r2 = rgb_src2[i+ridx], g2 = rgb_src2[i+1], b2 = rgb_src2[i+bIdx];

                cvtRGB82Yuv422<yIdx, uidx, vidx>(r1, g1, b1, r2, g2, b2, row);
            }
        }
    }
};

template<int bIdx, int uIdx, int yIdx, int scn>
inline void cvtRGBtoYUV422(uchar * dst_data, size_t dst_step, const uchar * src_data, size_t src_step,
                           int width, int height)
{
    RGB8toYUV422Invoker<bIdx, uIdx, yIdx, scn> converter(dst_data, dst_step, src_data, src_step, width);
    if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}


} // namespace anon


// 8u, 16u, 32f
void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isCbCr)
{
    CV_INSTRUMENT_REGION();

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2YCrCb_i<uchar>(scn, blueIdx, isCbCr));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2YCrCb_i<ushort>(scn, blueIdx, isCbCr));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2YCrCb_f<float>(scn, blueIdx, isCbCr));
}

void cvtYUVtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isCbCr)
{
    CV_INSTRUMENT_REGION();

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_i<uchar>(dcn, blueIdx, isCbCr));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_i<ushort>(dcn, blueIdx, isCbCr));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_f<float>(dcn, blueIdx, isCbCr));
}

// 4:2:0, two planes: Y, UV interleaved
// Y : [16, 235]; Cb, Cr: [16, 240] centered at 128
// 20-bit fixed-point arithmetics
typedef void (*cvt_2plane_yuv_ptr_t)(uchar *      /* dst_data   */,
                                     size_t       /* dst_step   */,
                                     int          /* dst_width  */,
                                     int          /* dst_height */,
                                     const uchar* /* _y1        */,
                                     size_t       /* _y1_step   */,
                                     const uchar* /* _uv        */,
                                     size_t       /* _uv_step   */);

void cvtTwoPlaneYUVtoBGR(const uchar * y_data, size_t y_step, const uchar * uv_data, size_t uv_step,
                         uchar * dst_data, size_t dst_step,
                         int dst_width, int dst_height,
                         int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    int blueIdx = swapBlue ? 2 : 0;

    cvt_2plane_yuv_ptr_t cvtPtr;
    switch(dcn*100 + blueIdx * 10 + uIdx)
    {
    case 300: cvtPtr = cvtYUV420sp2RGB<0, 0, 3>; break;
    case 301: cvtPtr = cvtYUV420sp2RGB<0, 1, 3>; break;
    case 320: cvtPtr = cvtYUV420sp2RGB<2, 0, 3>; break;
    case 321: cvtPtr = cvtYUV420sp2RGB<2, 1, 3>; break;
    case 400: cvtPtr = cvtYUV420sp2RGB<0, 0, 4>; break;
    case 401: cvtPtr = cvtYUV420sp2RGB<0, 1, 4>; break;
    case 420: cvtPtr = cvtYUV420sp2RGB<2, 0, 4>; break;
    case 421: cvtPtr = cvtYUV420sp2RGB<2, 1, 4>; break;
    default: CV_Error( cv::Error::StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };

    cvtPtr(dst_data, dst_step, dst_width, dst_height, y_data, y_step, uv_data, uv_step);
}

// 4:2:0, three planes in one array: Y, U, V
// Y : [16, 235]; Cb, Cr: [16, 240] centered at 128
// 20-bit fixed-point arithmetics
typedef void (*cvt_3plane_yuv_ptr_t)(uchar *      /* dst_data   */,
                                     size_t       /* dst_step   */,
                                     int          /* dst_width  */,
                                     int          /* dst_height */,
                                     size_t       /* _stride    */,
                                     const uchar* /* _y1        */,
                                     const uchar* /* _u         */,
                                     const uchar* /* _v         */,
                                     int          /* ustepIdx   */,
                                     int          /* vstepIdx   */);

void cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                 uchar * dst_data, size_t dst_step,
                                 int dst_width, int dst_height,
                                 int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    const uchar* u = src_data + src_step * static_cast<size_t>(dst_height);
    const uchar* v = src_data + src_step * static_cast<size_t>(dst_height + dst_height/4) + (dst_width/2) * ((dst_height % 4)/2);

    int ustepIdx = 0;
    int vstepIdx = dst_height % 4 == 2 ? 1 : 0;

    if(uIdx == 1) { std::swap(u ,v), std::swap(ustepIdx, vstepIdx); }
    int blueIdx = swapBlue ? 2 : 0;

    cvt_3plane_yuv_ptr_t cvtPtr;
    switch(dcn*10 + blueIdx)
    {
    case 30: cvtPtr = cvtYUV420p2RGB<0, 3>; break;
    case 32: cvtPtr = cvtYUV420p2RGB<2, 3>; break;
    case 40: cvtPtr = cvtYUV420p2RGB<0, 4>; break;
    case 42: cvtPtr = cvtYUV420p2RGB<2, 4>; break;
    default: CV_Error( cv::Error::StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };

    cvtPtr(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx);
}

// 4:2:0, three planes in one array: Y, U, V
// Y : [16, 235]; Cb, Cr: [16, 240] centered at 128
// 20-bit fixed-point arithmetics
void cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step,
                           uchar * dst_data, size_t dst_step,
                           int width, int height,
                           int scn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    uchar * uv_data = dst_data + dst_step * height;

    RGB8toYUV420pInvoker cvt(src_data, src_step, dst_data, uv_data, dst_step, width, height,
                             scn, swapBlue, uIdx == 2, false);

    if( width * height >= 320*240 )
        parallel_for_(Range(0, height/2), cvt);
    else
        cvt(Range(0, height/2));
}

// 4:2:0, two planes: Y, UV interleaved
// Y : [16, 235]; Cb, Cr: [16, 240] centered at 128
// 20-bit fixed-point arithmetics
void cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step,
                         uchar * y_data, uchar * uv_data, size_t dst_step,
                         int width, int height,
                         int scn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    RGB8toYUV420pInvoker cvt(src_data, src_step, y_data, uv_data, dst_step, width, height,
                             scn, swapBlue, uIdx == 2, true);

    if( width * height >= 320*240 )
        parallel_for_(Range(0, height/2), cvt);
    else
        cvt(Range(0, height/2));
}

// 4:2:2 interleaved
// Y : [16, 235]; Cb, Cr: [16, 240] centered at 128
// 20-bit fixed-point arithmetics
typedef void (*cvt_1plane_yuv_ptr_t)(uchar *       /* dst_data */,
                                     size_t        /* dst_step */,
                                     const uchar * /* src_data */,
                                     size_t        /* src_step */,
                                     int           /* width    */,
                                     int           /* height   */);

void cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int dcn, bool swapBlue, int uIdx, int ycn)
{
    CV_INSTRUMENT_REGION();

    cvt_1plane_yuv_ptr_t cvtPtr;
    int blueIdx = swapBlue ? 2 : 0;
    switch(dcn*1000 + blueIdx*100 + uIdx*10 + ycn)
    {
    case 3000: cvtPtr = cvtYUV422toRGB<0,0,0,3>; break;
    case 3001: cvtPtr = cvtYUV422toRGB<0,0,1,3>; break;
    case 3010: cvtPtr = cvtYUV422toRGB<0,1,0,3>; break;
    case 3200: cvtPtr = cvtYUV422toRGB<2,0,0,3>; break;
    case 3201: cvtPtr = cvtYUV422toRGB<2,0,1,3>; break;
    case 3210: cvtPtr = cvtYUV422toRGB<2,1,0,3>; break;
    case 4000: cvtPtr = cvtYUV422toRGB<0,0,0,4>; break;
    case 4001: cvtPtr = cvtYUV422toRGB<0,0,1,4>; break;
    case 4010: cvtPtr = cvtYUV422toRGB<0,1,0,4>; break;
    case 4200: cvtPtr = cvtYUV422toRGB<2,0,0,4>; break;
    case 4201: cvtPtr = cvtYUV422toRGB<2,0,1,4>; break;
    case 4210: cvtPtr = cvtYUV422toRGB<2,1,0,4>; break;
    default: CV_Error( cv::Error::StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };

    cvtPtr(dst_data, dst_step, src_data, src_step, width, height);
}

// 4:2:2 interleaved
// Y : [16, 235]; Cb, Cr: [16, 240] centered at 128
// 14-bit fixed-point arithmetics is used
void cvtOnePlaneBGRtoYUV(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int scn, bool swapBlue, int uIdx, int ycn)
{
    CV_INSTRUMENT_REGION();

    cvt_1plane_yuv_ptr_t cvtPtr;
    int blueIdx = swapBlue ? 2 : 0;
    switch(scn*1000 + blueIdx*100 + uIdx*10 + ycn)
    {
    case 3000: cvtPtr = cvtRGBtoYUV422<0,0,0,3>; break;
    case 3001: cvtPtr = cvtRGBtoYUV422<0,0,1,3>; break;
    case 3010: cvtPtr = cvtRGBtoYUV422<0,1,0,3>; break;
    case 3200: cvtPtr = cvtRGBtoYUV422<2,0,0,3>; break;
    case 3201: cvtPtr = cvtRGBtoYUV422<2,0,1,3>; break;
    case 3210: cvtPtr = cvtRGBtoYUV422<2,1,0,3>; break;
    case 4000: cvtPtr = cvtRGBtoYUV422<0,0,0,4>; break;
    case 4001: cvtPtr = cvtRGBtoYUV422<0,0,1,4>; break;
    case 4010: cvtPtr = cvtRGBtoYUV422<0,1,0,4>; break;
    case 4200: cvtPtr = cvtRGBtoYUV422<2,0,0,4>; break;
    case 4201: cvtPtr = cvtRGBtoYUV422<2,0,1,4>; break;
    case 4210: cvtPtr = cvtRGBtoYUV422<2,1,0,4>; break;
    default: CV_Error( cv::Error::StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };

    cvtPtr(dst_data, dst_step, src_data, src_step, width, height);
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
