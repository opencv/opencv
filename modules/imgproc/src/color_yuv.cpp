// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "color.hpp"

namespace cv
{

//constants for conversion from/to RGB and YUV, YCrCb according to BT.601

//to YCbCr
const float YCBF = 0.564f; // == 1/2/(1-B2YF)
const float YCRF = 0.713f; // == 1/2/(1-R2YF)
const int YCBI = 9241;  // == YCBF*16384
const int YCRI = 11682; // == YCRF*16384
//to YUV
const float B2UF = 0.492f;
const float R2VF = 0.877f;
const int B2UI = 8061;  // == B2UF*16384
const int R2VI = 14369; // == R2VF*16384
//from YUV
const float U2BF = 2.032f;
const float U2GF = -0.395f;
const float V2GF = -0.581f;
const float V2RF = 1.140f;
const int U2BI = 33292;
const int U2GI = -6472;
const int V2GI = -9519;
const int V2RI = 18678;
//from YCrCb
const float CB2BF = 1.773f;
const float CB2GF = -0.344f;
const float CR2GF = -0.714f;
const float CR2RF = 1.403f;
const int CB2BI = 29049;
const int CB2GI = -5636;
const int CR2GI = -11698;
const int CR2RI = 22987;

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

template<typename _Tp> struct RGB2YCrCb_f
{
    typedef _Tp channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, bool _isCrCb) : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF };
        static const float coeffs_yuv[] = { R2YF, G2YF, B2YF, R2VF, B2UF };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
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

#if CV_NEON

template <>
struct RGB2YCrCb_f<float>
{
    typedef float channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, bool _isCrCb) :
        srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF };
        static const float coeffs_yuv[] = { R2YF, G2YF, B2YF, R2VF, B2UF };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if(blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        v_c0 = vdupq_n_f32(coeffs[0]);
        v_c1 = vdupq_n_f32(coeffs[1]);
        v_c2 = vdupq_n_f32(coeffs[2]);
        v_c3 = vdupq_n_f32(coeffs[3]);
        v_c4 = vdupq_n_f32(coeffs[4]);
        v_delta = vdupq_n_f32(ColorChannel<float>::half());
    }

    void operator()(const float * src, float * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const float delta = ColorChannel<float>::half();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        n *= 3;

        if (scn == 3)
            for ( ; i <= n - 12; i += 12, src += 12)
            {
                float32x4x3_t v_src = vld3q_f32(src), v_dst;
                v_dst.val[0] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c0), v_src.val[1], v_c1), v_src.val[2], v_c2);
                v_dst.val[1+yuvOrder] = vmlaq_f32(v_delta, vsubq_f32(v_src.val[bidx^2], v_dst.val[0]), v_c3);
                v_dst.val[2-yuvOrder] = vmlaq_f32(v_delta, vsubq_f32(v_src.val[bidx], v_dst.val[0]), v_c4);

                vst3q_f32(dst + i, v_dst);
            }
        else
            for ( ; i <= n - 12; i += 12, src += 16)
            {
                float32x4x4_t v_src = vld4q_f32(src);
                float32x4x3_t v_dst;
                v_dst.val[0] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c0), v_src.val[1], v_c1), v_src.val[2], v_c2);
                v_dst.val[1+yuvOrder] = vmlaq_f32(v_delta, vsubq_f32(v_src.val[bidx^2], v_dst.val[0]), v_c3);
                v_dst.val[2-yuvOrder] = vmlaq_f32(v_delta, vsubq_f32(v_src.val[bidx], v_dst.val[0]), v_c4);

                vst3q_f32(dst + i, v_dst);
            }

        for ( ; i < n; i += 3, src += scn)
        {
            float Y = src[0]*C0 + src[1]*C1 + src[2]*C2;
            float Cr = (src[bidx^2] - Y)*C3 + delta;
            float Cb = (src[bidx] - Y)*C4 + delta;
            dst[i] = Y; dst[i+1+yuvOrder] = Cr; dst[i+2-yuvOrder] = Cb;
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    float coeffs[5];
    float32x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_delta;
};

#elif CV_SSE2

template <>
struct RGB2YCrCb_f<float>
{
    typedef float channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, bool _isCrCb) :
        srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF };
        static const float coeffs_yuv[] = { R2YF, G2YF, B2YF, R2VF, B2UF };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if (blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        v_c0 = _mm_set1_ps(coeffs[0]);
        v_c1 = _mm_set1_ps(coeffs[1]);
        v_c2 = _mm_set1_ps(coeffs[2]);
        v_c3 = _mm_set1_ps(coeffs[3]);
        v_c4 = _mm_set1_ps(coeffs[4]);
        v_delta = _mm_set1_ps(ColorChannel<float>::half());

        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
    }

    void process(__m128 v_r, __m128 v_g, __m128 v_b,
                 __m128 & v_y, __m128 & v_cr, __m128 & v_cb) const
    {
        v_y = _mm_mul_ps(v_r, v_c0);
        v_y = _mm_add_ps(v_y, _mm_mul_ps(v_g, v_c1));
        v_y = _mm_add_ps(v_y, _mm_mul_ps(v_b, v_c2));

        v_cr = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(blueIdx == 0 ? v_b : v_r, v_y), v_c3), v_delta);
        v_cb = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(blueIdx == 2 ? v_b : v_r, v_y), v_c4), v_delta);
    }

    void operator()(const float * src, float * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const float delta = ColorChannel<float>::half();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        n *= 3;

        if (haveSIMD)
        {
            for ( ; i <= n - 24; i += 24, src += 8 * scn)
            {
                __m128 v_r0 = _mm_loadu_ps(src);
                __m128 v_r1 = _mm_loadu_ps(src + 4);
                __m128 v_g0 = _mm_loadu_ps(src + 8);
                __m128 v_g1 = _mm_loadu_ps(src + 12);
                __m128 v_b0 = _mm_loadu_ps(src + 16);
                __m128 v_b1 = _mm_loadu_ps(src + 20);

                if (scn == 4)
                {
                    __m128 v_a0 = _mm_loadu_ps(src + 24);
                    __m128 v_a1 = _mm_loadu_ps(src + 28);
                    _mm_deinterleave_ps(v_r0, v_r1, v_g0, v_g1,
                                        v_b0, v_b1, v_a0, v_a1);
                }
                else
                    _mm_deinterleave_ps(v_r0, v_r1, v_g0, v_g1, v_b0, v_b1);

                __m128 v_y0, v_cr0, v_cb0;
                process(v_r0, v_g0, v_b0,
                        v_y0, v_cr0, v_cb0);

                __m128 v_y1, v_cr1, v_cb1;
                process(v_r1, v_g1, v_b1,
                        v_y1, v_cr1, v_cb1);

                if(isCrCb)
                    _mm_interleave_ps(v_y0, v_y1, v_cr0, v_cr1, v_cb0, v_cb1);
                else //YUV
                {
                    _mm_interleave_ps(v_y0, v_y1, v_cb0, v_cb1, v_cr0, v_cr1);
                }

                _mm_storeu_ps(dst + i, v_y0);
                _mm_storeu_ps(dst + i + 4, v_y1);
                _mm_storeu_ps(dst + i + 8  + yuvOrder*8, v_cr0);
                _mm_storeu_ps(dst + i + 12 + yuvOrder*8, v_cr1);
                _mm_storeu_ps(dst + i + 16 - yuvOrder*8, v_cb0);
                _mm_storeu_ps(dst + i + 20 - yuvOrder*8, v_cb1);
            }
        }

        for ( ; i < n; i += 3, src += scn)
        {
            float Y = src[0]*C0 + src[1]*C1 + src[2]*C2;
            float Cr = (src[bidx^2] - Y)*C3 + delta;
            float Cb = (src[bidx] - Y)*C4 + delta;
            dst[i] = Y; dst[i+1+yuvOrder] = Cr; dst[i+2-yuvOrder] = Cb;
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    float coeffs[5];
    __m128 v_c0, v_c1, v_c2, v_c3, v_c4, v_delta;
    bool haveSIMD;
};

#endif

template<typename _Tp> struct RGB2YCrCb_i
{
    typedef _Tp channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<_Tp>::half()*(1 << yuv_shift);
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<_Tp>(Y);
            dst[i+1+yuvOrder] = saturate_cast<_Tp>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<_Tp>(Cb);
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    int coeffs[5];
};

#if CV_NEON

template <>
struct RGB2YCrCb_i<uchar>
{
    typedef uchar channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if (blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        v_c0 = vdup_n_s16(coeffs[0]);
        v_c1 = vdup_n_s16(coeffs[1]);
        v_c2 = vdup_n_s16(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_c4 = vdupq_n_s32(coeffs[4]);
        v_delta = vdupq_n_s32(ColorChannel<uchar>::half()*(1 << yuv_shift));
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
    }

    void operator()(const uchar * src, uchar * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<uchar>::half()*(1 << yuv_shift);
        n *= 3;

        for ( ; i <= n - 24; i += 24, src += scn * 8)
        {
            uint8x8x3_t v_dst;
            int16x8x3_t v_src16;

            if (scn == 3)
            {
                uint8x8x3_t v_src = vld3_u8(src);
                v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
                v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
                v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));
            }
            else
            {
                uint8x8x4_t v_src = vld4_u8(src);
                v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
                v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
                v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));
            }

            int16x4x3_t v_src0;
            v_src0.val[0] = vget_low_s16(v_src16.val[0]);
            v_src0.val[1] = vget_low_s16(v_src16.val[1]);
            v_src0.val[2] = vget_low_s16(v_src16.val[2]);

            int32x4_t v_Y0 = vmlal_s16(vmlal_s16(vmull_s16(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y0 = vshrq_n_s32(vaddq_s32(v_Y0, v_delta2), yuv_shift);
            int32x4_t v_Cr0 = vmlaq_s32(v_delta, vsubq_s32(vmovl_s16(v_src0.val[bidx^2]), v_Y0), v_c3);
            v_Cr0 = vshrq_n_s32(vaddq_s32(v_Cr0, v_delta2), yuv_shift);
            int32x4_t v_Cb0 = vmlaq_s32(v_delta, vsubq_s32(vmovl_s16(v_src0.val[bidx]), v_Y0), v_c4);
            v_Cb0 = vshrq_n_s32(vaddq_s32(v_Cb0, v_delta2), yuv_shift);

            v_src0.val[0] = vget_high_s16(v_src16.val[0]);
            v_src0.val[1] = vget_high_s16(v_src16.val[1]);
            v_src0.val[2] = vget_high_s16(v_src16.val[2]);

            int32x4_t v_Y1 = vmlal_s16(vmlal_s16(vmull_s16(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y1 = vshrq_n_s32(vaddq_s32(v_Y1, v_delta2), yuv_shift);
            int32x4_t v_Cr1 = vmlaq_s32(v_delta, vsubq_s32(vmovl_s16(v_src0.val[bidx^2]), v_Y1), v_c3);
            v_Cr1 = vshrq_n_s32(vaddq_s32(v_Cr1, v_delta2), yuv_shift);
            int32x4_t v_Cb1 = vmlaq_s32(v_delta, vsubq_s32(vmovl_s16(v_src0.val[bidx]), v_Y1), v_c4);
            v_Cb1 = vshrq_n_s32(vaddq_s32(v_Cb1, v_delta2), yuv_shift);

            v_dst.val[0] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_Y0), vqmovn_s32(v_Y1)));
            v_dst.val[1+yuvOrder] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_Cr0), vqmovn_s32(v_Cr1)));
            v_dst.val[2-yuvOrder] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_Cb0), vqmovn_s32(v_Cb1)));

            vst3_u8(dst + i, v_dst);
        }

        for ( ; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<uchar>(Y);
            dst[i+1+yuvOrder] = saturate_cast<uchar>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<uchar>(Cb);
        }
    }
    int srccn, blueIdx, coeffs[5];
    bool isCrCb;
    int16x4_t v_c0, v_c1, v_c2;
    int32x4_t v_c3, v_c4, v_delta, v_delta2;
};

template <>
struct RGB2YCrCb_i<ushort>
{
    typedef ushort channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if (blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        v_c0 = vdupq_n_s32(coeffs[0]);
        v_c1 = vdupq_n_s32(coeffs[1]);
        v_c2 = vdupq_n_s32(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_c4 = vdupq_n_s32(coeffs[4]);
        v_delta = vdupq_n_s32(ColorChannel<ushort>::half()*(1 << yuv_shift));
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
    }

    void operator()(const ushort * src, ushort * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<ushort>::half()*(1 << yuv_shift);
        n *= 3;

        for ( ; i <= n - 24; i += 24, src += scn * 8)
        {
            uint16x8x3_t v_src, v_dst;
            int32x4x3_t v_src0;

            if (scn == 3)
                v_src = vld3q_u16(src);
            else
            {
                uint16x8x4_t v_src_ = vld4q_u16(src);
                v_src.val[0] = v_src_.val[0];
                v_src.val[1] = v_src_.val[1];
                v_src.val[2] = v_src_.val[2];
            }

            v_src0.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[0])));
            v_src0.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[1])));
            v_src0.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[2])));

            int32x4_t v_Y0 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y0 = vshrq_n_s32(vaddq_s32(v_Y0, v_delta2), yuv_shift);
            int32x4_t v_Cr0 = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx^2], v_Y0), v_c3);
            v_Cr0 = vshrq_n_s32(vaddq_s32(v_Cr0, v_delta2), yuv_shift);
            int32x4_t v_Cb0 = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx], v_Y0), v_c4);
            v_Cb0 = vshrq_n_s32(vaddq_s32(v_Cb0, v_delta2), yuv_shift);

            v_src0.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[0])));
            v_src0.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[1])));
            v_src0.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[2])));

            int32x4_t v_Y1 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y1 = vshrq_n_s32(vaddq_s32(v_Y1, v_delta2), yuv_shift);
            int32x4_t v_Cr1 = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx^2], v_Y1), v_c3);
            v_Cr1 = vshrq_n_s32(vaddq_s32(v_Cr1, v_delta2), yuv_shift);
            int32x4_t v_Cb1 = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx], v_Y1), v_c4);
            v_Cb1 = vshrq_n_s32(vaddq_s32(v_Cb1, v_delta2), yuv_shift);

            v_dst.val[0] = vcombine_u16(vqmovun_s32(v_Y0), vqmovun_s32(v_Y1));
            v_dst.val[1+yuvOrder] = vcombine_u16(vqmovun_s32(v_Cr0), vqmovun_s32(v_Cr1));
            v_dst.val[2-yuvOrder] = vcombine_u16(vqmovun_s32(v_Cb0), vqmovun_s32(v_Cb1));

            vst3q_u16(dst + i, v_dst);
        }

        for ( ; i <= n - 12; i += 12, src += scn * 4)
        {
            uint16x4x3_t v_dst;
            int32x4x3_t v_src0;

            if (scn == 3)
            {
                uint16x4x3_t v_src = vld3_u16(src);
                v_src0.val[0] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[0]));
                v_src0.val[1] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[1]));
                v_src0.val[2] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[2]));
            }
            else
            {
                uint16x4x4_t v_src = vld4_u16(src);
                v_src0.val[0] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[0]));
                v_src0.val[1] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[1]));
                v_src0.val[2] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[2]));
            }

            int32x4_t v_Y = vmlaq_s32(vmlaq_s32(vmulq_s32(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y = vshrq_n_s32(vaddq_s32(v_Y, v_delta2), yuv_shift);
            int32x4_t v_Cr = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx^2], v_Y), v_c3);
            v_Cr = vshrq_n_s32(vaddq_s32(v_Cr, v_delta2), yuv_shift);
            int32x4_t v_Cb = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx], v_Y), v_c4);
            v_Cb = vshrq_n_s32(vaddq_s32(v_Cb, v_delta2), yuv_shift);

            v_dst.val[0] = vqmovun_s32(v_Y);
            v_dst.val[1+yuvOrder] = vqmovun_s32(v_Cr);
            v_dst.val[2-yuvOrder] = vqmovun_s32(v_Cb);

            vst3_u16(dst + i, v_dst);
        }

        for ( ; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<ushort>(Y);
            dst[i+1+yuvOrder] = saturate_cast<ushort>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<ushort>(Cb);
        }
    }
    int srccn, blueIdx, coeffs[5];
    bool isCrCb;
    int32x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_delta, v_delta2;
};

#elif CV_SSE4_1

template <>
struct RGB2YCrCb_i<uchar>
{
    typedef uchar channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if (blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        short delta = 1 << (yuv_shift - 1);
        v_delta_16 = _mm_set1_epi16(delta);
        v_delta_32 = _mm_set1_epi32(delta);
        short delta2 = 1 + ColorChannel<uchar>::half() * 2;
        v_coeff = _mm_set_epi16(delta2, (short)coeffs[4], delta2, (short)coeffs[3], delta2, (short)coeffs[4], delta2, (short)coeffs[3]);
        if(isCrCb)
            v_shuffle2 = _mm_set_epi8(0x0, 0x0, 0x0, 0x0, 0xf, 0xe, 0xc, 0xb, 0xa, 0x8, 0x7, 0x6, 0x4, 0x3, 0x2, 0x0);
        else //if YUV
            v_shuffle2 = _mm_set_epi8(0x0, 0x0, 0x0, 0x0, 0xe, 0xf, 0xc, 0xa, 0xb, 0x8, 0x6, 0x7, 0x4, 0x2, 0x3, 0x0);
        haveSIMD = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    // 16u x 8
    void process(__m128i* v_rgb, __m128i & v_crgb,
                 __m128i* v_rb, uchar * dst) const
    {
        v_rgb[0] = _mm_madd_epi16(v_rgb[0], v_crgb);
        v_rgb[1] = _mm_madd_epi16(v_rgb[1], v_crgb);
        v_rgb[2] = _mm_madd_epi16(v_rgb[2], v_crgb);
        v_rgb[3] = _mm_madd_epi16(v_rgb[3], v_crgb);
        v_rgb[0] = _mm_hadd_epi32(v_rgb[0], v_rgb[1]);
        v_rgb[2] = _mm_hadd_epi32(v_rgb[2], v_rgb[3]);
        v_rgb[0] = _mm_add_epi32(v_rgb[0], v_delta_32);
        v_rgb[2] = _mm_add_epi32(v_rgb[2], v_delta_32);
        v_rgb[0] = _mm_srai_epi32(v_rgb[0], yuv_shift);
        v_rgb[2] = _mm_srai_epi32(v_rgb[2], yuv_shift);
        __m128i v_y = _mm_packs_epi32(v_rgb[0], v_rgb[2]);

        v_rb[0] = _mm_cvtepu8_epi16(v_rb[0]);
        v_rb[1] = _mm_cvtepu8_epi16(v_rb[1]);
        v_rb[0] = _mm_sub_epi16(v_rb[0], _mm_unpacklo_epi16(v_y, v_y));
        v_rb[1] = _mm_sub_epi16(v_rb[1], _mm_unpackhi_epi16(v_y, v_y));
        v_rgb[0] = _mm_unpacklo_epi16(v_rb[0], v_delta_16);
        v_rgb[1] = _mm_unpackhi_epi16(v_rb[0], v_delta_16);
        v_rgb[2] = _mm_unpacklo_epi16(v_rb[1], v_delta_16);
        v_rgb[3] = _mm_unpackhi_epi16(v_rb[1], v_delta_16);
        v_rgb[0] = _mm_madd_epi16(v_rgb[0], v_coeff);
        v_rgb[1] = _mm_madd_epi16(v_rgb[1], v_coeff);
        v_rgb[2] = _mm_madd_epi16(v_rgb[2], v_coeff);
        v_rgb[3] = _mm_madd_epi16(v_rgb[3], v_coeff);
        v_rgb[0] = _mm_srai_epi32(v_rgb[0], yuv_shift);
        v_rgb[1] = _mm_srai_epi32(v_rgb[1], yuv_shift);
        v_rgb[2] = _mm_srai_epi32(v_rgb[2], yuv_shift);
        v_rgb[3] = _mm_srai_epi32(v_rgb[3], yuv_shift);
        v_rgb[0] = _mm_packs_epi32(v_rgb[0], v_rgb[1]);
        v_rgb[2] = _mm_packs_epi32(v_rgb[2], v_rgb[3]);
        v_rgb[0] = _mm_packus_epi16(v_rgb[0], v_rgb[2]);

        v_rb[0] = _mm_unpacklo_epi16(v_y, v_rgb[0]);
        v_rb[1] = _mm_unpackhi_epi16(v_y, v_rgb[0]);

        v_rb[0] = _mm_shuffle_epi8(v_rb[0], v_shuffle2);
        v_rb[1] = _mm_shuffle_epi8(v_rb[1], v_shuffle2);
        v_rb[1] = _mm_alignr_epi8(v_rb[1], _mm_slli_si128(v_rb[0], 4), 12);

        _mm_storel_epi64((__m128i *)(dst), v_rb[0]);
        _mm_storeu_si128((__m128i *)(dst + 8), v_rb[1]);
    }

    void operator()(const uchar * src, uchar * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<uchar>::half()*(1 << yuv_shift);
        n *= 3;

        if (haveSIMD)
        {
            __m128i v_shuffle;
            __m128i v_crgb;
            if (scn == 4)
            {
                if (bidx == 0)
                {
                    v_shuffle = _mm_set_epi8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc, 0xe, 0x8, 0xa, 0x4, 0x6, 0x0, 0x2);
                }
                else
                {
                    v_shuffle = _mm_set_epi8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xe, 0xc, 0xa, 0x8, 0x6, 0x4, 0x2, 0x0);
                }
                v_crgb = _mm_set_epi16(0, (short)C2, (short)C1, (short)C0, 0, (short)C2, (short)C1, (short)C0);
                for ( ; i <= n - 24; i += 24, src += scn * 8)
                {
                    __m128i v_src[2];
                    v_src[0] = _mm_loadu_si128((__m128i const *)(src));
                    v_src[1] = _mm_loadu_si128((__m128i const *)(src + 16));

                    __m128i v_rgb[4];
                    v_rgb[0] = _mm_cvtepu8_epi16(v_src[0]);
                    v_rgb[1] = _mm_cvtepu8_epi16(_mm_srli_si128(v_src[0], 8));
                    v_rgb[2] = _mm_cvtepu8_epi16(v_src[1]);
                    v_rgb[3] = _mm_cvtepu8_epi16(_mm_srli_si128(v_src[1], 8));

                    __m128i v_rb[2];
                    v_rb[0] = _mm_shuffle_epi8(v_src[0], v_shuffle);
                    v_rb[1] = _mm_shuffle_epi8(v_src[1], v_shuffle);

                    process(v_rgb, v_crgb, v_rb, dst + i);
                }
            }
            else
            {
                if (bidx == 0)
                {
                    v_shuffle = _mm_set_epi8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9, 0xb, 0x6, 0x8, 0x3, 0x5, 0x0, 0x2);
                }
                else
                {
                    v_shuffle = _mm_set_epi8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xb, 0x9, 0x8, 0x6, 0x5, 0x3, 0x2, 0x0);
                }
                v_crgb = _mm_set_epi16(0, (short)C2, (short)C1, (short)C0, (short)C2, (short)C1, (short)C0, 0);
                for ( ; i <= n - 24; i += 24, src += scn * 8)
                {
                    __m128i v_src[2];
                    v_src[0] = _mm_loadu_si128((__m128i const *)(src));
                    v_src[1] = _mm_loadl_epi64((__m128i const *)(src + 16));

                    __m128i v_rgb[4];
                    v_rgb[0] = _mm_cvtepu8_epi16(_mm_slli_si128(v_src[0], 1));
                    v_rgb[1] = _mm_cvtepu8_epi16(_mm_srli_si128(v_src[0], 5));
                    v_rgb[2] = _mm_cvtepu8_epi16(_mm_alignr_epi8(v_src[1], v_src[0], 11));
                    v_rgb[3] = _mm_cvtepu8_epi16(_mm_srli_si128(v_src[1], 1));

                    __m128i v_rb[2];
                    v_rb[0] = _mm_shuffle_epi8(v_src[0], v_shuffle);
                    v_rb[1] = _mm_shuffle_epi8(_mm_alignr_epi8(v_src[1], v_src[0], 12), v_shuffle);

                    process(v_rgb, v_crgb, v_rb, dst + i);
                }
            }
        }

        for ( ; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<uchar>(Y);
            dst[i+1+yuvOrder] = saturate_cast<uchar>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<uchar>(Cb);
        }
    }

    __m128i v_delta_16, v_delta_32;
    __m128i v_coeff;
    __m128i v_shuffle2;
    int srccn, blueIdx, coeffs[5];
    bool isCrCb;
    bool haveSIMD;
};

template <>
struct RGB2YCrCb_i<ushort>
{
    typedef ushort channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if (blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        v_c0 = _mm_set1_epi32(coeffs[0]);
        v_c1 = _mm_set1_epi32(coeffs[1]);
        v_c2 = _mm_set1_epi32(coeffs[2]);
        v_c3 = _mm_set1_epi32(coeffs[3]);
        v_c4 = _mm_set1_epi32(coeffs[4]);
        v_delta2 = _mm_set1_epi32(1 << (yuv_shift - 1));
        v_delta = _mm_set1_epi32(ColorChannel<ushort>::half()*(1 << yuv_shift));
        v_delta = _mm_add_epi32(v_delta, v_delta2);
        v_zero = _mm_setzero_si128();

        haveSIMD = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    // 16u x 8
    void process(__m128i v_r, __m128i v_g, __m128i v_b,
                 __m128i & v_y, __m128i & v_cr, __m128i & v_cb) const
    {
        __m128i v_r_p = _mm_unpacklo_epi16(v_r, v_zero);
        __m128i v_g_p = _mm_unpacklo_epi16(v_g, v_zero);
        __m128i v_b_p = _mm_unpacklo_epi16(v_b, v_zero);

        __m128i v_y0 = _mm_add_epi32(_mm_mullo_epi32(v_r_p, v_c0),
                       _mm_add_epi32(_mm_mullo_epi32(v_g_p, v_c1),
                                     _mm_mullo_epi32(v_b_p, v_c2)));
        v_y0 = _mm_srli_epi32(_mm_add_epi32(v_delta2, v_y0), yuv_shift);

        __m128i v_cr0 = _mm_mullo_epi32(_mm_sub_epi32(blueIdx == 2 ? v_r_p : v_b_p, v_y0), v_c3);
        __m128i v_cb0 = _mm_mullo_epi32(_mm_sub_epi32(blueIdx == 0 ? v_r_p : v_b_p, v_y0), v_c4);
        v_cr0 = _mm_srai_epi32(_mm_add_epi32(v_delta, v_cr0), yuv_shift);
        v_cb0 = _mm_srai_epi32(_mm_add_epi32(v_delta, v_cb0), yuv_shift);

        v_r_p = _mm_unpackhi_epi16(v_r, v_zero);
        v_g_p = _mm_unpackhi_epi16(v_g, v_zero);
        v_b_p = _mm_unpackhi_epi16(v_b, v_zero);

        __m128i v_y1 = _mm_add_epi32(_mm_mullo_epi32(v_r_p, v_c0),
                       _mm_add_epi32(_mm_mullo_epi32(v_g_p, v_c1),
                                     _mm_mullo_epi32(v_b_p, v_c2)));
        v_y1 = _mm_srli_epi32(_mm_add_epi32(v_delta2, v_y1), yuv_shift);

        __m128i v_cr1 = _mm_mullo_epi32(_mm_sub_epi32(blueIdx == 2 ? v_r_p : v_b_p, v_y1), v_c3);
        __m128i v_cb1 = _mm_mullo_epi32(_mm_sub_epi32(blueIdx == 0 ? v_r_p : v_b_p, v_y1), v_c4);
        v_cr1 = _mm_srai_epi32(_mm_add_epi32(v_delta, v_cr1), yuv_shift);
        v_cb1 = _mm_srai_epi32(_mm_add_epi32(v_delta, v_cb1), yuv_shift);

        v_y = _mm_packus_epi32(v_y0, v_y1);
        v_cr = _mm_packus_epi32(v_cr0, v_cr1);
        v_cb = _mm_packus_epi32(v_cb0, v_cb1);
    }

    void operator()(const ushort * src, ushort * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<ushort>::half()*(1 << yuv_shift);
        n *= 3;

        if (haveSIMD)
        {
            for ( ; i <= n - 48; i += 48, src += scn * 16)
            {
                __m128i v_r0 = _mm_loadu_si128((__m128i const *)(src));
                __m128i v_r1 = _mm_loadu_si128((__m128i const *)(src + 8));
                __m128i v_g0 = _mm_loadu_si128((__m128i const *)(src + 16));
                __m128i v_g1 = _mm_loadu_si128((__m128i const *)(src + 24));
                __m128i v_b0 = _mm_loadu_si128((__m128i const *)(src + 32));
                __m128i v_b1 = _mm_loadu_si128((__m128i const *)(src + 40));

                if (scn == 4)
                {
                    __m128i v_a0 = _mm_loadu_si128((__m128i const *)(src + 48));
                    __m128i v_a1 = _mm_loadu_si128((__m128i const *)(src + 56));

                    _mm_deinterleave_epi16(v_r0, v_r1, v_g0, v_g1,
                                           v_b0, v_b1, v_a0, v_a1);
                }
                else
                    _mm_deinterleave_epi16(v_r0, v_r1, v_g0, v_g1, v_b0, v_b1);

                __m128i v_y0 = v_zero, v_cr0 = v_zero, v_cb0 = v_zero;
                process(v_r0, v_g0, v_b0,
                        v_y0, v_cr0, v_cb0);

                __m128i v_y1 = v_zero, v_cr1 = v_zero, v_cb1 = v_zero;
                process(v_r1, v_g1, v_b1,
                        v_y1, v_cr1, v_cb1);

                if(isCrCb)
                    _mm_interleave_epi16(v_y0, v_y1, v_cr0, v_cr1, v_cb0, v_cb1);
                else //YUV
                    _mm_interleave_epi16(v_y0, v_y1, v_cb0, v_cb1, v_cr0, v_cr1);

                _mm_storeu_si128((__m128i *)(dst + i), v_y0);
                _mm_storeu_si128((__m128i *)(dst + i + 8), v_y1);
                _mm_storeu_si128((__m128i *)(dst + i + 16 + yuvOrder*16), v_cr0);
                _mm_storeu_si128((__m128i *)(dst + i + 24 + yuvOrder*16), v_cr1);
                _mm_storeu_si128((__m128i *)(dst + i + 32 - yuvOrder*16), v_cb0);
                _mm_storeu_si128((__m128i *)(dst + i + 40 - yuvOrder*16), v_cb1);
            }
        }

        for ( ; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<ushort>(Y);
            dst[i+1+yuvOrder] = saturate_cast<ushort>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<ushort>(Cb);
        }
    }

    int srccn, blueIdx, coeffs[5];
    bool isCrCb;
    __m128i v_c0, v_c1, v_c2;
    __m128i v_c3, v_c4, v_delta, v_delta2;
    __m128i v_zero;
    bool haveSIMD;
};

#endif // CV_SSE4_1

template<typename _Tp> struct YCrCb2RGB_f
{
    typedef _Tp channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_cbr[] = {CR2RF, CR2GF, CB2GF, CB2BF};
        static const float coeffs_yuv[] = { V2RF,  V2GF,  U2GF,  U2BF};
        memcpy(coeffs, isCrCb ? coeffs_cbr : coeffs_yuv, 4*sizeof(coeffs[0]));
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

#if CV_NEON

template <>
struct YCrCb2RGB_f<float>
{
    typedef float channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_cbr[] = {CR2RF, CR2GF, CB2GF, CB2BF};
        static const float coeffs_yuv[] = { V2RF,  V2GF,  U2GF,  U2BF};
        memcpy(coeffs, isCrCb ? coeffs_cbr : coeffs_yuv, 4*sizeof(coeffs[0]));

        v_c0 = vdupq_n_f32(coeffs[0]);
        v_c1 = vdupq_n_f32(coeffs[1]);
        v_c2 = vdupq_n_f32(coeffs[2]);
        v_c3 = vdupq_n_f32(coeffs[3]);
        v_delta = vdupq_n_f32(ColorChannel<float>::half());
        v_alpha = vdupq_n_f32(ColorChannel<float>::max());
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const float delta = ColorChannel<float>::half(), alpha = ColorChannel<float>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;

        if (dcn == 3)
            for ( ; i <= n - 12; i += 12, dst += 12)
            {
                float32x4x3_t v_src = vld3q_f32(src + i), v_dst;
                float32x4_t v_Y = v_src.val[0], v_Cr = v_src.val[1+yuvOrder], v_Cb = v_src.val[2-yuvOrder];

                v_dst.val[bidx] = vmlaq_f32(v_Y, vsubq_f32(v_Cb, v_delta), v_c3);
                v_dst.val[1] = vaddq_f32(vmlaq_f32(vmulq_f32(vsubq_f32(v_Cb, v_delta), v_c2), vsubq_f32(v_Cr, v_delta), v_c1), v_Y);
                v_dst.val[bidx^2] = vmlaq_f32(v_Y, vsubq_f32(v_Cr, v_delta), v_c0);

                vst3q_f32(dst, v_dst);
            }
        else
            for ( ; i <= n - 12; i += 12, dst += 16)
            {
                float32x4x3_t v_src = vld3q_f32(src + i);
                float32x4x4_t v_dst;
                float32x4_t v_Y = v_src.val[0], v_Cr = v_src.val[1+yuvOrder], v_Cb = v_src.val[2-yuvOrder];

                v_dst.val[bidx] = vmlaq_f32(v_Y, vsubq_f32(v_Cb, v_delta), v_c3);
                v_dst.val[1] = vaddq_f32(vmlaq_f32(vmulq_f32(vsubq_f32(v_Cb, v_delta), v_c2), vsubq_f32(v_Cr, v_delta), v_c1), v_Y);
                v_dst.val[bidx^2] = vmlaq_f32(v_Y, vsubq_f32(v_Cr, v_delta), v_c0);
                v_dst.val[3] = v_alpha;

                vst4q_f32(dst, v_dst);
            }

        for ( ; i < n; i += 3, dst += dcn)
        {
            float Y = src[i], Cr = src[i+1+yuvOrder], Cb = src[i+2-yuvOrder];

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
    float32x4_t v_c0, v_c1, v_c2, v_c3, v_alpha, v_delta;
};

#elif CV_SSE2

template <>
struct YCrCb2RGB_f<float>
{
    typedef float channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_cbr[] = {CR2RF, CR2GF, CB2GF, CB2BF};
        static const float coeffs_yuv[] = { V2RF,  V2GF,  U2GF,  U2BF};
        memcpy(coeffs, isCrCb ? coeffs_cbr : coeffs_yuv, 4*sizeof(coeffs[0]));

        v_c0 = _mm_set1_ps(coeffs[0]);
        v_c1 = _mm_set1_ps(coeffs[1]);
        v_c2 = _mm_set1_ps(coeffs[2]);
        v_c3 = _mm_set1_ps(coeffs[3]);
        v_delta = _mm_set1_ps(ColorChannel<float>::half());
        v_alpha = _mm_set1_ps(ColorChannel<float>::max());

        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
    }

    void process(__m128 v_y, __m128 v_cr, __m128 v_cb,
                 __m128 & v_r, __m128 & v_g, __m128 & v_b) const
    {
        v_cb = _mm_sub_ps(v_cb, v_delta);
        v_cr = _mm_sub_ps(v_cr, v_delta);

        if (!isCrCb)
            std::swap(v_cb, v_cr);

        v_b = _mm_mul_ps(v_cb, v_c3);
        v_g = _mm_add_ps(_mm_mul_ps(v_cb, v_c2), _mm_mul_ps(v_cr, v_c1));
        v_r = _mm_mul_ps(v_cr, v_c0);

        v_b = _mm_add_ps(v_b, v_y);
        v_g = _mm_add_ps(v_g, v_y);
        v_r = _mm_add_ps(v_r, v_y);

        if (blueIdx == 0)
            std::swap(v_b, v_r);
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const float delta = ColorChannel<float>::half(), alpha = ColorChannel<float>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;

        if (haveSIMD)
        {
            for ( ; i <= n - 24; i += 24, dst += 8 * dcn)
            {
                __m128 v_y0 = _mm_loadu_ps(src + i);
                __m128 v_y1 = _mm_loadu_ps(src + i + 4);
                __m128 v_cr0 = _mm_loadu_ps(src + i + 8);
                __m128 v_cr1 = _mm_loadu_ps(src + i + 12);
                __m128 v_cb0 = _mm_loadu_ps(src + i + 16);
                __m128 v_cb1 = _mm_loadu_ps(src + i + 20);

                _mm_deinterleave_ps(v_y0, v_y1, v_cr0, v_cr1, v_cb0, v_cb1);

                __m128 v_r0, v_g0, v_b0;
                process(v_y0, v_cr0, v_cb0,
                        v_r0, v_g0, v_b0);

                __m128 v_r1, v_g1, v_b1;
                process(v_y1, v_cr1, v_cb1,
                        v_r1, v_g1, v_b1);

                __m128 v_a0 = v_alpha, v_a1 = v_alpha;

                if (dcn == 3)
                    _mm_interleave_ps(v_r0, v_r1, v_g0, v_g1, v_b0, v_b1);
                else
                    _mm_interleave_ps(v_r0, v_r1, v_g0, v_g1,
                                      v_b0, v_b1, v_a0, v_a1);

                _mm_storeu_ps(dst, v_r0);
                _mm_storeu_ps(dst + 4, v_r1);
                _mm_storeu_ps(dst + 8, v_g0);
                _mm_storeu_ps(dst + 12, v_g1);
                _mm_storeu_ps(dst + 16, v_b0);
                _mm_storeu_ps(dst + 20, v_b1);

                if (dcn == 4)
                {
                    _mm_storeu_ps(dst + 24, v_a0);
                    _mm_storeu_ps(dst + 28, v_a1);
                }
            }
        }

        for ( ; i < n; i += 3, dst += dcn)
        {
            float Y = src[i], Cr = src[i+1+yuvOrder], Cb = src[i+2-yuvOrder];

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

    __m128 v_c0, v_c1, v_c2, v_c3, v_alpha, v_delta;
    bool haveSIMD;
};

#endif

template<typename _Tp> struct YCrCb2RGB_i
{
    typedef _Tp channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 4*sizeof(coeffs[0]));
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

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

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

#if CV_NEON

template <>
struct YCrCb2RGB_i<uchar>
{
    typedef uchar channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 4*sizeof(coeffs[0]));

        v_c0 = vdupq_n_s32(coeffs[0]);
        v_c1 = vdupq_n_s32(coeffs[1]);
        v_c2 = vdupq_n_s32(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_delta = vdup_n_s16(ColorChannel<uchar>::half());
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const uchar delta = ColorChannel<uchar>::half(), alpha = ColorChannel<uchar>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;

        for ( ; i <= n - 24; i += 24, dst += dcn * 8)
        {
            uint8x8x3_t v_src = vld3_u8(src + i);
            int16x8x3_t v_src16;
            v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
            v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
            v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

            int16x4_t v_Y = vget_low_s16(v_src16.val[0]),
                      v_Cr = vget_low_s16(v_src16.val[1+yuvOrder]),
                      v_Cb = vget_low_s16(v_src16.val[2-yuvOrder]);

            int32x4_t v_b0 = vmulq_s32(v_c3, vsubl_s16(v_Cb, v_delta));
            v_b0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_b0, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g0 = vmlaq_s32(vmulq_s32(vsubl_s16(v_Cr, v_delta), v_c1), vsubl_s16(v_Cb, v_delta), v_c2);
            v_g0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_g0, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r0 = vmulq_s32(v_c0, vsubl_s16(v_Cr, v_delta));
            v_r0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_r0, v_delta2), yuv_shift), v_Y);

            v_Y = vget_high_s16(v_src16.val[0]);
            v_Cr = vget_high_s16(v_src16.val[1+yuvOrder]);
            v_Cb = vget_high_s16(v_src16.val[2-yuvOrder]);

            int32x4_t v_b1 = vmulq_s32(v_c3, vsubl_s16(v_Cb, v_delta));
            v_b1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_b1, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g1 = vmlaq_s32(vmulq_s32(vsubl_s16(v_Cr, v_delta), v_c1), vsubl_s16(v_Cb, v_delta), v_c2);
            v_g1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_g1, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r1 = vmulq_s32(v_c0, vsubl_s16(v_Cr, v_delta));
            v_r1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_r1, v_delta2), yuv_shift), v_Y);

            uint8x8_t v_b = vqmovun_s16(vcombine_s16(vmovn_s32(v_b0), vmovn_s32(v_b1)));
            uint8x8_t v_g = vqmovun_s16(vcombine_s16(vmovn_s32(v_g0), vmovn_s32(v_g1)));
            uint8x8_t v_r = vqmovun_s16(vcombine_s16(vmovn_s32(v_r0), vmovn_s32(v_r1)));

            if (dcn == 3)
            {
                uint8x8x3_t v_dst;
                v_dst.val[bidx] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[bidx^2] = v_r;
                vst3_u8(dst, v_dst);
            }
            else
            {
                uint8x8x4_t v_dst;
                v_dst.val[bidx] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[bidx^2] = v_r;
                v_dst.val[3] = v_alpha;
                vst4_u8(dst, v_dst);
            }
        }

        for ( ; i < n; i += 3, dst += dcn)
        {
            uchar Y = src[i];
            uchar Cr = src[i+1+yuvOrder];
            uchar Cb = src[i+2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

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

    int32x4_t v_c0, v_c1, v_c2, v_c3, v_delta2;
    int16x4_t v_delta;
    uint8x8_t v_alpha;
};

template <>
struct YCrCb2RGB_i<ushort>
{
    typedef ushort channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 4*sizeof(coeffs[0]));

        v_c0 = vdupq_n_s32(coeffs[0]);
        v_c1 = vdupq_n_s32(coeffs[1]);
        v_c2 = vdupq_n_s32(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_delta = vdupq_n_s32(ColorChannel<ushort>::half());
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
        v_alpha = vdupq_n_u16(ColorChannel<ushort>::max());
        v_alpha2 = vget_low_u16(v_alpha);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const ushort delta = ColorChannel<ushort>::half(), alpha = ColorChannel<ushort>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;

        for ( ; i <= n - 24; i += 24, dst += dcn * 8)
        {
            uint16x8x3_t v_src = vld3q_u16(src + i);

            int32x4_t v_Y = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[0]))),
                      v_Cr = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[1+yuvOrder]))),
                      v_Cb = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[2-yuvOrder])));

            int32x4_t v_b0 = vmulq_s32(v_c3, vsubq_s32(v_Cb, v_delta));
            v_b0 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_b0, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g0 = vmlaq_s32(vmulq_s32(vsubq_s32(v_Cr, v_delta), v_c1), vsubq_s32(v_Cb, v_delta), v_c2);
            v_g0 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_g0, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r0 = vmulq_s32(v_c0, vsubq_s32(v_Cr, v_delta));
            v_r0 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_r0, v_delta2), yuv_shift), v_Y);

            v_Y = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[0]))),
            v_Cr = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[1+yuvOrder]))),
            v_Cb = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[2-yuvOrder])));

            int32x4_t v_b1 = vmulq_s32(v_c3, vsubq_s32(v_Cb, v_delta));
            v_b1 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_b1, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g1 = vmlaq_s32(vmulq_s32(vsubq_s32(v_Cr, v_delta), v_c1), vsubq_s32(v_Cb, v_delta), v_c2);
            v_g1 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_g1, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r1 = vmulq_s32(v_c0, vsubq_s32(v_Cr, v_delta));
            v_r1 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_r1, v_delta2), yuv_shift), v_Y);

            uint16x8_t v_b = vcombine_u16(vqmovun_s32(v_b0), vqmovun_s32(v_b1));
            uint16x8_t v_g = vcombine_u16(vqmovun_s32(v_g0), vqmovun_s32(v_g1));
            uint16x8_t v_r = vcombine_u16(vqmovun_s32(v_r0), vqmovun_s32(v_r1));

            if (dcn == 3)
            {
                uint16x8x3_t v_dst;
                v_dst.val[bidx] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[bidx^2] = v_r;
                vst3q_u16(dst, v_dst);
            }
            else
            {
                uint16x8x4_t v_dst;
                v_dst.val[bidx] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[bidx^2] = v_r;
                v_dst.val[3] = v_alpha;
                vst4q_u16(dst, v_dst);
            }
        }

        for ( ; i <= n - 12; i += 12, dst += dcn * 4)
        {
            uint16x4x3_t v_src = vld3_u16(src + i);

            int32x4_t v_Y = vreinterpretq_s32_u32(vmovl_u16(v_src.val[0])),
                      v_Cr = vreinterpretq_s32_u32(vmovl_u16(v_src.val[1+yuvOrder])),
                      v_Cb = vreinterpretq_s32_u32(vmovl_u16(v_src.val[2-yuvOrder]));

            int32x4_t v_b = vmulq_s32(v_c3, vsubq_s32(v_Cb, v_delta));
            v_b = vaddq_s32(vshrq_n_s32(vaddq_s32(v_b, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g = vmlaq_s32(vmulq_s32(vsubq_s32(v_Cr, v_delta), v_c1), vsubq_s32(v_Cb, v_delta), v_c2);
            v_g = vaddq_s32(vshrq_n_s32(vaddq_s32(v_g, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r = vmulq_s32(vsubq_s32(v_Cr, v_delta), v_c0);
            v_r = vaddq_s32(vshrq_n_s32(vaddq_s32(v_r, v_delta2), yuv_shift), v_Y);

            uint16x4_t v_bd = vqmovun_s32(v_b);
            uint16x4_t v_gd = vqmovun_s32(v_g);
            uint16x4_t v_rd = vqmovun_s32(v_r);

            if (dcn == 3)
            {
                uint16x4x3_t v_dst;
                v_dst.val[bidx] = v_bd;
                v_dst.val[1] = v_gd;
                v_dst.val[bidx^2] = v_rd;
                vst3_u16(dst, v_dst);
            }
            else
            {
                uint16x4x4_t v_dst;
                v_dst.val[bidx] = v_bd;
                v_dst.val[1] = v_gd;
                v_dst.val[bidx^2] = v_rd;
                v_dst.val[3] = v_alpha2;
                vst4_u16(dst, v_dst);
            }
        }

        for ( ; i < n; i += 3, dst += dcn)
        {
            ushort Y = src[i];
            ushort Cr = src[i+1+yuvOrder];
            ushort Cb = src[i+2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

            dst[bidx] = saturate_cast<ushort>(b);
            dst[1] = saturate_cast<ushort>(g);
            dst[bidx^2] = saturate_cast<ushort>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    int coeffs[4];

    int32x4_t v_c0, v_c1, v_c2, v_c3, v_delta2, v_delta;
    uint16x8_t v_alpha;
    uint16x4_t v_alpha2;
};

#elif CV_SSE2

template <>
struct YCrCb2RGB_i<uchar>
{
    typedef uchar channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 4*sizeof(coeffs[0]));

        v_c0 = _mm_set1_epi16((short)coeffs[0]);
        v_c1 = _mm_set1_epi16((short)coeffs[1]);
        v_c2 = _mm_set1_epi16((short)coeffs[2]);
        v_c3 = _mm_set1_epi16((short)coeffs[3]);
        v_delta = _mm_set1_epi16(ColorChannel<uchar>::half());
        v_delta2 = _mm_set1_epi32(1 << (yuv_shift - 1));
        v_zero = _mm_setzero_si128();

        uchar alpha = ColorChannel<uchar>::max();
        v_alpha = _mm_set1_epi8(*(char *)&alpha);

        // when using YUV, one of coefficients is bigger than std::numeric_limits<short>::max(),
        //which is not appropriate for SSE
        useSSE = isCrCb;
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
    }

#if CV_SSE4_1
    // 16s x 8
    void process(__m128i* v_src, __m128i* v_shuffle,
                 __m128i* v_coeffs) const
    {
        __m128i v_ycrcb[3];
        v_ycrcb[0] = _mm_shuffle_epi8(v_src[0], v_shuffle[0]);
        v_ycrcb[1] = _mm_shuffle_epi8(_mm_alignr_epi8(v_src[1], v_src[0], 8), v_shuffle[0]);
        v_ycrcb[2] = _mm_shuffle_epi8(v_src[1], v_shuffle[0]);

        __m128i v_y[3];
        v_y[1] = _mm_shuffle_epi8(v_src[0], v_shuffle[1]);
        v_y[2] = _mm_srli_si128(_mm_shuffle_epi8(_mm_alignr_epi8(v_src[1], v_src[0], 15), v_shuffle[1]), 1);
        v_y[0] = _mm_unpacklo_epi8(v_y[1], v_zero);
        v_y[1] = _mm_unpackhi_epi8(v_y[1], v_zero);
        v_y[2] = _mm_unpacklo_epi8(v_y[2], v_zero);

        __m128i v_rgb[6];
        v_rgb[0] = _mm_unpacklo_epi8(v_ycrcb[0], v_zero);
        v_rgb[1] = _mm_unpackhi_epi8(v_ycrcb[0], v_zero);
        v_rgb[2] = _mm_unpacklo_epi8(v_ycrcb[1], v_zero);
        v_rgb[3] = _mm_unpackhi_epi8(v_ycrcb[1], v_zero);
        v_rgb[4] = _mm_unpacklo_epi8(v_ycrcb[2], v_zero);
        v_rgb[5] = _mm_unpackhi_epi8(v_ycrcb[2], v_zero);

        v_rgb[0] = _mm_sub_epi16(v_rgb[0], v_delta);
        v_rgb[1] = _mm_sub_epi16(v_rgb[1], v_delta);
        v_rgb[2] = _mm_sub_epi16(v_rgb[2], v_delta);
        v_rgb[3] = _mm_sub_epi16(v_rgb[3], v_delta);
        v_rgb[4] = _mm_sub_epi16(v_rgb[4], v_delta);
        v_rgb[5] = _mm_sub_epi16(v_rgb[5], v_delta);

        v_rgb[0] = _mm_madd_epi16(v_rgb[0], v_coeffs[0]);
        v_rgb[1] = _mm_madd_epi16(v_rgb[1], v_coeffs[1]);
        v_rgb[2] = _mm_madd_epi16(v_rgb[2], v_coeffs[2]);
        v_rgb[3] = _mm_madd_epi16(v_rgb[3], v_coeffs[0]);
        v_rgb[4] = _mm_madd_epi16(v_rgb[4], v_coeffs[1]);
        v_rgb[5] = _mm_madd_epi16(v_rgb[5], v_coeffs[2]);

        v_rgb[0] = _mm_add_epi32(v_rgb[0], v_delta2);
        v_rgb[1] = _mm_add_epi32(v_rgb[1], v_delta2);
        v_rgb[2] = _mm_add_epi32(v_rgb[2], v_delta2);
        v_rgb[3] = _mm_add_epi32(v_rgb[3], v_delta2);
        v_rgb[4] = _mm_add_epi32(v_rgb[4], v_delta2);
        v_rgb[5] = _mm_add_epi32(v_rgb[5], v_delta2);

        v_rgb[0] = _mm_srai_epi32(v_rgb[0], yuv_shift);
        v_rgb[1] = _mm_srai_epi32(v_rgb[1], yuv_shift);
        v_rgb[2] = _mm_srai_epi32(v_rgb[2], yuv_shift);
        v_rgb[3] = _mm_srai_epi32(v_rgb[3], yuv_shift);
        v_rgb[4] = _mm_srai_epi32(v_rgb[4], yuv_shift);
        v_rgb[5] = _mm_srai_epi32(v_rgb[5], yuv_shift);

        v_rgb[0] = _mm_packs_epi32(v_rgb[0], v_rgb[1]);
        v_rgb[2] = _mm_packs_epi32(v_rgb[2], v_rgb[3]);
        v_rgb[4] = _mm_packs_epi32(v_rgb[4], v_rgb[5]);

        v_rgb[0] = _mm_add_epi16(v_rgb[0], v_y[0]);
        v_rgb[2] = _mm_add_epi16(v_rgb[2], v_y[1]);
        v_rgb[4] = _mm_add_epi16(v_rgb[4], v_y[2]);

        v_src[0] = _mm_packus_epi16(v_rgb[0], v_rgb[2]);
        v_src[1] = _mm_packus_epi16(v_rgb[4], v_rgb[4]);
    }
#endif // CV_SSE4_1

    // 16s x 8
    void process(__m128i v_y, __m128i v_cr, __m128i v_cb,
                 __m128i & v_r, __m128i & v_g, __m128i & v_b) const
    {
        v_cr = _mm_sub_epi16(v_cr, v_delta);
        v_cb = _mm_sub_epi16(v_cb, v_delta);

        __m128i v_y_p = _mm_unpacklo_epi16(v_y, v_zero);

        __m128i v_mullo_3 = _mm_mullo_epi16(v_cb, v_c3);
        __m128i v_mullo_2 = _mm_mullo_epi16(v_cb, v_c2);
        __m128i v_mullo_1 = _mm_mullo_epi16(v_cr, v_c1);
        __m128i v_mullo_0 = _mm_mullo_epi16(v_cr, v_c0);

        __m128i v_mulhi_3 = _mm_mulhi_epi16(v_cb, v_c3);
        __m128i v_mulhi_2 = _mm_mulhi_epi16(v_cb, v_c2);
        __m128i v_mulhi_1 = _mm_mulhi_epi16(v_cr, v_c1);
        __m128i v_mulhi_0 = _mm_mulhi_epi16(v_cr, v_c0);

        __m128i v_b0 = _mm_srai_epi32(_mm_add_epi32(_mm_unpacklo_epi16(v_mullo_3, v_mulhi_3), v_delta2), yuv_shift);
        __m128i v_g0 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi16(v_mullo_2, v_mulhi_2),
                                                                  _mm_unpacklo_epi16(v_mullo_1, v_mulhi_1)), v_delta2),
                                      yuv_shift);
        __m128i v_r0 = _mm_srai_epi32(_mm_add_epi32(_mm_unpacklo_epi16(v_mullo_0, v_mulhi_0), v_delta2), yuv_shift);

        v_r0 = _mm_add_epi32(v_r0, v_y_p);
        v_g0 = _mm_add_epi32(v_g0, v_y_p);
        v_b0 = _mm_add_epi32(v_b0, v_y_p);

        v_y_p = _mm_unpackhi_epi16(v_y, v_zero);

        __m128i v_b1 = _mm_srai_epi32(_mm_add_epi32(_mm_unpackhi_epi16(v_mullo_3, v_mulhi_3), v_delta2), yuv_shift);
        __m128i v_g1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(_mm_unpackhi_epi16(v_mullo_2, v_mulhi_2),
                                                                  _mm_unpackhi_epi16(v_mullo_1, v_mulhi_1)), v_delta2),
                                      yuv_shift);
        __m128i v_r1 = _mm_srai_epi32(_mm_add_epi32(_mm_unpackhi_epi16(v_mullo_0, v_mulhi_0), v_delta2), yuv_shift);

        v_r1 = _mm_add_epi32(v_r1, v_y_p);
        v_g1 = _mm_add_epi32(v_g1, v_y_p);
        v_b1 = _mm_add_epi32(v_b1, v_y_p);

        v_r = _mm_packs_epi32(v_r0, v_r1);
        v_g = _mm_packs_epi32(v_g0, v_g1);
        v_b = _mm_packs_epi32(v_b0, v_b1);
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const uchar delta = ColorChannel<uchar>::half(), alpha = ColorChannel<uchar>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;

#if CV_SSE4_1
        if (checkHardwareSupport(CV_CPU_SSE4_1) && useSSE)
        {
            __m128i v_shuffle[2];
            v_shuffle[0] = _mm_set_epi8(0x8, 0x7, 0x7, 0x6, 0x6, 0x5, 0x5, 0x4, 0x4, 0x3, 0x3, 0x2, 0x2, 0x1, 0x1, 0x0);
            v_shuffle[1] = _mm_set_epi8(0xf, 0xc, 0xc, 0xc, 0x9, 0x9, 0x9, 0x6, 0x6, 0x6, 0x3, 0x3, 0x3, 0x0, 0x0, 0x0);
            __m128i v_coeffs[3];
            v_coeffs[0] = _mm_set_epi16((short)C0, 0, 0, (short)C3, (short)C2, (short)C1, (short)C0, 0);
            v_coeffs[1] = _mm_set_epi16((short)C2, (short)C1, (short)C0, 0, 0, (short)C3, (short)C2, (short)C1);
            v_coeffs[2] = _mm_set_epi16(0, (short)C3, (short)C2, (short)C1, (short)C0, 0, 0, (short)C3);

            if (dcn == 3)
            {
                if (bidx == 0)
                {
                    __m128i v_shuffle_dst = _mm_set_epi8(0xf, 0xc, 0xd, 0xe, 0x9, 0xa, 0xb, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2);
                    for ( ; i <= n - 24; i += 24, dst += dcn * 8)
                    {
                        __m128i v_src[2];
                        v_src[0] = _mm_loadu_si128((__m128i const *)(src + i));
                        v_src[1] = _mm_loadl_epi64((__m128i const *)(src + i + 16));

                        process(v_src, v_shuffle, v_coeffs);

                        __m128i v_dst[2];
                        v_dst[0] = _mm_shuffle_epi8(v_src[0], v_shuffle_dst);
                        v_dst[1] = _mm_shuffle_epi8(_mm_alignr_epi8(v_src[1], v_src[0], 15), v_shuffle_dst);

                        _mm_storeu_si128((__m128i *)(dst), _mm_alignr_epi8(v_dst[1], _mm_slli_si128(v_dst[0], 1), 1));
                        _mm_storel_epi64((__m128i *)(dst + 16), _mm_srli_si128(v_dst[1], 1));
                    }
                }
                else
                {
                    for ( ; i <= n - 24; i += 24, dst += dcn * 8)
                    {
                        __m128i v_src[2];
                        v_src[0] = _mm_loadu_si128((__m128i const *)(src + i));
                        v_src[1] = _mm_loadl_epi64((__m128i const *)(src + i + 16));

                        process(v_src, v_shuffle, v_coeffs);

                        _mm_storeu_si128((__m128i *)(dst), v_src[0]);
                        _mm_storel_epi64((__m128i *)(dst + 16), v_src[1]);
                    }
                }
            }
            else
            {
                if (bidx == 0)
                {
                    __m128i v_shuffle_dst = _mm_set_epi8(0x0, 0xa, 0xb, 0xc, 0x0, 0x7, 0x8, 0x9, 0x0, 0x4, 0x5, 0x6, 0x0, 0x1, 0x2, 0x3);

                    for ( ; i <= n - 24; i += 24, dst += dcn * 8)
                    {
                        __m128i v_src[2];
                        v_src[0] = _mm_loadu_si128((__m128i const *)(src + i));
                        v_src[1] = _mm_loadl_epi64((__m128i const *)(src + i + 16));

                        process(v_src, v_shuffle, v_coeffs);

                        _mm_storeu_si128((__m128i *)(dst), _mm_shuffle_epi8(_mm_alignr_epi8(v_src[0], v_alpha, 15), v_shuffle_dst));
                        _mm_storeu_si128((__m128i *)(dst + 16), _mm_shuffle_epi8(_mm_alignr_epi8(_mm_alignr_epi8(v_src[1], v_src[0], 12), v_alpha, 15), v_shuffle_dst));
                    }
                }
                else
                {
                    __m128i v_shuffle_dst = _mm_set_epi8(0x0, 0xc, 0xb, 0xa, 0x0, 0x9, 0x8, 0x7, 0x0, 0x6, 0x5, 0x4, 0x0, 0x3, 0x2, 0x1);

                    for ( ; i <= n - 24; i += 24, dst += dcn * 8)
                    {
                        __m128i v_src[2];
                        v_src[0] = _mm_loadu_si128((__m128i const *)(src + i));
                        v_src[1] = _mm_loadl_epi64((__m128i const *)(src + i + 16));

                        process(v_src, v_shuffle, v_coeffs);

                        _mm_storeu_si128((__m128i *)(dst), _mm_shuffle_epi8(_mm_alignr_epi8(v_src[0], v_alpha, 15), v_shuffle_dst));
                        _mm_storeu_si128((__m128i *)(dst + 16), _mm_shuffle_epi8(_mm_alignr_epi8(_mm_alignr_epi8(v_src[1], v_src[0], 12), v_alpha, 15), v_shuffle_dst));
                    }
                }
            }
        }
        else
#endif // CV_SSE4_1
        if (haveSIMD && useSSE)
        {
            for ( ; i <= n - 96; i += 96, dst += dcn * 32)
            {
                __m128i v_y0 = _mm_loadu_si128((__m128i const *)(src + i));
                __m128i v_y1 = _mm_loadu_si128((__m128i const *)(src + i + 16));
                __m128i v_cr0 = _mm_loadu_si128((__m128i const *)(src + i + 32));
                __m128i v_cr1 = _mm_loadu_si128((__m128i const *)(src + i + 48));
                __m128i v_cb0 = _mm_loadu_si128((__m128i const *)(src + i + 64));
                __m128i v_cb1 = _mm_loadu_si128((__m128i const *)(src + i + 80));

                _mm_deinterleave_epi8(v_y0, v_y1, v_cr0, v_cr1, v_cb0, v_cb1);

                __m128i v_r_0 = v_zero, v_g_0 = v_zero, v_b_0 = v_zero;
                process(_mm_unpacklo_epi8(v_y0, v_zero),
                        _mm_unpacklo_epi8(v_cr0, v_zero),
                        _mm_unpacklo_epi8(v_cb0, v_zero),
                        v_r_0, v_g_0, v_b_0);

                __m128i v_r_1 = v_zero, v_g_1 = v_zero, v_b_1 = v_zero;
                process(_mm_unpackhi_epi8(v_y0, v_zero),
                        _mm_unpackhi_epi8(v_cr0, v_zero),
                        _mm_unpackhi_epi8(v_cb0, v_zero),
                        v_r_1, v_g_1, v_b_1);

                __m128i v_r0 = _mm_packus_epi16(v_r_0, v_r_1);
                __m128i v_g0 = _mm_packus_epi16(v_g_0, v_g_1);
                __m128i v_b0 = _mm_packus_epi16(v_b_0, v_b_1);

                process(_mm_unpacklo_epi8(v_y1, v_zero),
                        _mm_unpacklo_epi8(v_cr1, v_zero),
                        _mm_unpacklo_epi8(v_cb1, v_zero),
                        v_r_0, v_g_0, v_b_0);

                process(_mm_unpackhi_epi8(v_y1, v_zero),
                        _mm_unpackhi_epi8(v_cr1, v_zero),
                        _mm_unpackhi_epi8(v_cb1, v_zero),
                        v_r_1, v_g_1, v_b_1);

                __m128i v_r1 = _mm_packus_epi16(v_r_0, v_r_1);
                __m128i v_g1 = _mm_packus_epi16(v_g_0, v_g_1);
                __m128i v_b1 = _mm_packus_epi16(v_b_0, v_b_1);

                if (bidx == 0)
                {
                    std::swap(v_r0, v_b0);
                    std::swap(v_r1, v_b1);
                }

                __m128i v_a0 = v_alpha, v_a1 = v_alpha;

                if (dcn == 3)
                    _mm_interleave_epi8(v_r0, v_r1, v_g0, v_g1, v_b0, v_b1);
                else
                    _mm_interleave_epi8(v_r0, v_r1, v_g0, v_g1,
                                        v_b0, v_b1, v_a0, v_a1);

                _mm_storeu_si128((__m128i *)(dst), v_r0);
                _mm_storeu_si128((__m128i *)(dst + 16), v_r1);
                _mm_storeu_si128((__m128i *)(dst + 32), v_g0);
                _mm_storeu_si128((__m128i *)(dst + 48), v_g1);
                _mm_storeu_si128((__m128i *)(dst + 64), v_b0);
                _mm_storeu_si128((__m128i *)(dst + 80), v_b1);

                if (dcn == 4)
                {
                    _mm_storeu_si128((__m128i *)(dst + 96), v_a0);
                    _mm_storeu_si128((__m128i *)(dst + 112), v_a1);
                }
            }
        }

        for ( ; i < n; i += 3, dst += dcn)
        {
            uchar Y = src[i];
            uchar Cr = src[i+1+yuvOrder];
            uchar Cb = src[i+2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

            dst[bidx] = saturate_cast<uchar>(b);
            dst[1] = saturate_cast<uchar>(g);
            dst[bidx^2] = saturate_cast<uchar>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[4];
    bool isCrCb;
    bool useSSE, haveSIMD;

    __m128i v_c0, v_c1, v_c2, v_c3, v_delta2;
    __m128i v_delta, v_alpha, v_zero;
};

#endif // CV_SSE2


///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////

const int ITUR_BT_601_CY = 1220542;
const int ITUR_BT_601_CUB = 2116026;
const int ITUR_BT_601_CUG = -409993;
const int ITUR_BT_601_CVG = -852492;
const int ITUR_BT_601_CVR = 1673527;
const int ITUR_BT_601_SHIFT = 20;

// Coefficients for RGB to YUV420p conversion
const int ITUR_BT_601_CRY =  269484;
const int ITUR_BT_601_CGY =  528482;
const int ITUR_BT_601_CBY =  102760;
const int ITUR_BT_601_CRU = -155188;
const int ITUR_BT_601_CGU = -305135;
const int ITUR_BT_601_CBU =  460324;
const int ITUR_BT_601_CGV = -385875;
const int ITUR_BT_601_CBV = -74448;

template<int bIdx, int uIdx>
struct YUV420sp2RGB888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *muv;
    size_t stride;

    YUV420sp2RGB888Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _uv)
        : dst_data(_dst_data), dst_step(_dst_step), width(_dst_width), my1(_y1), muv(_uv), stride(_stride) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        //R = 1.164(Y - 16) + 1.596(V - 128)
        //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
        //B = 1.164(Y - 16)                  + 2.018(U - 128)

        //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

        const uchar* y1 = my1 + rangeBegin * stride, *uv = muv + rangeBegin * stride / 2;

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, uv += stride)
        {
            uchar* row1 = dst_data + dst_step * j;
            uchar* row2 = dst_data + dst_step * (j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width; i += 2, row1 += 6, row2 += 6)
            {
                int u = int(uv[i + 0 + uIdx]) - 128;
                int v = int(uv[i + 1 - uIdx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(y1[i + 1]) - 16) * ITUR_BT_601_CY;
                row1[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);

                int y10 = std::max(0, int(y2[i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);

                int y11 = std::max(0, int(y2[i + 1]) - 16) * ITUR_BT_601_CY;
                row2[5-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[4]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[3+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx, int uIdx>
struct YUV420sp2RGBA8888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *muv;
    size_t stride;

    YUV420sp2RGBA8888Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _uv)
        : dst_data(_dst_data), dst_step(_dst_step), width(_dst_width), my1(_y1), muv(_uv), stride(_stride) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        //R = 1.164(Y - 16) + 1.596(V - 128)
        //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
        //B = 1.164(Y - 16)                  + 2.018(U - 128)

        //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

        const uchar* y1 = my1 + rangeBegin * stride, *uv = muv + rangeBegin * stride / 2;

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, uv += stride)
        {
            uchar* row1 = dst_data + dst_step * j;
            uchar* row2 = dst_data + dst_step * (j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width; i += 2, row1 += 8, row2 += 8)
            {
                int u = int(uv[i + 0 + uIdx]) - 128;
                int v = int(uv[i + 1 - uIdx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[3]      = uchar(0xff);

                int y01 = std::max(0, int(y1[i + 1]) - 16) * ITUR_BT_601_CY;
                row1[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[7]      = uchar(0xff);

                int y10 = std::max(0, int(y2[i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[3]      = uchar(0xff);

                int y11 = std::max(0, int(y2[i + 1]) - 16) * ITUR_BT_601_CY;
                row2[6-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[5]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[4+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[7]      = uchar(0xff);
            }
        }
    }
};

template<int bIdx>
struct YUV420p2RGB888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *mu, *mv;
    size_t stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGB888Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
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

            for (int i = 0; i < width / 2; i += 1, row1 += 6, row2 += 6)
            {
                int u = int(u1[i]) - 128;
                int v = int(v1[i]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[2 * i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row1[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);

                int y10 = std::max(0, int(y2[2 * i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);

                int y11 = std::max(0, int(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row2[5-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[4]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[3+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx>
struct YUV420p2RGBA8888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *mu, *mv;
    size_t  stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGBA8888Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
        : dst_data(_dst_data), dst_step(_dst_step), width(_dst_width), my1(_y1), mu(_u), mv(_v), stride(_stride), ustepIdx(_ustepIdx), vstepIdx(_vstepIdx) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

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

            for (int i = 0; i < width / 2; i += 1, row1 += 8, row2 += 8)
            {
                int u = int(u1[i]) - 128;
                int v = int(v1[i]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[2 * i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[3]      = uchar(0xff);

                int y01 = std::max(0, int(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row1[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[7]      = uchar(0xff);

                int y10 = std::max(0, int(y2[2 * i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[3]      = uchar(0xff);

                int y11 = std::max(0, int(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row2[6-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[5]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[4+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[7]      = uchar(0xff);
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION (320*240)

template<int bIdx, int uIdx>
inline void cvtYUV420sp2RGB(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _uv)
{
    YUV420sp2RGB888Invoker<bIdx, uIdx> converter(dst_data, dst_step, dst_width, _stride, _y1,  _uv);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

template<int bIdx, int uIdx>
inline void cvtYUV420sp2RGBA(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _uv)
{
    YUV420sp2RGBA8888Invoker<bIdx, uIdx> converter(dst_data, dst_step, dst_width, _stride, _y1,  _uv);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

template<int bIdx>
inline void cvtYUV420p2RGB(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGB888Invoker<bIdx> converter(dst_data, dst_step, dst_width, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

template<int bIdx>
inline void cvtYUV420p2RGBA(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGBA8888Invoker<bIdx> converter(dst_data, dst_step, dst_width, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

///////////////////////////////////// RGB -> YUV420p /////////////////////////////////////

struct RGB888toYUV420pInvoker: public ParallelLoopBody
{
    RGB888toYUV420pInvoker(const uchar * _src_data, size_t _src_step,
                           uchar * _y_data, uchar * _uv_data, size_t _dst_step,
                           int _src_width, int _src_height, int _scn, bool swapBlue_, bool swapUV_, bool interleaved_)
        : src_data(_src_data), src_step(_src_step),
          y_data(_y_data), uv_data(_uv_data), dst_step(_dst_step),
          src_width(_src_width), src_height(_src_height),
          scn(_scn), swapBlue(swapBlue_), swapUV(swapUV_), interleaved(interleaved_) { }

    void operator()(const Range& rowRange) const CV_OVERRIDE
    {
        const int w = src_width;
        const int h = src_height;
        const int cn = scn;
        for( int i = rowRange.start; i < rowRange.end; i++ )
        {
            const uchar* brow0 = src_data + src_step * (2 * i);
            const uchar* grow0 = brow0 + 1;
            const uchar* rrow0 = brow0 + 2;
            const uchar* brow1 = src_data + src_step * (2 * i + 1);
            const uchar* grow1 = brow1 + 1;
            const uchar* rrow1 = brow1 + 2;
            if (swapBlue)
            {
                std::swap(brow0, rrow0);
                std::swap(brow1, rrow1);
            }

            uchar* y = y_data + dst_step * (2*i);
            uchar* u;
            uchar* v;
            if (interleaved)
            {
                u = uv_data + dst_step * i;
                v = uv_data + dst_step * i + 1;
            }
            else
            {
                u = uv_data + dst_step * (i/2) + (i % 2) * (w/2);
                v = uv_data + dst_step * ((i + h/2)/2) + ((i + h/2) % 2) * (w/2);
            }

            if (swapUV)
            {
                std::swap(u, v);
            }

            for( int j = 0, k = 0; j < w * cn; j += 2 * cn, k++ )
            {
                int r00 = rrow0[j];      int g00 = grow0[j];      int b00 = brow0[j];
                int r01 = rrow0[cn + j]; int g01 = grow0[cn + j]; int b01 = brow0[cn + j];
                int r10 = rrow1[j];      int g10 = grow1[j];      int b10 = brow1[j];
                int r11 = rrow1[cn + j]; int g11 = grow1[cn + j]; int b11 = brow1[cn + j];

                const int shifted16 = (16 << ITUR_BT_601_SHIFT);
                const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
                int y00 = ITUR_BT_601_CRY * r00 + ITUR_BT_601_CGY * g00 + ITUR_BT_601_CBY * b00 + halfShift + shifted16;
                int y01 = ITUR_BT_601_CRY * r01 + ITUR_BT_601_CGY * g01 + ITUR_BT_601_CBY * b01 + halfShift + shifted16;
                int y10 = ITUR_BT_601_CRY * r10 + ITUR_BT_601_CGY * g10 + ITUR_BT_601_CBY * b10 + halfShift + shifted16;
                int y11 = ITUR_BT_601_CRY * r11 + ITUR_BT_601_CGY * g11 + ITUR_BT_601_CBY * b11 + halfShift + shifted16;

                y[2*k + 0]            = saturate_cast<uchar>(y00 >> ITUR_BT_601_SHIFT);
                y[2*k + 1]            = saturate_cast<uchar>(y01 >> ITUR_BT_601_SHIFT);
                y[2*k + dst_step + 0] = saturate_cast<uchar>(y10 >> ITUR_BT_601_SHIFT);
                y[2*k + dst_step + 1] = saturate_cast<uchar>(y11 >> ITUR_BT_601_SHIFT);

                const int shifted128 = (128 << ITUR_BT_601_SHIFT);
                int u00 = ITUR_BT_601_CRU * r00 + ITUR_BT_601_CGU * g00 + ITUR_BT_601_CBU * b00 + halfShift + shifted128;
                int v00 = ITUR_BT_601_CBU * r00 + ITUR_BT_601_CGV * g00 + ITUR_BT_601_CBV * b00 + halfShift + shifted128;

                if (interleaved)
                {
                    u[k*2] = saturate_cast<uchar>(u00 >> ITUR_BT_601_SHIFT);
                    v[k*2] = saturate_cast<uchar>(v00 >> ITUR_BT_601_SHIFT);
                }
                else
                {
                    u[k] = saturate_cast<uchar>(u00 >> ITUR_BT_601_SHIFT);
                    v[k] = saturate_cast<uchar>(v00 >> ITUR_BT_601_SHIFT);
                }
            }
        }
    }

    void convert() const
    {
        if( src_width * src_height >= 320*240 )
            parallel_for_(Range(0, src_height/2), *this);
        else
            operator()(Range(0, src_height/2));
    }

private:
    RGB888toYUV420pInvoker& operator=(const RGB888toYUV420pInvoker&);

    const uchar * src_data;
    size_t src_step;
    uchar *y_data, *uv_data;
    size_t dst_step;
    int src_width;
    int src_height;
    const int scn;
    bool swapBlue;
    bool swapUV;
    bool interleaved;
};


///////////////////////////////////// YUV422 -> RGB /////////////////////////////////////

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGB888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    const uchar * src_data;
    size_t src_step;
    int width;

    YUV422toRGB888Invoker(uchar * _dst_data, size_t _dst_step,
                          const uchar * _src_data, size_t _src_step,
                          int _width)
        : dst_data(_dst_data), dst_step(_dst_step), src_data(_src_data), src_step(_src_step), width(_width) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src_data + rangeBegin * src_step;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += src_step)
        {
            uchar* row = dst_data + dst_step * j;

            for (int i = 0; i < 2 * width; i += 4, row += 6)
            {
                int u = int(yuv_src[i + uidx]) - 128;
                int v = int(yuv_src[i + vidx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
                row[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
                row[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGBA8888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    const uchar * src_data;
    size_t src_step;
    int width;

    YUV422toRGBA8888Invoker(uchar * _dst_data, size_t _dst_step,
                            const uchar * _src_data, size_t _src_step,
                            int _width)
        : dst_data(_dst_data), dst_step(_dst_step), src_data(_src_data), src_step(_src_step), width(_width) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src_data + rangeBegin * src_step;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += src_step)
        {
            uchar* row = dst_data + dst_step * j;

            for (int i = 0; i < 2 * width; i += 4, row += 8)
            {
                int u = int(yuv_src[i + uidx]) - 128;
                int v = int(yuv_src[i + vidx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
                row[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row[3]      = uchar(0xff);

                int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
                row[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row[7]      = uchar(0xff);
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION (320*240)

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGB(uchar * dst_data, size_t dst_step, const uchar * src_data, size_t src_step,
                           int width, int height)
{
    YUV422toRGB888Invoker<bIdx, uIdx, yIdx> converter(dst_data, dst_step, src_data, src_step, width);
    if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGBA(uchar * dst_data, size_t dst_step, const uchar * src_data, size_t src_step,
                           int width, int height)
{
    YUV422toRGBA8888Invoker<bIdx, uIdx, yIdx> converter(dst_data, dst_step, src_data, src_step, width);
    if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}

//
// HAL functions
//

namespace hal
{

// 8u, 16u, 32f
void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isCbCr)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoYUV, cv_hal_cvtBGRtoYUV, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isCbCr);

#if defined(HAVE_IPP)
#if !IPP_DISABLE_RGB_YUV
    CV_IPP_CHECK()
    {
        if (scn == 3 && depth == CV_8U && swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralFunctor((ippiGeneralFunc)ippiRGBToYUV_8u_C3R)))
                return;
        }
        else if (scn == 3 && depth == CV_8U && !swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                         (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 2, 1, 0, depth)))
                return;
        }
        else if (scn == 4 && depth == CV_8U && swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                         (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 0, 1, 2, depth)))
                return;
        }
        else if (scn == 4 && depth == CV_8U && !swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                         (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 2, 1, 0, depth)))
                return;
        }
    }
#endif
#endif

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

    CALL_HAL(cvtYUVtoBGR, cv_hal_cvtYUVtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isCbCr);


#if defined(HAVE_IPP)
#if !IPP_DISABLE_YUV_RGB
    CV_IPP_CHECK()
    {
        if (dcn == 3 && depth == CV_8U && swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R)))
                return;
        }
        else if (dcn == 3 && depth == CV_8U && !swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                                   ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)))
                return;
        }
        else if (dcn == 4 && depth == CV_8U && swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                                   ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)))
                return;
        }
        else if (dcn == 4 && depth == CV_8U && !swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                                   ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)))
                return;
        }
    }
#endif
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_i<uchar>(dcn, blueIdx, isCbCr));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_i<ushort>(dcn, blueIdx, isCbCr));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_f<float>(dcn, blueIdx, isCbCr));
}

void cvtTwoPlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int dst_width, int dst_height,
                         int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtTwoPlaneYUVtoBGR, cv_hal_cvtTwoPlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
    const uchar* uv = src_data + src_step * static_cast<size_t>(dst_height);
    cvtTwoPlaneYUVtoBGR(src_data, uv, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
}

void cvtTwoPlaneYUVtoBGR(const uchar * y_data, const uchar * uv_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int dst_width, int dst_height,
                         int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    // TODO: add hal replacement method
    int blueIdx = swapBlue ? 2 : 0;
    switch(dcn*100 + blueIdx * 10 + uIdx)
    {
    case 300: cvtYUV420sp2RGB<0, 0> (dst_data, dst_step, dst_width, dst_height, src_step, y_data, uv_data); break;
    case 301: cvtYUV420sp2RGB<0, 1> (dst_data, dst_step, dst_width, dst_height, src_step, y_data, uv_data); break;
    case 320: cvtYUV420sp2RGB<2, 0> (dst_data, dst_step, dst_width, dst_height, src_step, y_data, uv_data); break;
    case 321: cvtYUV420sp2RGB<2, 1> (dst_data, dst_step, dst_width, dst_height, src_step, y_data, uv_data); break;
    case 400: cvtYUV420sp2RGBA<0, 0>(dst_data, dst_step, dst_width, dst_height, src_step, y_data, uv_data); break;
    case 401: cvtYUV420sp2RGBA<0, 1>(dst_data, dst_step, dst_width, dst_height, src_step, y_data, uv_data); break;
    case 420: cvtYUV420sp2RGBA<2, 0>(dst_data, dst_step, dst_width, dst_height, src_step, y_data, uv_data); break;
    case 421: cvtYUV420sp2RGBA<2, 1>(dst_data, dst_step, dst_width, dst_height, src_step, y_data, uv_data); break;
    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };
}

void cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                  uchar * dst_data, size_t dst_step,
                                  int dst_width, int dst_height,
                                  int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtThreePlaneYUVtoBGR, cv_hal_cvtThreePlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
    const uchar* u = src_data + src_step * static_cast<size_t>(dst_height);
    const uchar* v = src_data + src_step * static_cast<size_t>(dst_height + dst_height/4) + (dst_width/2) * ((dst_height % 4)/2);

    int ustepIdx = 0;
    int vstepIdx = dst_height % 4 == 2 ? 1 : 0;

    if(uIdx == 1) { std::swap(u ,v), std::swap(ustepIdx, vstepIdx); }
    int blueIdx = swapBlue ? 2 : 0;

    switch(dcn*10 + blueIdx)
    {
    case 30: cvtYUV420p2RGB<0>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx); break;
    case 32: cvtYUV420p2RGB<2>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx); break;
    case 40: cvtYUV420p2RGBA<0>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx); break;
    case 42: cvtYUV420p2RGBA<2>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx); break;
    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };
}

void cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step,
                           uchar * dst_data, size_t dst_step,
                           int width, int height,
                           int scn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoThreePlaneYUV, cv_hal_cvtBGRtoThreePlaneYUV, src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, uIdx);
    uchar * uv_data = dst_data + dst_step * height;
    RGB888toYUV420pInvoker(src_data, src_step, dst_data, uv_data, dst_step, width, height, scn, swapBlue, uIdx == 2, false).convert();
}

void cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step,
                         uchar * y_data, uchar * uv_data, size_t dst_step,
                         int width, int height,
                         int scn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    // TODO: add hal replacement method
    RGB888toYUV420pInvoker(src_data, src_step, y_data, uv_data, dst_step, width, height, scn, swapBlue, uIdx == 2, true).convert();
}

void cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int dcn, bool swapBlue, int uIdx, int ycn)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtOnePlaneYUVtoBGR, cv_hal_cvtOnePlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, uIdx, ycn);
    int blueIdx = swapBlue ? 2 : 0;
    switch(dcn*1000 + blueIdx*100 + uIdx*10 + ycn)
    {
    case 3000: cvtYUV422toRGB<0,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3001: cvtYUV422toRGB<0,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3010: cvtYUV422toRGB<0,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3200: cvtYUV422toRGB<2,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3201: cvtYUV422toRGB<2,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3210: cvtYUV422toRGB<2,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4000: cvtYUV422toRGBA<0,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4001: cvtYUV422toRGBA<0,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4010: cvtYUV422toRGBA<0,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4200: cvtYUV422toRGBA<2,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4201: cvtYUV422toRGBA<2,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4210: cvtYUV422toRGBA<2,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };
}

} // namespace hal

//
// OCL calls
//

#ifdef HAVE_OPENCL

bool oclCvtColorYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("YUV2RGB", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d", dcn, bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2YUV( InputArray _src, OutputArray _dst, int bidx )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("RGB2YUV", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=3 -D bidx=%d", bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtcolorYCrCb2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx)
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("YCrCb2RGB", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d", dcn, bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2YCrCb( InputArray _src, OutputArray _dst, int bidx)
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("RGB2YCrCb", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=3 -D bidx=%d", bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorOnePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx, int yidx )
{
    OclHelper< Set<2>, Set<3, 4>, Set<CV_8U> > h(_src, _dst, dcn);

    bool optimized = _src.offset() % 4 == 0 && _src.step() % 4 == 0;
    if(!h.createKernel("YUV2RGB_422", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D uidx=%d -D yidx=%d%s", dcn, bidx, uidx, yidx,
                       optimized ? " -D USE_OPTIMIZED_LOAD" : "")))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorYUV2Gray_420( InputArray _src, OutputArray _dst )
{
    OclHelper< Set<1>, Set<1>, Set<CV_8U>, FROM_YUV> h(_src, _dst, 1);

    h.src.rowRange(0, _dst.rows()).copyTo(_dst);
    return true;
}

bool oclCvtColorTwoPlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx )
{
    OclHelper< Set<1>, Set<3, 4>, Set<CV_8U>, FROM_YUV > h(_src, _dst, dcn);

    if(!h.createKernel("YUV2RGB_NVx", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D uidx=%d", dcn, bidx, uidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorThreePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx )
{
    OclHelper< Set<1>, Set<3, 4>, Set<CV_8U>, FROM_YUV > h(_src, _dst, dcn);

    if(!h.createKernel("YUV2RGB_YV12_IYUV", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D uidx=%d%s", dcn, bidx, uidx,
                       _src.isContinuous() ? " -D SRC_CONT" : "")))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2ThreePlaneYUV( InputArray _src, OutputArray _dst, int bidx, int uidx )
{
    OclHelper< Set<3, 4>, Set<1>, Set<CV_8U>, TO_YUV > h(_src, _dst, 1);

    if(!h.createKernel("RGB2YUV_YV12_IYUV", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=1 -D bidx=%d -D uidx=%d", bidx, uidx)))
    {
        return false;
    }

    return h.run();
}

#endif

//
// HAL calls
//

void cvtColorBGR2YUV(InputArray _src, OutputArray _dst, bool swapb, bool crcb)
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoYUV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, crcb);
}

void cvtColorYUV2BGR(InputArray _src, OutputArray _dst, int dcn, bool swapb, bool crcb)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtYUVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, crcb);
}

void cvtColorOnePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx, int ycn)
{
    CvtHelper< Set<2>, Set<3, 4>, Set<CV_8U> > h(_src, _dst, dcn);

    hal::cvtOnePlaneYUVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                             dcn, swapb, uidx, ycn);
}

void cvtColorYUV2Gray_ch( InputArray _src, OutputArray _dst, int coi )
{
    CV_Assert( _src.channels() == 2 && _src.depth() == CV_8U );

    extractChannel(_src, _dst, coi);
}

void cvtColorBGR2ThreePlaneYUV( InputArray _src, OutputArray _dst, bool swapb, int uidx)
{
    CvtHelper< Set<3, 4>, Set<1>, Set<CV_8U>, TO_YUV > h(_src, _dst, 1);

    hal::cvtBGRtoThreePlaneYUV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                               h.scn, swapb, uidx);
}

void cvtColorYUV2Gray_420( InputArray _src, OutputArray _dst )
{
    CvtHelper< Set<1>, Set<1>, Set<CV_8U>, FROM_YUV > h(_src, _dst, 1);

#ifdef HAVE_IPP
#if IPP_VERSION_X100 >= 201700
    if (CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C1R_L, h.src.data, (IppSizeL)h.src.step, h.dst.data, (IppSizeL)h.dst.step,
                              ippiSizeL(h.dstSz.width, h.dstSz.height)) >= 0)
        return;
#endif
#endif
    h.src(Range(0, h.dstSz.height), Range::all()).copyTo(h.dst);
}

void cvtColorThreePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<1>, Set<3, 4>, Set<CV_8U>, FROM_YUV> h(_src, _dst, dcn);

    hal::cvtThreePlaneYUVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.dst.cols, h.dst.rows,
                               dcn, swapb, uidx);
}

// http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
// http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples

void cvtColorTwoPlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx )
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<1>, Set<3, 4>, Set<CV_8U>, FROM_YUV> h(_src, _dst, dcn);

    hal::cvtTwoPlaneYUVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.dst.cols, h.dst.rows,
                             dcn, swapb, uidx);
}

void cvtColorTwoPlaneYUV2BGRpair( InputArray _ysrc, InputArray _uvsrc, OutputArray _dst, int dcn, bool swapb, int uidx )
{
    int stype = _ysrc.type();
    int depth = CV_MAT_DEPTH(stype);
    Size ysz = _ysrc.size(), uvs = _uvsrc.size();
    CV_Assert( dcn == 3 || dcn == 4 );
    CV_Assert( depth == CV_8U );
    CV_Assert( ysz.width == uvs.width * 2 && ysz.height == uvs.height * 2 );

    Mat ysrc = _ysrc.getMat(), uvsrc = _uvsrc.getMat();

    _dst.create( ysz, CV_MAKETYPE(depth, dcn));
    Mat dst = _dst.getMat();

    hal::cvtTwoPlaneYUVtoBGR(ysrc.data, uvsrc.data, ysrc.step,
                             dst.data, dst.step, dst.cols, dst.rows,
                             dcn, swapb, uidx);
}

} // namespace cv
