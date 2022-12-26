// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/lib/NN/OpConv_Winograd.fx).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "fast_convolution.hpp"

namespace cv { namespace dnn {

#if CV_NEON || CV_SIMD128 || CV_TRY_AVX2
enum { VEC_ALIGN = 32, DFT_TYPE = CV_32F }; // Memory alignment.

static void
_fx_winograd_accum_f32(const float* inwptr, const float* wptr,
                       float* outbuf, int Cg, int iblock)
 {
#if CV_NEON && CV_NEON_AARCH64
    CV_Assert(_FX_WINO_IBLOCK == 6 && _FX_WINO_KBLOCK == 4);
    if (iblock > 3)
    {
        for (int atom_id = 0; atom_id < _FX_WINO_NATOMS_F32; atom_id++,
                outbuf += _FX_WINO_ATOM_F32)
        {
            float32x4_t s00 = vdupq_n_f32(0.f), s01 = s00, s02 = s00, s03 = s00, s04 = s00, s05 = s00;
            float32x4_t s10 = vdupq_n_f32(0.f), s11 = s00, s12 = s00, s13 = s00, s14 = s00, s15 = s00;
            float32x4_t s20 = vdupq_n_f32(0.f), s21 = s00, s22 = s00, s23 = s00, s24 = s00, s25 = s00;
            float32x4_t s30 = vdupq_n_f32(0.f), s31 = s00, s32 = s00, s33 = s00, s34 = s00, s35 = s00;
            for (int c = 0; c < Cg; c++, inwptr += _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32,
                                         wptr += _FX_WINO_KBLOCK*_FX_WINO_ATOM_F32) {
                float32x4_t w0 = vld1q_f32(wptr), w1 = vld1q_f32(wptr + 4);
                float32x4_t w2 = vld1q_f32(wptr + 8), w3 = vld1q_f32(wptr + 12);
                float32x4_t x0, x1;
                x0 = vld1q_f32(inwptr);
                x1 = vld1q_f32(inwptr + 4);
                s00 = vfmaq_f32(s00, w0, x0);
                s01 = vfmaq_f32(s01, w0, x1);
                s10 = vfmaq_f32(s10, w1, x0);
                s11 = vfmaq_f32(s11, w1, x1);
                s20 = vfmaq_f32(s20, w2, x0);
                s21 = vfmaq_f32(s21, w2, x1);
                s30 = vfmaq_f32(s30, w3, x0);
                s31 = vfmaq_f32(s31, w3, x1);
                x0 = vld1q_f32(inwptr + 8);
                x1 = vld1q_f32(inwptr + 12);
                s02 = vfmaq_f32(s02, w0, x0);
                s03 = vfmaq_f32(s03, w0, x1);
                s12 = vfmaq_f32(s12, w1, x0);
                s13 = vfmaq_f32(s13, w1, x1);
                s22 = vfmaq_f32(s22, w2, x0);
                s23 = vfmaq_f32(s23, w2, x1);
                s32 = vfmaq_f32(s32, w3, x0);
                s33 = vfmaq_f32(s33, w3, x1);
                x0 = vld1q_f32(inwptr + 16);
                x1 = vld1q_f32(inwptr + 20);
                s04 = vfmaq_f32(s04, w0, x0);
                s05 = vfmaq_f32(s05, w0, x1);
                s14 = vfmaq_f32(s14, w1, x0);
                s15 = vfmaq_f32(s15, w1, x1);
                s24 = vfmaq_f32(s24, w2, x0);
                s25 = vfmaq_f32(s25, w2, x1);
                s34 = vfmaq_f32(s34, w3, x0);
                s35 = vfmaq_f32(s35, w3, x1);
            }

            vst1q_f32(outbuf, s00);
            vst1q_f32(outbuf + 1*64, s01);
            vst1q_f32(outbuf + 2*64, s02);
            vst1q_f32(outbuf + 3*64, s03);
            vst1q_f32(outbuf + 4*64, s04);
            vst1q_f32(outbuf + 5*64, s05);

            vst1q_f32(outbuf + 6*64, s10);
            vst1q_f32(outbuf + 7*64, s11);
            vst1q_f32(outbuf + 8*64, s12);
            vst1q_f32(outbuf + 9*64, s13);
            vst1q_f32(outbuf + 10*64, s14);
            vst1q_f32(outbuf + 11*64, s15);

            vst1q_f32(outbuf + 12*64, s20);
            vst1q_f32(outbuf + 13*64, s21);
            vst1q_f32(outbuf + 14*64, s22);
            vst1q_f32(outbuf + 15*64, s23);
            vst1q_f32(outbuf + 16*64, s24);
            vst1q_f32(outbuf + 17*64, s25);

            vst1q_f32(outbuf + 18*64, s30);
            vst1q_f32(outbuf + 19*64, s31);
            vst1q_f32(outbuf + 20*64, s32);
            vst1q_f32(outbuf + 21*64, s33);
            vst1q_f32(outbuf + 22*64, s34);
            vst1q_f32(outbuf + 23*64, s35);
        }
    }
    else
    {
        for (int atom_id = 0; atom_id < _FX_WINO_NATOMS_F32; atom_id++,
                outbuf += _FX_WINO_ATOM_F32)
        {
            float32x4_t s00 = vdupq_n_f32(0.f), s01 = s00, s02 = s00;
            float32x4_t s10 = vdupq_n_f32(0.f), s11 = s00, s12 = s00;
            float32x4_t s20 = vdupq_n_f32(0.f), s21 = s00, s22 = s00;
            float32x4_t s30 = vdupq_n_f32(0.f), s31 = s00, s32 = s00;
            for (int c = 0; c < Cg; c++, inwptr += _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32,
                                         wptr += _FX_WINO_KBLOCK*_FX_WINO_ATOM_F32) {
                float32x4_t w0 = vld1q_f32(wptr), w1 = vld1q_f32(wptr + 4);
                float32x4_t w2 = vld1q_f32(wptr + 8), w3 = vld1q_f32(wptr + 12);
                float32x4_t x0, x1, x2;
                x0 = vld1q_f32(inwptr);
                x1 = vld1q_f32(inwptr + 4);
                x2 = vld1q_f32(inwptr + 8);
                s00 = vfmaq_f32(s00, w0, x0);
                s01 = vfmaq_f32(s01, w0, x1);
                s02 = vfmaq_f32(s02, w0, x2);
                s10 = vfmaq_f32(s10, w1, x0);
                s11 = vfmaq_f32(s11, w1, x1);
                s12 = vfmaq_f32(s12, w1, x2);
                s20 = vfmaq_f32(s20, w2, x0);
                s21 = vfmaq_f32(s21, w2, x1);
                s22 = vfmaq_f32(s22, w2, x2);
                s30 = vfmaq_f32(s30, w3, x0);
                s31 = vfmaq_f32(s31, w3, x1);
                s32 = vfmaq_f32(s32, w3, x2);
            }

            vst1q_f32(outbuf, s00);
            vst1q_f32(outbuf + 1*64, s01);
            vst1q_f32(outbuf + 2*64, s02);
            vst1q_f32(outbuf + 6*64, s10);
            vst1q_f32(outbuf + 7*64, s11);
            vst1q_f32(outbuf + 8*64, s12);
            vst1q_f32(outbuf + 12*64, s20);
            vst1q_f32(outbuf + 13*64, s21);
            vst1q_f32(outbuf + 14*64, s22);
            vst1q_f32(outbuf + 18*64, s30);
            vst1q_f32(outbuf + 19*64, s31);
            vst1q_f32(outbuf + 20*64, s32);
        }
    }
#elif CV_SIMD128
    CV_Assert(_FX_WINO_IBLOCK == 3 && _FX_WINO_KBLOCK == 4);
    for (int atom_id = 0; atom_id < _FX_WINO_NATOMS_F32; atom_id++,
            outbuf += _FX_WINO_ATOM_F32)
    {
        v_float32x4 s00 = v_setzero_f32(), s01 = s00, s02 = s00;
        v_float32x4 s10 = v_setzero_f32(), s11 = s00, s12 = s00;
        v_float32x4 s20 = v_setzero_f32(), s21 = s00, s22 = s00;
        v_float32x4 s30 = v_setzero_f32(), s31 = s00, s32 = s00;

        for (int c = 0; c < Cg; c++, inwptr += _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32,
                                     wptr += _FX_WINO_KBLOCK*_FX_WINO_ATOM_F32)
        {
            v_float32x4 x0, x1, x2;
            x0 = v_load(inwptr);
            x1 = v_load(inwptr + 4);
            x2 = v_load(inwptr + 8);

            v_float32x4 w0 = v_load(wptr);
            s00 = v_fma(w0, x0, s00);
            s01 = v_fma(w0, x1, s01);
            s02 = v_fma(w0, x2, s02);

            w0 = v_load(wptr + 4);
            s10 = v_fma(w0, x0, s10);
            s11 = v_fma(w0, x1, s11);
            s12 = v_fma(w0, x2, s12);

            w0 = v_load(wptr + 8);
            s20 = v_fma(w0, x0, s20);
            s21 = v_fma(w0, x1, s21);
            s22 = v_fma(w0, x2, s22);

            w0 = v_load(wptr + 12);
            s30 = v_fma(w0, x0, s30);
            s31 = v_fma(w0, x1, s31);
            s32 = v_fma(w0, x2, s32);
        }

        v_store(outbuf, s00);
        v_store(outbuf + 1*64, s01);
        v_store(outbuf + 2*64, s02);
        v_store(outbuf + 3*64, s10);
        v_store(outbuf + 4*64, s11);
        v_store(outbuf + 5*64, s12);
        v_store(outbuf + 6*64, s20);
        v_store(outbuf + 7*64, s21);
        v_store(outbuf + 8*64, s22);
        v_store(outbuf + 9*64, s30);
        v_store(outbuf + 10*64, s31);
        v_store(outbuf + 11*64, s32);
    }
#else
    for (int atom_id = 0; atom_id < _FX_WINO_NATOMS_F32;
                    atom_id++, outbuf += _FX_WINO_ATOM_F32)
    {
        float sumbuf[_FX_WINO_IBLOCK*_FX_WINO_KBLOCK*_FX_WINO_ATOM_F32];
        memset(sumbuf, 0, sizeof(sumbuf));
        for (int c = 0; c < Cg; c++, inwptr += _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32,
                                     wptr += _FX_WINO_KBLOCK*_FX_WINO_ATOM_F32)
        {
            for (int i = 0; i < _FX_WINO_KBLOCK; i++)
            {
                for (int j = 0; j < _FX_WINO_IBLOCK; j++)
                {
                    int i_ = i*_FX_WINO_ATOM_F32;
                    int j_ = j*_FX_WINO_ATOM_F32;
                    int ij_ = i_*_FX_WINO_IBLOCK + j_;
                    float s0 = inwptr[j_ + 0]*wptr[i_ + 0];
                    float s1 = inwptr[j_ + 1]*wptr[i_ + 1];
                    float s2 = inwptr[j_ + 2]*wptr[i_ + 2];
                    float s3 = inwptr[j_ + 3]*wptr[i_ + 3];
                    sumbuf[ij_ + 0] += s0;
                    sumbuf[ij_ + 1] += s1;
                    sumbuf[ij_ + 2] += s2;
                    sumbuf[ij_ + 3] += s3;
                }
            }
        }
        for (int ij = 0; ij < _FX_WINO_KBLOCK*_FX_WINO_IBLOCK; ij++)
        {
            int ij_ = ij*_FX_WINO_ATOM_F32;
            int ij_out = ij*_FX_WINO_AREA;
            outbuf[ij_out + 0] = sumbuf[ij_ + 0];
            outbuf[ij_out + 1] = sumbuf[ij_ + 1];
            outbuf[ij_out + 2] = sumbuf[ij_ + 2];
            outbuf[ij_out + 3] = sumbuf[ij_ + 3];
        }
    }
#endif
}

#if CV_NEON
#define T4x4(a, b, c, d, tr0, tr1) \
    tr0 = vtrnq_f32(a, b); \
    tr1 = vtrnq_f32(c, d); \
    a = vcombine_f32(vget_low_f32(tr0.val[0]), vget_low_f32(tr1.val[0])); \
    b = vcombine_f32(vget_low_f32(tr0.val[1]), vget_low_f32(tr1.val[1])); \
    c = vcombine_f32(vget_high_f32(tr0.val[0]), vget_high_f32(tr1.val[0])); \
    d = vcombine_f32(vget_high_f32(tr0.val[1]), vget_high_f32(tr1.val[1]))
#endif

/*Input transform*/
static void
_fx_winograd_BtXB_8x8_f32(const float* inptr, int inpstep,
                          float* outptr, int Cg)
{
#if CV_NEON && CV_NEON_AARCH64
    float32x4_t x00 = vld1q_f32(inptr), x01 = vld1q_f32(inptr + 4);
    float32x4_t x10 = vld1q_f32(inptr + inpstep), x11 = vld1q_f32(inptr + inpstep + 4);
    float32x4_t x20 = vld1q_f32(inptr + inpstep*2), x21 = vld1q_f32(inptr + inpstep*2 + 4);
    float32x4_t x30 = vld1q_f32(inptr + inpstep*3), x31 = vld1q_f32(inptr + inpstep*3 + 4);
    float32x4_t x40 = vld1q_f32(inptr + inpstep*4), x41 = vld1q_f32(inptr + inpstep*4 + 4);
    float32x4_t x50 = vld1q_f32(inptr + inpstep*5), x51 = vld1q_f32(inptr + inpstep*5 + 4);
    float32x4_t x60 = vld1q_f32(inptr + inpstep*6), x61 = vld1q_f32(inptr + inpstep*6 + 4);
    float32x4_t x70 = vld1q_f32(inptr + inpstep*7), x71 = vld1q_f32(inptr + inpstep*7 + 4);

    float32x4_t z00, z01, z10, z11, z20, z21, z30, z31, z40, z41, z50, z51, z60, z61, z70, z71;

    {
        /* Y[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*X */
        /* Y[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*X */
        float32x4_t q5_25 = vdupq_n_f32(5.25f), t00, t01, t10, t11;
        t00 = vsubq_f32(x40, x20);
        t01 = vsubq_f32(x41, x21);
        t10 = vsubq_f32(x30, x50);
        t11 = vsubq_f32(x31, x51);
        float32x4_t y00 = vfmaq_f32(vsubq_f32(x00, x60), t00, q5_25);
        float32x4_t y01 = vfmaq_f32(vsubq_f32(x01, x61), t01, q5_25);
        float32x4_t y70 = vfmaq_f32(vsubq_f32(x70, x10), t10, q5_25);
        float32x4_t y71 = vfmaq_f32(vsubq_f32(x71, x11), t11, q5_25);

        /* Y[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*X */
        /* Y[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*X */
        float32x4_t qm4_25 = vdupq_n_f32(-4.25f);
        t00 = vfmaq_f32(vaddq_f32(x10, x50), x30, qm4_25);
        t01 = vfmaq_f32(vaddq_f32(x11, x51), x31, qm4_25);
        t10 = vfmaq_f32(vaddq_f32(x20, x60), x40, qm4_25);
        t11 = vfmaq_f32(vaddq_f32(x21, x61), x41, qm4_25);

        float32x4_t y10 = vaddq_f32(t00, t10), y11 = vaddq_f32(t01, t11);
        float32x4_t y20 = vsubq_f32(t10, t00), y21 = vsubq_f32(t11, t01);

        /* Y[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*X */
        /* Y[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*X */
        float32x4_t q0_5 = vdupq_n_f32(0.5f), q0_25 = vdupq_n_f32(0.25f);
        float32x4_t qm2_5 = vdupq_n_f32(-2.5f), qm1_25 = vdupq_n_f32(-1.25f);
        t00 = vfmaq_f32(vaddq_f32(x50, x50), x10, q0_5);
        t01 = vfmaq_f32(vaddq_f32(x51, x51), x11, q0_5);
        t10 = vfmaq_f32(x60, x20, q0_25);
        t11 = vfmaq_f32(x61, x21, q0_25);
        t00 = vfmaq_f32(t00, x30, qm2_5);
        t01 = vfmaq_f32(t01, x31, qm2_5);
        t10 = vfmaq_f32(t10, x40, qm1_25);
        t11 = vfmaq_f32(t11, x41, qm1_25);

        float32x4_t y30 = vaddq_f32(t00, t10), y31 = vaddq_f32(t01, t11);
        float32x4_t y40 = vsubq_f32(t10, t00), y41 = vsubq_f32(t11, t01);

        /* Y[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*X */
        /* Y[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*X */
        float32x4_t q4 = vdupq_n_f32(4.f), qm5 = vdupq_n_f32(-5.f);
        t00 = vfmaq_f32(vaddq_f32(x10, x10), x50, q0_5);
        t01 = vfmaq_f32(vaddq_f32(x11, x11), x51, q0_5);
        t10 = vfmaq_f32(x60, x20, q4);
        t11 = vfmaq_f32(x61, x21, q4);
        t00 = vfmaq_f32(t00, x30, qm2_5);
        t01 = vfmaq_f32(t01, x31, qm2_5);
        t10 = vfmaq_f32(t10, x40, qm5);
        t11 = vfmaq_f32(t11, x41, qm5);

        float32x4_t y50 = vaddq_f32(t00, t10), y51 = vaddq_f32(t01, t11);
        float32x4_t y60 = vsubq_f32(t10, t00), y61 = vsubq_f32(t11, t01);

        /* transpose 8x8 matrix in-place with some renumeration of the elements: */
        /* Y:              */
        /*        y00 y01  */
        /*        y10 y11  */
        /*        ...      */
        /*        y70 y71  */
        /*   Y':           */
        /*        y00 y40  */
        /*        y10 y50  */
        /*        y20 y60  */
        /*        y30 y70  */
        /*        y01 y41  */
        /*        y11 y51  */
        /*        y21 y61  */
        /*        y31 y71  */
        /*    in other words, y40 <-> y01, y50 <-> y11, y60 <-> y21, y70 <-> y31 */
        float32x4x2_t tr0, tr1;

        T4x4(y00, y10, y20, y30, tr0, tr1);
        T4x4(y01, y11, y21, y31, tr0, tr1);
        T4x4(y40, y50, y60, y70, tr0, tr1);
        T4x4(y41, y51, y61, y71, tr0, tr1);

        /* Z[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*Y */
        /* Z[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*Y */
        t00 = vsubq_f32(y01, y20);
        t01 = vsubq_f32(y41, y60);
        t10 = vsubq_f32(y30, y11);
        t11 = vsubq_f32(y70, y51);
        z00 = vfmaq_f32(vsubq_f32(y00, y21), t00, q5_25);
        z01 = vfmaq_f32(vsubq_f32(y40, y61), t01, q5_25);
        z70 = vfmaq_f32(vsubq_f32(y31, y10), t10, q5_25);
        z71 = vfmaq_f32(vsubq_f32(y71, y50), t11, q5_25);

        /* Z[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*Y */
        /* Z[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*Y */
        t00 = vfmaq_f32(vaddq_f32(y10, y11), y30, qm4_25);
        t01 = vfmaq_f32(vaddq_f32(y50, y51), y70, qm4_25);
        t10 = vfmaq_f32(vaddq_f32(y20, y21), y01, qm4_25);
        t11 = vfmaq_f32(vaddq_f32(y60, y61), y41, qm4_25);

        z10 = vaddq_f32(t00, t10); z11 = vaddq_f32(t01, t11);
        z20 = vsubq_f32(t10, t00); z21 = vsubq_f32(t11, t01);

        /* Z[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*Y */
        /* Z[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*Y */
        t00 = vfmaq_f32(vaddq_f32(y11, y11), y10, q0_5);
        t01 = vfmaq_f32(vaddq_f32(y51, y51), y50, q0_5);
        t10 = vfmaq_f32(y21, y20, q0_25);
        t11 = vfmaq_f32(y61, y60, q0_25);
        t00 = vfmaq_f32(t00, y30, qm2_5);
        t01 = vfmaq_f32(t01, y70, qm2_5);
        t10 = vfmaq_f32(t10, y01, qm1_25);
        t11 = vfmaq_f32(t11, y41, qm1_25);

        z30 = vaddq_f32(t00, t10); z31 = vaddq_f32(t01, t11);
        z40 = vsubq_f32(t10, t00); z41 = vsubq_f32(t11, t01);

        /* Z[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*Y */
        /* Z[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*Y */
        t00 = vfmaq_f32(vaddq_f32(y10, y10), y11, q0_5);
        t01 = vfmaq_f32(vaddq_f32(y50, y50), y51, q0_5);
        t10 = vfmaq_f32(y21, y20, q4);
        t11 = vfmaq_f32(y61, y60, q4);
        t00 = vfmaq_f32(t00, y30, qm2_5);
        t01 = vfmaq_f32(t01, y70, qm2_5);
        t10 = vfmaq_f32(t10, y01, qm5);
        t11 = vfmaq_f32(t11, y41, qm5);

        z50 = vaddq_f32(t00, t10); z51 = vaddq_f32(t01, t11);
        z60 = vsubq_f32(t10, t00); z61 = vsubq_f32(t11, t01);
    }

    const int outstep = _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32*Cg;

    vst1q_f32(outptr, z00);
    vst1q_f32(outptr + outstep, z01);
    vst1q_f32(outptr + outstep*2, z10);
    vst1q_f32(outptr + outstep*3, z11);
    vst1q_f32(outptr + outstep*4, z20);
    vst1q_f32(outptr + outstep*5, z21);
    vst1q_f32(outptr + outstep*6, z30);
    vst1q_f32(outptr + outstep*7, z31);
    vst1q_f32(outptr + outstep*8, z40);
    vst1q_f32(outptr + outstep*9, z41);
    vst1q_f32(outptr + outstep*10, z50);
    vst1q_f32(outptr + outstep*11, z51);
    vst1q_f32(outptr + outstep*12, z60);
    vst1q_f32(outptr + outstep*13, z61);
    vst1q_f32(outptr + outstep*14, z70);
    vst1q_f32(outptr + outstep*15, z71);
#elif CV_SIMD128
    v_float32x4 x00 = v_load(inptr), x01 = v_load(inptr + 4);
    v_float32x4 x10 = v_load(inptr + inpstep), x11 = v_load(inptr + inpstep + 4);
    v_float32x4 x20 = v_load(inptr + inpstep*2), x21 = v_load(inptr + inpstep*2 + 4);
    v_float32x4 x30 = v_load(inptr + inpstep*3), x31 = v_load(inptr + inpstep*3 + 4);
    v_float32x4 x40 = v_load(inptr + inpstep*4), x41 = v_load(inptr + inpstep*4 + 4);
    v_float32x4 x50 = v_load(inptr + inpstep*5), x51 = v_load(inptr + inpstep*5 + 4);
    v_float32x4 x60 = v_load(inptr + inpstep*6), x61 = v_load(inptr + inpstep*6 + 4);
    v_float32x4 x70 = v_load(inptr + inpstep*7), x71 = v_load(inptr + inpstep*7 + 4);

    v_float32x4 z00, z01, z10, z11, z20, z21, z30, z31, z40, z41, z50, z51, z60, z61, z70, z71;

    {
        /* Y[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*X */
        /* Y[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*X */
        v_float32x4 q5_25 = v_setall_f32(5.25f), t00, t01, t10, t11;
        t00 = x40 - x20;
        t01 = x41 - x21;
        t10 = x30 - x50;
        t11 = x31 - x51;
        v_float32x4 y00 = v_fma(t00, q5_25, x00 - x60);
        v_float32x4 y01 = v_fma(t01, q5_25, x01 - x61);
        v_float32x4 y70 = v_fma(t10, q5_25, x70 - x10);
        v_float32x4 y71 = v_fma(t11, q5_25, x71 - x11);

        /* Y[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*X */
        /* Y[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*X */
        v_float32x4 qm4_25 = v_setall_f32(-4.25f);
        t00 = v_fma(x30, qm4_25, x10 + x50);
        t01 = v_fma(x31, qm4_25, x11 + x51);
        t10 = v_fma(x40, qm4_25, x20 + x60);
        t11 = v_fma(x41, qm4_25, x21 + x61);

        v_float32x4 y10 = t00 + t10, y11 = t01 + t11;
        v_float32x4 y20 = t10 - t00, y21 = t11 - t01;

        /* Y[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*X */
        /* Y[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*X */
        v_float32x4 q0_5 = v_setall_f32(0.5f), q0_25 = v_setall_f32(0.25f);
        v_float32x4 qm2_5 = v_setall_f32(-2.5f), qm1_25 = v_setall_f32(-1.25f);
        t00 = v_fma(x10, q0_5, x50 + x50);
        t01 = v_fma(x11, q0_5, x51 + x51);
        t10 = v_fma(x20, q0_25, x60);
        t11 = v_fma(x21, q0_25, x61);
        t00 = v_fma(x30, qm2_5, t00);
        t01 = v_fma(x31, qm2_5, t01);
        t10 = v_fma(x40, qm1_25, t10);
        t11 = v_fma(x41, qm1_25, t11);

        v_float32x4 y30 = t00 + t10, y31 = t01 + t11;
        v_float32x4 y40 = t10 - t00, y41 = t11 - t01;

        /* Y[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*X */
        /* Y[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*X */
        v_float32x4 q4 = v_setall_f32(4.f), qm5 = v_setall_f32(-5.f);
        t00 = v_fma(x50, q0_5, x10 + x10);
        t01 = v_fma(x51, q0_5, x11 + x11);
        t10 = v_fma(x20, q4   , x60);
        t11 = v_fma(x21, q4   , x61);
        t00 = v_fma(x30, qm2_5, t00);
        t01 = v_fma(x31, qm2_5, t01);
        t10 = v_fma(x40, qm5  , t10);
        t11 = v_fma(x41, qm5  , t11);

        v_float32x4 y50 = t00 + t10, y51 = t01 + t11;
        v_float32x4 y60 = t10 - t00, y61 = t11 - t01;

        /* transpose 8x8 matrix in-place with some renumeration of the elements: */
        /* Y:              */
        /*        y00 y01  */
        /*        y10 y11  */
        /*        ...      */
        /*        y70 y71  */
        /*   Y':           */
        /*        y00 y40  */
        /*        y10 y50  */
        /*        y20 y60  */
        /*        y30 y70  */
        /*        y01 y41  */
        /*        y11 y51  */
        /*        y21 y61  */
        /*        y31 y71  */
        /*    in other words, y40 <-> y01, y50 <-> y11, y60 <-> y21, y70 <-> y31 */

        v_transpose4x4(y00, y10, y20, y30, y00, y10, y20, y30);
        v_transpose4x4(y01, y11, y21, y31, y01, y11, y21, y31);
        v_transpose4x4(y40, y50, y60, y70, y40, y50, y60, y70);
        v_transpose4x4(y41, y51, y61, y71, y41, y51, y61, y71);

        /* Z[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*Y */
        /* Z[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*Y */
        t00 = y01 - y20;
        t01 = y41 - y60;
        t10 = y30 - y11;
        t11 = y70 - y51;
        z00 = v_fma(t00, q5_25, y00 - y21);
        z01 = v_fma(t01, q5_25, y40 - y61);
        z70 = v_fma(t10, q5_25, y31 - y10);
        z71 = v_fma(t11, q5_25, y71 - y50);

        /* Z[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*Y */
        /* Z[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*Y */
        t00 = v_fma(y30, qm4_25, y10 + y11);
        t01 = v_fma(y70, qm4_25, y50 + y51);
        t10 = v_fma(y01, qm4_25, y20 + y21);
        t11 = v_fma(y41, qm4_25, y60 + y61);

        z10 = t00 + t10; z11 = t01 + t11;
        z20 = t10 - t00; z21 = t11 - t01;

        /* Z[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*Y */
        /* Z[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*Y */
        t00 = v_fma(y10, q0_5, y11 + y11);
        t01 = v_fma(y50, q0_5, y51 + y51);
        t10 = v_fma(y20, q0_25, y21);
        t11 = v_fma(y60, q0_25, y61);
        t00 = v_fma(y30, qm2_5, t00);
        t01 = v_fma(y70, qm2_5, t01);
        t10 = v_fma(y01, qm1_25, t10);
        t11 = v_fma(y41, qm1_25, t11);

        z30 = t00 + t10; z31 = t01 + t11;
        z40 = t10 - t00; z41 = t11 - t01;

        /* Z[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*Y */
        /* Z[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*Y */
        t00 = v_fma(y11, q0_5, y10 + y10);
        t01 = v_fma(y51, q0_5, y50 + y50);
        t10 = v_fma(y20, q4, y21);
        t11 = v_fma(y60, q4, y61);
        t00 = v_fma(y30, qm2_5, t00);
        t01 = v_fma(y70, qm2_5, t01);
        t10 = v_fma(y01, qm5, t10);
        t11 = v_fma(y41, qm5, t11);

        z50 = t00 + t10; z51 = t01 + t11;
        z60 = t10 - t00; z61 = t11 - t01;
    }

    const int outstep = _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32*Cg;

    v_store(outptr, z00);
    v_store(outptr + outstep, z01);
    v_store(outptr + outstep*2, z10);
    v_store(outptr + outstep*3, z11);
    v_store(outptr + outstep*4, z20);
    v_store(outptr + outstep*5, z21);
    v_store(outptr + outstep*6, z30);
    v_store(outptr + outstep*7, z31);
    v_store(outptr + outstep*8, z40);
    v_store(outptr + outstep*9, z41);
    v_store(outptr + outstep*10, z50);
    v_store(outptr + outstep*11, z51);
    v_store(outptr + outstep*12, z60);
    v_store(outptr + outstep*13, z61);
    v_store(outptr + outstep*14, z70);
    v_store(outptr + outstep*15, z71);
#else
#error "Only SIMD128, AVX2 and NEON are supported in Winograd."
#endif
}

/*  Inverse Winograd 8x8 transform:
    out = (A'*inp*A)', where
    inp is input 8x8 FP32 matrix,
    A' is
    [1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f,
     0.f, 1.f, -1.f, 2.f, -2.f, 0.5f, -0.5f, 0.f,
     0.f, 1.f, 1.f, 4.f, 4.f, 0.25f, 0.25f, 0.f,
     0.f, 1.f, -1.f, 8.f, -8.f, 0.125f, -0.125f, 0.f,
     0.f, 1.f, 1.f, 16.f, 16.f, 1.f/16, 1.f/16, 0.f,
     0.f, 1.f, -1.f, 32.f, -32.f, 1.f/32, -1.f/32, 1.f]

    inp is pre-loaded into xij registers,
    out will be stored in zij, where (0<=i<=7 for x, 0<=i<=5 for z), 0<=j<=1.

    After the inverse transform is done, we add bias,
    optionally add results from the earlier tensors (by-pass),
    optionally apply activation function and then
    store the final results.

    Note that both _FX_WINOGRAD_FWD_8x8() and
    _FX_WINOGRAD_INV_8x8() produce tranposed output.
    That is, after both forward and then inverse transformation,
    we get non-transposed result.
    Of course, for the correct work of Winograd-based convolution,
    the Winograd-transformed weights should also be transposed.
    init_conv() (see OpConv.fx) takes care of that.
*/
static void
_fx_winograd_AtXA_8x8_f32(const float* inptr, int inpstep,
                          float* bpptr, int bpstep, float* outptr, int outstep,
                          float bias, float minval, float maxval, bool ifMinMaxAct)
{
#if CV_NEON && CV_NEON_AARCH64
    float32x4_t x00 = vld1q_f32(inptr), x01 = vld1q_f32(inptr + 4);
    float32x4_t x10 = vld1q_f32(inptr + inpstep), x11 = vld1q_f32(inptr + inpstep + 4);
    float32x4_t x20 = vld1q_f32(inptr + inpstep*2), x21 = vld1q_f32(inptr + inpstep*2 + 4);
    float32x4_t x30 = vld1q_f32(inptr + inpstep*3), x31 = vld1q_f32(inptr + inpstep*3 + 4);
    float32x4_t x40 = vld1q_f32(inptr + inpstep*4), x41 = vld1q_f32(inptr + inpstep*4 + 4);
    float32x4_t x50 = vld1q_f32(inptr + inpstep*5), x51 = vld1q_f32(inptr + inpstep*5 + 4);
    float32x4_t x60 = vld1q_f32(inptr + inpstep*6), x61 = vld1q_f32(inptr + inpstep*6 + 4);
    float32x4_t x70 = vld1q_f32(inptr + inpstep*7), x71 = vld1q_f32(inptr + inpstep*7 + 4);
    float32x4_t z00, z01, z10, z11, z20, z21, z30, z31, z40, z41, z50, z51;

    {
        float32x4_t s12_0, s12_1, s34_0, s34_1, s56_0, s56_1;
        s12_0 = vaddq_f32(x10, x20); s12_1 = vaddq_f32(x11, x21);
        s34_0 = vaddq_f32(x30, x40); s34_1 = vaddq_f32(x31, x41);
        s56_0 = vaddq_f32(x50, x60); s56_1 = vaddq_f32(x51, x61);

        float32x4_t y00 = vaddq_f32(vaddq_f32(vaddq_f32(x00, s12_0), s34_0), s56_0);
        float32x4_t y01 = vaddq_f32(vaddq_f32(vaddq_f32(x01, s12_1), s34_1), s56_1);
        float32x4_t y20 = vfmaq_n_f32(vfmaq_n_f32(s12_0, s34_0, 4.0f), s56_0, 0.25f);
        float32x4_t y21 = vfmaq_n_f32(vfmaq_n_f32(s12_1, s34_1, 4.0f), s56_1, 0.25f);
        float32x4_t y40 = vfmaq_n_f32(vfmaq_n_f32(s12_0, s34_0, 16.0f), s56_0, 1.f/16);
        float32x4_t y41 = vfmaq_n_f32(vfmaq_n_f32(s12_1, s34_1, 16.0f), s56_1, 1.f/16);

        s12_0 = vsubq_f32(x10, x20); s12_1 = vsubq_f32(x11, x21);
        s34_0 = vsubq_f32(x30, x40); s34_1 = vsubq_f32(x31, x41);
        s56_0 = vsubq_f32(x50, x60); s56_1 = vsubq_f32(x51, x61);

        float32x4_t y50 = vfmaq_n_f32(vfmaq_n_f32(vaddq_f32(x70, s12_0),
                                      s34_0, 32.f), s56_0, 1.f/32);
        float32x4_t y51 = vfmaq_n_f32(vfmaq_n_f32(vaddq_f32(x71, s12_1),
                                      s34_1, 32.f), s56_1, 1.f/32);
        float32x4_t y10 = vfmaq_n_f32(vfmaq_n_f32(s12_0, s34_0, 2.0f), s56_0, 0.5f);
        float32x4_t y11 = vfmaq_n_f32(vfmaq_n_f32(s12_1, s34_1, 2.0f), s56_1, 0.5f);
        float32x4_t y30 = vfmaq_n_f32(vfmaq_n_f32(s12_0, s34_0, 8.0f), s56_0, 0.125f);
        float32x4_t y31 = vfmaq_n_f32(vfmaq_n_f32(s12_1, s34_1, 8.0f), s56_1, 0.125f);
        float32x4_t y60 = vdupq_n_f32(0.f), y61 = y60, y70 = y60, y71 = y60;

        /* transpose 8x8 matrix in-place with some renumeration of the elements: */
        /*  Y: */
        /*        y00 y01 */
        /*        y10 y11 */
        /*        ... */
        /*        y50 y51 */
        /*        0   0 */
        /*        0   0 */
        /*   Y': */
        /*        y00 y40 */
        /*        y10 y50 */
        /*        y20 y60 */
        /*        y30 y70 */
        /*        y01 y41 */
        /*        y11 y51 */
        /*        y21 y61 */
        /*        y31 y71 */
        /*    in other words, y40 <-> y01, y50 <-> y11, y60 <-> y21, y70 <-> y31 */
        float32x4x2_t tr0, tr1;

        T4x4(y00, y10, y20, y30, tr0, tr1);
        T4x4(y01, y11, y21, y31, tr0, tr1);
        T4x4(y40, y50, y60, y70, tr0, tr1);
        T4x4(y41, y51, y61, y71, tr0, tr1);

        s12_0 = vaddq_f32(y10, y20); s12_1 = vaddq_f32(y50, y60);
        s34_0 = vaddq_f32(y30, y01); s34_1 = vaddq_f32(y70, y41);
        s56_0 = vaddq_f32(y11, y21); s56_1 = vaddq_f32(y51, y61);

        z00 = vaddq_f32(vaddq_f32(vaddq_f32(y00, s12_0), s34_0), s56_0);
        z01 = vaddq_f32(vaddq_f32(vaddq_f32(y40, s12_1), s34_1), s56_1);
        z20 = vfmaq_n_f32(vfmaq_n_f32(s12_0, s34_0, 4.0f), s56_0, 0.25f);
        z21 = vfmaq_n_f32(vfmaq_n_f32(s12_1, s34_1, 4.0f), s56_1, 0.25f);
        z40 = vfmaq_n_f32(vfmaq_n_f32(s12_0, s34_0, 16.0f), s56_0, 1.f/16);
        z41 = vfmaq_n_f32(vfmaq_n_f32(s12_1, s34_1, 16.0f), s56_1, 1.f/16);

        s12_0 = vsubq_f32(y10, y20); s12_1 = vsubq_f32(y50, y60);
        s34_0 = vsubq_f32(y30, y01); s34_1 = vsubq_f32(y70, y41);
        s56_0 = vsubq_f32(y11, y21); s56_1 = vsubq_f32(y51, y61);

        z50 = vfmaq_n_f32(vfmaq_n_f32(vaddq_f32(y31, s12_0),
                          s34_0, 32.f), s56_0, 1.f/32);
        z51 = vfmaq_n_f32(vfmaq_n_f32(vaddq_f32(y71, s12_1),
                          s34_1, 32.f), s56_1, 1.f/32);
        z10 = vfmaq_n_f32(vfmaq_n_f32(s12_0, s34_0, 2.0f), s56_0, 0.5f);
        z11 = vfmaq_n_f32(vfmaq_n_f32(s12_1, s34_1, 2.0f), s56_1, 0.5f);
        z30 = vfmaq_n_f32(vfmaq_n_f32(s12_0, s34_0, 8.0f), s56_0, 0.125f);
        z31 = vfmaq_n_f32(vfmaq_n_f32(s12_1, s34_1, 8.0f), s56_1, 0.125f);
        float32x4_t vbias = vdupq_n_f32(bias);

        z00 = vaddq_f32(z00, vbias);
        z01 = vaddq_f32(z01, vbias);
        z10 = vaddq_f32(z10, vbias);
        z11 = vaddq_f32(z11, vbias);
        z20 = vaddq_f32(z20, vbias);
        z21 = vaddq_f32(z21, vbias);
        z30 = vaddq_f32(z30, vbias);
        z31 = vaddq_f32(z31, vbias);
        z40 = vaddq_f32(z40, vbias);
        z41 = vaddq_f32(z41, vbias);
        z50 = vaddq_f32(z50, vbias);
        z51 = vaddq_f32(z51, vbias);
    }

    if (bpptr)
    {
        float32x2_t zhalf = vdup_n_f32(0.f);
        z00 = vaddq_f32(z00, vld1q_f32(bpptr));
        z01 = vaddq_f32(z01, vcombine_f32(vld1_f32(bpptr + 4), zhalf));
        z10 = vaddq_f32(z10, vld1q_f32(bpptr + bpstep));
        z11 = vaddq_f32(z11, vcombine_f32(vld1_f32(bpptr + bpstep + 4), zhalf));
        z20 = vaddq_f32(z20, vld1q_f32(bpptr + bpstep*2));
        z21 = vaddq_f32(z21, vcombine_f32(vld1_f32(bpptr + bpstep*2 + 4), zhalf));
        z30 = vaddq_f32(z30, vld1q_f32(bpptr + bpstep*3));
        z31 = vaddq_f32(z31, vcombine_f32(vld1_f32(bpptr + bpstep*3 + 4), zhalf));
        z40 = vaddq_f32(z40, vld1q_f32(bpptr + bpstep*4));
        z41 = vaddq_f32(z41, vcombine_f32(vld1_f32(bpptr + bpstep*4 + 4), zhalf));
        z50 = vaddq_f32(z50, vld1q_f32(bpptr + bpstep*5));
        z51 = vaddq_f32(z51, vcombine_f32(vld1_f32(bpptr + bpstep*5 + 4), zhalf));
    }

    if (ifMinMaxAct)
    {
        float32x4_t vmax = vdupq_n_f32(maxval);
        float32x4_t vmin = vdupq_n_f32(minval);

        z00 = vminq_f32(vmaxq_f32(z00, vmin), vmax);
        z01 = vminq_f32(vmaxq_f32(z01, vmin), vmax);
        z10 = vminq_f32(vmaxq_f32(z10, vmin), vmax);
        z11 = vminq_f32(vmaxq_f32(z11, vmin), vmax);
        z20 = vminq_f32(vmaxq_f32(z20, vmin), vmax);
        z21 = vminq_f32(vmaxq_f32(z21, vmin), vmax);
        z30 = vminq_f32(vmaxq_f32(z30, vmin), vmax);
        z31 = vminq_f32(vmaxq_f32(z31, vmin), vmax);
        z40 = vminq_f32(vmaxq_f32(z40, vmin), vmax);
        z41 = vminq_f32(vmaxq_f32(z41, vmin), vmax);
        z50 = vminq_f32(vmaxq_f32(z50, vmin), vmax);
        z51 = vminq_f32(vmaxq_f32(z51, vmin), vmax);
    }

    vst1q_f32(outptr, z00);
    vst1_f32(outptr + 4, vget_low_f32(z01));
    vst1q_f32(outptr + outstep, z10);
    vst1_f32(outptr + outstep + 4, vget_low_f32(z11));
    vst1q_f32(outptr + outstep*2, z20);
    vst1_f32(outptr + outstep*2 + 4, vget_low_f32(z21));
    vst1q_f32(outptr + outstep*3, z30);
    vst1_f32(outptr + outstep*3 + 4, vget_low_f32(z31));
    vst1q_f32(outptr + outstep*4, z40);
    vst1_f32(outptr + outstep*4 + 4, vget_low_f32(z41));
    vst1q_f32(outptr + outstep*5, z50);
    vst1_f32(outptr + outstep*5 + 4, vget_low_f32(z51));
#elif CV_SIMD128
    v_float32x4 x00 = v_load(inptr), x01 = v_load(inptr + 4);
    v_float32x4 x10 = v_load(inptr + inpstep), x11 = v_load(inptr + inpstep + 4);
    v_float32x4 x20 = v_load(inptr + inpstep*2), x21 = v_load(inptr + inpstep*2 + 4);
    v_float32x4 x30 = v_load(inptr + inpstep*3), x31 = v_load(inptr + inpstep*3 + 4);
    v_float32x4 x40 = v_load(inptr + inpstep*4), x41 = v_load(inptr + inpstep*4 + 4);
    v_float32x4 x50 = v_load(inptr + inpstep*5), x51 = v_load(inptr + inpstep*5 + 4);
    v_float32x4 x60 = v_load(inptr + inpstep*6), x61 = v_load(inptr + inpstep*6 + 4);
    v_float32x4 x70 = v_load(inptr + inpstep*7), x71 = v_load(inptr + inpstep*7 + 4);
    v_float32x4 z00, z01, z10, z11, z20, z21, z30, z31, z40, z41, z50, z51;

    {
        v_float32x4 s12_0, s12_1, s34_0, s34_1, s56_0, s56_1;
        s12_0 = x10 + x20; s12_1 = x11 + x21;
        s34_0 = x30 + x40; s34_1 = x31 + x41;
        s56_0 = x50 + x60; s56_1 = x51 + x61;

        v_float32x4 y00 = x00 + s12_0 + s34_0 + s56_0;
        v_float32x4 y01 = x01 + s12_1 + s34_1 + s56_1;

        v_float32x4 a0 = v_setall_f32(0.25f), a1 = v_setall_f32(4.0f);
        v_float32x4 y20 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y21 = v_fma(s56_1, a0 ,v_fma(s34_1, a1, s12_1) );

        a0 = v_setall_f32(1.f/16), a1 = v_setall_f32(16.0f);
        v_float32x4 y40 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y41 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        s12_0 = x10 - x20; s12_1 = x11 - x21;
        s34_0 = x30 - x40; s34_1 = x31 - x41;
        s56_0 = x50 - x60; s56_1 = x51 - x61;

        a0 = v_setall_f32(1.f/32), a1 = v_setall_f32(32.f);
        v_float32x4 y50 = v_fma(s56_0, a0, v_fma(s34_0, a1, x70 + s12_0));
        v_float32x4 y51 = v_fma(s56_1, a0, v_fma(s34_1, a1, x71 + s12_1));

        a0 = v_setall_f32(0.5f), a1 = v_setall_f32(2.f);
        v_float32x4 y10 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y11 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(0.125f), a1 = v_setall_f32(8.f);
        v_float32x4 y30 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y31 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        v_float32x4 y60 = v_setall_f32(0.f), y61 = y60, y70 = y60, y71 = y60;

        /* transpose 8x8 matrix in-place with some renumeration of the elements: */
        /*  Y: */
        /*        y00 y01 */
        /*        y10 y11 */
        /*        ... */
        /*        y50 y51 */
        /*        0   0 */
        /*        0   0 */
        /*   Y': */
        /*        y00 y40 */
        /*        y10 y50 */
        /*        y20 y60 */
        /*        y30 y70 */
        /*        y01 y41 */
        /*        y11 y51 */
        /*        y21 y61 */
        /*        y31 y71 */
        /*    in other words, y40 <-> y01, y50 <-> y11, y60 <-> y21, y70 <-> y31 */

        v_transpose4x4(y00, y10, y20, y30, y00, y10, y20, y30);
        v_transpose4x4(y01, y11, y21, y31, y01, y11, y21, y31);
        v_transpose4x4(y40, y50, y60, y70, y40, y50, y60, y70);
        v_transpose4x4(y41, y51, y61, y71, y41, y51, y61, y71);

        s12_0 = y10 + y20; s12_1 = y50 + y60;
        s34_0 = y30 + y01; s34_1 = y70 + y41;
        s56_0 = y11 + y21; s56_1 = y51 + y61;

        z00 = y00 + s12_0 + s34_0 + s56_0;
        z01 = y40 + s12_1 + s34_1 + s56_1;

        a0 = v_setall_f32(0.25f), a1 = v_setall_f32(4.0f);
        z20 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z21 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(1.f/16), a1 = v_setall_f32(16.0f);
        z40 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z41 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        s12_0 = y10 - y20; s12_1 = y50 - y60;
        s34_0 = y30 - y01; s34_1 = y70 - y41;
        s56_0 = y11 - y21; s56_1 = y51 - y61;

        a0 = v_setall_f32(1.f/32), a1 = v_setall_f32(32.0f);
        z50 = v_fma(s56_0, a0, v_fma(s34_0, a1, y31 + s12_0));
        z51 = v_fma(s56_1, a0, v_fma(s34_1, a1, y71 + s12_1));

        a0 = v_setall_f32(0.5f), a1 = v_setall_f32(2.0f);
        z10 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z11 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(0.125f), a1 = v_setall_f32(8.0f);
        z30 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z31 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        v_float32x4 vbias = v_setall_f32(bias);
        z00 += vbias;
        z01 += vbias;
        z10 += vbias;
        z11 += vbias;
        z20 += vbias;
        z21 += vbias;
        z30 += vbias;
        z31 += vbias;
        z40 += vbias;
        z41 += vbias;
        z50 += vbias;
        z51 += vbias;
    }

    if (bpptr)
    {
        z00 += v_load(bpptr);
        z01 += v_load_low(bpptr + 4);
        z10 += v_load(bpptr + bpstep);
        z11 += v_load_low(bpptr + bpstep + 4);
        z20 += v_load(bpptr + bpstep*2);
        z21 += v_load_low(bpptr + bpstep*2 + 4);
        z30 += v_load(bpptr + bpstep*3);
        z31 += v_load_low(bpptr + bpstep*3 + 4);
        z40 += v_load(bpptr + bpstep*4);
        z41 += v_load_low(bpptr + bpstep*4 + 4);
        z50 += v_load(bpptr + bpstep*5);
        z51 += v_load_low(bpptr + bpstep*5 + 4);
    }

    if (ifMinMaxAct)
    {
        v_float32x4 vmax = v_setall_f32(maxval);
        v_float32x4 vmin = v_setall_f32(minval);

        z00 = v_min(v_max(z00, vmin), vmax);
        z01 = v_min(v_max(z01, vmin), vmax);
        z10 = v_min(v_max(z10, vmin), vmax);
        z11 = v_min(v_max(z11, vmin), vmax);
        z20 = v_min(v_max(z20, vmin), vmax);
        z21 = v_min(v_max(z21, vmin), vmax);
        z30 = v_min(v_max(z30, vmin), vmax);
        z31 = v_min(v_max(z31, vmin), vmax);
        z40 = v_min(v_max(z40, vmin), vmax);
        z41 = v_min(v_max(z41, vmin), vmax);
        z50 = v_min(v_max(z50, vmin), vmax);
        z51 = v_min(v_max(z51, vmin), vmax);
    }

    v_store(outptr, z00);
    v_store_low(outptr + 4, z01);
    v_store(outptr + outstep, z10);
    v_store_low(outptr + outstep + 4, z11);
    v_store(outptr + outstep*2, z20);
    v_store_low(outptr + outstep*2 + 4, z21);
    v_store(outptr + outstep*3, z30);
    v_store_low(outptr + outstep*3 + 4, z31);
    v_store(outptr + outstep*4, z40);
    v_store_low(outptr + outstep*4 + 4, z41);
    v_store(outptr + outstep*5, z50);
    v_store_low(outptr + outstep*5 + 4, z51);
#else
#error "Only SIMD128, AVX2 and NEON are supported in Winograd."
#endif
}

int runWinograd63(InputArray _input, InputArray _fusedAddMat, OutputArray _output, const Ptr<FastConv>& conv,
                  int ntasks, float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    Mat fusedAddMat = _fusedAddMat.getMat();

    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);
    CV_Assert(inputShape.size() == 4 && outputShape.size() == 4);

    int N = inputShape[0], C = inputShape[1], Hi = inputShape[2], Wi = inputShape[3];  // [N, C, H, W]
    int K = conv->K;
    int H0 = outputShape[2], W0 = outputShape[3];

    int pad_top = conv->pad_top;
    int pad_left = conv->pad_left;

    int ngroups = conv->ngroups, Cg = C/ngroups, Kg = K/ngroups;
    int Kg_nblocks = (Kg + _FX_WINO_KBLOCK - 1)/_FX_WINO_KBLOCK;
    const size_t inp_planesize = (size_t)Hi*Wi;
    const size_t out_planesize = (size_t)H0*W0;

    int blocks_per_row = (W0+_FX_WINO_STEP-1)/_FX_WINO_STEP;
    int blocks_per_plane = ((H0+_FX_WINO_STEP-1)/_FX_WINO_STEP)*blocks_per_row;
    int blocks_per_plane_aligned = ((blocks_per_plane +
                                     _FX_WINO_IBLOCK-1)/_FX_WINO_IBLOCK)*_FX_WINO_IBLOCK;

    size_t totalbufsize = (size_t)N*C*blocks_per_plane_aligned*_FX_WINO_AREA;

    AutoBuffer<float> _buf;
    _buf.allocate(totalbufsize + VEC_ALIGN);
    float* wbuf_all = alignPtr(_buf.data(), VEC_ALIGN);

    float* inp = input.ptr<float>();
    float* out = output.ptr<float>();

    float* fusedAddPtr = fusedAddMat.empty() ? nullptr : fusedAddMat.ptr<float>();

    // Phase 1. compute forward Winograd transforms for all input blocks,
    // all input planes, all samples in the batch.
    // [TODO]: maybe, if there are too many input channels, it makes sense to
    // transform only part of input channels at once and then compute the partial
    // accumulated sums (i.e. update the output buffers several times,
    // rather than compute them in one pass).
    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        int nc0 = (N*C)*task_id/ntasks;
        int nc1 = (N*C)*(task_id+1)/ntasks;
        for(; nc0 < nc1; nc0++)
        {
            int n = nc0 / C;
            int c = nc0 - n*C;
            int g = c / Cg;
            c -= g*Cg;
            for (int block_id = 0; block_id < blocks_per_plane; block_id += _FX_WINO_IBLOCK)
            {
                for (int db = 0; db < _FX_WINO_IBLOCK; db++)
                {
                    size_t inwofs = ((n*ngroups + g)*blocks_per_plane_aligned +
                                     block_id)*Cg*_FX_WINO_AREA +
                                    (c*_FX_WINO_IBLOCK + db)*_FX_WINO_ATOM_F32;
                    float* inwptr = (float*)wbuf_all + inwofs;

                    if (block_id + db < blocks_per_plane)
                    {
                        int y0 = (block_id + db) / blocks_per_row;
                        int x0 = (block_id + db) - y0 * blocks_per_row;
                        y0 = y0*_FX_WINO_STEP - pad_top;
                        x0 = x0*_FX_WINO_STEP - pad_left;
                        bool partial = y0 < 0 || y0 + _FX_WINO_SIZE > Hi ||
                                       x0 < 0 || x0 + _FX_WINO_SIZE > Wi;
                        int dx1 = 0, dx2 = _FX_WINO_SIZE, dy1 = 0, dy2 = _FX_WINO_SIZE;
                        int inpstep = Wi;

                        float inpbuf[_FX_WINO_AREA];
                        float* inptr0 = (float*)inp + nc0*inp_planesize + y0*Wi + x0;
                        float* inptr = inptr0;

                        if (partial)
                        {
                            memset(inpbuf, 0, sizeof(inpbuf));
                            dy1 = -y0 > 0 ? -y0 : 0;
                            dy2 = Hi - y0 < _FX_WINO_SIZE ? Hi - y0 : _FX_WINO_SIZE;

                            if (dy2 < dy1) {dy2 = dy1 = 0;}
                            dx1 = -x0 > 0 ? -x0 : 0;
                            dx2 = Wi - x0 < _FX_WINO_SIZE ? Wi - x0 : _FX_WINO_SIZE;

                            if (dx2 < dx1) {dx2 = dx1 = 0;}
                            inptr0 -= y0*Wi + x0;

                            if (dx1 < dx2 && dy1 < dy2)
                            {
                                for(int dy = dy1; dy < dy2; dy++)
                                    memcpy(&inpbuf[dy*_FX_WINO_SIZE + dx1],
                                           inptr0 + (y0+dy)*Wi + (x0+dx1),
                                           (dx2-dx1)*sizeof(inpbuf[0]));
                            }

                            inptr = inpbuf;
                            inpstep = _FX_WINO_SIZE;
                        }
#if CV_TRY_AVX2
                        if (conv->useAVX2)
                            opt_AVX2::_fx_winograd_BtXB_8x8_f32(inptr, inpstep, inwptr, Cg);
                        else
#endif
                        _fx_winograd_BtXB_8x8_f32(inptr, inpstep, inwptr, Cg);
                    }
                    else
                    {
                        for (int i = 0; i < _FX_WINO_NATOMS_F32; i++, inwptr += _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32)
                            memset(inwptr, 0, _FX_WINO_ATOM_F32*sizeof(inwptr[0]));
                    }
                }
            }
        }
    }});

    // Phase 2. compute elemwise-weighted sums of transformed blocks,
    // apply inverse Winograd transforms to the sums,
    // add bias, apply activation function if any and store the results.
    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        size_t out_wbuf_size = _FX_WINO_AREA*_FX_WINO_KBLOCK*_FX_WINO_IBLOCK;
        size_t outbuf_size = _FX_WINO_AREA;
        AutoBuffer<float> out_wbuf_, outbuf_;
        out_wbuf_.allocate(out_wbuf_size + VEC_ALIGN);
        float* out_wbuf = alignPtr(out_wbuf_.data(), VEC_ALIGN);
        outbuf_.allocate(outbuf_size + VEC_ALIGN);
        float* outbuf = alignPtr(outbuf_.data(), VEC_ALIGN);

        memset(out_wbuf, 0, out_wbuf_size * sizeof(float));
        memset(outbuf, 0, outbuf_size * sizeof(float));

        int ngk0 = (int)(((int64_t)N*Kg_nblocks*ngroups)*task_id/ntasks);
        int ngk1 = (int)(((int64_t)N*Kg_nblocks*ngroups)*(task_id+1)/ntasks);

        for(; ngk0 < ngk1; ngk0++)
        {
            int n = ngk0 / (Kg_nblocks*ngroups);
            int gk0 = ngk0 % (Kg_nblocks*ngroups);
            int g = gk0 / Kg_nblocks;
            int k0 = (gk0 % Kg_nblocks)*_FX_WINO_KBLOCK;
            int k1 = k0 + _FX_WINO_KBLOCK <= Kg ? k0 + _FX_WINO_KBLOCK : Kg;

            for (int block_id0 = 0; block_id0 < blocks_per_plane; block_id0 += _FX_WINO_IBLOCK)
            {
                int block_id1 = block_id0 + _FX_WINO_IBLOCK;
                block_id1 = block_id1 < blocks_per_plane ? block_id1 : blocks_per_plane;
                size_t inwofs = ((n*ngroups + g)*blocks_per_plane_aligned + block_id0)*Cg*_FX_WINO_AREA;
                size_t wofs = (g*Kg_nblocks*_FX_WINO_KBLOCK + k0)*Cg*_FX_WINO_AREA;

                float* inwptr = wbuf_all + inwofs;
                const float* wptr = conv->weightsWinoBufPtr + wofs;

#if CV_TRY_AVX2
                if (conv->useAVX2)
                    opt_AVX2::_fx_winograd_accum_f32(inwptr, wptr, out_wbuf, Cg, block_id1 - block_id0);
                else
#endif
                _fx_winograd_accum_f32(inwptr, wptr, out_wbuf, Cg, block_id1 - block_id0);
                for (int k = k0; k < k1; k++)
                {
                    float biasv = conv->biasBuf[g*Kg + k];
                    for (int block_id = block_id0; block_id < block_id1; block_id++)
                    {
                        int y0 = block_id / blocks_per_row;
                        int x0 = block_id - y0 * blocks_per_row;
                        y0 = y0*_FX_WINO_STEP;
                        x0 = x0*_FX_WINO_STEP;
                        int dy1 = H0 - y0;
                        if (dy1 > _FX_WINO_STEP) dy1 = _FX_WINO_STEP;
                        int dx1 = W0 - x0;
                        if (dx1 > _FX_WINO_STEP) dx1 = _FX_WINO_STEP;
                        assert(dx1 > 0 && dy1 > 0);
                        bool partial = activ || dy1 < _FX_WINO_STEP || dx1 < _FX_WINO_STEP;
                        size_t outofs = (n*K + g*Kg + k)*out_planesize + y0*W0 + x0;
                        int outstep = W0;

                        float* outptr0 = (float*)out + outofs;
                        float* pbptr0 = fusedAddPtr ? fusedAddPtr + outofs : nullptr;
                        float *outptr = outptr0, *bpptr = pbptr0;

                        if (partial)
                        {
                            outptr = outbuf;
                            outstep = _FX_WINO_SIZE;
                            if (pbptr0)
                            {
                                bpptr = outbuf;
                                for (int y = 0; y < dy1; y++)
                                    memcpy(outbuf + y*_FX_WINO_SIZE, pbptr0 + y*W0,
                                           dx1*sizeof(pbptr0[0]));
                            }
                        }
#if CV_TRY_AVX2
                        if (conv->useAVX2)
                            opt_AVX2::_fx_winograd_AtXA_8x8_f32(out_wbuf + ((k - k0)*_FX_WINO_IBLOCK + (block_id - block_id0))*_FX_WINO_AREA, _FX_WINO_SIZE,
                                                                bpptr, outstep, outptr, outstep, biasv, minval, maxval, ifMinMaxAct);
                        else
#endif
                        _fx_winograd_AtXA_8x8_f32(out_wbuf + ((k - k0)*_FX_WINO_IBLOCK + (block_id - block_id0))*_FX_WINO_AREA, _FX_WINO_SIZE,
                                                  bpptr, outstep, outptr, outstep, biasv, minval, maxval, ifMinMaxAct);
                        if (partial)
                        {
                            if (activ)
                                activ->forwardSlice(outptr, outptr, _FX_WINO_SIZE*_FX_WINO_STEP, 0, g*Kg + k, g*Kg + k + 1);
                            for (int y = 0; y < dy1; y++)
                                memcpy(outptr0 + y*W0, outptr + y*_FX_WINO_SIZE,dx1*sizeof(outptr0[0]));
                        }
                    }
                }
            }
        }
    }});
    return 1;
}

#else

int runWinograd63(InputArray _input, InputArray _fusedAddMat, OutputArray _output, const Ptr<FastConv>& conv,
                  int ntasks, float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct)
{
    return 0;
}
#endif
}} // namespace cv::dnn
