// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "convolution.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {

// NEON code work around.
namespace opt_NEON
{

#if CV_NEON && CV_NEON_AARCH64

/* Accumulate */
void winofunc_accum_F32(const float* inwptr, const float* wptr, float* outbuf, int Cg, int iblock,
                            const int winoIblock, const int winoKblock, const int winoAtomF32, const int winoNatomF32)
{
    CV_Assert(winoIblock == 6 && winoKblock == 4 && winoAtomF32 == 4);
    if (iblock > 3)
    {
        for (int atom_id = 0; atom_id < winoNatomF32; atom_id++,
                outbuf += winoAtomF32)
        {
            float32x4_t s00 = vdupq_n_f32(0.f), s01 = s00, s02 = s00, s03 = s00, s04 = s00, s05 = s00;
            float32x4_t s10 = vdupq_n_f32(0.f), s11 = s00, s12 = s00, s13 = s00, s14 = s00, s15 = s00;
            float32x4_t s20 = vdupq_n_f32(0.f), s21 = s00, s22 = s00, s23 = s00, s24 = s00, s25 = s00;
            float32x4_t s30 = vdupq_n_f32(0.f), s31 = s00, s32 = s00, s33 = s00, s34 = s00, s35 = s00;
            for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF32,
                                         wptr += winoKblock*winoAtomF32) {
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
        for (int atom_id = 0; atom_id < winoNatomF32; atom_id++,
                outbuf += winoAtomF32)
        {
            float32x4_t s00 = vdupq_n_f32(0.f), s01 = s00, s02 = s00;
            float32x4_t s10 = vdupq_n_f32(0.f), s11 = s00, s12 = s00;
            float32x4_t s20 = vdupq_n_f32(0.f), s21 = s00, s22 = s00;
            float32x4_t s30 = vdupq_n_f32(0.f), s31 = s00, s32 = s00;
            for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF32,
                                         wptr += winoKblock*winoAtomF32) {
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
}

#undef T4x4
#define T4x4(a, b, c, d, tr0, tr1) \
    tr0 = vtrnq_f32(a, b); \
    tr1 = vtrnq_f32(c, d); \
    a = vcombine_f32(vget_low_f32(tr0.val[0]), vget_low_f32(tr1.val[0])); \
    b = vcombine_f32(vget_low_f32(tr0.val[1]), vget_low_f32(tr1.val[1])); \
    c = vcombine_f32(vget_high_f32(tr0.val[0]), vget_high_f32(tr1.val[0])); \
    d = vcombine_f32(vget_high_f32(tr0.val[1]), vget_high_f32(tr1.val[1]))

/*Input transform*/
void winofunc_BtXB_8x8_F32(const float* inptr, int inpstep,
                          float* outptr, int Cg, const int winoIblock, const int winoAtomF32)
{
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

    const int outstep = winoIblock*winoAtomF32*Cg;

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
}

/*Output transform*/
void winofunc_AtXA_8x8_F32(const float* inptr, int inpstep,
                          float* bpptr, int bpstep, float* outptr, int outstep,
                          float bias, float minval, float maxval, bool ifMinMaxAct)
{
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
}

#endif
}

}} // namespace
