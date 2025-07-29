// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/hal/intrin.hpp"
#include "convolution.hpp"

// === dispatched calls (implemented here)

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

cv::dnn::Winofunc getWinofunc_F32();
cv::dnn::Winofunc getWinofunc_F16();

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::dnn::

// === implementation

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN


#if defined(CV_CPU_COMPILE_AVX) && CV_CPU_COMPILE_AVX || defined(CV_CPU_COMPILE_AVX2) && CV_CPU_COMPILE_AVX2


#if !CV_FMA3 // AVX workaround
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

void impl_accum_F32(const uchar* inwptr_, const uchar* wptr_, uchar* outbuf_, int Cg, int iblock,
                            const int winoIblock, const int winoKblock, const int winoAtomF32, const int winoNatomF32)
{
    const float * inwptr = (float*)inwptr_;
    const float * wptr = (float*)wptr_;
    float * outbuf = (float*)outbuf_;

    CV_Assert(winoIblock == 6 && winoKblock == 4 && winoAtomF32 == 8);
    if (iblock > 3)
    {
        for (int atom_id = 0; atom_id < winoNatomF32; atom_id++,
                outbuf += winoAtomF32)
        {
            __m256 s00 = _mm256_set1_ps(0.f), s01 = s00, s02 = s00, s03 = s00, s04 = s00, s05 = s00;
            __m256 s10 = _mm256_set1_ps(0.f), s11 = s00, s12 = s00, s13 = s00, s14 = s00, s15 = s00;
            __m256 s20 = _mm256_set1_ps(0.f), s21 = s00, s22 = s00, s23 = s00, s24 = s00, s25 = s00;
            __m256 s30 = _mm256_set1_ps(0.f), s31 = s00, s32 = s00, s33 = s00, s34 = s00, s35 = s00;
            for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF32,
                                         wptr += winoKblock*winoAtomF32)
            {
                __m256 w0 = _mm256_load_ps(wptr), w1 = _mm256_load_ps(wptr + 8);
                __m256 w2 = _mm256_load_ps(wptr + 16), w3 = _mm256_load_ps(wptr + 24);
                __m256 x0, x1;
                x0 = _mm256_load_ps(inwptr);
                x1 = _mm256_load_ps(inwptr + 8);
                s00 = _mm256_fmadd_ps(w0, x0, s00);
                s01 = _mm256_fmadd_ps(w0, x1, s01);
                s10 = _mm256_fmadd_ps(w1, x0, s10);
                s11 = _mm256_fmadd_ps(w1, x1, s11);
                s20 = _mm256_fmadd_ps(w2, x0, s20);
                s21 = _mm256_fmadd_ps(w2, x1, s21);
                s30 = _mm256_fmadd_ps(w3, x0, s30);
                s31 = _mm256_fmadd_ps(w3, x1, s31);
                x0 = _mm256_load_ps(inwptr + 16);
                x1 = _mm256_load_ps(inwptr + 24);
                s02 = _mm256_fmadd_ps(w0, x0, s02);
                s03 = _mm256_fmadd_ps(w0, x1, s03);
                s12 = _mm256_fmadd_ps(w1, x0, s12);
                s13 = _mm256_fmadd_ps(w1, x1, s13);
                s22 = _mm256_fmadd_ps(w2, x0, s22);
                s23 = _mm256_fmadd_ps(w2, x1, s23);
                s32 = _mm256_fmadd_ps(w3, x0, s32);
                s33 = _mm256_fmadd_ps(w3, x1, s33);
                x0 = _mm256_load_ps(inwptr + 32);
                x1 = _mm256_load_ps(inwptr + 40);
                s04 = _mm256_fmadd_ps(w0, x0, s04);
                s05 = _mm256_fmadd_ps(w0, x1, s05);
                s14 = _mm256_fmadd_ps(w1, x0, s14);
                s15 = _mm256_fmadd_ps(w1, x1, s15);
                s24 = _mm256_fmadd_ps(w2, x0, s24);
                s25 = _mm256_fmadd_ps(w2, x1, s25);
                s34 = _mm256_fmadd_ps(w3, x0, s34);
                s35 = _mm256_fmadd_ps(w3, x1, s35);
            }

            _mm256_store_ps(outbuf, s00);
            _mm256_store_ps(outbuf + 1*64, s01);
            _mm256_store_ps(outbuf + 2*64, s02);
            _mm256_store_ps(outbuf + 3*64, s03);
            _mm256_store_ps(outbuf + 4*64, s04);
            _mm256_store_ps(outbuf + 5*64, s05);

            _mm256_store_ps(outbuf + 6*64, s10);
            _mm256_store_ps(outbuf + 7*64, s11);
            _mm256_store_ps(outbuf + 8*64, s12);
            _mm256_store_ps(outbuf + 9*64, s13);
            _mm256_store_ps(outbuf + 10*64, s14);
            _mm256_store_ps(outbuf + 11*64, s15);

            _mm256_store_ps(outbuf + 12*64, s20);
            _mm256_store_ps(outbuf + 13*64, s21);
            _mm256_store_ps(outbuf + 14*64, s22);
            _mm256_store_ps(outbuf + 15*64, s23);
            _mm256_store_ps(outbuf + 16*64, s24);
            _mm256_store_ps(outbuf + 17*64, s25);

            _mm256_store_ps(outbuf + 18*64, s30);
            _mm256_store_ps(outbuf + 19*64, s31);
            _mm256_store_ps(outbuf + 20*64, s32);
            _mm256_store_ps(outbuf + 21*64, s33);
            _mm256_store_ps(outbuf + 22*64, s34);
            _mm256_store_ps(outbuf + 23*64, s35);
        }
    }
    else
    {
        for (int atom_id = 0; atom_id < winoNatomF32; atom_id++,
                outbuf += winoAtomF32)
        {
            __m256 s00 = _mm256_set1_ps(0.f), s01 = s00, s02 = s00;
            __m256 s10 = _mm256_set1_ps(0.f), s11 = s00, s12 = s00;
            __m256 s20 = _mm256_set1_ps(0.f), s21 = s00, s22 = s00;
            __m256 s30 = _mm256_set1_ps(0.f), s31 = s00, s32 = s00;
            for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF32,
                                         wptr += winoKblock*winoAtomF32) {
                __m256 w0 = _mm256_load_ps(wptr), w1 = _mm256_load_ps(wptr + 8);
                __m256 w2 = _mm256_load_ps(wptr + 16), w3 = _mm256_load_ps(wptr + 24);
                __m256 x0, x1, x2;
                x0 = _mm256_load_ps(inwptr);
                x1 = _mm256_load_ps(inwptr + 8);
                x2 = _mm256_load_ps(inwptr + 16);
                s00 = _mm256_fmadd_ps(w0, x0, s00);
                s01 = _mm256_fmadd_ps(w0, x1, s01);
                s02 = _mm256_fmadd_ps(w0, x2, s02);
                s10 = _mm256_fmadd_ps(w1, x0, s10);
                s11 = _mm256_fmadd_ps(w1, x1, s11);
                s12 = _mm256_fmadd_ps(w1, x2, s12);
                s20 = _mm256_fmadd_ps(w2, x0, s20);
                s21 = _mm256_fmadd_ps(w2, x1, s21);
                s22 = _mm256_fmadd_ps(w2, x2, s22);
                s30 = _mm256_fmadd_ps(w3, x0, s30);
                s31 = _mm256_fmadd_ps(w3, x1, s31);
                s32 = _mm256_fmadd_ps(w3, x2, s32);
            }

            _mm256_store_ps(outbuf, s00);
            _mm256_store_ps(outbuf + 1*64, s01);
            _mm256_store_ps(outbuf + 2*64, s02);
            _mm256_store_ps(outbuf + 6*64, s10);
            _mm256_store_ps(outbuf + 7*64, s11);
            _mm256_store_ps(outbuf + 8*64, s12);
            _mm256_store_ps(outbuf + 12*64, s20);
            _mm256_store_ps(outbuf + 13*64, s21);
            _mm256_store_ps(outbuf + 14*64, s22);
            _mm256_store_ps(outbuf + 18*64, s30);
            _mm256_store_ps(outbuf + 19*64, s31);
            _mm256_store_ps(outbuf + 20*64, s32);
        }
    }
    _mm256_zeroupper();
}

static inline
void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

/*Input transform*/
void impl_BtXB_8x8_F32(const float* inptr, int inpstep,
                               uchar* outptr_, int Cg, const int winoIblock, const int winoAtomF32)
{
    float * outptr = (float*)outptr_;

    __m256 x00 = _mm256_loadu_ps(inptr);
    __m256 x10 = _mm256_loadu_ps(inptr + inpstep);
    __m256 x20 = _mm256_loadu_ps(inptr + inpstep*2);
    __m256 x30 = _mm256_loadu_ps(inptr + inpstep*3);
    __m256 x40 = _mm256_loadu_ps(inptr + inpstep*4);
    __m256 x50 = _mm256_loadu_ps(inptr + inpstep*5);
    __m256 x60 = _mm256_loadu_ps(inptr + inpstep*6);
    __m256 x70 = _mm256_loadu_ps(inptr + inpstep*7);

    __m256 z00, z10, z20, z30, z40, z50, z60, z70;

    {
        /* Y[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*X */
        /* Y[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*X */
        __m256 q5_25 = _mm256_set1_ps(5.25f), t00, t10;
        t00 = _mm256_sub_ps(x40, x20);
        t10 = _mm256_sub_ps(x30, x50);

        __m256 y00 = _mm256_fmadd_ps(t00, q5_25, _mm256_sub_ps(x00, x60));
        __m256 y70 = _mm256_fmadd_ps(t10, q5_25, _mm256_sub_ps(x70, x10));

        /* Y[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*X */
        /* Y[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*X */
        __m256 qm4_25 = _mm256_set1_ps(-4.25f);
        t00 = _mm256_fmadd_ps(x30, qm4_25, _mm256_add_ps(x10, x50));
        t10 = _mm256_fmadd_ps(x40, qm4_25, _mm256_add_ps(x20, x60));

        __m256 y10 = _mm256_add_ps(t00, t10);
        __m256 y20 = _mm256_sub_ps(t10, t00);

        /* Y[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*X */
        /* Y[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*X */
        __m256 q0_5 = _mm256_set1_ps(0.5f), q0_25 = _mm256_set1_ps(0.25f);
        __m256 qm2_5 = _mm256_set1_ps(-2.5f), qm1_25 = _mm256_set1_ps(-1.25f);
        t00 = _mm256_fmadd_ps(x10, q0_5, _mm256_add_ps(x50, x50));
        t10 = _mm256_fmadd_ps(x20, q0_25, x60);
        t00 = _mm256_fmadd_ps(x30, qm2_5, t00);
        t10 = _mm256_fmadd_ps(x40, qm1_25, t10);

        __m256 y30 = _mm256_add_ps(t00, t10);
        __m256 y40 = _mm256_sub_ps(t10, t00);

        /* Y[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*X */
        /* Y[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*X */
        __m256 q4 = _mm256_set1_ps(4.f), qm5 = _mm256_set1_ps(-5.f);
        t00 = _mm256_fmadd_ps(x50, q0_5, _mm256_add_ps(x10, x10));
        t10 = _mm256_fmadd_ps(x20, q4   , x60);
        t00 = _mm256_fmadd_ps(x30, qm2_5, t00);
        t10 = _mm256_fmadd_ps(x40, qm5  , t10);

        __m256 y50 = _mm256_add_ps(t00, t10);
        __m256 y60 = _mm256_sub_ps(t10, t00);

        /* transpose 8x8 matrix in-place with some renumeration of the elements: */
        transpose8_ps(y00, y10, y20, y30, y40, y50, y60, y70);

        /* Z[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*Y */
        /* Z[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*Y */
        t00 = _mm256_sub_ps(y40, y20);
        t10 = _mm256_sub_ps(y30, y50);
        z00 = _mm256_fmadd_ps(t00, q5_25, _mm256_sub_ps(y00, y60));
        z70 = _mm256_fmadd_ps(t10, q5_25, _mm256_sub_ps(y70, y10));

        /* Z[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*Y */
        /* Z[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*Y */
        t00 = _mm256_fmadd_ps(y30, qm4_25, _mm256_add_ps(y10, y50));
        t10 = _mm256_fmadd_ps(y40, qm4_25, _mm256_add_ps(y20, y60));
        z10 = _mm256_add_ps(t00, t10);
        z20 = _mm256_sub_ps(t10, t00);

        /* Z[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*Y */
        /* Z[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*Y */
        t00 = _mm256_fmadd_ps(y10, q0_5, _mm256_add_ps(y50, y50));
        t10 = _mm256_fmadd_ps(y20, q0_25, y60);
        t00 = _mm256_fmadd_ps(y30, qm2_5, t00);
        t10 = _mm256_fmadd_ps(y40, qm1_25, t10);

        z30 = _mm256_add_ps(t00, t10);
        z40 = _mm256_sub_ps(t10, t00);

        /* Z[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*Y */
        /* Z[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*Y */
        t00 = _mm256_fmadd_ps(y50, q0_5, _mm256_add_ps(y10, y10));
        t10 = _mm256_fmadd_ps(y20, q4, y60);
        t00 = _mm256_fmadd_ps(y30, qm2_5, t00);
        t10 = _mm256_fmadd_ps(y40, qm5, t10);

        z50 = _mm256_add_ps(t00, t10);
        z60 = _mm256_sub_ps(t10, t00);
    }

    const int outstep = winoIblock*winoAtomF32*Cg;

    _mm256_storeu_ps(outptr, z00);
    _mm256_storeu_ps(outptr + outstep, z10);
    _mm256_storeu_ps(outptr + outstep*2, z20);
    _mm256_storeu_ps(outptr + outstep*3, z30);
    _mm256_storeu_ps(outptr + outstep*4, z40);
    _mm256_storeu_ps(outptr + outstep*5, z50);
    _mm256_storeu_ps(outptr + outstep*6, z60);
    _mm256_storeu_ps(outptr + outstep*7, z70);
    _mm256_zeroupper();
}

#define STORE6_ELE_FROM_16(ptr, z00, lowM, highM) \
    lowM = _mm256_castps256_ps128(z00); \
    highM = _mm256_extractf128_ps(z00, 1); \
    _mm_storeu_ps(ptr, lowM); \
    _mm_storel_epi64((__m128i*)(ptr + 4), _mm_castps_si128(highM))

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
*/
void impl_AtXA_8x8_F32(const uchar* inptr_, int inpstep,
                          float* bpptr, int bpstep, float* outptr, int outstep,
                          float bias, float minval, float maxval, bool ifMinMaxAct)
{
    const float * inptr = (float*)inptr_;

    __m256 x00 = _mm256_load_ps(inptr);
    __m256 x10 = _mm256_load_ps(inptr + inpstep);
    __m256 x20 = _mm256_load_ps(inptr + inpstep*2);
    __m256 x30 = _mm256_load_ps(inptr + inpstep*3);
    __m256 x40 = _mm256_load_ps(inptr + inpstep*4);
    __m256 x50 = _mm256_load_ps(inptr + inpstep*5);
    __m256 x60 = _mm256_load_ps(inptr + inpstep*6);
    __m256 x70 = _mm256_load_ps(inptr + inpstep*7);
    __m256 z00, z10, z20, z30, z40, z50;

    {
        __m256 s12_0, s34_0, s56_0;
        s12_0 = _mm256_add_ps(x10, x20);
        s34_0 = _mm256_add_ps(x30, x40);
        s56_0 = _mm256_add_ps(x50, x60);

        __m256 y00 = _mm256_add_ps(x00, _mm256_add_ps(s12_0, _mm256_add_ps(s34_0, s56_0)));
        __m256 y20 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(0.25f), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(4.0f), s12_0));
        __m256 y40 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(1.f/16), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(16.0f), s12_0));

        s12_0 = _mm256_sub_ps(x10, x20);
        s34_0 = _mm256_sub_ps(x30, x40);
        s56_0 = _mm256_sub_ps(x50, x60);
        __m256 y50 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(1.f/32), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(32.f), _mm256_add_ps(x70, s12_0)));
        __m256 y10 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(0.5f), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(2.f), s12_0));
        __m256 y30 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(0.125f), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(8.f), s12_0));
        __m256 y60 = _mm256_set1_ps(0.f), y70 = y60;

        /* transpose 8x8 matrix in-place with some renumeration of the elements: */

        transpose8_ps(y00, y10, y20, y30, y40, y50, y60, y70);

        s12_0 = _mm256_add_ps(y10, y20);
        s34_0 = _mm256_add_ps(y30, y40);
        s56_0 = _mm256_add_ps(y50, y60);

        z00 = _mm256_add_ps(y00, _mm256_add_ps(s12_0, _mm256_add_ps(s34_0, s56_0)));
        z20 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(0.25f), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(4.0f), s12_0));
        z40 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(1.f/16), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(16.0f), s12_0));

        s12_0 = _mm256_sub_ps(y10, y20);
        s34_0 = _mm256_sub_ps(y30, y40);
        s56_0 = _mm256_sub_ps(y50, y60);

        z50 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(1.f/32), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(32.0f), _mm256_add_ps(y70, s12_0)));
        z10 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(0.5f), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(2.0f), s12_0));
        z30 = _mm256_fmadd_ps(s56_0, _mm256_set1_ps(0.125f), _mm256_fmadd_ps(s34_0, _mm256_set1_ps(8.0f), s12_0));

        __m256 vbias = _mm256_set1_ps(bias);
        z00 = _mm256_add_ps(vbias, z00);
        z10 = _mm256_add_ps(vbias, z10);
        z20 = _mm256_add_ps(vbias, z20);
        z30 = _mm256_add_ps(vbias, z30);
        z40 = _mm256_add_ps(vbias, z40);
        z50 = _mm256_add_ps(vbias, z50);
    }

    if (bpptr)
    {
        z00 = _mm256_add_ps(z00, _mm256_loadu_ps(bpptr));
        z10 = _mm256_add_ps(z10, _mm256_loadu_ps(bpptr + bpstep));
        z20 = _mm256_add_ps(z20, _mm256_loadu_ps(bpptr + bpstep*2));
        z30 = _mm256_add_ps(z30, _mm256_loadu_ps(bpptr + bpstep*3));
        z40 = _mm256_add_ps(z40, _mm256_loadu_ps(bpptr + bpstep*4));
        z50 = _mm256_add_ps(z50, _mm256_loadu_ps(bpptr + bpstep*5));
    }

    if (ifMinMaxAct)
    {
        __m256 vmax = _mm256_set1_ps(maxval);
        __m256 vmin = _mm256_set1_ps(minval);

        z00 = _mm256_min_ps(_mm256_max_ps(z00, vmin), vmax);
        z10 = _mm256_min_ps(_mm256_max_ps(z10, vmin), vmax);
        z20 = _mm256_min_ps(_mm256_max_ps(z20, vmin), vmax);
        z30 = _mm256_min_ps(_mm256_max_ps(z30, vmin), vmax);
        z40 = _mm256_min_ps(_mm256_max_ps(z40, vmin), vmax);
        z50 = _mm256_min_ps(_mm256_max_ps(z50, vmin), vmax);
    }

    __m128 lowM, highM;
    STORE6_ELE_FROM_16(outptr, z00, lowM, highM);
    STORE6_ELE_FROM_16(outptr + outstep, z10, lowM, highM);
    STORE6_ELE_FROM_16(outptr + outstep * 2, z20, lowM, highM);
    STORE6_ELE_FROM_16(outptr + outstep * 3, z30, lowM, highM);
    STORE6_ELE_FROM_16(outptr + outstep * 4, z40, lowM, highM);
    STORE6_ELE_FROM_16(outptr + outstep * 5, z50, lowM, highM);
    _mm256_zeroupper();
}

cv::dnn::Winofunc getWinofunc_F32()
{
    return {&impl_accum_F32, &impl_BtXB_8x8_F32, &impl_AtXA_8x8_F32, 6, 8, 4};
}


// end of AVX/AVX2
#elif defined(CV_CPU_COMPILE_NEON) && CV_CPU_COMPILE_NEON && defined(CV_NEON_AARCH64) && CV_NEON_AARCH64


/* Accumulate */
void impl_accum_F32(const uchar* inwptr_, const uchar* wptr_, uchar* outbuf_, int Cg, int iblock,
                            const int winoIblock, const int winoKblock, const int winoAtomF32, const int winoNatomF32)
{
    const float * inwptr = (float*)inwptr_;
    const float * wptr = (float*)wptr_;
    float * outbuf = (float*)outbuf_;

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
void impl_BtXB_8x8_F32(const float* inptr, int inpstep,
                          uchar* outptr_, int Cg, const int winoIblock, const int winoAtomF32)
{
    float * outptr = (float*)outptr_;

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
void impl_AtXA_8x8_F32(const uchar* inptr_, int inpstep,
                          float* bpptr, int bpstep, float* outptr, int outstep,
                          float bias, float minval, float maxval, bool ifMinMaxAct)
{
    const float * inptr = (float*)inptr_;

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

cv::dnn::Winofunc getWinofunc_F32()
{
    return {&impl_accum_F32, &impl_BtXB_8x8_F32, &impl_AtXA_8x8_F32, 6, 4, 4};
}


// end of NEON/AArch64
#elif CV_SIMD128


void impl_accum_F32(const uchar* inwptr_, const uchar* wptr_, uchar* outbuf_, int Cg, int iblock,
                            const int winoIblock, const int winoKblock, const int winoAtomF32, const int winoNatomF32)
{
    const float * inwptr = (float*)inwptr_;
    const float * wptr = (float*)wptr_;
    float * outbuf = (float*)outbuf_;
#if 1
    CV_Assert(winoIblock == 3 && winoKblock == 4 && winoAtomF32 == 4);
    for (int atom_id = 0; atom_id < winoNatomF32; atom_id++,
            outbuf += winoAtomF32)
    {
        v_float32x4 s00 = v_setzero_f32(), s01 = s00, s02 = s00;
        v_float32x4 s10 = v_setzero_f32(), s11 = s00, s12 = s00;
        v_float32x4 s20 = v_setzero_f32(), s21 = s00, s22 = s00;
        v_float32x4 s30 = v_setzero_f32(), s31 = s00, s32 = s00;

        for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF32,
                                     wptr += winoKblock*winoAtomF32)
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
    // Naive C++ code, the code should never be run here.
    for (int atom_id = 0; atom_id < winoNatomF32;
                atom_id++, outbuf += winoAtomF32)
    {
        float sumbuf[winoIblock*winoKblock*winoAtomF32];
        memset(sumbuf, 0, sizeof(sumbuf));
        for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF32,
                                     wptr += winoKblock*winoAtomF32)
        {
            for (int i = 0; i < winoKblock; i++)
            {
                for (int j = 0; j < winoIblock; j++)
                {
                    int i_ = i*winoAtomF32;
                    int j_ = j*winoAtomF32;
                    int ij_ = i_*winoIblock + j_;
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
        for (int ij = 0; ij < winoKblock*winoIblock; ij++)
        {
            int ij_ = ij*winoAtomF32;
            int ij_out = ij*CONV_WINO_AREA;
            outbuf[ij_out + 0] = sumbuf[ij_ + 0];
            outbuf[ij_out + 1] = sumbuf[ij_ + 1];
            outbuf[ij_out + 2] = sumbuf[ij_ + 2];
            outbuf[ij_out + 3] = sumbuf[ij_ + 3];
        }
    }
#endif
}

/*Input transform*/
void impl_BtXB_8x8_F32(const float* inptr, int inpstep, uchar* outptr_, int Cg, const int winoIblock, const int winoAtomF32)
{
    float * outptr = (float*)outptr_;

    CV_Assert(winoIblock == 3 && winoAtomF32 == 4);
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
        t00 = v_sub(x40, x20);
        t01 = v_sub(x41, x21);
        t10 = v_sub(x30, x50);
        t11 = v_sub(x31, x51);
        v_float32x4 y00 = v_fma(t00, q5_25, v_sub(x00, x60));
        v_float32x4 y01 = v_fma(t01, q5_25, v_sub(x01, x61));
        v_float32x4 y70 = v_fma(t10, q5_25, v_sub(x70, x10));
        v_float32x4 y71 = v_fma(t11, q5_25, v_sub(x71, x11));

        /* Y[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*X */
        /* Y[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*X */
        v_float32x4 qm4_25 = v_setall_f32(-4.25f);
        t00 = v_fma(x30, qm4_25, v_add(x10, x50));
        t01 = v_fma(x31, qm4_25, v_add(x11, x51));
        t10 = v_fma(x40, qm4_25, v_add(x20, x60));
        t11 = v_fma(x41, qm4_25, v_add(x21, x61));

        v_float32x4 y10 = v_add(t00, t10), y11 = v_add(t01, t11);
        v_float32x4 y20 = v_sub(t10, t00), y21 = v_sub(t11, t01);

        /* Y[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*X */
        /* Y[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*X */
        v_float32x4 q0_5 = v_setall_f32(0.5f), q0_25 = v_setall_f32(0.25f);
        v_float32x4 qm2_5 = v_setall_f32(-2.5f), qm1_25 = v_setall_f32(-1.25f);
        t00 = v_fma(x10, q0_5, v_add(x50, x50));
        t01 = v_fma(x11, q0_5, v_add(x51, x51));
        t10 = v_fma(x20, q0_25, x60);
        t11 = v_fma(x21, q0_25, x61);
        t00 = v_fma(x30, qm2_5, t00);
        t01 = v_fma(x31, qm2_5, t01);
        t10 = v_fma(x40, qm1_25, t10);
        t11 = v_fma(x41, qm1_25, t11);

        v_float32x4 y30 = v_add(t00, t10), y31 = v_add(t01, t11);
        v_float32x4 y40 = v_sub(t10, t00), y41 = v_sub(t11, t01);

        /* Y[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*X */
        /* Y[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*X */
        v_float32x4 q4 = v_setall_f32(4.f), qm5 = v_setall_f32(-5.f);
        t00 = v_fma(x50, q0_5, v_add(x10, x10));
        t01 = v_fma(x51, q0_5, v_add(x11, x11));
        t10 = v_fma(x20, q4   , x60);
        t11 = v_fma(x21, q4   , x61);
        t00 = v_fma(x30, qm2_5, t00);
        t01 = v_fma(x31, qm2_5, t01);
        t10 = v_fma(x40, qm5  , t10);
        t11 = v_fma(x41, qm5  , t11);

        v_float32x4 y50 = v_add(t00, t10), y51 = v_add(t01, t11);
        v_float32x4 y60 = v_sub(t10, t00), y61 = v_sub(t11, t01);

        /* transpose 8x8 matrix with v_transpose4x4 */

        v_float32x4 y000, y100, y200, y300, y010, y110, y210, y310, y400, y500, y600, y700, y410, y510, y610, y710;
        v_transpose4x4(y00, y10, y20, y30, y000, y100, y200, y300);
        v_transpose4x4(y01, y11, y21, y31, y010, y110, y210, y310);
        v_transpose4x4(y40, y50, y60, y70, y400, y500, y600, y700);
        v_transpose4x4(y41, y51, y61, y71, y410, y510, y610, y710);

        /* Z[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*Y */
        /* Z[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*Y */
        t00 = v_sub(y010, y200);
        t01 = v_sub(y410, y600);
        t10 = v_sub(y300, y110);
        t11 = v_sub(y700, y510);
        z00 = v_fma(t00, q5_25, v_sub(y000, y210));
        z01 = v_fma(t01, q5_25, v_sub(y400, y610));
        z70 = v_fma(t10, q5_25, v_sub(y310, y100));
        z71 = v_fma(t11, q5_25, v_sub(y710, y500));

        /* Z[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*Y */
        /* Z[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*Y */
        t00 = v_fma(y300, qm4_25, v_add(y100, y110));
        t01 = v_fma(y700, qm4_25, v_add(y500, y510));
        t10 = v_fma(y010, qm4_25, v_add(y200, y210));
        t11 = v_fma(y410, qm4_25, v_add(y600, y610));

        z10 = v_add(t00, t10); z11 = v_add(t01, t11);
        z20 = v_sub(t10, t00); z21 = v_sub(t11, t01);

        /* Z[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*Y */
        /* Z[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*Y */
        t00 = v_fma(y100, q0_5, v_add(y110, y110));
        t01 = v_fma(y500, q0_5, v_add(y510, y510));
        t10 = v_fma(y200, q0_25, y210);
        t11 = v_fma(y600, q0_25, y610);
        t00 = v_fma(y300, qm2_5, t00);
        t01 = v_fma(y700, qm2_5, t01);
        t10 = v_fma(y010, qm1_25, t10);
        t11 = v_fma(y410, qm1_25, t11);

        z30 = v_add(t00, t10); z31 = v_add(t01, t11);
        z40 = v_sub(t10, t00); z41 = v_sub(t11, t01);

        /* Z[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*Y */
        /* Z[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*Y */
        t00 = v_fma(y110, q0_5, v_add(y100, y100));
        t01 = v_fma(y510, q0_5, v_add(y500, y500));
        t10 = v_fma(y200, q4, y210);
        t11 = v_fma(y600, q4, y610);
        t00 = v_fma(y300, qm2_5, t00);
        t01 = v_fma(y700, qm2_5, t01);
        t10 = v_fma(y010, qm5, t10);
        t11 = v_fma(y410, qm5, t11);

        z50 = v_add(t00, t10); z51 = v_add(t01, t11);
        z60 = v_sub(t10, t00); z61 = v_sub(t11, t01);
    }

    const int outstep = winoIblock*winoAtomF32*Cg;

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
}

/*Output transform*/
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

    That is, after both forward and then inverse transformation,
    we get non-transposed result.
    Of course, for the correct work of Winograd-based convolution,
    the Winograd-transformed weights should also be transposed.
    init_conv() (see OpConv.fx) takes care of that.
*/
void impl_AtXA_8x8_F32(const uchar* inptr_, int inpstep,
                          float* bpptr, int bpstep, float* outptr, int outstep,
                          float bias, float minval, float maxval, bool ifMinMaxAct)
{
    const float * inptr = (float*)inptr_;

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
        s12_0 = v_add(x10, x20); s12_1 = v_add(x11, x21);
        s34_0 = v_add(x30, x40); s34_1 = v_add(x31, x41);
        s56_0 = v_add(x50, x60); s56_1 = v_add(x51, x61);

        v_float32x4 y00 = v_add(v_add(v_add(x00, s12_0), s34_0), s56_0);
        v_float32x4 y01 = v_add(v_add(v_add(x01, s12_1), s34_1), s56_1);

        v_float32x4 a0 = v_setall_f32(0.25f), a1 = v_setall_f32(4.0f);
        v_float32x4 y20 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y21 = v_fma(s56_1, a0 ,v_fma(s34_1, a1, s12_1) );

        a0 = v_setall_f32(1.f/16), a1 = v_setall_f32(16.0f);
        v_float32x4 y40 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y41 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        s12_0 = v_sub(x10, x20); s12_1 = v_sub(x11, x21);
        s34_0 = v_sub(x30, x40); s34_1 = v_sub(x31, x41);
        s56_0 = v_sub(x50, x60); s56_1 = v_sub(x51, x61);

        a0 = v_setall_f32(1.f/32), a1 = v_setall_f32(32.f);
        v_float32x4 y50 = v_fma(s56_0, a0, v_fma(s34_0, a1, v_add(x70, s12_0)));
        v_float32x4 y51 = v_fma(s56_1, a0, v_fma(s34_1, a1, v_add(x71, s12_1)));

        a0 = v_setall_f32(0.5f), a1 = v_setall_f32(2.f);
        v_float32x4 y10 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y11 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(0.125f), a1 = v_setall_f32(8.f);
        v_float32x4 y30 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y31 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        v_float32x4 y60 = v_setall_f32(0.f), y61 = y60, y70 = y60, y71 = y60;

        /* transpose 8x8 matrix with v_transpose4x4 */

        v_float32x4 y000, y100, y200, y300, y010, y110, y210, y310, y400, y500, y600, y700, y410, y510, y610, y710;
        v_transpose4x4(y00, y10, y20, y30, y000, y100, y200, y300);
        v_transpose4x4(y01, y11, y21, y31, y010, y110, y210, y310);
        v_transpose4x4(y40, y50, y60, y70, y400, y500, y600, y700);
        v_transpose4x4(y41, y51, y61, y71, y410, y510, y610, y710);

        s12_0 = v_add(y100, y200); s12_1 = v_add(y500, y600);
        s34_0 = v_add(y300, y010); s34_1 = v_add(y700, y410);
        s56_0 = v_add(y110, y210); s56_1 = v_add(y510, y610);

        z00 = v_add(v_add(v_add(y000, s12_0), s34_0), s56_0);
        z01 = v_add(v_add(v_add(y400, s12_1), s34_1), s56_1);

        a0 = v_setall_f32(0.25f), a1 = v_setall_f32(4.0f);
        z20 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z21 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(1.f/16), a1 = v_setall_f32(16.0f);
        z40 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z41 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        s12_0 = v_sub(y100, y200); s12_1 = v_sub(y500, y600);
        s34_0 = v_sub(y300, y010); s34_1 = v_sub(y700, y410);
        s56_0 = v_sub(y110, y210); s56_1 = v_sub(y510, y610);

        a0 = v_setall_f32(1.f/32), a1 = v_setall_f32(32.0f);
        z50 = v_fma(s56_0, a0, v_fma(s34_0, a1, v_add(y310, s12_0)));
        z51 = v_fma(s56_1, a0, v_fma(s34_1, a1, v_add(y710, s12_1)));
        a0 = v_setall_f32(0.5f), a1 = v_setall_f32(2.0f);
        z10 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z11 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(0.125f), a1 = v_setall_f32(8.0f);
        z30 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z31 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        v_float32x4 vbias = v_setall_f32(bias);
        z00 = v_add(z00, vbias);
        z01 = v_add(z01, vbias);
        z10 = v_add(z10, vbias);
        z11 = v_add(z11, vbias);
        z20 = v_add(z20, vbias);
        z21 = v_add(z21, vbias);
        z30 = v_add(z30, vbias);
        z31 = v_add(z31, vbias);
        z40 = v_add(z40, vbias);
        z41 = v_add(z41, vbias);
        z50 = v_add(z50, vbias);
        z51 = v_add(z51, vbias);
    }

    if (bpptr)
    {
        z00 = v_add(z00, v_load(bpptr));
        z01 = v_add(z01, v_load_low(bpptr + 4));
        z10 = v_add(z10, v_load(bpptr + bpstep));
        z11 = v_add(z11, v_load_low(bpptr + bpstep + 4));
        z20 = v_add(z20, v_load(bpptr + bpstep * 2));
        z21 = v_add(z21, v_load_low(bpptr + bpstep * 2 + 4));
        z30 = v_add(z30, v_load(bpptr + bpstep * 3));
        z31 = v_add(z31, v_load_low(bpptr + bpstep * 3 + 4));
        z40 = v_add(z40, v_load(bpptr + bpstep * 4));
        z41 = v_add(z41, v_load_low(bpptr + bpstep * 4 + 4));
        z50 = v_add(z50, v_load(bpptr + bpstep * 5));
        z51 = v_add(z51, v_load_low(bpptr + bpstep * 5 + 4));
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
}

cv::dnn::Winofunc getWinofunc_F32()
{
    return {&impl_accum_F32, &impl_BtXB_8x8_F32, &impl_AtXA_8x8_F32, 3, 4, 4};
}


// end of CV_SIMD128
#else


cv::dnn::Winofunc getWinofunc_F32()
{
    return cv::dnn::Winofunc::empty();
}


// end of fallback
#endif

//==============================================================================

// FP16, currently, only ARMv8 may support it
#if defined(CV_CPU_COMPILE_NEON_FP16) && CV_CPU_COMPILE_NEON_FP16

#undef T4x4
#define T4x4(a, b, c, d, tr0, tr1) \
    tr0 = vtrnq_f32(a, b); \
    tr1 = vtrnq_f32(c, d); \
    a = vcombine_f32(vget_low_f32(tr0.val[0]), vget_low_f32(tr1.val[0])); \
    b = vcombine_f32(vget_low_f32(tr0.val[1]), vget_low_f32(tr1.val[1])); \
    c = vcombine_f32(vget_high_f32(tr0.val[0]), vget_high_f32(tr1.val[0])); \
    d = vcombine_f32(vget_high_f32(tr0.val[1]), vget_high_f32(tr1.val[1]))

/* Accumulate */
void impl_accum_F16(const uchar* inwptr_, const uchar* wptr_, uchar* outbuf_, int Cg, int iblock,
                        const int winoIblock, const int winoKblock, const int winoAtomF16, const int winoNatomF16)
{
    const __fp16* inwptr = (const __fp16*)inwptr_;
    const __fp16* wptr = (const __fp16*)wptr_;
    __fp16* outbuf = (__fp16*)outbuf_;

    CV_Assert(winoIblock == 6 && winoKblock == 4 && winoAtomF16 == 8);

    if (iblock > 3)
    {
        for (int atom_id = 0; atom_id < winoNatomF16; atom_id++, outbuf += winoAtomF16)
        {
            float16x8_t s00 = vdupq_n_f16(0.f), s01 = s00, s02 = s00, s03 = s00, s04 = s00, s05 = s00;
            float16x8_t s10 = vdupq_n_f16(0.f), s11 = s00, s12 = s00, s13 = s00, s14 = s00, s15 = s00;
            float16x8_t s20 = vdupq_n_f16(0.f), s21 = s00, s22 = s00, s23 = s00, s24 = s00, s25 = s00;
            float16x8_t s30 = vdupq_n_f16(0.f), s31 = s00, s32 = s00, s33 = s00, s34 = s00, s35 = s00;

            for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF16,
                                         wptr += winoKblock*winoAtomF16)
            {
                float16x8_t w0 = vld1q_f16(wptr), w1 = vld1q_f16(wptr + 8);
                float16x8_t w2 = vld1q_f16(wptr + 16), w3 = vld1q_f16(wptr + 24);

                float16x8_t x0, x1, x2;
                x0 = vld1q_f16(inwptr);
                x1 = vld1q_f16(inwptr + 8);
                x2 = vld1q_f16(inwptr + 16);

                s00 = vfmaq_f16(s00, w0, x0);
                s01 = vfmaq_f16(s01, w0, x1);
                s02 = vfmaq_f16(s02, w0, x2);

                s10 = vfmaq_f16(s10, w1, x0);
                s11 = vfmaq_f16(s11, w1, x1);
                s12 = vfmaq_f16(s12, w1, x2);

                s20 = vfmaq_f16(s20, w2, x0);
                s21 = vfmaq_f16(s21, w2, x1);
                s22 = vfmaq_f16(s22, w2, x2);

                s30 = vfmaq_f16(s30, w3, x0);
                s31 = vfmaq_f16(s31, w3, x1);
                s32 = vfmaq_f16(s32, w3, x2);

                x0 = vld1q_f16(inwptr + 24);
                x1 = vld1q_f16(inwptr + 32);
                x2 = vld1q_f16(inwptr + 40);

                s03 = vfmaq_f16(s03, w0, x0);
                s04 = vfmaq_f16(s04, w0, x1);
                s05 = vfmaq_f16(s05, w0, x2);

                s13 = vfmaq_f16(s13, w1, x0);
                s14 = vfmaq_f16(s14, w1, x1);
                s15 = vfmaq_f16(s15, w1, x2);

                s23 = vfmaq_f16(s23, w2, x0);
                s24 = vfmaq_f16(s24, w2, x1);
                s25 = vfmaq_f16(s25, w2, x2);

                s33 = vfmaq_f16(s33, w3, x0);
                s34 = vfmaq_f16(s34, w3, x1);
                s35 = vfmaq_f16(s35, w3, x2);
            }

            vst1q_f16(outbuf, s00);
            vst1q_f16(outbuf + 1*64, s01);
            vst1q_f16(outbuf + 2*64, s02);
            vst1q_f16(outbuf + 3*64, s03);
            vst1q_f16(outbuf + 4*64, s04);
            vst1q_f16(outbuf + 5*64, s05);

            vst1q_f16(outbuf + 6*64, s10);
            vst1q_f16(outbuf + 7*64, s11);
            vst1q_f16(outbuf + 8*64, s12);
            vst1q_f16(outbuf + 9*64, s13);
            vst1q_f16(outbuf + 10*64, s14);
            vst1q_f16(outbuf + 11*64, s15);

            vst1q_f16(outbuf + 12*64, s20);
            vst1q_f16(outbuf + 13*64, s21);
            vst1q_f16(outbuf + 14*64, s22);
            vst1q_f16(outbuf + 15*64, s23);
            vst1q_f16(outbuf + 16*64, s24);
            vst1q_f16(outbuf + 17*64, s25);

            vst1q_f16(outbuf + 18*64, s30);
            vst1q_f16(outbuf + 19*64, s31);
            vst1q_f16(outbuf + 20*64, s32);
            vst1q_f16(outbuf + 21*64, s33);
            vst1q_f16(outbuf + 22*64, s34);
            vst1q_f16(outbuf + 23*64, s35);
        }
    }
    else
    {
        for (int atom_id = 0; atom_id < winoNatomF16; atom_id++,
                outbuf += winoAtomF16)
        {
            float16x8_t s00 = vdupq_n_f16(0.f), s01 = s00, s02 = s00;
            float16x8_t s10 = vdupq_n_f16(0.f), s11 = s00, s12 = s00;
            float16x8_t s20 = vdupq_n_f16(0.f), s21 = s00, s22 = s00;
            float16x8_t s30 = vdupq_n_f16(0.f), s31 = s00, s32 = s00;

            for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF16,
                                         wptr += winoKblock*winoAtomF16)
            {
                float16x8_t w0 = vld1q_f16(wptr), w1 = vld1q_f16(wptr + 8);
                float16x8_t w2 = vld1q_f16(wptr + 16), w3 = vld1q_f16(wptr + 24);
                float16x8_t x0, x1, x2;

                x0 = vld1q_f16(inwptr);
                x1 = vld1q_f16(inwptr + 8);
                x2 = vld1q_f16(inwptr + 16);

                s00 = vfmaq_f16(s00, w0, x0);
                s01 = vfmaq_f16(s01, w0, x1);
                s02 = vfmaq_f16(s02, w0, x2);

                s10 = vfmaq_f16(s10, w1, x0);
                s11 = vfmaq_f16(s11, w1, x1);
                s12 = vfmaq_f16(s12, w1, x2);

                s20 = vfmaq_f16(s20, w2, x0);
                s21 = vfmaq_f16(s21, w2, x1);
                s22 = vfmaq_f16(s22, w2, x2);

                s30 = vfmaq_f16(s30, w3, x0);
                s31 = vfmaq_f16(s31, w3, x1);
                s32 = vfmaq_f16(s32, w3, x2);
            }

            vst1q_f16(outbuf, s00);
            vst1q_f16(outbuf + 1*64, s01);
            vst1q_f16(outbuf + 2*64, s02);

            vst1q_f16(outbuf + 6*64, s10);
            vst1q_f16(outbuf + 7*64, s11);
            vst1q_f16(outbuf + 8*64, s12);

            vst1q_f16(outbuf + 12*64, s20);
            vst1q_f16(outbuf + 13*64, s21);
            vst1q_f16(outbuf + 14*64, s22);

            vst1q_f16(outbuf + 18*64, s30);
            vst1q_f16(outbuf + 19*64, s31);
            vst1q_f16(outbuf + 20*64, s32);
        }
    }
}

/*Input transform*/
//NOTE: Since we don't have the fully fp16 support. Current work around is that we need packing the data and
// convert it to FP16 in input transform stage. And at output transform stage we will convert it back to FP32.
void impl_BtXB_8x8_F16(const float * inptr, int inpstep,
                           uchar * outptr_, int Cg, const int winoIblock, const int winoAtomF16)
{
    __fp16* outptr = (__fp16*)outptr_;

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
        // Y[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*X
        // Y[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*X
        float32x4_t q5_25 = vdupq_n_f32(5.25f), t00, t01, t10, t11;
        t00 = vsubq_f32(x40, x20);
        t01 = vsubq_f32(x41, x21);
        t10 = vsubq_f32(x30, x50);
        t11 = vsubq_f32(x31, x51);
        float32x4_t y00 = vfmaq_f32(vsubq_f32(x00, x60), t00, q5_25);
        float32x4_t y01 = vfmaq_f32(vsubq_f32(x01, x61), t01, q5_25);
        float32x4_t y70 = vfmaq_f32(vsubq_f32(x70, x10), t10, q5_25);
        float32x4_t y71 = vfmaq_f32(vsubq_f32(x71, x11), t11, q5_25);

        // Y[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*X
        // Y[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*X
        float32x4_t qm4_25 = vdupq_n_f32(-4.25f);
        t00 = vfmaq_f32(vaddq_f32(x10, x50), x30, qm4_25);
        t01 = vfmaq_f32(vaddq_f32(x11, x51), x31, qm4_25);
        t10 = vfmaq_f32(vaddq_f32(x20, x60), x40, qm4_25);
        t11 = vfmaq_f32(vaddq_f32(x21, x61), x41, qm4_25);

        float32x4_t y10 = vaddq_f32(t00, t10), y11 = vaddq_f32(t01, t11);
        float32x4_t y20 = vsubq_f32(t10, t00), y21 = vsubq_f32(t11, t01);

        // Y[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*X
        // Y[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*X
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

        // Y[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*X
        // Y[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*X
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

        // transpose 8x8 matrix in-place with some renumeration of the elements:
        // Y:
        //        y00 y01
        //        y10 y11
        //        ...
        //        y70 y71
        // Y':
        //        y00 y40
        //        y10 y50
        //        y20 y60
        //        y30 y70
        //        y01 y41
        //        y11 y51
        //        y21 y61
        //        y31 y71
        // in other words, y40 <-> y01, y50 <-> y11, y60 <-> y21, y70 <-> y31
        float32x4x2_t tr0, tr1;

        T4x4(y00, y10, y20, y30, tr0, tr1);
        T4x4(y01, y11, y21, y31, tr0, tr1);
        T4x4(y40, y50, y60, y70, tr0, tr1);
        T4x4(y41, y51, y61, y71, tr0, tr1);

        // Z[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*Y
        // Z[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*Y
        t00 = vsubq_f32(y01, y20);
        t01 = vsubq_f32(y41, y60);
        t10 = vsubq_f32(y30, y11);
        t11 = vsubq_f32(y70, y51);
        z00 = vfmaq_f32(vsubq_f32(y00, y21), t00, q5_25);
        z01 = vfmaq_f32(vsubq_f32(y40, y61), t01, q5_25);
        z70 = vfmaq_f32(vsubq_f32(y31, y10), t10, q5_25);
        z71 = vfmaq_f32(vsubq_f32(y71, y50), t11, q5_25);

        // Z[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*Y
        // Z[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*Y
        t00 = vfmaq_f32(vaddq_f32(y10, y11), y30, qm4_25);
        t01 = vfmaq_f32(vaddq_f32(y50, y51), y70, qm4_25);
        t10 = vfmaq_f32(vaddq_f32(y20, y21), y01, qm4_25);
        t11 = vfmaq_f32(vaddq_f32(y60, y61), y41, qm4_25);

        z10 = vaddq_f32(t00, t10); z11 = vaddq_f32(t01, t11);
        z20 = vsubq_f32(t10, t00); z21 = vsubq_f32(t11, t01);

        // Z[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*Y
        // Z[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*Y
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

        // Z[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*Y
        // Z[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*Y
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

    const int outstep = winoIblock*winoAtomF16*Cg;

    vst1_f16(outptr, vcvt_f16_f32(z00));
    vst1_f16(outptr + 4, vcvt_f16_f32(z01));
    vst1_f16(outptr + outstep, vcvt_f16_f32(z10));
    vst1_f16(outptr + outstep + 4, vcvt_f16_f32(z11));
    vst1_f16(outptr + outstep*2, vcvt_f16_f32(z20));
    vst1_f16(outptr + outstep*2 + 4, vcvt_f16_f32(z21));
    vst1_f16(outptr + outstep*3, vcvt_f16_f32(z30));
    vst1_f16(outptr + outstep*3 + 4, vcvt_f16_f32(z31));
    vst1_f16(outptr + outstep*4, vcvt_f16_f32(z40));
    vst1_f16(outptr + outstep*4 + 4, vcvt_f16_f32(z41));
    vst1_f16(outptr + outstep*5, vcvt_f16_f32(z50));
    vst1_f16(outptr + outstep*5 + 4, vcvt_f16_f32(z51));
    vst1_f16(outptr + outstep*6, vcvt_f16_f32(z60));
    vst1_f16(outptr + outstep*6 + 4, vcvt_f16_f32(z61));
    vst1_f16(outptr + outstep*7, vcvt_f16_f32(z70));
    vst1_f16(outptr + outstep*7 + 4, vcvt_f16_f32(z71));
}

// Output transform
void impl_AtXA_8x8_F16(const uchar* inptr_, int inpstep,
                           float * bpptr, int bpstep, float* outptr, int outstep,
                           float bias, float minval, float maxval, bool ifMinMaxAct)
{
    const __fp16* inptr = (const __fp16*)inptr_;

    float32x4_t x00 = vcvt_f32_f16(vld1_f16(inptr)), x01 = vcvt_f32_f16(vld1_f16(inptr + 4));
    float32x4_t x10 = vcvt_f32_f16(vld1_f16(inptr + inpstep)), x11 = vcvt_f32_f16(vld1_f16(inptr + inpstep + 4));
    float32x4_t x20 = vcvt_f32_f16(vld1_f16(inptr + inpstep*2)), x21 = vcvt_f32_f16(vld1_f16(inptr + inpstep*2 + 4));
    float32x4_t x30 = vcvt_f32_f16(vld1_f16(inptr + inpstep*3)), x31 = vcvt_f32_f16(vld1_f16(inptr + inpstep*3 + 4));
    float32x4_t x40 = vcvt_f32_f16(vld1_f16(inptr + inpstep*4)), x41 = vcvt_f32_f16(vld1_f16(inptr + inpstep*4 + 4));
    float32x4_t x50 = vcvt_f32_f16(vld1_f16(inptr + inpstep*5)), x51 = vcvt_f32_f16(vld1_f16(inptr + inpstep*5 + 4));
    float32x4_t x60 = vcvt_f32_f16(vld1_f16(inptr + inpstep*6)), x61 = vcvt_f32_f16(vld1_f16(inptr + inpstep*6 + 4));
    float32x4_t x70 = vcvt_f32_f16(vld1_f16(inptr + inpstep*7)), x71 = vcvt_f32_f16(vld1_f16(inptr + inpstep*7 + 4));
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

        // transpose 8x8 matrix in-place with some renumeration of the elements:
        // Y:
        //        y00 y01
        //        y10 y11
        //        ...
        //        y50 y51
        //        0   0
        //        0   0
        // Y':
        //        y00 y40
        //        y10 y50
        //        y20 y60
        //        y30 y70
        //        y01 y41
        //        y11 y51
        //        y21 y61
        //        y31 y71
        // in other words, y40 <-> y01, y50 <-> y11, y60 <-> y21, y70 <-> y31
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

cv::dnn::Winofunc getWinofunc_F16()
{
    return {&impl_accum_F16, &impl_BtXB_8x8_F16, &impl_AtXA_8x8_F16, 6, 8, 2};
}

// end of NEON_FP16
#else

cv::dnn::Winofunc getWinofunc_F16()
{
    return cv::dnn::Winofunc::empty();
}


// end of fallback
#endif


CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::dnn::

#endif // !CV_CPU_DECLARATIONS_ONLY
