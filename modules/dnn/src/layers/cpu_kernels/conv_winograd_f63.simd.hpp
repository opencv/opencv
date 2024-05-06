// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

/* Accumulate */
void winofunc_accum_F32(const float* inwptr, const float* wptr, float* outbuf, int Cg, int iblock,
                            const int winoIblock, const int winoKblock, const int winoAtomF32, const int winoNatomF32);

/*Input transform*/
void winofunc_BtXB_8x8_F32(const float* inptr, int inpstep,
                               float* outptr, int Cg, const int winoIblock, const int winoAtomF32);

/*Output transform*/
void winofunc_AtXA_8x8_F32(const float* inptr, int inpstep,
                               float* bpptr, int bpstep, float* outptr, int outstep,
                               float bias, float minval, float maxval, bool ifMinMaxAct);

// FP 16 branch, only ARMv8 supports.
void winofunc_accum_F16(const char* _inwptr, const char* _wptr, char* _outbuf, int Cg, int iblock,
                        const int winoIblock, const int winoKblock, const int winoAtomF16, const int winoNatomF16);
void winofunc_BtXB_8x8_F16(const float * inptr, int inpstep,
                           char * _outptr, int Cg, const int winoIblock, const int winoAtomF16);
void winofunc_AtXA_8x8_F16(const char* inptr, int inpstep,
                           float * bpptr, int bpstep, float* outptr, int outstep,
                           float bias, float minval, float maxval, bool ifMinMaxAct);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY)

#if CV_AVX

#if !CV_FMA3 // AVX workaround
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

void winofunc_accum_F32(const float* inwptr, const float* wptr, float* outbuf, int Cg, int iblock,
                            const int winoIblock, const int winoKblock, const int winoAtomF32, const int winoNatomF32)
{
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
void winofunc_BtXB_8x8_F32(const float* inptr, int inpstep,
                               float* outptr, int Cg, const int winoIblock, const int winoAtomF32)
{
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
void winofunc_AtXA_8x8_F32(const float* inptr, int inpstep,
                          float* bpptr, int bpstep, float* outptr, int outstep,
                          float bias, float minval, float maxval, bool ifMinMaxAct)
{

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

#endif // CV_AVX

// FP16, currently, only ARMv8 may support it
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#undef T4x4
#define T4x4(a, b, c, d, tr0, tr1) \
    tr0 = vtrnq_f32(a, b); \
    tr1 = vtrnq_f32(c, d); \
    a = vcombine_f32(vget_low_f32(tr0.val[0]), vget_low_f32(tr1.val[0])); \
    b = vcombine_f32(vget_low_f32(tr0.val[1]), vget_low_f32(tr1.val[1])); \
    c = vcombine_f32(vget_high_f32(tr0.val[0]), vget_high_f32(tr1.val[0])); \
    d = vcombine_f32(vget_high_f32(tr0.val[1]), vget_high_f32(tr1.val[1]))

/* Accumulate */
void winofunc_accum_F16(const char* _inwptr, const char* _wptr, char* _outbuf, int Cg, int iblock,
                        const int winoIblock, const int winoKblock, const int winoAtomF16, const int winoNatomF16)
{
    const __fp16* inwptr = (const __fp16*)_inwptr;
    const __fp16* wptr = (const __fp16*)_wptr;
    __fp16* outbuf = (__fp16*)_outbuf;

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
void winofunc_BtXB_8x8_F16(const float * inptr, int inpstep,
                           char * _outptr, int Cg, const int winoIblock, const int winoAtomF16)
{
    __fp16* outptr = (__fp16*)_outptr;
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
void winofunc_AtXA_8x8_F16(const char* _inptr, int inpstep,
                           float * bpptr, int bpstep, float* outptr, int outstep,
                           float bias, float minval, float maxval, bool ifMinMaxAct)
{
    const __fp16* inptr = (const __fp16*)_inptr;

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
#endif
#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
