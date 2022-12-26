// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "fast_convolution.hpp"

namespace cv {
namespace dnn {
namespace opt_AVX2
{
#if CV_TRY_AVX2
void convBlockMR1(int np, const float* a, const float* b, float *c, const float bias, bool init_c,
          const float minval, const float maxval, bool ifMinMaxAct)
{
#if CONV_NR == 24
    __m256 c0 = _mm256_set1_ps(bias), c1 = c0, c2 = c0;

    for (int p = 0; p < np; p++, a++, b += CONV_NR)
    {
        __m256 a0 = _mm256_set1_ps(a[0]);
        __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + 8), b2 = _mm256_loadu_ps(b + 16);

        c0 = _mm256_fmadd_ps(b0, a0, c0);
        c1 = _mm256_fmadd_ps(b1, a0, c1);
        c2 = _mm256_fmadd_ps(b2, a0, c2);
    }

    if (init_c)
    {
        c0 = _mm256_add_ps(_mm256_loadu_ps(c), c0);
        c1 = _mm256_add_ps(_mm256_loadu_ps(c + 8), c1);
        c2 = _mm256_add_ps(_mm256_loadu_ps(c + 16), c2);
    }

     if (ifMinMaxAct)
    {
        __m256 vmax = _mm256_set1_ps(maxval);
        __m256 vmin = _mm256_set1_ps(minval);

        c0 = _mm256_min_ps(_mm256_max_ps(c0, vmin), vmax);
        c1 = _mm256_min_ps(_mm256_max_ps(c1, vmin), vmax);
        c2 = _mm256_min_ps(_mm256_max_ps(c2, vmin), vmax);
    }

    _mm256_storeu_ps(c, c0);
    _mm256_storeu_ps(c + 8, c1);
    _mm256_storeu_ps(c + 16, c2);
    _mm256_zeroupper();
#else
#error "unsupported CONV_NR in convBlockMR1."
#endif
}

void convBlock_AVX2(int np, const float* a, const float* b, float* c, int ldc, bool init_c)
{
#if CONV_MR == 4 && CONV_NR == 24
    __m256 c00 = _mm256_set1_ps(0.f), c01 = c00, c02 = c00;
    __m256 c10 = c00, c11 = c00, c12 = c00;
    __m256 c20 = c00, c21 = c00, c22 = c00;
    __m256 c30 = c00, c31 = c00, c32 = c00;

    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();

    for (int p = 0; p < np; p++, a += CONV_MR, b += CONV_NR)
    {
        a0 = _mm256_set1_ps(a[0]), a1 = _mm256_set1_ps(a[1]);
        b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8), b2 = _mm256_load_ps(b + 16);

        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);
        c02 = _mm256_fmadd_ps(b2, a0, c02);

        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);
        c12 = _mm256_fmadd_ps(b2, a1, c12);

        a0 = _mm256_set1_ps(a[2]), a1 = _mm256_set1_ps(a[3]);

        c20 = _mm256_fmadd_ps(b0, a0, c20);
        c21 = _mm256_fmadd_ps(b1, a0, c21);
        c22 = _mm256_fmadd_ps(b2, a0, c22);

        c30 = _mm256_fmadd_ps(b0, a1, c30);
        c31 = _mm256_fmadd_ps(b1, a1, c31);
        c32 = _mm256_fmadd_ps(b2, a1, c32);
    }

    if (!init_c)
    {
        c00 = _mm256_add_ps(c00, _mm256_load_ps(c));
        c01 = _mm256_add_ps(c01, _mm256_load_ps(c + 8));
        c02 = _mm256_add_ps(c02, _mm256_load_ps(c + 16));

        c10 = _mm256_add_ps(c10, _mm256_load_ps(c + ldc));
        c11 = _mm256_add_ps(c11, _mm256_load_ps(c + ldc + 8));
        c12 = _mm256_add_ps(c12, _mm256_load_ps(c + ldc + 16));

        c20 = _mm256_add_ps(c20, _mm256_load_ps(c + ldc*2));
        c21 = _mm256_add_ps(c21, _mm256_load_ps(c + ldc*2 + 8));
        c22 = _mm256_add_ps(c22, _mm256_load_ps(c + ldc*2 + 16));

        c30 = _mm256_add_ps(c30, _mm256_load_ps(c + ldc*3));
        c31 = _mm256_add_ps(c31, _mm256_load_ps(c + ldc*3 + 8));
        c32 = _mm256_add_ps(c32, _mm256_load_ps(c + ldc*3 + 16));
    }

    _mm256_storeu_ps(c, c00), _mm256_storeu_ps(c+8, c01), _mm256_storeu_ps(c+16, c02);
    _mm256_storeu_ps(c + ldc, c10), _mm256_storeu_ps(c + ldc + 8, c11), _mm256_storeu_ps(c + ldc + 16, c12);
    _mm256_storeu_ps(c + ldc*2, c20), _mm256_storeu_ps(c + ldc*2 + 8, c21), _mm256_storeu_ps(c + ldc*2 + 16, c22);
    _mm256_storeu_ps(c + ldc*3, c30), _mm256_storeu_ps(c + ldc*3 + 8, c31), _mm256_storeu_ps(c + ldc*3 + 16, c32);
    _mm256_zeroupper();
#else
#error "unsupported CONV_MR and/or CONV_NR in convBlock_AVX2."
#endif
}

void _fx_winograd_accum_f32(const float* inwptr, const float* wptr,
                       float* outbuf, int Cg, int iblock)
{
    CV_Assert(_FX_WINO_IBLOCK == 6 && _FX_WINO_KBLOCK == 4);// && _FX_WINO_ATOM_F32 == 8);
    if (iblock > 3)
    {
        for (int atom_id = 0; atom_id < _FX_WINO_NATOMS_F32; atom_id++,
                outbuf += _FX_WINO_ATOM_F32)
        {
            __m256 s00 = _mm256_set1_ps(0.f), s01 = s00, s02 = s00, s03 = s00, s04 = s00, s05 = s00;
            __m256 s10 = _mm256_set1_ps(0.f), s11 = s00, s12 = s00, s13 = s00, s14 = s00, s15 = s00;
            __m256 s20 = _mm256_set1_ps(0.f), s21 = s00, s22 = s00, s23 = s00, s24 = s00, s25 = s00;
            __m256 s30 = _mm256_set1_ps(0.f), s31 = s00, s32 = s00, s33 = s00, s34 = s00, s35 = s00;
            for (int c = 0; c < Cg; c++, inwptr += _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32,
                                         wptr += _FX_WINO_KBLOCK*_FX_WINO_ATOM_F32)
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
        for (int atom_id = 0; atom_id < _FX_WINO_NATOMS_F32; atom_id++,
                outbuf += _FX_WINO_ATOM_F32)
        {
            __m256 s00 = _mm256_set1_ps(0.f), s01 = s00, s02 = s00;
            __m256 s10 = _mm256_set1_ps(0.f), s11 = s00, s12 = s00;
            __m256 s20 = _mm256_set1_ps(0.f), s21 = s00, s22 = s00;
            __m256 s30 = _mm256_set1_ps(0.f), s31 = s00, s32 = s00;
            for (int c = 0; c < Cg; c++, inwptr += _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32,
                                         wptr += _FX_WINO_KBLOCK*_FX_WINO_ATOM_F32) {
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
void _fx_winograd_BtXB_8x8_f32(const float* inptr, int inpstep, float* outptr, int Cg)
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

    const int outstep = _FX_WINO_IBLOCK*_FX_WINO_ATOM_F32*Cg;

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
void _fx_winograd_AtXA_8x8_f32(const float* inptr, int inpstep,
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

#endif
} // namespace opt_AVX2
} // namespace dnn
} // namespace cv