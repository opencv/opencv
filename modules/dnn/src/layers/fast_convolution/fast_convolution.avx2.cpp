// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "fast_convolution.hpp"

namespace cv {
namespace opt_AVX2
{
#if CV_TRY_AVX2
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

void depthWiseBlock_AVX2(const float *inptr, float *outptr, const float *weights, float biasval, int *ofstab, int *yxtab,
                    float minval, float maxval, int Hi, int Wi, int H0, int W0, int ksize, int pad_top, int pad_left,
                    int dilation_y, int stride_x, int stride_y, int inner_xleft, int inner_xright, int inner_ytop,
                    int inner_ybottom, bool ifMinMaxAct, bool useSIMD, bool is3x3)
{
    __m256 vminval = _mm256_set1_ps(minval);
    __m256 vmaxval = _mm256_set1_ps(maxval);

    __m256 w0 = _mm256_setzero_ps(),
        w1 = w0, w2 = w0, w3 = w0, w4 = w0, w5 = w0, w6 = w0, w7 = w0, w8 = w0, vbias = w0;

    if (useSIMD)
    {
        vbias = _mm256_set1_ps(biasval);
        if (is3x3)
        {
            w0 = _mm256_set1_ps(weights[0]);
            w1 = _mm256_set1_ps(weights[1]);
            w2 = _mm256_set1_ps(weights[2]);
            w3 = _mm256_set1_ps(weights[3]);
            w4 = _mm256_set1_ps(weights[4]);
            w5 = _mm256_set1_ps(weights[5]);
            w6 = _mm256_set1_ps(weights[6]);
            w7 = _mm256_set1_ps(weights[7]);
            w8 = _mm256_set1_ps(weights[8]);
        }
    }

    int dy0 = 1;
    for (int y0 = 0; y0 < H0; y0 += dy0, outptr += W0 * dy0)
    {
        dy0 = inner_ytop <= y0 && y0 + 3 < inner_ybottom && is3x3 && stride_y == 1 && dilation_y == 1
              ? 3 : 1;

        int x0 = 0, x1 = y0 >= inner_ytop && y0 < inner_ybottom ? inner_xleft : W0;
        int yi_ = y0 * stride_y - pad_top;

        for (;;)
        {
            float s_0, s_1, s_2;
            if (dy0 == 3)
            {
                for (; x0 < x1; x0++)
                {
                    int xi_ = x0 * stride_x - pad_left;
                    s_0 = s_1 = s_2 = biasval;
                    for (int k = 0; k < ksize; k++)
                    {
                        int dy = yxtab[k * 2];
                        int yi = yi_ + dy;
                        int xi = xi_ + yxtab[k * 2 + 1];
                        float w = weights[k];

                        if ((unsigned) xi < (unsigned) Wi)
                        {
                            s_0 += inptr[yi * Wi + xi] * w;
                            s_1 += inptr[(yi + 1) * Wi + xi] * w;
                            s_2 += inptr[(yi + 2) * Wi + xi] * w;
                        }
                    }
                    if (ifMinMaxAct)
                    {
                        s_0 = std::min(std::max(s_0, minval), maxval);
                        s_1 = std::min(std::max(s_1, minval), maxval);
                        s_2 = std::min(std::max(s_2, minval), maxval);
                    }

                    outptr[x0] = s_0;
                    outptr[x0 + W0] = s_1;
                    outptr[x0 + W0 * 2] = s_2;
                }
            }
            else
            {
                for (; x0 < x1; x0++)
                {
                    int xi_ = x0 * stride_x - pad_left;
                    s_0 = biasval;
                    for (int k = 0; k < ksize; k++) {
                        int dy = yxtab[k * 2];
                        int yi = yi_ + dy;
                        int xi = xi_ + yxtab[k * 2 + 1];
                        float w = weights[k];
                        if (((unsigned) yi < (unsigned) Hi) & ((unsigned) xi < (unsigned) Wi))
                            s_0 += inptr[yi * Wi + xi] * w;
                    }
                    if (ifMinMaxAct)
                        s_0 = std::min(std::max(s_0, minval), maxval);
                    outptr[x0] = s_0;
                }
            }
            if (x0 == W0)
                break;
            x1 = inner_xright;

            if (useSIMD)
            {
                if (is3x3)
                {
                    if (dy0 == 3)
                    {
                        for (; x0 <= x1 - FAST_VEC_NLANES; x0 += FAST_VEC_NLANES)
                        {
                            int xi_ = x0 * stride_x - pad_left;
                            const float *inptr_xi = inptr + Wi * yi_ + xi_;

                            __m256 s0, s1, s2;
                            __m256 x00 = _mm256_loadu_ps(inptr_xi);
                            __m256 x01 = _mm256_loadu_ps(inptr_xi + 1);
                            __m256 x02 = _mm256_loadu_ps(inptr_xi + 2);

                            __m256 x10 = _mm256_loadu_ps(inptr_xi + Wi);
                            __m256 x11 = _mm256_loadu_ps(inptr_xi + Wi + 1);
                            __m256 x12 = _mm256_loadu_ps(inptr_xi + Wi + 2);

                            __m256 x20 = _mm256_loadu_ps(inptr_xi + Wi * 2);
                            __m256 x21 = _mm256_loadu_ps(inptr_xi + Wi * 2 + 1);
                            __m256 x22 = _mm256_loadu_ps(inptr_xi + Wi * 2 + 2);

                            __m256 x30 = _mm256_loadu_ps(inptr_xi + Wi * 3);
                            __m256 x31 = _mm256_loadu_ps(inptr_xi + Wi * 3 + 1);
                            __m256 x32 = _mm256_loadu_ps(inptr_xi + Wi * 3 + 2);

                            __m256 x40 = _mm256_loadu_ps(inptr_xi + Wi * 4);
                            __m256 x41 = _mm256_loadu_ps(inptr_xi + Wi * 4 + 1);
                            __m256 x42 = _mm256_loadu_ps(inptr_xi + Wi * 4 + 2);

                            s0 = _mm256_fmadd_ps(x00, w0, vbias);
                            s1 = _mm256_fmadd_ps(x10, w0, vbias);
                            s2 = _mm256_fmadd_ps(x20, w0, vbias);

                            s0 = _mm256_fmadd_ps(x01, w1, s0);
                            s1 = _mm256_fmadd_ps(x11, w1, s1);
                            s2 = _mm256_fmadd_ps(x21, w1, s2);

                            s0 = _mm256_fmadd_ps(x02, w2, s0);
                            s1 = _mm256_fmadd_ps(x12, w2, s1);
                            s2 = _mm256_fmadd_ps(x22, w2, s2);

                            s0 = _mm256_fmadd_ps(x10, w3, s0);
                            s1 = _mm256_fmadd_ps(x20, w3, s1);
                            s2 = _mm256_fmadd_ps(x30, w3, s2);

                            s0 = _mm256_fmadd_ps(x11, w4, s0);
                            s1 = _mm256_fmadd_ps(x21, w4, s1);
                            s2 = _mm256_fmadd_ps(x31, w4, s2);

                            s0 = _mm256_fmadd_ps(x12, w5, s0);
                            s1 = _mm256_fmadd_ps(x22, w5, s1);
                            s2 = _mm256_fmadd_ps(x32, w5, s2);

                            s0 = _mm256_fmadd_ps(x20, w6, s0);
                            s1 = _mm256_fmadd_ps(x30, w6, s1);
                            s2 = _mm256_fmadd_ps(x40, w6, s2);

                            s0 = _mm256_fmadd_ps(x21, w7, s0);
                            s1 = _mm256_fmadd_ps(x31, w7, s1);
                            s2 = _mm256_fmadd_ps(x41, w7, s2);

                            s0 = _mm256_fmadd_ps(x22, w8, s0);
                            s1 = _mm256_fmadd_ps(x32, w8, s1);
                            s2 = _mm256_fmadd_ps(x42, w8, s2);

                            if (ifMinMaxAct)
                            {
                                s0 = _mm256_min_ps(_mm256_max_ps(s0, vminval), vmaxval);
                                s1 = _mm256_min_ps(_mm256_max_ps(s1, vminval), vmaxval);
                                s2 = _mm256_min_ps(_mm256_max_ps(s2, vminval), vmaxval);
                            }

                            _mm256_storeu_ps(outptr + x0, s0);
                            _mm256_storeu_ps(outptr + W0 + x0, s1);
                            _mm256_storeu_ps(outptr + W0 * 2 + x0, s2);
                        }
                    }
                    else
                    {
                        for (; x0 <= x1 - FAST_VEC_NLANES; x0 += FAST_VEC_NLANES)
                        {
                            int xi_ = x0 * stride_x - pad_left;
                            const float *inptr_xi = inptr + Wi * yi_ + xi_;
                            __m256 s0 = _mm256_fmadd_ps(_mm256_loadu_ps(inptr_xi + ofstab[0]), w0, vbias);
                            __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(inptr_xi + ofstab[1]), w1);
                            __m256 s2 = _mm256_mul_ps(_mm256_loadu_ps(inptr_xi + ofstab[2]), w2);

                            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(inptr_xi + ofstab[3]), w3, s0);
                            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(inptr_xi + ofstab[4]), w4, s1);
                            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(inptr_xi + ofstab[5]), w5, s2);

                            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(inptr_xi + ofstab[6]), w6, s0);
                            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(inptr_xi + ofstab[7]), w7, s1);
                            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(inptr_xi + ofstab[8]), w8, s2);

                            s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), s2);

                            if (ifMinMaxAct)
                                s0 = _mm256_min_ps(_mm256_max_ps(s0, vminval), vmaxval);
                            _mm256_storeu_ps(outptr + x0, s0);
                        }
                    }
                }
                else
                {
                    for (; x0 <= x1 - FAST_VEC_NLANES; x0 += FAST_VEC_NLANES)
                    {
                        int xi_ = x0 * stride_x - pad_left, k = 0;
                        const float *inptr_xi = inptr + Wi * yi_ + xi_;
                        __m256 s0 = vbias;
                        for (; k <= ksize - 4; k += 4)
                        {
                            __m256 v0 = _mm256_loadu_ps(inptr_xi + ofstab[k]);
                            __m256 v1 = _mm256_loadu_ps(inptr_xi + ofstab[k + 1]);
                            __m256 v2 = _mm256_loadu_ps(inptr_xi + ofstab[k + 2]);
                            __m256 v3 = _mm256_loadu_ps(inptr_xi + ofstab[k + 3]);

                            __m256 ww0 = _mm256_set1_ps(weights[k]);
                            __m256 ww1 = _mm256_set1_ps(weights[k+1]);
                            __m256 ww2 = _mm256_set1_ps(weights[k+2]);
                            __m256 ww3 = _mm256_set1_ps(weights[k+3]);

                            s0 = _mm256_fmadd_ps(v0, ww0, s0);
                            s0 = _mm256_fmadd_ps(v1, ww1, s0);
                            s0 = _mm256_fmadd_ps(v2, ww2, s0);
                            s0 = _mm256_fmadd_ps(v3, ww3, s0);
                        }
                        for (; k < ksize; k++)
                            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(inptr_xi + ofstab[k]),
                                                 _mm256_set1_ps(weights[k]), s0);

                        if (ifMinMaxAct)
                            s0 = _mm256_min_ps(_mm256_max_ps(s0, vminval), vmaxval);
                        _mm256_storeu_ps(outptr + x0, s0);
                    }
                }
            }

            if (dy0 == 3)
            {
                for (; x0 < x1; x0++)
                {
                    int xi_ = x0 * stride_x - pad_left;
                    const float *inptr_xi = inptr + W0 * yi_ + xi_;
                    s_0 = s_1 = s_2 = biasval;
                    for (int k = 0; k < ksize; k++) {
                        int inp_ofs = ofstab[k];
                        float w = weights[k];
                        s_0 += inptr_xi[inp_ofs] * w;
                        s_1 += inptr_xi[inp_ofs + Wi] * w;
                        s_2 += inptr_xi[inp_ofs + Wi * 2] * w;
                    }
                    if (ifMinMaxAct)
                    {
                        s_0 = std::min(std::max(s_0, minval), maxval);
                        s_1 = std::min(std::max(s_1, minval), maxval);
                        s_2 = std::min(std::max(s_2, minval), maxval);
                    }

                    outptr[x0] = s_0;
                    outptr[x0 + W0] = s_1;
                    outptr[x0 + W0 * 2] = s_2;
                }
            }
            else
            {
                for (; x0 < x1; x0++)
                {
                    int xi_ = x0 * stride_x - pad_left;
                    const float *inptr_xi = inptr + Wi * yi_ + xi_;
                    s_0 = biasval;
                    for (int k = 0; k < ksize; k++)
                    {
                        s_0 += inptr_xi[ofstab[k]] * weights[k];
                    }
                    if (ifMinMaxAct)
                        s_0 = std::min(std::max(s_0, minval), maxval);
                    outptr[x0] = s_0;
                }
            }
            x1 = W0;
        }
    }
}
#endif
} // namespace opt_AVX2
} // namespace cv