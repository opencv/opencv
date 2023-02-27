// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX

#if !CV_FMA3 // AVX workaround
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR)
{
    CV_Assert(convMR == 4 && convNR == 24);
    __m256 c00 = _mm256_set1_ps(0.f), c01 = c00, c02 = c00;
    __m256 c10 = c00, c11 = c00, c12 = c00;
    __m256 c20 = c00, c21 = c00, c22 = c00;
    __m256 c30 = c00, c31 = c00, c32 = c00;

    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();

    for (int p = 0; p < np; p++, a += convMR, b += convNR)
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
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

// NEON code work around.
namespace opt_NEON
{
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_NEON
    void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR);
#endif
}
}} // namespace cv::dnn
