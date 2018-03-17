// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "convert.hpp"

namespace cv
{
namespace opt_AVX2
{

void cvtScale_s16s32f32Line_AVX2(const short* src, int* dst, float scale, float shift, int width)
{
    int x = 0;

    __m256 scale256 = _mm256_set1_ps(scale);
    __m256 shift256 = _mm256_set1_ps(shift);
    const int shuffle = 0xD8;

    for (; x <= width - 16; x += 16)
    {
        __m256i v_src = _mm256_loadu_si256((const __m256i *)(src + x));
        v_src = _mm256_permute4x64_epi64(v_src, shuffle);
        __m256i v_src_lo = _mm256_srai_epi32(_mm256_unpacklo_epi16(v_src, v_src), 16);
        __m256i v_src_hi = _mm256_srai_epi32(_mm256_unpackhi_epi16(v_src, v_src), 16);
        __m256 v_dst0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v_src_lo), scale256), shift256);
        __m256 v_dst1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v_src_hi), scale256), shift256);
        _mm256_storeu_si256((__m256i *)(dst + x), _mm256_cvtps_epi32(v_dst0));
        _mm256_storeu_si256((__m256i *)(dst + x + 8), _mm256_cvtps_epi32(v_dst1));
    }

    for (; x < width; x++)
        dst[x] = saturate_cast<int>(src[x] * scale + shift);
}

}
} // cv::
/* End of file. */
