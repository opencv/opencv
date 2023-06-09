#include <stdio.h>

#if defined __F16C__ || (defined _MSC_VER && _MSC_VER >= 1700 && defined __AVX__) || (defined __INTEL_COMPILER && defined __AVX__)
#include <immintrin.h>
int test()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    short dst[8];
    __m128 v_src = _mm_load_ps(src);
    __m128i v_dst = _mm_cvtps_ph(v_src, 0);
    _mm_storel_epi64((__m128i*)dst, v_dst);
    return (int)dst[0];
}
#elif defined __GNUC__ && (defined __arm__ || defined __aarch64__)
#include "arm_neon.h"
int test()
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    short dst[8];
    float32x4_t v_src = *(float32x4_t*)src;
    float16x4_t v_dst = vcvt_f16_f32(v_src);
    float16x8_t x = vcombine_f16(v_dst, v_dst);

    x = vfmaq_laneq_f16(x, x, x, 7);
    x = vfmaq_f16(x, x, x);

    return (int)dst[0];
#else
    return 0;
#endif
}
#else
#error "FP16 is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
