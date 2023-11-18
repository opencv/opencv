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
    const float src[] = { 0.0f, 1.0f, 2.0f, 3.0f };
    short dst[4];
    float32x4_t v_src = vld1q_f32(src);
    float16x4_t v_dst = vcvt_f16_f32(v_src);
    vst1_f16((__fp16*)dst, v_dst);
    return dst[0] + dst[1] + dst[2] + dst[3];
}
#else
#error "FP16 is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
