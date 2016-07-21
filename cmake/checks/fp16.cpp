#include <stdio.h>

#if defined __F16C__ || (defined _MSC_VER && _MSC_VER >= 1700)
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
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    short dst[8];
    float32x4_t v_src = *(float32x4_t*)src;
    float16x4_t v_dst = vcvt_f16_f32(v_src);
    *(float16x4_t*)dst = v_dst;
    return (int)dst[0];
}
#else
#error "FP16 is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
