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
#elif (defined __GNUC__ && (defined __arm__ || defined __aarch64__))
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
#elif defined _MSC_VER && (defined _M_ARM64 || defined _M_ARM64EC)
// OpenCV is built with _ARM64_DISTINCT_NEON_TYPES, which hides the vcvt_*_f16 wrappers.
#define _ARM64_DISTINCT_NEON_TYPES
#include "arm_neon.h"
int test()
{
    short dst[8] = { 0 };
    __n128 v_src = neon_fcvtl_32(neon_ld1m_16((const __int16*)dst));
    neon_st1m_16((__int16*)dst, neon_fcvtn_32(v_src));
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
