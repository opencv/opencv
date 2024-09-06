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
#elif (defined __GNUC__ && (defined __arm__ || defined __aarch64__)) /*|| (defined _MSC_VER && defined _M_ARM64)*/
// Windows + ARM64 case disabled: https://github.com/opencv/opencv/issues/25052
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
#elif (defined __riscv_zvfhmin && __riscv_zvfhmin) || (defined __riscv_zvfh && __riscv_zvfh)
#include <riscv_vector.h>

int test()
{
    const _Float16 input1[] = {0.5f, 1.5f, 2.5f, 3.5f};
    const float input2[] = {-0.5f, -1.5f, -2.5f, -3.5f};
    short dst[4];

    size_t vl =  __riscv_vsetvl_e16m1(4);

    vfloat16m1_t in_f16 = __riscv_vle16_v_f16m1(input1, vl);
    vfloat32m2_t in_f32 = __riscv_vle32_v_f32m2(input2, vl);

    vfloat32m2_t cvt_f32 = __riscv_vfwcvt_f_f_v_f32m2(in_f16, vl);
    vfloat32m2_t res_f32 = __riscv_vfadd(in_f32, cvt_f32, vl);
    vfloat16m1_t res_f16 = __riscv_vfncvt_f_f_w_f16m1(res_f32, vl);

    __riscv_vse16_v_f16m1((_Float16*)dst, res_f16, vl);
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
