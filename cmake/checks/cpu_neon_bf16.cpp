#if (defined __GNUC__ && (defined __arm__ || defined __aarch64__)) || (defined _MSC_VER && defined _M_ARM64)
#include <stdio.h>
#include "arm_neon.h"

/*#if defined __clang__
#pragma clang attribute push (__attribute__((target("bf16"))), apply_to=function)
#elif defined GCC
#pragma GCC push_options
#pragma GCC target("armv8.2-a", "bf16")
#endif*/
bfloat16x8_t vld1q_as_bf16(const float* src)
{
    float32x4_t s0 = vld1q_f32(src), s1 = vld1q_f32(src + 4);
    return vcombine_bf16(vcvt_bf16_f32(s0), vcvt_bf16_f32(s1));
}

void vprintreg(const char* name, const float32x4_t& r)
{
    float data[4];
    vst1q_f32(data, r);
    printf("%s: (%.2f, %.2f, %.2f, %.2f)\n",
        name, data[0], data[1], data[2], data[3]);
}

void test()
{
    const float src1[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    const float src2[] = { 1.f, 3.f, 6.f, 10.f, 15.f, 21.f, 28.f, 36.f };
    bfloat16x8_t s1 = vld1q_as_bf16(src1), s2 = vld1q_as_bf16(src2);
    float32x4_t d = vbfdotq_f32(vdupq_n_f32(0.f), s1, s2);
    vprintreg("(s1[0]*s2[0] + s1[1]*s2[1], ... s1[6]*s2[6] + s1[7]*s2[7])", d);
}
/*#if defined __clang__
#pragma clang attribute pop
#elif defined GCC
#pragma GCC pop_options
#endif*/
#else
#error "BF16 is not supported"
#endif

int main()
{
    test();
    return 0;
}
