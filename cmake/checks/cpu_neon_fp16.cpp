#include <stdio.h>

#if (defined __GNUC__ && (defined __arm__ || defined __aarch64__)) || (defined _MSC_VER && (defined _M_ARM64 || defined _M_ARM64EC)
#include "arm_neon.h"

float16x8_t vld1q_as_f16(const float* src)
{
    float32x4_t s0 = vld1q_f32(src), s1 = vld1q_f32(src + 4);
    return vcombine_f16(vcvt_f16_f32(s0), vcvt_f16_f32(s1));
}

void vprintreg(const char* name, const float16x8_t& r)
{
    float data[8];
    vst1q_f32(data, vcvt_f32_f16(vget_low_f16(r)));
    vst1q_f32(data + 4, vcvt_f32_f16(vget_high_f16(r)));
    printf("%s: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f)\n",
        name, data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7]);
}

void test()
{
    const float src1[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    const float src2[] = { 1.f, 3.f, 6.f, 10.f, 15.f, 21.f, 28.f, 36.f };
    float16x8_t s1 = vld1q_as_f16(src1), s2 = vld1q_as_f16(src2);
    float16x8_t d = vsubq_f16(s1, s1);
    d = vfmaq_laneq_f16(d, s1, s2, 0);
    d = vfmaq_laneq_f16(d, s1, s2, 1);
    d = vfmaq_laneq_f16(d, s1, s2, 2);
    d = vfmaq_laneq_f16(d, s1, s2, 3);
    d = vfmaq_laneq_f16(d, s1, s2, 4);
    d = vfmaq_laneq_f16(d, s1, s2, 5);
    d = vfmaq_laneq_f16(d, s1, s2, 6);
    d = vfmaq_laneq_f16(d, s1, s2, 7);
    vprintreg("s1*s2[0]+s1*s2[1] + ... + s1*s2[7]", d);
}
#else
#error "FP16 is not supported"
#endif

int main()
{
    test();
    return 0;
}
