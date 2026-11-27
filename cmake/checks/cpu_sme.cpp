#include <stdio.h>

#if defined(__ARM_FEATURE_SME) && defined(__aarch64__)
#  include <arm_sme.h>
#  define CV_SME 1
#endif

#if defined(CV_SME)
// One fp32 outer product into ZA, entered from normal (non-streaming) code.
__arm_new("za") __arm_locally_streaming
static float test_mopa(const float* a, const float* b, float* c)
{
    svbool_t pg = svptrue_b32();
    svzero_za();
    svmopa_za32_f32_m(0, pg, pg, svld1_f32(pg, a), svld1_f32(pg, b));
    svst1_hor_za32(0, 0, pg, c);
    return c[0];
}

__arm_locally_streaming
static int test_svl()
{
    return (int)svcntw();
}

int test()
{
    float a[64] = { 1.f }, b[64] = { 2.f }, c[64] = { 0.f };
    return (int)test_mopa(a, b, c) + test_svl();
}
#else
#error "SME is not supported"
#endif

int main()
{
    printf("%d\n", test());
    return 0;
}
