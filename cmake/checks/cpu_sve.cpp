#include <stdio.h>

#if defined(__ARM_FEATURE_SVE)
#  include <arm_sve.h>
#  define CV_SVE 1
#endif

#if defined(CV_SVE)
int test()
{
    const float src[1024] = {0.0};
    svbool_t pg = svptrue_b32();
    svfloat32_t val = svld1(pg, src);
    return (int)svlastb_f32(pg, val);
}
#else
#error "SVE is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
