#include <stdio.h>
#include <arm_neon.h>

#if !defined(__ARM_FEATURE_DOTPROD)
#error "NEON dot product is not supported"
#endif

int test()
{
    const uint8x8_t d0 = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint32x2_t d1 = {1024, 2048};
    uint32x2_t result = vdot_u32(d1, d0, d0);
    return vget_lane_u32(result, 0);
}

int main()
{
  printf("%d\n", test());
  return 0;
}
