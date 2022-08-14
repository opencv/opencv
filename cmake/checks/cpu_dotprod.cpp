#include <stdio.h>

#if defined __GNUC__ && (defined __arm__ || defined __aarch64__)
#include "arm_neon.h"
int test()
{
    const unsigned int src[] = { 0, 0, 0, 0 };
    unsigned int dst[4];
    uint32x4_t v_src = *(uint32x4_t*)src;
    uint8x16_t v_m0 = *(uint8x16_t*)src;
    uint8x16_t v_m1 = *(uint8x16_t*)src;
    uint32x4_t v_dst = vdotq_u32(v_src, v_m0, v_m1);
    *(uint32x4_t*)dst = v_dst;
    return (int)dst[0];
}
#else
#error "DOTPROD is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
