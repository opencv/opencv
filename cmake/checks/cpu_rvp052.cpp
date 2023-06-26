#include <stdio.h>

#if defined(__riscv)
#  include <nds_intrinsic.h>
#  define CV_RVP052 1
#endif

#if defined CV_RVP052
unsigned long test()
{
    unsigned long a = 0xABCDFFCDAFCDFBCD;
    unsigned long b = 0x1111111111111111;
    unsigned long c = __nds__ukadd16(a, b);
    return c;
}
#else
#error "RISC-V Andes Packed-SIMD extension(RVP) is not supported"
#endif

int main()
{
  printf("%lx\n", test());
  return 0;
}
