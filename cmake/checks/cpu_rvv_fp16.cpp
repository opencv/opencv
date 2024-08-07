#include <stdio.h>

#if defined(__riscv) && __riscv && defined (__riscv_zvfh) && __riscv_zvfh
#  include <riscv_vector.h>

int test()
{
    const _Float16 input1[] = {0.5f, 1.5f, 2.5f, 3.5f};
    const _Float16 input2[] = {-0.5f, -1.5f, -2.5f, -3.5f};

    size_t vl =  __riscv_vsetvl_e16m1(4);
    vfloat16m1_t vec1 = __riscv_vle16_v_f16m1(input1, vl);
    vfloat16m1_t vec2 = __riscv_vle16_v_f16m1(input2, vl);
    vfloat16m1_t result = __riscv_vfadd_vv_f16m1(vec1, vec2, vl);
    return (int)__riscv_vfmv_f_s_f16m1_f16(result);
}
#else
#error "RISC-V Vector Extension with Half-Precision Floating-Point (zvfh) is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
