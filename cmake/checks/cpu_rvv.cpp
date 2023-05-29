#include <stdio.h>

#if defined(__riscv)
#  include <riscv_vector.h>
#  define CV_RVV 1
#endif

#if defined CV_RVV
#if defined(__riscv_v_intrinsic) &&  __riscv_v_intrinsic>10999
#define vreinterpret_v_u64m1_u8m1 __riscv_vreinterpret_v_u64m1_u8m1
#define vle64_v_u64m1 __riscv_vle64_v_u64m1
#define vle32_v_f32m1 __riscv_vle32_v_f32m1
#define vfmv_f_s_f32m1_f32 __riscv_vfmv_f_s_f32m1_f32
#endif
int test()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint64_t ptr[2] = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
    vuint8m1_t a = vreinterpret_v_u64m1_u8m1(vle64_v_u64m1(ptr, 2));
    vfloat32m1_t val = vle32_v_f32m1((const float*)(src), 4);
    return (int)vfmv_f_s_f32m1_f32(val);
}
#else
#error "RISC-V vector extension(RVV) is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
