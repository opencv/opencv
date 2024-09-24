#include <stdio.h>

#if !defined(__riscv) || !defined(__riscv_v)
#error "RISC-V or vector extension(RVV) is not supported by the compiler"
#endif

#if !defined(__THEAD_VERSION__) && defined(__riscv_v_intrinsic) && __riscv_v_intrinsic < 12000
#error "Wrong intrinsics version, v0.12 or higher is required for gcc or clang"
#endif

#include <riscv_vector.h>

#ifdef __THEAD_VERSION__
int test()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint64_t ptr[2] = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
    vuint8m1_t a = vreinterpret_v_u64m1_u8m1(vle64_v_u64m1(ptr, 2));
    vfloat32m1_t val = vle32_v_f32m1((const float*)(src), 4);
    return (int)vfmv_f_s_f32m1_f32(val);
}
#else
int test()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint64_t ptr[2] = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
    vuint8m1_t a = __riscv_vreinterpret_v_u64m1_u8m1(__riscv_vle64_v_u64m1(ptr, 2));
    vfloat32m1_t val = __riscv_vle32_v_f32m1((const float*)(src), 4);
    return (int)__riscv_vfmv_f_s_f32m1_f32(val);
}
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
