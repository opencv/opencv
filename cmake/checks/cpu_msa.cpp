#include <stdio.h>

#if defined(__mips_msa)
#  include <msa.h>
#  define CV_MSA 1
#endif

#if defined CV_MSA
int test()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    v4f32 val = (v4f32)__msa_ld_w((const float*)(src), 0);
    return __msa_copy_s_w(__builtin_msa_ftint_s_w (val), 0);
}
#else
#error "MSA is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
