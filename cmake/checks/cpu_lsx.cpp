#include <stdio.h>
#include <lsxintrin.h>

int test()
{
    const float src[] = { 0.0f, 1.0f, 2.0f, 3.0f};
    v4f32 val = (v4f32)__lsx_vld((const float*)(src), 0);
    return __lsx_vpickve2gr_w(__lsx_vftint_w_s(val), 3);
}

int main()
{
  printf("%d\n", test());
  return 0;
}
