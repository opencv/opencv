#include <stdio.h>

#if defined(__loongarch_asx)
#  include <lasxintrin.h>
#  define CV_LASX 1
#endif

#if defined CV_LASX
int test()
{
    const float src[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
    v8f32 val = (v8f32)__lasx_xvld((const float*)(src), 0);
    return __lasx_xvpickve2gr_w(__lasx_xvftint_w_s (val), 7);
}
#else
#error "LASX is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
