#include <stdio.h>

#if defined _WIN32 && (defined(_M_ARM) || defined(_M_ARM64))
# include <Intrin.h>
# include <arm_neon.h>
# define CV_NEON 1
#elif defined(__ARM_NEON__) || (defined (__ARM_NEON) && defined(__aarch64__))
#  include <arm_neon.h>
#  define CV_NEON 1
#endif

// MSVC 2019 bug. Details: https://github.com/opencv/opencv/pull/16027
void test_aliased_type(const uint8x16_t& a) { }
void test_aliased_type(const int8x16_t& a) { }

#if defined CV_NEON
int test()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float32x4_t val = vld1q_f32((const float32_t*)(src));
    return (int)vgetq_lane_f32(val, 0);
}
#else
#error "NEON is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
