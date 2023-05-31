#ifdef _MSC_VER
#  include <nmmintrin.h>
#  if defined(_M_X64)
#    define CV_POPCNT_U64 _mm_popcnt_u64
#  endif
#  define CV_POPCNT_U32 _mm_popcnt_u32
#elif defined(__POPCNT__)
#  include <popcntintrin.h>
#  if defined(__x86_64__)
#    define CV_POPCNT_U64 __builtin_popcountll
#  endif
#  define CV_POPCNT_U32 __builtin_popcount
#else
#  error "__POPCNT__ is not defined by compiler"
#endif

int main()
{
#ifdef CV_POPCNT_U64
    int i = CV_POPCNT_U64(1);
#endif
    int j = CV_POPCNT_U32(1);
    return 0;
}
