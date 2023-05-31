#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>
void test()
{
    __m512i zmm = _mm512_setzero_si512();
#if defined __GNUC__ && defined __x86_64__
    asm volatile ("" : : : "zmm16", "zmm17", "zmm18", "zmm19");
#endif
}
#else
#error "AVX512 is not supported"
#endif
int main() { return 0; }
