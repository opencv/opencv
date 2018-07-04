#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>
void test()
{
    __m512i zmm = _mm512_setzero_si512();
    __m256i a = _mm256_setzero_si256();
    __m256i b = _mm256_abs_epi64(a); // VL
    __m512i c = _mm512_abs_epi8(zmm); // BW
    __m512i d = _mm512_broadcast_i32x8(b); // DQ
#if defined __GNUC__ && defined __x86_64__
    asm volatile ("" : : : "zmm16", "zmm17", "zmm18", "zmm19");
#endif
}
#else
#error "AVX512-SKX is not supported"
#endif
int main() { return 0; }
