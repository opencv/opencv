#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>
void test()
{
    __m512i a, b, c;
    a = _mm512_popcnt_epi8(a);
    a = _mm512_shrdv_epi64(a, b, c);
    a = _mm512_popcnt_epi64(a);
}
#else
#error "AVX512-ICL is not supported"
#endif
int main() { return 0; }