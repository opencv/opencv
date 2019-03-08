#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>
void test()
{
    __m512i a, b, c;
    a = _mm512_madd52hi_epu64(a, b, c);
    a = _mm512_permutexvar_epi8(a, b);
}
#else
#error "AVX512-CNL is not supported"
#endif
int main() { return 0; }