#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>
void test()
{
    __m512i a, b, c;
    a = _mm512_dpwssd_epi32(a, b, c);            // VNNI
}
#else
#error "AVX512-CLX is not supported"
#endif
int main() { return 0; }
