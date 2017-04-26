#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>
void test()
{
    __m512i zmm = _mm512_setzero_si512();
}
#else
#error "AVX512 is not supported"
#endif
int main() { return 0; }
