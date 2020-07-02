#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>
void test()
{
    __m512 a, b, c, d, e;
    __m512i ai, bi, ci, di, ei, fi;
    __m128  *mem;
    __m128i *memi;
    __mmask16 m;
    a  = _mm512_4fnmadd_ps(a, b, c, d, e, mem);
    ai = _mm512_4dpwssd_epi32(ai, bi, ci, di, ei, memi);
    ai = _mm512_popcnt_epi64(ai);
}
#else
#error "AVX512-KNM is not supported"
#endif
int main() { return 0; }