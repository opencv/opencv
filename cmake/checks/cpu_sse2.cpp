#include <emmintrin.h>

inline __m128i _v128_comgt_epu32(const __m128i& a, const __m128i& b)
{
    const __m128i delta = _mm_set1_epi32((int)0x80000000);
    return _mm_cmpgt_epi32(_mm_xor_si128(a, delta), _mm_xor_si128(b, delta));
}

int main()
{
    __m128i a, b, c;
    a = _mm_set1_epi32(0x00000000);
    b = _mm_set1_epi32(0x0000ffff);
    c = _v128_comgt_epu32(a, b);
    return 0;
}
