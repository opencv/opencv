#include <x86intrin.h>

int main()
{
    __m128i a = _mm_setzero_si128(), b = _mm_setzero_si128();
    __m128i c = _mm_comge_epu32(a, b);
    return 0;
}