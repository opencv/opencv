// Checks whether the assembler can encode AVX-VNNI, not just the compiler. See #29840.
#include <immintrin.h>

__attribute__((target("avx2,avxvnni")))
static int check_avxvnni()
{
    __m256i a = _mm256_setzero_si256();
    __m256i b = _mm256_setzero_si256();
    __m256i c = _mm256_setzero_si256();
    __m256i r = _mm256_dpbusd_epi32(a, b, c);
    return _mm256_extract_epi32(r, 0);
}

int main()
{
    return check_avxvnni();
}
