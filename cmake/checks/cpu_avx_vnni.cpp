#if !defined __AVXVNNI__
#error "__AVXVNNI__ define is missing"
#endif
#include <immintrin.h>

// Knowing the intrinsic is not enough: some compiler/binutils combinations accept
// -mavxvnni but cannot encode VPDPBUSD, so the build breaks at assembly time (#29840).
// Results are written through a pointer so the instruction survives optimization.
void test(const void* src, void* dst)
{
    __m256i a = _mm256_loadu_si256((const __m256i*)src);
    __m256i r = _mm256_dpbusd_epi32(_mm256_setzero_si256(), a, a);
    _mm256_storeu_si256((__m256i*)dst, r);
}
int main() { return 0; }
