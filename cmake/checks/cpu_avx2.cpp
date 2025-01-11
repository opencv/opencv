#if !defined __AVX2__
#error "__AVX2__ define is missing"
#endif
#include <immintrin.h>
void test()
{
    int data[8] = {0,0,0,0, 0,0,0,0};
    __m256i a = _mm256_loadu_si256((const __m256i *)data);
    __m256i b = _mm256_bslli_epi128(a, 1);  // available in GCC 4.9.3+
}
int main() { return 0; }
