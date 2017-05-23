#if !defined __AVX__ // MSVC supports this flag since MSVS 2013
#error "__AVX__ define is missing"
#endif
#include <immintrin.h>
void test()
{
    __m256 a = _mm256_set1_ps(0.0f);
}
int main() { return 0; }
