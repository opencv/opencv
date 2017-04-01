#if !defined __AVX2__ // MSVC supports this flag since MSVS 2013
#error "__AVX2__ define is missing"
#endif
#include <immintrin.h>
void test()
{
    int data[8] = {0,0,0,0, 0,0,0,0};
    __m256i a = _mm256_loadu_si256((const __m256i *)data);
}
int main() { return 0; }
