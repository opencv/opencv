#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>

void test()
{
    int* base;
    __m512i idx;
    __mmask16 m16;
    __m512 f;
    _mm512_mask_prefetch_i32gather_ps(idx, m16, base, 1, _MM_HINT_T1);
    f = _mm512_rsqrt28_ps(f);
}
#else
#error "AVX512-KNL is not supported"
#endif
int main() { return 0; }