#include <pmmintrin.h>
int main() {
    __m128 u, v;
    u = _mm_set1_ps(0.0f);
    v = _mm_moveldup_ps(u); // SSE3
    return 0;
}
