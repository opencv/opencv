#include <tmmintrin.h>
const double v = 0;
int main() {
    __m128i a = _mm_setzero_si128();
    __m128i b = _mm_abs_epi32(a);
    return 0;
}
