#include <nmmintrin.h>
int main() {
    int i = _mm_popcnt_u64(1);
    return 0;
}
