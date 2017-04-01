#include <nmmintrin.h>
#ifndef _MSC_VER
#include <popcntintrin.h>
#endif
int main() {
    int i = _mm_popcnt_u64(1);
    return 0;
}
