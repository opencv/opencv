#include <nmmintrin.h>

int main()
{
    unsigned int res = _mm_crc32_u8(1, 2);
    return 0;
}
