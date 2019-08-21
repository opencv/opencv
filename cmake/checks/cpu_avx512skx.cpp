#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>

// Workaround for problem with GCC 5-6 in -O0 mode
struct v_uint32x16
{
    __m512i val;
    explicit v_uint32x16(__m512i v) : val(v) {}
};
inline v_uint32x16 operator << (const v_uint32x16& a, int imm)
{
    return v_uint32x16(_mm512_slli_epi32(a.val, imm));
}

void test()
{
    __m512i zmm = _mm512_setzero_si512();
    __m256i a = _mm256_setzero_si256();
    __m256i b = _mm256_abs_epi64(a); // VL
    __m512i c = _mm512_abs_epi8(zmm); // BW
    __m512i d = _mm512_broadcast_i32x8(b); // DQ
    v_uint32x16 e(d); e = e << 10;
    __m512i f = _mm512_packus_epi32(d,d);
#if defined __GNUC__ && defined __x86_64__
    asm volatile ("" : : : "zmm16", "zmm17", "zmm18", "zmm19");
#endif
}

#else
#error "AVX512-SKX is not supported"
#endif
int main() { return 0; }
