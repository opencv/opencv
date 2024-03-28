/* crc32_fold_vpclmulqdq_tpl.h -- VPCMULQDQ-based CRC32 folding template.
 * Copyright Wangyang Guo (wangyang.guo@intel.com)
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef COPY
static size_t fold_16_vpclmulqdq_copy(__m128i *xmm_crc0, __m128i *xmm_crc1,
    __m128i *xmm_crc2, __m128i *xmm_crc3, uint8_t *dst, const uint8_t *src, size_t len) {
#else
static size_t fold_16_vpclmulqdq(__m128i *xmm_crc0, __m128i *xmm_crc1,
    __m128i *xmm_crc2, __m128i *xmm_crc3, const uint8_t *src, size_t len,
    __m128i init_crc, int32_t first) {
    __m512i zmm_initial = _mm512_zextsi128_si512(init_crc);
#endif
    __m512i zmm_t0, zmm_t1, zmm_t2, zmm_t3;
    __m512i zmm_crc0, zmm_crc1, zmm_crc2, zmm_crc3;
    __m512i z0, z1, z2, z3;
    size_t len_tmp = len;
    const __m512i zmm_fold4 = _mm512_set4_epi32(
        0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596);
    const __m512i zmm_fold16 = _mm512_set4_epi32(
        0x00000001, 0x1542778a, 0x00000001, 0x322d1430);

    // zmm register init
    zmm_crc0 = _mm512_setzero_si512();
    zmm_t0 = _mm512_loadu_si512((__m512i *)src);
#ifndef COPY
    XOR_INITIAL512(zmm_t0);
#endif
    zmm_crc1 = _mm512_loadu_si512((__m512i *)src + 1);
    zmm_crc2 = _mm512_loadu_si512((__m512i *)src + 2);
    zmm_crc3 = _mm512_loadu_si512((__m512i *)src + 3);

    /* already have intermediate CRC in xmm registers
        * fold4 with 4 xmm_crc to get zmm_crc0
    */
    zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc0, 0);
    zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc1, 1);
    zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc2, 2);
    zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc3, 3);
    z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
    zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
    zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_t0, 0x96);

#ifdef COPY
    _mm512_storeu_si512((__m512i *)dst, zmm_t0);
    _mm512_storeu_si512((__m512i *)dst + 1, zmm_crc1);
    _mm512_storeu_si512((__m512i *)dst + 2, zmm_crc2);
    _mm512_storeu_si512((__m512i *)dst + 3, zmm_crc3);
    dst += 256;
#endif
    len -= 256;
    src += 256;

    // fold-16 loops
    while (len >= 256) {
        zmm_t0 = _mm512_loadu_si512((__m512i *)src);
        zmm_t1 = _mm512_loadu_si512((__m512i *)src + 1);
        zmm_t2 = _mm512_loadu_si512((__m512i *)src + 2);
        zmm_t3 = _mm512_loadu_si512((__m512i *)src + 3);

        z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold16, 0x01);
        z1 = _mm512_clmulepi64_epi128(zmm_crc1, zmm_fold16, 0x01);
        z2 = _mm512_clmulepi64_epi128(zmm_crc2, zmm_fold16, 0x01);
        z3 = _mm512_clmulepi64_epi128(zmm_crc3, zmm_fold16, 0x01);

        zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold16, 0x10);
        zmm_crc1 = _mm512_clmulepi64_epi128(zmm_crc1, zmm_fold16, 0x10);
        zmm_crc2 = _mm512_clmulepi64_epi128(zmm_crc2, zmm_fold16, 0x10);
        zmm_crc3 = _mm512_clmulepi64_epi128(zmm_crc3, zmm_fold16, 0x10);

        zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_t0, 0x96);
        zmm_crc1 = _mm512_ternarylogic_epi32(zmm_crc1, z1, zmm_t1, 0x96);
        zmm_crc2 = _mm512_ternarylogic_epi32(zmm_crc2, z2, zmm_t2, 0x96);
        zmm_crc3 = _mm512_ternarylogic_epi32(zmm_crc3, z3, zmm_t3, 0x96);

#ifdef COPY
        _mm512_storeu_si512((__m512i *)dst, zmm_t0);
        _mm512_storeu_si512((__m512i *)dst + 1, zmm_t1);
        _mm512_storeu_si512((__m512i *)dst + 2, zmm_t2);
        _mm512_storeu_si512((__m512i *)dst + 3, zmm_t3);
        dst += 256;
#endif
        len -= 256;
        src += 256;
    }
    // zmm_crc[0,1,2,3] -> zmm_crc0
    z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
    zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
    zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc1, 0x96);

    z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
    zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
    zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc2, 0x96);

    z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
    zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
    zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc3, 0x96);

    // zmm_crc0 -> xmm_crc[0, 1, 2, 3]
    *xmm_crc0 = _mm512_extracti32x4_epi32(zmm_crc0, 0);
    *xmm_crc1 = _mm512_extracti32x4_epi32(zmm_crc0, 1);
    *xmm_crc2 = _mm512_extracti32x4_epi32(zmm_crc0, 2);
    *xmm_crc3 = _mm512_extracti32x4_epi32(zmm_crc0, 3);

    return (len_tmp - len);  // return n bytes processed
}
