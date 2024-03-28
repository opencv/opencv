/* crc32 for POWER8 using VSX instructions
 * Copyright (C) 2021 IBM Corporation
 *
 * Author: Rogerio Alves <rogealve@br.ibm.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * Calculate the checksum of data that is 16 byte aligned and a multiple of
 * 16 bytes.
 *
 * The first step is to reduce it to 1024 bits. We do this in 8 parallel
 * chunks in order to mask the latency of the vpmsum instructions. If we
 * have more than 32 kB of data to checksum we repeat this step multiple
 * times, passing in the previous 1024 bits.
 *
 * The next step is to reduce the 1024 bits to 64 bits. This step adds
 * 32 bits of 0s to the end - this matches what a CRC does. We just
 * calculate constants that land the data in this 32 bits.
 *
 * We then use fixed point Barrett reduction to compute a mod n over GF(2)
 * for n = CRC using POWER8 instructions. We use x = 32.
 *
 * http://en.wikipedia.org/wiki/Barrett_reduction
 *
 * This code uses gcc vector builtins instead using assembly directly.
 */

#include <altivec.h>
#include "zendian.h"
#include "zbuild.h"

#include "crc32_constants.h"
#include "crc32_braid_tbl.h"

#if defined (__clang__)
#include "fallback_builtins.h"
#endif

#define MAX_SIZE    32768
#define VMX_ALIGN	16
#define VMX_ALIGN_MASK	(VMX_ALIGN-1)

static unsigned int crc32_align(unsigned int crc, const unsigned char *p, unsigned long len) {
    while (len--)
        crc = crc_table[(crc ^ *p++) & 0xff] ^ (crc >> 8);
    return crc;
}

static unsigned int ALIGNED_(32) __crc32_vpmsum(unsigned int crc, const void* p, unsigned long len);

Z_INTERNAL uint32_t crc32_power8(uint32_t crc, const unsigned char *p, size_t _len) {
    unsigned int prealign;
    unsigned int tail;

    unsigned long len = (unsigned long) _len;

    if (p == (const unsigned char *) 0x0)
        return 0;

    crc ^= 0xffffffff;

    if (len < VMX_ALIGN + VMX_ALIGN_MASK) {
        crc = crc32_align(crc, p, len);
        goto out;
    }

    if ((unsigned long)p & VMX_ALIGN_MASK) {
        prealign = VMX_ALIGN - ((unsigned long)p & VMX_ALIGN_MASK);
        crc = crc32_align(crc, p, prealign);
        len -= prealign;
        p += prealign;
    }

    crc = __crc32_vpmsum(crc, p, len & ~VMX_ALIGN_MASK);

    tail = len & VMX_ALIGN_MASK;
    if (tail) {
        p += len & ~VMX_ALIGN_MASK;
        crc = crc32_align(crc, p, tail);
    }

out:
    crc ^= 0xffffffff;

    return crc;
}

/* When we have a load-store in a single-dispatch group and address overlap
 * such that forward is not allowed (load-hit-store) the group must be flushed.
 * A group ending NOP prevents the flush.
 */
#define GROUP_ENDING_NOP __asm__("ori 2,2,0" ::: "memory")

#if BYTE_ORDER == BIG_ENDIAN
#define BYTESWAP_DATA
#endif

#ifdef BYTESWAP_DATA
#define VEC_PERM(vr, va, vb, vc) vr = vec_perm(va, vb, (__vector unsigned char) vc)
#if BYTE_ORDER == LITTLE_ENDIAN
/* Byte reverse permute constant LE. */
static const __vector unsigned long long vperm_const ALIGNED_(16) = { 0x08090A0B0C0D0E0FUL, 0x0001020304050607UL };
#else
static const __vector unsigned long long vperm_const ALIGNED_(16) = { 0x0F0E0D0C0B0A0908UL, 0X0706050403020100UL };
#endif
#else
#define VEC_PERM(vr, va, vb, vc)
#endif

static unsigned int ALIGNED_(32) __crc32_vpmsum(unsigned int crc, const void* p, unsigned long len) {

    const __vector unsigned long long vzero = {0,0};
    const __vector unsigned long long vones = {0xffffffffffffffffUL, 0xffffffffffffffffUL};

    const __vector unsigned long long vmask_32bit =
        (__vector unsigned long long)vec_sld((__vector unsigned char)vzero, (__vector unsigned char)vones, 4);

    const __vector unsigned long long vmask_64bit =
        (__vector unsigned long long)vec_sld((__vector unsigned char)vzero, (__vector unsigned char)vones, 8);

    __vector unsigned long long vcrc;

    __vector unsigned long long vconst1, vconst2;

    /* vdata0-vdata7 will contain our data (p). */
    __vector unsigned long long vdata0, vdata1, vdata2, vdata3, vdata4, vdata5, vdata6, vdata7;

    /* v0-v7 will contain our checksums */
    __vector unsigned long long v0 = {0,0};
    __vector unsigned long long v1 = {0,0};
    __vector unsigned long long v2 = {0,0};
    __vector unsigned long long v3 = {0,0};
    __vector unsigned long long v4 = {0,0};
    __vector unsigned long long v5 = {0,0};
    __vector unsigned long long v6 = {0,0};
    __vector unsigned long long v7 = {0,0};


    /* Vector auxiliary variables. */
    __vector unsigned long long va0, va1, va2, va3, va4, va5, va6, va7;

    unsigned int offset; /* Constant table offset. */

    unsigned long i; /* Counter. */
    unsigned long chunks;

    unsigned long block_size;
    int next_block = 0;

    /* Align by 128 bits. The last 128 bit block will be processed at end. */
    unsigned long length = len & 0xFFFFFFFFFFFFFF80UL;

    vcrc = (__vector unsigned long long)__builtin_pack_vector_int128(0UL, crc);

    /* Short version. */
    if (len < 256) {
        /* Calculate where in the constant table we need to start. */
        offset = 256 - len;

        vconst1 = vec_ld(offset, vcrc_short_const);
        vdata0 = vec_ld(0, (__vector unsigned long long*) p);
        VEC_PERM(vdata0, vdata0, vconst1, vperm_const);

        /* xor initial value */
        vdata0 = vec_xor(vdata0, vcrc);

        vdata0 = (__vector unsigned long long) __builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata0, (__vector unsigned int)vconst1);
        v0 = vec_xor(v0, vdata0);

        for (i = 16; i < len; i += 16) {
            vconst1 = vec_ld(offset + i, vcrc_short_const);
            vdata0 = vec_ld(i, (__vector unsigned long long*) p);
            VEC_PERM(vdata0, vdata0, vconst1, vperm_const);
            vdata0 = (__vector unsigned long long) __builtin_crypto_vpmsumw(
                (__vector unsigned int)vdata0, (__vector unsigned int)vconst1);
            v0 = vec_xor(v0, vdata0);
        }
    } else {

        /* Load initial values. */
        vdata0 = vec_ld(0, (__vector unsigned long long*) p);
        vdata1 = vec_ld(16, (__vector unsigned long long*) p);

        VEC_PERM(vdata0, vdata0, vdata0, vperm_const);
        VEC_PERM(vdata1, vdata1, vdata1, vperm_const);

        vdata2 = vec_ld(32, (__vector unsigned long long*) p);
        vdata3 = vec_ld(48, (__vector unsigned long long*) p);

        VEC_PERM(vdata2, vdata2, vdata2, vperm_const);
        VEC_PERM(vdata3, vdata3, vdata3, vperm_const);

        vdata4 = vec_ld(64, (__vector unsigned long long*) p);
        vdata5 = vec_ld(80, (__vector unsigned long long*) p);

        VEC_PERM(vdata4, vdata4, vdata4, vperm_const);
        VEC_PERM(vdata5, vdata5, vdata5, vperm_const);

        vdata6 = vec_ld(96, (__vector unsigned long long*) p);
        vdata7 = vec_ld(112, (__vector unsigned long long*) p);

        VEC_PERM(vdata6, vdata6, vdata6, vperm_const);
        VEC_PERM(vdata7, vdata7, vdata7, vperm_const);

        /* xor in initial value */
        vdata0 = vec_xor(vdata0, vcrc);

        p = (char *)p + 128;

        do {
            /* Checksum in blocks of MAX_SIZE. */
            block_size = length;
            if (block_size > MAX_SIZE) {
                block_size = MAX_SIZE;
            }

            length = length - block_size;

            /*
             * Work out the offset into the constants table to start at. Each
             * constant is 16 bytes, and it is used against 128 bytes of input
             * data - 128 / 16 = 8
             */
            offset = (MAX_SIZE/8) - (block_size/8);
            /* We reduce our final 128 bytes in a separate step */
            chunks = (block_size/128)-1;

            vconst1 = vec_ld(offset, vcrc_const);

            va0 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata0,
                                           (__vector unsigned long long)vconst1);
            va1 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata1,
                                           (__vector unsigned long long)vconst1);
            va2 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata2,
                                           (__vector unsigned long long)vconst1);
            va3 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata3,
                                           (__vector unsigned long long)vconst1);
            va4 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata4,
                                           (__vector unsigned long long)vconst1);
            va5 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata5,
                                           (__vector unsigned long long)vconst1);
            va6 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata6,
                                           (__vector unsigned long long)vconst1);
            va7 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata7,
                                           (__vector unsigned long long)vconst1);

            if (chunks > 1) {
                offset += 16;
                vconst2 = vec_ld(offset, vcrc_const);
                GROUP_ENDING_NOP;

                vdata0 = vec_ld(0, (__vector unsigned long long*) p);
                VEC_PERM(vdata0, vdata0, vdata0, vperm_const);

                vdata1 = vec_ld(16, (__vector unsigned long long*) p);
                VEC_PERM(vdata1, vdata1, vdata1, vperm_const);

                vdata2 = vec_ld(32, (__vector unsigned long long*) p);
                VEC_PERM(vdata2, vdata2, vdata2, vperm_const);

                vdata3 = vec_ld(48, (__vector unsigned long long*) p);
                VEC_PERM(vdata3, vdata3, vdata3, vperm_const);

                vdata4 = vec_ld(64, (__vector unsigned long long*) p);
                VEC_PERM(vdata4, vdata4, vdata4, vperm_const);

                vdata5 = vec_ld(80, (__vector unsigned long long*) p);
                VEC_PERM(vdata5, vdata5, vdata5, vperm_const);

                vdata6 = vec_ld(96, (__vector unsigned long long*) p);
                VEC_PERM(vdata6, vdata6, vdata6, vperm_const);

                vdata7 = vec_ld(112, (__vector unsigned long long*) p);
                VEC_PERM(vdata7, vdata7, vdata7, vperm_const);

                p = (char *)p + 128;

                /*
                 * main loop. Each iteration calculates the CRC for a 128-byte
                 * block.
                 */
                for (i = 0; i < chunks-2; i++) {
                    vconst1 = vec_ld(offset, vcrc_const);
                    offset += 16;
                    GROUP_ENDING_NOP;

                    v0 = vec_xor(v0, va0);
                    va0 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata0,
                                                   (__vector unsigned long long)vconst2);
                    vdata0 = vec_ld(0, (__vector unsigned long long*) p);
                    VEC_PERM(vdata0, vdata0, vdata0, vperm_const);
                    GROUP_ENDING_NOP;

                    v1 = vec_xor(v1, va1);
                    va1 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata1,
                                                   (__vector unsigned long long)vconst2);
                    vdata1 = vec_ld(16, (__vector unsigned long long*) p);
                    VEC_PERM(vdata1, vdata1, vdata1, vperm_const);
                    GROUP_ENDING_NOP;

                    v2 = vec_xor(v2, va2);
                    va2 = __builtin_crypto_vpmsumd((__vector unsigned long long)
                                                   vdata2, (__vector unsigned long long)vconst2);
                    vdata2 = vec_ld(32, (__vector unsigned long long*) p);
                    VEC_PERM(vdata2, vdata2, vdata2, vperm_const);
                    GROUP_ENDING_NOP;

                    v3 = vec_xor(v3, va3);
                    va3 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata3,
                                                   (__vector unsigned long long)vconst2);
                    vdata3 = vec_ld(48, (__vector unsigned long long*) p);
                    VEC_PERM(vdata3, vdata3, vdata3, vperm_const);

                    vconst2 = vec_ld(offset, vcrc_const);
                    GROUP_ENDING_NOP;

                    v4 = vec_xor(v4, va4);
                    va4 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata4,
                                                   (__vector unsigned long long)vconst1);
                    vdata4 = vec_ld(64, (__vector unsigned long long*) p);
                    VEC_PERM(vdata4, vdata4, vdata4, vperm_const);
                    GROUP_ENDING_NOP;

                    v5 = vec_xor(v5, va5);
                    va5 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata5,
                                                   (__vector unsigned long long)vconst1);
                    vdata5 = vec_ld(80, (__vector unsigned long long*) p);
                    VEC_PERM(vdata5, vdata5, vdata5, vperm_const);
                    GROUP_ENDING_NOP;

                    v6 = vec_xor(v6, va6);
                    va6 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata6,
                                                   (__vector unsigned long long)vconst1);
                    vdata6 = vec_ld(96, (__vector unsigned long long*) p);
                    VEC_PERM(vdata6, vdata6, vdata6, vperm_const);
                    GROUP_ENDING_NOP;

                    v7 = vec_xor(v7, va7);
                    va7 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata7,
                                                   (__vector unsigned long long)vconst1);
                    vdata7 = vec_ld(112, (__vector unsigned long long*) p);
                    VEC_PERM(vdata7, vdata7, vdata7, vperm_const);

                    p = (char *)p + 128;
                }

                /* First cool down */
                vconst1 = vec_ld(offset, vcrc_const);
                offset += 16;

                v0 = vec_xor(v0, va0);
                va0 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata0,
                                               (__vector unsigned long long)vconst1);
                GROUP_ENDING_NOP;

                v1 = vec_xor(v1, va1);
                va1 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata1,
                                               (__vector unsigned long long)vconst1);
                GROUP_ENDING_NOP;

                v2 = vec_xor(v2, va2);
                va2 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata2,
                                               (__vector unsigned long long)vconst1);
                GROUP_ENDING_NOP;

                v3 = vec_xor(v3, va3);
                va3 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata3,
                                               (__vector unsigned long long)vconst1);
                GROUP_ENDING_NOP;

                v4 = vec_xor(v4, va4);
                va4 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata4,
                                               (__vector unsigned long long)vconst1);
                GROUP_ENDING_NOP;

                v5 = vec_xor(v5, va5);
                va5 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata5,
                                               (__vector unsigned long long)vconst1);
                GROUP_ENDING_NOP;

                v6 = vec_xor(v6, va6);
                va6 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata6,
                                               (__vector unsigned long long)vconst1);
                GROUP_ENDING_NOP;

                v7 = vec_xor(v7, va7);
                va7 = __builtin_crypto_vpmsumd((__vector unsigned long long)vdata7,
                                               (__vector unsigned long long)vconst1);
            }/* else */

            /* Second cool down. */
            v0 = vec_xor(v0, va0);
            v1 = vec_xor(v1, va1);
            v2 = vec_xor(v2, va2);
            v3 = vec_xor(v3, va3);
            v4 = vec_xor(v4, va4);
            v5 = vec_xor(v5, va5);
            v6 = vec_xor(v6, va6);
            v7 = vec_xor(v7, va7);

            /*
             * vpmsumd produces a 96 bit result in the least significant bits
             * of the register. Since we are bit reflected we have to shift it
             * left 32 bits so it occupies the least significant bits in the
             * bit reflected domain.
             */
            v0 = (__vector unsigned long long)vec_sld((__vector unsigned char)v0,
                                                      (__vector unsigned char)vzero, 4);
            v1 = (__vector unsigned long long)vec_sld((__vector unsigned char)v1,
                                                      (__vector unsigned char)vzero, 4);
            v2 = (__vector unsigned long long)vec_sld((__vector unsigned char)v2,
                                                      (__vector unsigned char)vzero, 4);
            v3 = (__vector unsigned long long)vec_sld((__vector unsigned char)v3,
                                                      (__vector unsigned char)vzero, 4);
            v4 = (__vector unsigned long long)vec_sld((__vector unsigned char)v4,
                                                      (__vector unsigned char)vzero, 4);
            v5 = (__vector unsigned long long)vec_sld((__vector unsigned char)v5,
                                                      (__vector unsigned char)vzero, 4);
            v6 = (__vector unsigned long long)vec_sld((__vector unsigned char)v6,
                                                      (__vector unsigned char)vzero, 4);
            v7 = (__vector unsigned long long)vec_sld((__vector unsigned char)v7,
                                                      (__vector unsigned char)vzero, 4);

            /* xor with the last 1024 bits. */
            va0 = vec_ld(0, (__vector unsigned long long*) p);
            VEC_PERM(va0, va0, va0, vperm_const);

            va1 = vec_ld(16, (__vector unsigned long long*) p);
            VEC_PERM(va1, va1, va1, vperm_const);

            va2 = vec_ld(32, (__vector unsigned long long*) p);
            VEC_PERM(va2, va2, va2, vperm_const);

            va3 = vec_ld(48, (__vector unsigned long long*) p);
            VEC_PERM(va3, va3, va3, vperm_const);

            va4 = vec_ld(64, (__vector unsigned long long*) p);
            VEC_PERM(va4, va4, va4, vperm_const);

            va5 = vec_ld(80, (__vector unsigned long long*) p);
            VEC_PERM(va5, va5, va5, vperm_const);

            va6 = vec_ld(96, (__vector unsigned long long*) p);
            VEC_PERM(va6, va6, va6, vperm_const);

            va7 = vec_ld(112, (__vector unsigned long long*) p);
            VEC_PERM(va7, va7, va7, vperm_const);

            p = (char *)p + 128;

            vdata0 = vec_xor(v0, va0);
            vdata1 = vec_xor(v1, va1);
            vdata2 = vec_xor(v2, va2);
            vdata3 = vec_xor(v3, va3);
            vdata4 = vec_xor(v4, va4);
            vdata5 = vec_xor(v5, va5);
            vdata6 = vec_xor(v6, va6);
            vdata7 = vec_xor(v7, va7);

            /* Check if we have more blocks to process */
            next_block = 0;
            if (length != 0) {
                next_block = 1;

                /* zero v0-v7 */
                v0 = vec_xor(v0, v0);
                v1 = vec_xor(v1, v1);
                v2 = vec_xor(v2, v2);
                v3 = vec_xor(v3, v3);
                v4 = vec_xor(v4, v4);
                v5 = vec_xor(v5, v5);
                v6 = vec_xor(v6, v6);
                v7 = vec_xor(v7, v7);
            }
            length = length + 128;

        } while (next_block);

        /* Calculate how many bytes we have left. */
        length = (len & 127);

        /* Calculate where in (short) constant table we need to start. */
        offset = 128 - length;

        v0 = vec_ld(offset, vcrc_short_const);
        v1 = vec_ld(offset + 16, vcrc_short_const);
        v2 = vec_ld(offset + 32, vcrc_short_const);
        v3 = vec_ld(offset + 48, vcrc_short_const);
        v4 = vec_ld(offset + 64, vcrc_short_const);
        v5 = vec_ld(offset + 80, vcrc_short_const);
        v6 = vec_ld(offset + 96, vcrc_short_const);
        v7 = vec_ld(offset + 112, vcrc_short_const);

        offset += 128;

        v0 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata0, (__vector unsigned int)v0);
        v1 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata1, (__vector unsigned int)v1);
        v2 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata2, (__vector unsigned int)v2);
        v3 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata3, (__vector unsigned int)v3);
        v4 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata4, (__vector unsigned int)v4);
        v5 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata5, (__vector unsigned int)v5);
        v6 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata6, (__vector unsigned int)v6);
        v7 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
            (__vector unsigned int)vdata7, (__vector unsigned int)v7);

        /* Now reduce the tail (0-112 bytes). */
        for (i = 0; i < length; i+=16) {
            vdata0 = vec_ld(i,(__vector unsigned long long*)p);
            VEC_PERM(vdata0, vdata0, vdata0, vperm_const);
            va0 = vec_ld(offset + i,vcrc_short_const);
            va0 = (__vector unsigned long long)__builtin_crypto_vpmsumw(
                (__vector unsigned int)vdata0, (__vector unsigned int)va0);
            v0 = vec_xor(v0, va0);
        }

        /* xor all parallel chunks together. */
        v0 = vec_xor(v0, v1);
        v2 = vec_xor(v2, v3);
        v4 = vec_xor(v4, v5);
        v6 = vec_xor(v6, v7);

        v0 = vec_xor(v0, v2);
        v4 = vec_xor(v4, v6);

        v0 = vec_xor(v0, v4);
    }

    /* Barrett Reduction */
    vconst1 = vec_ld(0, v_Barrett_const);
    vconst2 = vec_ld(16, v_Barrett_const);

    v1 = (__vector unsigned long long)vec_sld((__vector unsigned char)v0,
                                              (__vector unsigned char)v0, 8);
    v0 = vec_xor(v1,v0);

    /* shift left one bit */
    __vector unsigned char vsht_splat = vec_splat_u8 (1);
    v0 = (__vector unsigned long long)vec_sll((__vector unsigned char)v0, vsht_splat);

    v0 = vec_and(v0, vmask_64bit);

    /*
     * The reflected version of Barrett reduction. Instead of bit
     * reflecting our data (which is expensive to do), we bit reflect our
     * constants and our algorithm, which means the intermediate data in
     * our vector registers goes from 0-63 instead of 63-0. We can reflect
     * the algorithm because we don't carry in mod 2 arithmetic.
     */

    /* bottom 32 bits of a */
    v1 = vec_and(v0, vmask_32bit);

    /* ma */
    v1 = __builtin_crypto_vpmsumd((__vector unsigned long long)v1,
                                  (__vector unsigned long long)vconst1);

    /* bottom 32bits of ma */
    v1 = vec_and(v1, vmask_32bit);
    /* qn */
    v1 = __builtin_crypto_vpmsumd((__vector unsigned long long)v1,
                                  (__vector unsigned long long)vconst2);
    /* a - qn, subtraction is xor in GF(2) */
    v0 = vec_xor (v0, v1);

    /*
     * Since we are bit reflected, the result (ie the low 32 bits) is in
     * the high 32 bits. We just need to shift it left 4 bytes
     * V0 [ 0 1 X 3 ]
     * V0 [ 0 X 2 3 ]
     */

    /* shift result into top 64 bits of */
    v0 = (__vector unsigned long long)vec_sld((__vector unsigned char)v0,
                                              (__vector unsigned char)vzero, 4);

#if BYTE_ORDER == BIG_ENDIAN
    return v0[0];
#else
    return v0[1];
#endif
}
