/* slide_hash_armv6.c -- Optimized hash table shifting for ARMv6 with support for SIMD instructions
 * Copyright (C) 2023 Cameron Cawley
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#if defined(ARM_SIMD)
#include "acle_intrins.h"
#include "zbuild.h"
#include "deflate.h"

/* SIMD version of hash_chain rebase */
static inline void slide_hash_chain(Pos *table, uint32_t entries, uint16_t wsize) {
    Z_REGISTER uint16x2_t v;
    uint16x2_t p0, p1, p2, p3;
    Z_REGISTER size_t n;

    size_t size = entries*sizeof(table[0]);
    Assert((size % (sizeof(uint16x2_t) * 4) == 0), "hash table size err");

    Assert(sizeof(Pos) == 2, "Wrong Pos size");
    v = wsize | (wsize << 16);

    n = size / (sizeof(uint16x2_t) * 4);
    do {
        p0 = *((const uint16x2_t *)(table));
        p1 = *((const uint16x2_t *)(table+2));
        p2 = *((const uint16x2_t *)(table+4));
        p3 = *((const uint16x2_t *)(table+6));
        p0 = __uqsub16(p0, v);
        p1 = __uqsub16(p1, v);
        p2 = __uqsub16(p2, v);
        p3 = __uqsub16(p3, v);
        *((uint16x2_t *)(table)) = p0;
        *((uint16x2_t *)(table+2)) = p1;
        *((uint16x2_t *)(table+4)) = p2;
        *((uint16x2_t *)(table+6)) = p3;
        table += 8;
    } while (--n);
}

Z_INTERNAL void slide_hash_armv6(deflate_state *s) {
    unsigned int wsize = s->w_size;

    slide_hash_chain(s->head, HASH_SIZE, wsize);
    slide_hash_chain(s->prev, wsize, wsize);
}
#endif
