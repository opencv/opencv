/*
 * AVX2 optimized hash slide, based on Intel's slide_sse implementation
 *
 * Copyright (C) 2017 Intel Corporation
 * Authors:
 *   Arjan van de Ven   <arjan@linux.intel.com>
 *   Jim Kukunas        <james.t.kukunas@linux.intel.com>
 *   Mika T. Lindqvist  <postmaster@raasu.org>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#include "../../zbuild.h"
#include "../../deflate.h"

#include <immintrin.h>

static inline void slide_hash_chain(Pos *table, uint32_t entries, const __m256i wsize) {
    table += entries;
    table -= 16;

    do {
        __m256i value, result;

        value = _mm256_loadu_si256((__m256i *)table);
        result = _mm256_subs_epu16(value, wsize);
        _mm256_storeu_si256((__m256i *)table, result);

        table -= 16;
        entries -= 16;
    } while (entries > 0);
}

Z_INTERNAL void slide_hash_avx2(deflate_state *s) {
    uint16_t wsize = (uint16_t)s->w_size;
    const __m256i ymm_wsize = _mm256_set1_epi16((short)wsize);

    slide_hash_chain(s->head, HASH_SIZE, ymm_wsize);
    slide_hash_chain(s->prev, wsize, ymm_wsize);
}
