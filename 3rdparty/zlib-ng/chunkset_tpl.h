/* chunkset_tpl.h -- inline functions to copy small data chunks.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include <stdlib.h>

#if CHUNK_SIZE == 32 && defined(X86_SSSE3)
extern uint8_t* chunkmemset_ssse3(uint8_t *out, unsigned dist, unsigned len);
#endif

/* Returns the chunk size */
Z_INTERNAL uint32_t CHUNKSIZE(void) {
    return sizeof(chunk_t);
}

/* Behave like memcpy, but assume that it's OK to overwrite at least
   chunk_t bytes of output even if the length is shorter than this,
   that the length is non-zero, and that `from` lags `out` by at least
   sizeof chunk_t bytes (or that they don't overlap at all or simply that
   the distance is less than the length of the copy).

   Aside from better memory bus utilisation, this means that short copies
   (chunk_t bytes or fewer) will fall straight through the loop
   without iteration, which will hopefully make the branch prediction more
   reliable. */
#ifndef HAVE_CHUNKCOPY
static inline uint8_t* CHUNKCOPY(uint8_t *out, uint8_t const *from, unsigned len) {
    Assert(len > 0, "chunkcopy should never have a length 0");
    chunk_t chunk;
    int32_t align = ((len - 1) % sizeof(chunk_t)) + 1;
    loadchunk(from, &chunk);
    storechunk(out, &chunk);
    out += align;
    from += align;
    len -= align;
    while (len > 0) {
        loadchunk(from, &chunk);
        storechunk(out, &chunk);
        out += sizeof(chunk_t);
        from += sizeof(chunk_t);
        len -= sizeof(chunk_t);
    }
    return out;
}
#endif

/* Perform short copies until distance can be rewritten as being at least
   sizeof chunk_t.

   This assumes that it's OK to overwrite at least the first
   2*sizeof(chunk_t) bytes of output even if the copy is shorter than this.
   This assumption holds because inflate_fast() starts every iteration with at
   least 258 bytes of output space available (258 being the maximum length
   output from a single token; see inflate_fast()'s assumptions below). */
#ifndef HAVE_CHUNKUNROLL
static inline uint8_t* CHUNKUNROLL(uint8_t *out, unsigned *dist, unsigned *len) {
    unsigned char const *from = out - *dist;
    chunk_t chunk;
    while (*dist < *len && *dist < sizeof(chunk_t)) {
        loadchunk(from, &chunk);
        storechunk(out, &chunk);
        out += *dist;
        *len -= *dist;
        *dist += *dist;
    }
    return out;
}
#endif

#ifndef HAVE_CHUNK_MAG
/* Loads a magazine to feed into memory of the pattern */
static inline chunk_t GET_CHUNK_MAG(uint8_t *buf, uint32_t *chunk_rem, uint32_t dist) {
        /* This code takes string of length dist from "from" and repeats
         * it for as many times as can fit in a chunk_t (vector register) */
        uint32_t cpy_dist;
        uint32_t bytes_remaining = sizeof(chunk_t);
        chunk_t chunk_load;
        uint8_t *cur_chunk = (uint8_t *)&chunk_load;
        while (bytes_remaining) {
            cpy_dist = MIN(dist, bytes_remaining);
            memcpy(cur_chunk, buf, cpy_dist);
            bytes_remaining -= cpy_dist;
            cur_chunk += cpy_dist;
            /* This allows us to bypass an expensive integer division since we're effectively
             * counting in this loop, anyway */
            *chunk_rem = cpy_dist;
        }

        return chunk_load;
}
#endif

/* Copy DIST bytes from OUT - DIST into OUT + DIST * k, for 0 <= k < LEN/DIST.
   Return OUT + LEN. */
Z_INTERNAL uint8_t* CHUNKMEMSET(uint8_t *out, unsigned dist, unsigned len) {
    /* Debug performance related issues when len < sizeof(uint64_t):
       Assert(len >= sizeof(uint64_t), "chunkmemset should be called on larger chunks"); */
    Assert(dist > 0, "chunkmemset cannot have a distance 0");
    /* Only AVX2 */
#if CHUNK_SIZE == 32 && defined(X86_SSSE3)
    if (len <= 16) {
        return chunkmemset_ssse3(out, dist, len);
    }
#endif

    uint8_t *from = out - dist;

    if (dist == 1) {
        memset(out, *from, len);
        return out + len;
    } else if (dist > sizeof(chunk_t)) {
        return CHUNKCOPY(out, out - dist, len);
    }

    chunk_t chunk_load;
    uint32_t chunk_mod = 0;

    /* TODO: possibly build up a permutation table for this if not an even modulus */
#ifdef HAVE_CHUNKMEMSET_2
    if (dist == 2) {
        chunkmemset_2(from, &chunk_load);
    } else
#endif
#ifdef HAVE_CHUNKMEMSET_4
    if (dist == 4) {
        chunkmemset_4(from, &chunk_load);
    } else
#endif
#ifdef HAVE_CHUNKMEMSET_8
    if (dist == 8) {
        chunkmemset_8(from, &chunk_load);
    } else if (dist == sizeof(chunk_t)) {
        loadchunk(from, &chunk_load);
    } else
#endif
    {
        chunk_load = GET_CHUNK_MAG(from, &chunk_mod, dist);
    }

    /* If we're lucky enough and dist happens to be an even modulus of our vector length,
     * we can do two stores per loop iteration, which for most ISAs, especially x86, is beneficial */
    if (chunk_mod == 0) {
        while (len >= (2 * sizeof(chunk_t))) {
            storechunk(out, &chunk_load);
            storechunk(out + sizeof(chunk_t), &chunk_load);
            out += 2 * sizeof(chunk_t);
            len -= 2 * sizeof(chunk_t);
        }
    }

    /* If we don't have a "dist" length that divides evenly into a vector
     * register, we can write the whole vector register but we need only
     * advance by the amount of the whole string that fits in our chunk_t.
     * If we do divide evenly into the vector length, adv_amount = chunk_t size*/
    uint32_t adv_amount = sizeof(chunk_t) - chunk_mod;
    while (len >= sizeof(chunk_t)) {
        storechunk(out, &chunk_load);
        len -= adv_amount;
        out += adv_amount;
    }

    if (len) {
        memcpy(out, &chunk_load, len);
        out += len;
    }

    return out;
}

Z_INTERNAL uint8_t* CHUNKMEMSET_SAFE(uint8_t *out, unsigned dist, unsigned len, unsigned left) {
#if !defined(UNALIGNED64_OK)
#  if !defined(UNALIGNED_OK)
    static const uint32_t align_mask = 7;
#  else
    static const uint32_t align_mask = 3;
#  endif
#endif

    len = MIN(len, left);
    uint8_t *from = out - dist;
#if !defined(UNALIGNED64_OK)
    while (((uintptr_t)out & align_mask) && (len > 0)) {
        *out++ = *from++;
        --len;
        --left;
    }
#endif
    if (left < (unsigned)(3 * sizeof(chunk_t))) {
        while (len > 0) {
            *out++ = *from++;
            --len;
        }
        return out;
    }
    if (len)
        return CHUNKMEMSET(out, dist, len);

    return out;
}
