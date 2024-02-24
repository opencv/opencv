/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_compress.h"
#include "internal_decompress.h"

#include "internal_coding.h"
#include "internal_structs.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "openexr_compression.h"

#if defined __SSE2__ || (_MSC_VER >= 1300 && (_M_IX86 || _M_X64))
#    define IMF_HAVE_SSE2 1
#    include <emmintrin.h>
#    include <mmintrin.h>
#endif
#if defined __SSE4_1__
#    define IMF_HAVE_SSE4_1 1
#    include <smmintrin.h>
#endif
#if defined(__aarch64__)
#    define IMF_HAVE_NEON_AARCH64 1
#    include <arm_neon.h>
#endif

/**************************************/

#ifdef IMF_HAVE_SSE4_1
static void
reconstruct (uint8_t* buf, uint64_t outSize)
{
    static const uint64_t bytesPerChunk = sizeof (__m128i);
    const uint64_t        vOutSize      = outSize / bytesPerChunk;
    const __m128i         c             = _mm_set1_epi8 (-128);
    const __m128i         shuffleMask   = _mm_set1_epi8 (15);
    __m128i *             vBuf, vPrev;
    uint8_t               prev;

    /*
     * The first element doesn't have its high bit flipped during compression,
     * so it must not be flipped here.  To make the SIMD loop nice and
     * uniform, we pre-flip the bit so that the loop will unflip it again.
     */
    buf[0] += -128;
    vBuf  = (__m128i*) buf;
    vPrev = _mm_setzero_si128 ();

    for (uint64_t i = 0; i < vOutSize; ++i)
    {
        __m128i d = _mm_add_epi8 (_mm_loadu_si128 (vBuf), c);

        /* Compute the prefix sum of elements. */
        d = _mm_add_epi8 (d, _mm_slli_si128 (d, 1));
        d = _mm_add_epi8 (d, _mm_slli_si128 (d, 2));
        d = _mm_add_epi8 (d, _mm_slli_si128 (d, 4));
        d = _mm_add_epi8 (d, _mm_slli_si128 (d, 8));
        d = _mm_add_epi8 (d, vPrev);

        _mm_storeu_si128 (vBuf++, d);

        // Broadcast the high byte in our result to all lanes of the prev
        // value for the next iteration.
        vPrev = _mm_shuffle_epi8 (d, shuffleMask);
    }

    prev = _mm_extract_epi8 (vPrev, 15);
    for (uint64_t i = vOutSize * bytesPerChunk; i < outSize; ++i)
    {
        uint8_t d = prev + buf[i] - 128;
        buf[i]    = d;
        prev      = d;
    }
}
#elif defined(IMF_HAVE_NEON_AARCH64)
static void
reconstruct (uint8_t* buf, uint64_t outSize)
{
    static const uint64_t bytesPerChunk = sizeof (uint8x16_t);
    const uint64_t        vOutSize      = outSize / bytesPerChunk;
    const uint8x16_t      c             = vdupq_n_u8 (-128);
    const uint8x16_t      shuffleMask   = vdupq_n_u8 (15);
    const uint8x16_t      zero          = vdupq_n_u8 (0);
    uint8_t*              vBuf;
    uint8x16_t            vPrev;
    uint8_t               prev;

    /*
     * The first element doesn't have its high bit flipped during compression,
     * so it must not be flipped here.  To make the SIMD loop nice and
     * uniform, we pre-flip the bit so that the loop will unflip it again.
     */
    buf[0] += -128;
    vBuf  = buf;
    vPrev = vdupq_n_u8 (0);

    for (uint64_t i = 0; i < vOutSize; ++i)
    {
        uint8x16_t d = vaddq_u8 (vld1q_u8 (vBuf), c);

        /* Compute the prefix sum of elements. */
        d = vaddq_u8 (d, vextq_u8 (zero, d, 16 - 1));
        d = vaddq_u8 (d, vextq_u8 (zero, d, 16 - 2));
        d = vaddq_u8 (d, vextq_u8 (zero, d, 16 - 4));
        d = vaddq_u8 (d, vextq_u8 (zero, d, 16 - 8));
        d = vaddq_u8 (d, vPrev);

        vst1q_u8 (vBuf, d);
        vBuf += sizeof (uint8x16_t);

        // Broadcast the high byte in our result to all lanes of the prev
        // value for the next iteration.
        vPrev = vqtbl1q_u8 (d, shuffleMask);
    }

    prev = vgetq_lane_u8 (vPrev, 15);
    for (uint64_t i = vOutSize * bytesPerChunk; i < outSize; ++i)
    {
        uint8_t d = prev + buf[i] - 128;
        buf[i]    = d;
        prev      = d;
    }
}
#else
static void
reconstruct (uint8_t* buf, uint64_t sz)
{
    uint8_t* t    = buf + 1;
    uint8_t* stop = buf + sz;
    while (t < stop)
    {
        int d = (int) (t[-1]) + (int) (t[0]) - 128;
        t[0]  = (uint8_t) d;
        ++t;
    }
}
#endif

/**************************************/

#ifdef IMF_HAVE_SSE2
static void
interleave (uint8_t* out, const uint8_t* source, uint64_t outSize)
{
    static const uint64_t bytesPerChunk = 2 * sizeof (__m128i);
    const uint64_t        vOutSize      = outSize / bytesPerChunk;
    const __m128i*        v1            = (const __m128i*) source;
    const __m128i*        v2   = (const __m128i*) (source + (outSize + 1) / 2);
    __m128i*              vOut = (__m128i*) out;
    const uint8_t *       t1, *t2;
    uint8_t*              sOut;

    for (uint64_t i = 0; i < vOutSize; ++i)
    {
        __m128i a  = _mm_loadu_si128 (v1++);
        __m128i b  = _mm_loadu_si128 (v2++);
        __m128i lo = _mm_unpacklo_epi8 (a, b);
        __m128i hi = _mm_unpackhi_epi8 (a, b);

        _mm_storeu_si128 (vOut++, lo);
        _mm_storeu_si128 (vOut++, hi);
    }

    t1   = (const uint8_t*) v1;
    t2   = (const uint8_t*) v2;
    sOut = (uint8_t*) vOut;

    for (uint64_t i = vOutSize * bytesPerChunk; i < outSize; ++i)
        *(sOut++) = (i % 2 == 0) ? *(t1++) : *(t2++);
}

#elif defined(IMF_HAVE_NEON_AARCH64)
static void
interleave (uint8_t* out, const uint8_t* source, uint64_t outSize)
{
    static const uint64_t bytesPerChunk = 2 * sizeof (uint8x16_t);
    const uint64_t        vOutSize      = outSize / bytesPerChunk;
    const uint8_t*        v1            = source;
    const uint8_t*        v2            = source + (outSize + 1) / 2;

    for (uint64_t i = 0; i < vOutSize; ++i)
    {
        uint8x16_t a = vld1q_u8 (v1);
        v1 += sizeof (uint8x16_t);
        uint8x16_t b = vld1q_u8 (v2);
        v2 += sizeof (uint8x16_t);
        uint8x16_t lo = vzip1q_u8 (a, b);
        uint8x16_t hi = vzip2q_u8 (a, b);

        vst1q_u8 (out, lo);
        out += sizeof (uint8x16_t);
        vst1q_u8 (out, hi);
        out += sizeof (uint8x16_t);
    }

    for (uint64_t i = vOutSize * bytesPerChunk; i < outSize; ++i)
        *(out++) = (i % 2 == 0) ? *(v1++) : *(v2++);
}

#else

static void
interleave (uint8_t* out, const uint8_t* source, uint64_t outSize)
{
    const uint8_t* t1   = source;
    const uint8_t* t2   = source + (outSize + 1) / 2;
    uint8_t*       s    = out;
    uint8_t* const stop = s + outSize;

    while (1)
    {
        if (s < stop)
            *(s++) = *(t1++);
        else
            break;

        if (s < stop)
            *(s++) = *(t2++);
        else
            break;
    }
}

#endif

/**************************************/

void
internal_zip_reconstruct_bytes (uint8_t* out, uint8_t* source, uint64_t count)
{
    reconstruct (source, count);
    interleave (out, source, count);
}

/**************************************/

void
internal_zip_deconstruct_bytes (
    uint8_t* scratch, const uint8_t* source, uint64_t count)
{
    int            p;
    uint8_t*       t1   = scratch;
    uint8_t*       t2   = t1 + (count + 1) / 2;
    const uint8_t* raw  = source;
    const uint8_t* stop = raw + count;

    /* reorder */
    while (raw < stop)
    {
        *(t1++) = *(raw++);
        if (raw < stop) *(t2++) = *(raw++);
    }

    /* reorder */
    t1 = scratch;
    t2 = t1 + count;
    t1++;
    p = (int) t1[-1];
    while (t1 < t2)
    {
        int d = (int) (t1[0]) - p + (128 + 256);
        p     = (int) t1[0];
        t1[0] = (uint8_t) d;
        ++t1;
    }
}

/**************************************/

static exr_result_t
undo_zip_impl (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size,
    void*                  scratch_data,
    uint64_t               scratch_size)
{
    size_t       actual_out_bytes;
    exr_result_t res;

    if (scratch_size < uncompressed_size) return EXR_ERR_INVALID_ARGUMENT;

    res = exr_uncompress_buffer (
        decode->context,
        compressed_data,
        comp_buf_size,
        scratch_data,
        scratch_size,
        &actual_out_bytes);

    if (res == EXR_ERR_SUCCESS)
    {
        if (actual_out_bytes == uncompressed_size)
        {
            internal_zip_reconstruct_bytes (
                uncompressed_data, scratch_data, actual_out_bytes);
        }
        else
            res = EXR_ERR_CORRUPT_CHUNK;
    }

    return res;
}

/**************************************/

exr_result_t
internal_exr_undo_zip (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size)
{
    exr_result_t rv;
    uint64_t     scratchbufsz = uncompressed_size;
    if (comp_buf_size > scratchbufsz) scratchbufsz = comp_buf_size;

    rv = internal_decode_alloc_buffer (
        decode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(decode->scratch_buffer_1),
        &(decode->scratch_alloc_size_1),
        scratchbufsz);
    if (rv != EXR_ERR_SUCCESS) return rv;
    return undo_zip_impl (
        decode,
        compressed_data,
        comp_buf_size,
        uncompressed_data,
        uncompressed_size,
        decode->scratch_buffer_1,
        decode->scratch_alloc_size_1);
}

/**************************************/

static exr_result_t
apply_zip_impl (exr_encode_pipeline_t* encode)
{
    int          level;
    size_t       compbufsz;
    exr_result_t rv;

    rv = exr_get_zip_compression_level (
        encode->context, encode->part_index, &level);
    if (rv != EXR_ERR_SUCCESS) return rv;

    internal_zip_deconstruct_bytes (
        encode->scratch_buffer_1, encode->packed_buffer, encode->packed_bytes);

    rv = exr_compress_buffer (
        encode->context,
        level,
        encode->scratch_buffer_1,
        encode->packed_bytes,
        encode->compressed_buffer,
        encode->compressed_alloc_size,
        &compbufsz);

    if (rv == EXR_ERR_SUCCESS)
    {
        if (compbufsz > encode->packed_bytes)
        {
            memcpy (
                encode->compressed_buffer,
                encode->packed_buffer,
                encode->packed_bytes);
            compbufsz = encode->packed_bytes;
        }
        encode->compressed_bytes = compbufsz;
    }
    else
    {
        const struct _internal_exr_context* pctxt = EXR_CCTXT (encode->context);
        if (pctxt)
            pctxt->print_error (
                pctxt,
                rv,
                "Unable to compress buffer %" PRIu64 " -> %" PRIu64
                " @ level %d",
                encode->packed_bytes,
                (uint64_t) encode->compressed_alloc_size,
                level);
    }

    return rv;
}

exr_result_t
internal_exr_apply_zip (exr_encode_pipeline_t* encode)
{
    exr_result_t rv;

    rv = internal_encode_alloc_buffer (
        encode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(encode->scratch_buffer_1),
        &(encode->scratch_alloc_size_1),
        encode->packed_bytes);
    if (rv != EXR_ERR_SUCCESS)
    {
        const struct _internal_exr_context* pctxt = EXR_CCTXT (encode->context);
        if (pctxt)
            pctxt->print_error (
                pctxt,
                rv,
                "Unable to allocate scratch buffer for deflate of %" PRIu64
                " bytes",
                encode->packed_bytes);
        return rv;
    }

    return apply_zip_impl (encode);
}
