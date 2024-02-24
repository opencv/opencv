/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_compress.h"
#include "internal_decompress.h"

#include "internal_coding.h"

#include <stdio.h>
#include <string.h>

#define MIN_RUN_LENGTH 3
#define MAX_RUN_LENGTH 127

uint64_t
internal_rle_compress (
    void* out, uint64_t outbytes, const void* src, uint64_t srcbytes)
{
    int8_t*       cbuf = out;
    const int8_t* runs = src;
    const int8_t* end  = runs + srcbytes;
    const int8_t* rune = runs + 1;
    uint64_t      outb = 0;

    while (runs < end)
    {
        uint8_t curcount = 0;
        while (rune < end && *runs == *rune && curcount < MAX_RUN_LENGTH)
        {
            ++rune;
            ++curcount;
        }

        if (curcount >= (MIN_RUN_LENGTH - 1))
        {
            cbuf[outb++] = (int8_t) curcount;
            cbuf[outb++] = *runs;

            runs = rune;
        }
        else
        {
            /* incompressible */
            ++curcount;
            while (rune < end &&
                   ((rune + 1 >= end || *rune != *(rune + 1)) ||
                    (rune + 2 >= end || *(rune + 1) != *(rune + 2))) &&
                   curcount < MAX_RUN_LENGTH)
            {
                ++curcount;
                ++rune;
            }
            cbuf[outb++] = (int8_t) (-((int) curcount));
            while (runs < rune)
                cbuf[outb++] = *runs++;
        }
        ++rune;
        if (outb >= outbytes) break;
    }
    return outb;
}

/**************************************/

static void
reorder_and_predict (void* scratch, const void* packed, uint64_t packedbytes)
{
    int8_t*       t1   = scratch;
    int8_t*       t2   = t1 + (packedbytes + 1) / 2;
    const int8_t* in   = packed;
    const int8_t* stop = in + packedbytes;
    int           d, p;

    while (in < stop)
    {
        *(t1++) = *(in++);
        if (in < stop) *(t2++) = *(in++);
    }

    t1   = scratch;
    stop = t1 + packedbytes;
    p    = *(t1++);
    while (t1 < stop)
    {
        d     = (int) (*t1) - p + (128 + 256);
        p     = *t1;
        *t1++ = (int8_t) (d);
    }
}

exr_result_t
internal_exr_apply_rle (exr_encode_pipeline_t* encode)
{
    exr_result_t rv;
    uint64_t     outb, srcb;

    srcb = encode->packed_bytes;

    rv = internal_encode_alloc_buffer (
        encode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(encode->scratch_buffer_1),
        &(encode->scratch_alloc_size_1),
        srcb);
    if (rv != EXR_ERR_SUCCESS) return rv;

    reorder_and_predict (encode->scratch_buffer_1, encode->packed_buffer, srcb);

    outb = internal_rle_compress (
        encode->compressed_buffer,
        encode->compressed_alloc_size,
        encode->scratch_buffer_1,
        srcb);

    if (outb >= srcb)
    {
        memcpy (encode->compressed_buffer, encode->packed_buffer, srcb);
        outb = srcb;
    }
    encode->compressed_bytes = outb;
    return EXR_ERR_SUCCESS;
}

/**************************************/

uint64_t
internal_rle_decompress (
    uint8_t* out, uint64_t outsz, const uint8_t* src, uint64_t packsz)
{
    const int8_t* in          = (const int8_t*) src;
    uint8_t*      dst         = (uint8_t*) out;
    uint64_t      unpackbytes = 0;
    uint64_t      outbytes    = 0;

    while (unpackbytes < packsz)
    {
        if (*in < 0)
        {
            uint64_t count = (uint64_t) (-((int) *in++));
            ++unpackbytes;
            if (unpackbytes + count > packsz) return EXR_ERR_CORRUPT_CHUNK;
            if (outbytes + count > outsz) return EXR_ERR_CORRUPT_CHUNK;

            memcpy (dst, in, count);
            in += count;
            dst += count;
            unpackbytes += count;
            outbytes += count;
        }
        else
        {
            uint64_t count = (uint64_t) (*in++);
            if (unpackbytes + 2 > packsz) return EXR_ERR_CORRUPT_CHUNK;
            unpackbytes += 2;

            ++count;
            if (outbytes + count > outsz) return EXR_ERR_CORRUPT_CHUNK;

            memset (dst, *(const uint8_t*) in, count);
            dst += count;
            outbytes += count;
            ++in;
        }
    }
    return outbytes;
}

static void
unpredict_and_reorder (void* out, void* scratch, uint64_t packedbytes)
{
    int8_t*       t1   = scratch;
    int8_t*       t2   = t1 + (packedbytes + 1) / 2;
    int8_t*       s    = out;
    const int8_t* stop = t1 + packedbytes;

    ++t1;
    while (t1 < stop)
    {
        int d = (int) (t1[-1]) + (int) (t1[0]) - 128;
        t1[0] = (int8_t) d;
        ++t1;
    }

    t1   = scratch;
    stop = s + packedbytes;
    while (s < stop)
    {
        *(s++) = *(t1++);
        if (s < stop) *(s++) = *(t2++);
    }
}

exr_result_t
internal_exr_undo_rle (
    exr_decode_pipeline_t* decode,
    const void*            src,
    uint64_t               packsz,
    void*                  out,
    uint64_t               outsz)
{
    exr_result_t rv;
    uint64_t     unpackb;
    rv = internal_decode_alloc_buffer (
        decode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(decode->scratch_buffer_1),
        &(decode->scratch_alloc_size_1),
        outsz);
    if (rv != EXR_ERR_SUCCESS) return rv;

    unpackb =
        internal_rle_decompress (decode->scratch_buffer_1, outsz, src, packsz);
    if (unpackb != outsz) return EXR_ERR_CORRUPT_CHUNK;

    unpredict_and_reorder (out, decode->scratch_buffer_1, outsz);
    return EXR_ERR_SUCCESS;
}
