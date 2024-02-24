/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_compress.h"
#include "internal_decompress.h"

#include "internal_coding.h"
#include "internal_xdr.h"

#include <string.h>
#include "openexr_compression.h"

/**************************************/

static inline uint32_t
float_to_float24 (float f)
{
    union
    {
        float    f;
        uint32_t i;
    } u;
    uint32_t s, e, m, i;

    u.f = f;

    //
    // Disassemble the 32-bit floating point number, f,
    // into sign, s, exponent, e, and significand, m.
    //

    s = u.i & 0x80000000;
    e = u.i & 0x7f800000;
    m = u.i & 0x007fffff;

    if (e == 0x7f800000)
    {
        if (m)
        {
            //
            // F is a NAN; we preserve the sign bit and
            // the 15 leftmost bits of the significand,
            // with one exception: If the 15 leftmost
            // bits are all zero, the NAN would turn
            // into an infinity, so we have to set at
            // least one bit in the significand.
            //

            m >>= 8;
            i = (e >> 8) | m | (m == 0);
        }
        else
        {
            //
            // F is an infinity.
            //

            i = e >> 8;
        }
    }
    else
    {
        //
        // F is finite, round the significand to 15 bits.
        //

        i = ((e | m) + (m & 0x00000080)) >> 8;

        if (i >= 0x7f8000)
        {
            //
            // F was close to FLT_MAX, and the significand was
            // rounded up, resulting in an exponent overflow.
            // Avoid the overflow by truncating the significand
            // instead of rounding it.
            //

            i = (e | m) >> 8;
        }
    }

    return (s >> 8) | i;
}

/**************************************/

static exr_result_t
apply_pxr24_impl (exr_encode_pipeline_t* encode)
{
    uint8_t*       out    = encode->scratch_buffer_1;
    uint64_t       nOut   = 0;
    const uint8_t* lastIn = encode->packed_buffer;
    size_t         compbufsz;
    exr_result_t   rv;

    for (int y = 0; y < encode->chunk.height; ++y)
    {
        int cury = y + encode->chunk.start_y;

        for (int c = 0; c < encode->channel_count; ++c)
        {
            const exr_coding_channel_info_t* curc   = encode->channels + c;
            int                              w      = curc->width;
            uint64_t                         nBytes = (uint64_t) (w);

            if (curc->height == 0 ||
                (curc->y_samples > 1 && (cury % curc->y_samples) != 0))
                continue;

            switch (curc->data_type)
            {
                case EXR_PIXEL_UINT: {
                    uint8_t*        ptr[4];
                    uint32_t        prevPixel = 0;
                    const uint32_t* din       = (const uint32_t*) (lastIn);

                    nBytes *= sizeof (uint32_t);
                    if (nOut + nBytes > encode->scratch_alloc_size_1)
                        return EXR_ERR_OUT_OF_MEMORY;
                    nOut += nBytes;
                    lastIn += nBytes;

                    ptr[0] = out;
                    out += w;
                    ptr[1] = out;
                    out += w;
                    ptr[2] = out;
                    out += w;
                    ptr[3] = out;
                    out += w;

                    for (int x = 0; x < w; ++x)
                    {
                        uint32_t pixel = unaligned_load32 (din);
                        uint32_t diff  = pixel - prevPixel;
                        prevPixel      = pixel;

                        ++din;
                        *(ptr[0]++) = (uint8_t) (diff >> 24);
                        *(ptr[1]++) = (uint8_t) (diff >> 16);
                        *(ptr[2]++) = (uint8_t) (diff >> 8);
                        *(ptr[3]++) = (uint8_t) (diff);
                    }
                    break;
                }
                case EXR_PIXEL_HALF: {
                    uint8_t*        ptr[2];
                    uint32_t        prevPixel = 0;
                    const uint16_t* din       = (const uint16_t*) (lastIn);

                    nBytes *= sizeof (uint16_t);
                    if (nOut + nBytes > encode->scratch_alloc_size_1)
                        return EXR_ERR_OUT_OF_MEMORY;
                    nOut += nBytes;
                    lastIn += nBytes;

                    ptr[0] = out;
                    out += w;
                    ptr[1] = out;
                    out += w;

                    for (int x = 0; x < w; ++x)
                    {
                        uint32_t pixel = (uint32_t) unaligned_load16 (din);
                        uint32_t diff  = pixel - prevPixel;
                        prevPixel      = pixel;

                        ++din;
                        *(ptr[0]++) = (uint8_t) (diff >> 8);
                        *(ptr[1]++) = (uint8_t) (diff);
                    }
                    break;
                }
                case EXR_PIXEL_FLOAT: {
                    uint8_t*     ptr[3];
                    uint32_t     prevPixel = 0;
                    const float* din       = (const float*) (lastIn);

                    nBytes *= 3;
                    if (nOut + nBytes > encode->scratch_alloc_size_1)
                        return EXR_ERR_OUT_OF_MEMORY;
                    nOut += nBytes;
                    lastIn += w * 4;

                    ptr[0] = out;
                    out += w;
                    ptr[1] = out;
                    out += w;
                    ptr[2] = out;
                    out += w;

                    for (int x = 0; x < w; ++x)
                    {
                        union
                        {
                            uint32_t i;
                            float    f;
                        } v;
                        uint32_t pixel24, diff;
                        v.i       = unaligned_load32 (din);
                        pixel24   = float_to_float24 (v.f);
                        diff      = pixel24 - prevPixel;
                        prevPixel = pixel24;

                        ++din;
                        *(ptr[0]++) = (uint8_t) (diff >> 16);
                        *(ptr[1]++) = (uint8_t) (diff >> 8);
                        *(ptr[2]++) = (uint8_t) (diff);
                    }
                    break;
                }
                default: return EXR_ERR_INVALID_ARGUMENT;
            }
        }
    }

    rv = exr_compress_buffer (
        encode->context,
        -1,
        encode->scratch_buffer_1,
        nOut,
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
    return rv;
}

exr_result_t
internal_exr_apply_pxr24 (exr_encode_pipeline_t* encode)
{
    exr_result_t rv;
    rv = internal_encode_alloc_buffer (
        encode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(encode->scratch_buffer_1),
        &(encode->scratch_alloc_size_1),
        encode->packed_bytes);
    if (rv != EXR_ERR_SUCCESS) return rv;

    return apply_pxr24_impl (encode);
}

/**************************************/

static exr_result_t
undo_pxr24_impl (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size,
    void*                  scratch_data,
    uint64_t               scratch_size)
{
    size_t         outSize;
    exr_result_t   rstat;
    uint8_t*       out    = uncompressed_data;
    uint64_t       nOut   = 0;
    uint64_t       nDec   = 0;
    const uint8_t* lastIn = scratch_data;

    if (scratch_size < uncompressed_size) return EXR_ERR_INVALID_ARGUMENT;

    rstat = exr_uncompress_buffer (
        decode->context,
        compressed_data,
        comp_buf_size,
        scratch_data,
        uncompressed_size,
        &outSize);

    if (rstat != EXR_ERR_SUCCESS) return rstat;

    for (int y = 0; y < decode->chunk.height; ++y)
    {
        int cury = y + decode->chunk.start_y;

        for (int c = 0; c < decode->channel_count; ++c)
        {
            const exr_coding_channel_info_t* curc = decode->channels + c;
            int                              w    = curc->width;
            uint64_t                         nBytes =
                (uint64_t) (w) * (uint64_t) (curc->bytes_per_element);

            if (curc->height == 0 ||
                (curc->y_samples > 1 && (cury % curc->y_samples) != 0))
                continue;

            if (nOut + nBytes > uncompressed_size) return EXR_ERR_OUT_OF_MEMORY;

            switch (curc->data_type)
            {
                case EXR_PIXEL_UINT: {
                    const uint8_t* ptr[4];
                    uint32_t       pixel = 0;
                    uint32_t*      dout  = (uint32_t*) (out);

                    ptr[0] = lastIn;
                    lastIn += w;
                    ptr[1] = lastIn;
                    lastIn += w;
                    ptr[2] = lastIn;
                    lastIn += w;
                    ptr[3] = lastIn;
                    lastIn += w;

                    if (nDec + nBytes > outSize) return EXR_ERR_CORRUPT_CHUNK;

                    for (int x = 0; x < w; ++x)
                    {
                        uint32_t diff =
                            (((uint32_t) (*(ptr[0]++)) << 24) |
                             ((uint32_t) (*(ptr[1]++)) << 16) |
                             ((uint32_t) (*(ptr[2]++)) << 8) |
                             ((uint32_t) (*(ptr[3]++))));
                        pixel += diff;
                        unaligned_store32 (dout, pixel);
                        ++dout;
                    }
                    nDec += nBytes;
                    break;
                }
                case EXR_PIXEL_HALF: {
                    const uint8_t* ptr[2];
                    uint32_t       pixel = 0;
                    uint16_t*      dout  = (uint16_t*) (out);

                    ptr[0] = lastIn;
                    lastIn += w;
                    ptr[1] = lastIn;
                    lastIn += w;

                    if (nDec + nBytes > outSize) return EXR_ERR_CORRUPT_CHUNK;

                    for (int x = 0; x < w; ++x)
                    {
                        uint32_t diff =
                            (((uint32_t) (*(ptr[0]++)) << 8) |
                             ((uint32_t) (*(ptr[1]++))));
                        pixel += diff;
                        unaligned_store16 (dout, (uint16_t) pixel);
                        ++dout;
                    }
                    nDec += nBytes;
                    break;
                }
                case EXR_PIXEL_FLOAT: {
                    const uint8_t* ptr[3];
                    uint32_t       pixel = 0;
                    uint32_t*      dout  = (uint32_t*) (out);

                    ptr[0] = lastIn;
                    lastIn += w;
                    ptr[1] = lastIn;
                    lastIn += w;
                    ptr[2] = lastIn;
                    lastIn += w;

                    if (nDec + (uint64_t) (w * 3) > outSize)
                        return EXR_ERR_CORRUPT_CHUNK;

                    for (int x = 0; x < w; ++x)
                    {
                        uint32_t diff =
                            (((uint32_t) (*(ptr[0]++)) << 24) |
                             ((uint32_t) (*(ptr[1]++)) << 16) |
                             ((uint32_t) (*(ptr[2]++)) << 8));
                        pixel += diff;
                        unaligned_store32 (dout, pixel);
                        ++dout;
                    }
                    nDec += (uint64_t) (w * 3);
                    break;
                }
                default: return EXR_ERR_INVALID_ARGUMENT;
            }
            out += nBytes;
            nOut += nBytes;
        }
    }
    return EXR_ERR_SUCCESS;
}

exr_result_t
internal_exr_undo_pxr24 (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size)
{
    exr_result_t rv;
    rv = internal_decode_alloc_buffer (
        decode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(decode->scratch_buffer_1),
        &(decode->scratch_alloc_size_1),
        uncompressed_size);
    if (rv != EXR_ERR_SUCCESS) return rv;
    return undo_pxr24_impl (
        decode,
        compressed_data,
        comp_buf_size,
        uncompressed_data,
        uncompressed_size,
        decode->scratch_buffer_1,
        decode->scratch_alloc_size_1);
}
