/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_compress.h"
#include "internal_decompress.h"

#include "internal_coding.h"
#include "internal_xdr.h"

#include <string.h>

/**************************************/

extern const uint16_t* exrcore_expTable;
extern const uint16_t* exrcore_logTable;

static inline void
convertFromLinear (uint16_t s[16])
{
    for (int i = 0; i < 16; ++i)
        s[i] = exrcore_expTable[s[i]];
}

static inline void
convertToLinear (uint16_t s[16])
{
    for (int i = 0; i < 16; ++i)
        s[i] = exrcore_logTable[s[i]];
}

/**************************************/

static inline int
shiftAndRound (int x, int shift)
{
    int a, b;
    //
    // Compute
    //
    //     y = x * pow (2, -shift),
    //
    // then round y to the nearest integer.
    // In case of a tie, where y is exactly
    // halfway between two integers, round
    // to the even one.
    //

    x <<= 1;
    a = (1 << shift) - 1;
    shift += 1;
    b = (x >> shift) & 1;
    return (x + a + b) >> shift;
}

/*
 * Pack a block of 4 by 4 16-bit pixels (32 bytes) into
 * either 14 or 3 bytes.
 *
 *
 * Integers s[0] ... s[15] represent floating-point numbers
 * in what is essentially a sign-magnitude format.  Convert
 * s[0] .. s[15] into a new set of integers, t[0] ... t[15],
 * such that if t[i] is greater than t[j], the floating-point
 * number that corresponds to s[i] is always greater than
 * the floating-point number that corresponds to s[j].
 *
 * Also, replace any bit patterns that represent NaNs or
 * infinities with bit patterns that represent floating-point
 * zeroes.
 *
 *	bit pattern	floating-point		bit pattern
 *	in s[i]		value			in t[i]
 *
 *  0x7fff		NAN			0x8000
 *  0x7ffe		NAN			0x8000
 *	  ...					  ...
 *  0x7c01		NAN			0x8000
 *  0x7c00		+infinity		0x8000
 *  0x7bff		+HALF_MAX		0xfbff
 *  0x7bfe					0xfbfe
 *  0x7bfd					0xfbfd
 *	  ...					  ...
 *  0x0002		+2 * HALF_MIN		0x8002
 *  0x0001		+HALF_MIN		0x8001
 *  0x0000		+0.0			0x8000
 *  0x8000		-0.0			0x7fff
 *  0x8001		-HALF_MIN		0x7ffe
 *  0x8002		-2 * HALF_MIN		0x7ffd
 *	  ...					  ...
 *  0xfbfd					0x0f02
 *  0xfbfe					0x0401
 *  0xfbff		-HALF_MAX		0x0400
 *  0xfc00		-infinity		0x8000
 *  0xfc01		NAN			0x8000
 *	  ...					  ...
 *  0xfffe		NAN			0x8000
 *  0xffff		NAN			0x8000
 */
static int
pack (const uint16_t s[16], uint8_t b[14], int flatfields, int exactmax)
{
    int      d[16];
    int      r[15];
    int      rMin;
    int      rMax;
    uint16_t t[16];
    uint16_t tMax;
    int      shift = -1;

    const int bias = 0x20;

    for (int i = 0; i < 16; ++i)
    {
        if ((s[i] & 0x7c00) == 0x7c00)
            t[i] = 0x8000;
        else if (s[i] & 0x8000)
            t[i] = ~s[i];
        else
            t[i] = s[i] | 0x8000;
    }

    // find max
    tMax = 0;
    for (int i = 0; i < 16; ++i)
        if (tMax < t[i]) tMax = t[i];

    //
    // Compute a set of running differences, r[0] ... r[14]:
    // Find a shift value such that after rounding off the
    // rightmost bits and shifting all differences are between
    // -32 and +31.  Then bias the differences so that they
    // end up between 0 and 63.
    //

    do
    {
        shift += 1;

        //
        // Compute absolute differences, d[0] ... d[15],
        // between tMax and t[0] ... t[15].
        //
        // Shift and round the absolute differences.
        //

        for (int i = 0; i < 16; ++i)
            d[i] = shiftAndRound (tMax - t[i], shift);

        //
        // Convert d[0] .. d[15] into running differences
        //

        r[0] = d[0] - d[4] + bias;
        r[1] = d[4] - d[8] + bias;
        r[2] = d[8] - d[12] + bias;

        r[3] = d[0] - d[1] + bias;
        r[4] = d[4] - d[5] + bias;
        r[5] = d[8] - d[9] + bias;
        r[6] = d[12] - d[13] + bias;

        r[7]  = d[1] - d[2] + bias;
        r[8]  = d[5] - d[6] + bias;
        r[9]  = d[9] - d[10] + bias;
        r[10] = d[13] - d[14] + bias;

        r[11] = d[2] - d[3] + bias;
        r[12] = d[6] - d[7] + bias;
        r[13] = d[10] - d[11] + bias;
        r[14] = d[14] - d[15] + bias;

        rMin = r[0];
        rMax = r[0];

        for (int i = 1; i < 15; ++i)
        {
            if (rMin > r[i]) rMin = r[i];

            if (rMax < r[i]) rMax = r[i];
        }
    } while (rMin < 0 || rMax > 0x3f);

    if (rMin == bias && rMax == bias && flatfields)
    {
        //
        // Special case - all pixels have the same value.
        // We encode this in 3 instead of 14 bytes by
        // storing the value 0xfc in the third output byte,
        // which cannot occur in the 14-byte encoding.
        //

        b[0] = (uint8_t) (t[0] >> 8);
        b[1] = (uint8_t) t[0];
        b[2] = 0xfc;

        return 3;
    }

    if (exactmax)
    {
        //
        // Adjust t[0] so that the pixel whose value is equal
        // to tMax gets represented as accurately as possible.
        //

        t[0] = tMax - (uint16_t) (d[0] << shift);
    }

    //
    // Pack t[0], shift and r[0] ... r[14] into 14 bytes:
    //

    b[0]  = (uint8_t) (t[0] >> 8);
    b[1]  = (uint8_t) t[0];
    b[2]  = (uint8_t) ((shift << 2) | (r[0] >> 4));
    b[3]  = (uint8_t) ((r[0] << 4) | (r[1] >> 2));
    b[4]  = (uint8_t) ((r[1] << 6) | r[2]);
    b[5]  = (uint8_t) ((r[3] << 2) | (r[4] >> 4));
    b[6]  = (uint8_t) ((r[4] << 4) | (r[5] >> 2));
    b[7]  = (uint8_t) ((r[5] << 6) | r[6]);
    b[8]  = (uint8_t) ((r[7] << 2) | (r[8] >> 4));
    b[9]  = (uint8_t) ((r[8] << 4) | (r[9] >> 2));
    b[10] = (uint8_t) ((r[9] << 6) | r[10]);
    b[11] = (uint8_t) ((r[11] << 2) | (r[12] >> 4));
    b[12] = (uint8_t) ((r[12] << 4) | (r[13] >> 2));
    b[13] = (uint8_t) ((r[13] << 6) | r[14]);

    return 14;
}

/**************************************/

static inline void
unpack14 (const uint8_t b[14], uint16_t s[16])
{
    uint16_t shift, bias;
    s[0] = ((uint16_t) (b[0] << 8)) | ((uint16_t) b[1]);

    shift = (b[2] >> 2);
    bias  = (uint16_t) (0x20u << shift);

    s[4] =
        (uint16_t) ((uint32_t) s[0] + (uint32_t) ((((uint32_t) (b[2] << 4) | (uint32_t) (b[3] >> 4)) & 0x3fu) << shift) - bias);
    s[8] =
        (uint16_t) ((uint32_t) s[4] + (uint32_t) ((((uint32_t) (b[3] << 2) | (uint32_t) (b[4] >> 6)) & 0x3fu) << shift) - bias);
    s[12] =
        (uint16_t) ((uint32_t) s[8] + (uint32_t) ((uint32_t) (b[4] & 0x3fu) << shift) - bias);

    s[1] =
        (uint16_t) ((uint32_t) s[0] + (uint32_t) ((uint32_t) (b[5] >> 2) << shift) - bias);
    s[5] =
        (uint16_t) ((uint32_t) s[4] + (uint32_t) ((((uint32_t) (b[5] << 4) | (uint32_t) (b[6] >> 4)) & 0x3fu) << shift) - bias);
    s[9] =
        (uint16_t) ((uint32_t) s[8] + (uint32_t) ((((uint32_t) (b[6] << 2) | (uint32_t) (b[7] >> 6)) & 0x3fu) << shift) - bias);
    s[13] =
        (uint16_t) ((uint32_t) s[12] + (uint32_t) ((uint32_t) (b[7] & 0x3fu) << shift) - bias);

    s[2] =
        (uint16_t) ((uint32_t) s[1] + (uint32_t) ((uint32_t) (b[8] >> 2) << shift) - bias);
    s[6] =
        (uint16_t) ((uint32_t) s[5] + (uint32_t) ((((uint32_t) (b[8] << 4) | (uint32_t) (b[9] >> 4)) & 0x3fu) << shift) - bias);
    s[10] =
        (uint16_t) ((uint32_t) s[9] + (uint32_t) ((((uint32_t) (b[9] << 2) | (uint32_t) (b[10] >> 6)) & 0x3fu) << shift) - bias);
    s[14] =
        (uint16_t) ((uint32_t) s[13] + (uint32_t) ((uint32_t) (b[10] & 0x3fu) << shift) - bias);

    s[3] =
        (uint16_t) ((uint32_t) s[2] + (uint32_t) ((uint32_t) (b[11] >> 2) << shift) - bias);
    s[7] =
        (uint16_t) ((uint32_t) s[6] + (uint32_t) ((((uint32_t) (b[11] << 4) | (uint32_t) (b[12] >> 4)) & 0x3fu) << shift) - bias);
    s[11] =
        (uint16_t) ((uint32_t) s[10] + (uint32_t) ((((uint32_t) (b[12] << 2) | (uint32_t) (b[13] >> 6)) & 0x3fu) << shift) - bias);
    s[15] =
        (uint16_t) ((uint32_t) s[14] + (uint32_t) ((uint32_t) (b[13] & 0x3fu) << shift) - bias);

    for (int i = 0; i < 16; ++i)
    {
        if (s[i] & 0x8000)
            s[i] &= 0x7fff;
        else
            s[i] = ~s[i];
    }
}

static inline void
unpack3 (const uint8_t b[3], uint16_t s[16])
{
    s[0] = ((uint16_t) (b[0] << 8)) | ((uint16_t) b[1]);

    if (s[0] & 0x8000)
        s[0] &= 0x7fff;
    else
        s[0] = ~s[0];

    for (int i = 1; i < 16; ++i)
        s[i] = s[0];
}

/**************************************/

static exr_result_t
compress_b44_impl (exr_encode_pipeline_t* encode, int flat_field)
{
    uint8_t*       out  = encode->compressed_buffer;
    uint64_t       nOut = 0;
    uint8_t *      scratch, *tmp;
    const uint8_t* packed;
    int            nx, ny, wcount;
    uint64_t       bpl, nBytes;
    exr_result_t   rv;

    rv = internal_encode_alloc_buffer (
        encode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(encode->scratch_buffer_1),
        &(encode->scratch_alloc_size_1),
        encode->packed_bytes);
    if (rv != EXR_ERR_SUCCESS) return rv;

    nOut   = 0;
    packed = encode->packed_buffer;
    for (int y = 0; y < encode->chunk.height; ++y)
    {
        int cury = y + encode->chunk.start_y;

        scratch = encode->scratch_buffer_1;
        for (int c = 0; c < encode->channel_count; ++c)
        {
            const exr_coding_channel_info_t* curc = encode->channels + c;

            nx     = curc->width;
            ny     = curc->height;
            bpl    = ((uint64_t) (nx)) * (uint64_t) (curc->bytes_per_element);
            nBytes = ((uint64_t) (ny)) * bpl;

            if (nBytes == 0) continue;

            tmp = scratch;
            if (curc->y_samples > 1)
            {
                if ((cury % curc->y_samples) != 0)
                {
                    scratch += nBytes;
                    continue;
                }
                tmp += ((uint64_t) (y / curc->y_samples)) * bpl;
            }
            else { tmp += ((uint64_t) y) * bpl; }

            memcpy (tmp, packed, bpl);
            if (curc->data_type == EXR_PIXEL_HALF) priv_to_native16 (tmp, nx);
            packed += bpl;
            scratch += nBytes;
        }
    }

    nOut    = 0;
    scratch = encode->scratch_buffer_1;
    for (int c = 0; c < encode->channel_count; ++c)
    {
        const exr_coding_channel_info_t* curc = encode->channels + c;

        nx     = curc->width;
        ny     = curc->height;
        bpl    = (uint64_t) (nx) * (uint64_t) (curc->bytes_per_element);
        nBytes = ((uint64_t) (ny)) * bpl;

        if (nBytes == 0) continue;

        if (curc->data_type != EXR_PIXEL_HALF)
        {
            if (nOut + nBytes > encode->compressed_alloc_size)
                return EXR_ERR_OUT_OF_MEMORY;
            memcpy (out, scratch, nBytes);
            out += nBytes;
            scratch += nBytes;
            nOut += nBytes;
            continue;
        }

        for (int y = 0; y < ny; y += 4)
        {
            //
            // Copy the next 4x4 pixel block into array s.
            // If the width, cd.nx, or the height, cd.ny, of
            // the pixel data in _tmpBuffer is not divisible
            // by 4, then pad the data by repeating the
            // rightmost column and the bottom row.
            //
            uint16_t *row0, *row1, *row2, *row3;

            row0 = (uint16_t*) scratch;
            row0 += y * nx;

            row1 = row0 + nx;
            row2 = row1 + nx;
            row3 = row2 + nx;

            if (y + 3 >= ny)
            {
                if (y + 1 >= ny) row1 = row0;
                if (y + 2 >= ny) row2 = row1;

                row3 = row2;
            }

            for (int x = 0; x < nx; x += 4)
            {
                uint16_t s[16];

                if (x + 3 >= nx)
                {
                    int n = nx - x;

                    for (int i = 0; i < 4; ++i)
                    {
                        int j = i;
                        if (j > n - 1) j = n - 1;

                        s[i + 0]  = row0[j];
                        s[i + 4]  = row1[j];
                        s[i + 8]  = row2[j];
                        s[i + 12] = row3[j];
                    }
                }
                else
                {
                    memcpy (&s[0], row0, 4 * sizeof (uint16_t));
                    memcpy (&s[4], row1, 4 * sizeof (uint16_t));
                    memcpy (&s[8], row2, 4 * sizeof (uint16_t));
                    memcpy (&s[12], row3, 4 * sizeof (uint16_t));
                }

                row0 += 4;
                row1 += 4;
                row2 += 4;
                row3 += 4;

                //
                // Compress the contents of array s and append the
                // results to the output buffer.
                //

                if (curc->p_linear) convertFromLinear (s);

                wcount = pack (s, out, flat_field, !(curc->p_linear));
                out += wcount;
                nOut += (uint64_t) wcount;
                if (nOut + 14 > encode->compressed_alloc_size)
                    return EXR_ERR_OUT_OF_MEMORY;
            }
        }
        scratch += nBytes;
    }

    encode->compressed_bytes = nOut;
    return rv;
}

exr_result_t
internal_exr_apply_b44 (exr_encode_pipeline_t* encode)
{
    return compress_b44_impl (encode, 0);
}

exr_result_t
internal_exr_apply_b44a (exr_encode_pipeline_t* encode)
{
    return compress_b44_impl (encode, 1);
}

/**************************************/

static exr_result_t
uncompress_b44_impl (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncomp_buf_size)
{
    const uint8_t* in      = compressed_data;
    uint8_t*       out     = uncompressed_data;
    uint8_t*       scratch = decode->scratch_buffer_1;
    uint8_t*       tmp;
    uint16_t *     row0, *row1, *row2, *row3;
    uint64_t       n, nBytes, bpl = 0, bIn = 0;
    int            nx, ny;
    uint16_t       s[16];

    for (int c = 0; c < decode->channel_count; ++c)
    {
        const exr_coding_channel_info_t* curc = decode->channels + c;
        nx                                    = curc->width;
        ny                                    = curc->height;
        nBytes = (uint64_t) (ny) * (uint64_t) (nx) *
                 (uint64_t) (curc->bytes_per_element);

        if (nBytes == 0) continue;

        if (curc->data_type != EXR_PIXEL_HALF)
        {
            if (bIn + nBytes > comp_buf_size) return EXR_ERR_OUT_OF_MEMORY;
            memcpy (scratch, in, nBytes);
            in += nBytes;
            bIn += nBytes;
            scratch += nBytes;
            continue;
        }

        for (int y = 0; y < ny; y += 4)
        {
            row0 = (uint16_t*) scratch;
            row0 += y * nx;
            row1 = row0 + nx;
            row2 = row1 + nx;
            row3 = row2 + nx;
            for (int x = 0; x < nx; x += 4)
            {
                if (bIn + 3 > comp_buf_size) return EXR_ERR_OUT_OF_MEMORY;

                /* check if 3-byte encoded flat field */
                if (in[2] >= (13 << 2))
                {
                    unpack3 (in, s);
                    in += 3;
                    bIn += 3;
                }
                else
                {
                    if (bIn + 14 > comp_buf_size) return EXR_ERR_OUT_OF_MEMORY;
                    unpack14 (in, s);
                    in += 14;
                    bIn += 14;
                }

                if (curc->p_linear) convertToLinear (s);

                priv_from_native16 (s, 16);

                n = (x + 3 < nx) ? 4 * sizeof (uint16_t)
                                 : (uint64_t) (nx - x) * sizeof (uint16_t);
                if (y + 3 < ny)
                {
                    memcpy (row0, &s[0], n);
                    memcpy (row1, &s[4], n);
                    memcpy (row2, &s[8], n);
                    memcpy (row3, &s[12], n);
                }
                else
                {
                    memcpy (row0, &s[0], n);
                    if (y + 1 < ny) memcpy (row1, &s[4], n);
                    if (y + 2 < ny) memcpy (row2, &s[8], n);
                }
                row0 += 4;
                row1 += 4;
                row2 += 4;
                row3 += 4;
            }
        }
        scratch += nBytes;
    }

    /* now put it back so each scanline has channel data */
    bIn = 0;
    for (int y = 0; y < decode->chunk.height; ++y)
    {
        int cury = y + decode->chunk.start_y;

        scratch = decode->scratch_buffer_1;
        for (int c = 0; c < decode->channel_count; ++c)
        {
            const exr_coding_channel_info_t* curc = decode->channels + c;

            nx     = curc->width;
            ny     = curc->height;
            bpl    = ((uint64_t) (nx)) * (uint64_t) (curc->bytes_per_element);
            nBytes = ((uint64_t) (ny)) * bpl;

            if (nBytes == 0) continue;

            tmp = scratch;
            if (curc->y_samples > 1)
            {
                if ((cury % curc->y_samples) != 0)
                {
                    scratch += nBytes;
                    continue;
                }
                tmp += ((uint64_t) (y / curc->y_samples)) * bpl;
            }
            else
                tmp += ((uint64_t) y) * bpl;

            if (bIn + bpl > uncomp_buf_size) return EXR_ERR_OUT_OF_MEMORY;

            memcpy (out, tmp, bpl);

            bIn += bpl;
            out += bpl;
            scratch += nBytes;
        }
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static uint64_t
compute_scratch_buffer_size (
    exr_decode_pipeline_t* decode, uint64_t uncompressed_size)
{
    const exr_coding_channel_info_t* curc;
    int                              nx, ny;
    uint64_t                         ret  = uncompressed_size;
    uint64_t                         comp = 0;

    for (int c = 0; c < decode->channel_count; ++c)
    {
        curc = decode->channels + c;

        nx = curc->width;
        ny = curc->height;

        if (nx % 4) nx += 4 - nx % 4;
        if (ny % 4) ny += 4 - ny % 4;

        comp += (uint64_t) (ny) * (uint64_t) (nx) *
                (uint64_t) (curc->bytes_per_element);
    }
    if (comp > ret) ret = comp;
    return ret;
}

/**************************************/

exr_result_t
internal_exr_undo_b44 (
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
        compute_scratch_buffer_size (decode, uncompressed_size));
    if (rv != EXR_ERR_SUCCESS) return rv;

    return uncompress_b44_impl (
        decode,
        compressed_data,
        comp_buf_size,
        uncompressed_data,
        uncompressed_size);
}

exr_result_t
internal_exr_undo_b44a (
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
        compute_scratch_buffer_size (decode, uncompressed_size));
    if (rv != EXR_ERR_SUCCESS) return rv;

    return uncompress_b44_impl (
        decode,
        compressed_data,
        comp_buf_size,
        uncompressed_data,
        uncompressed_size);
}
