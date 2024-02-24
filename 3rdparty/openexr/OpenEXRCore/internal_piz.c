/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_compress.h"
#include "internal_decompress.h"

#include "internal_coding.h"
#include "internal_huf.h"
#include "internal_xdr.h"

#include <string.h>

/**************************************/

#define USHORT_RANGE (1 << 16)
#define BITMAP_SIZE (USHORT_RANGE >> 3)

static void
bitmapFromData (
    const uint16_t* data,
    uint64_t        nData,
    uint8_t*        bitmap,
    uint16_t*       minNonZero,
    uint16_t*       maxNonZero)
{
    uint16_t mnnz = BITMAP_SIZE - 1;
    uint16_t mxnz = 0;

    for (int i = 0; i < BITMAP_SIZE; ++i)
        bitmap[i] = 0;

    for (uint64_t i = 0; i < nData; ++i)
        bitmap[data[i] >> 3] |= (1 << (data[i] & 7));

    bitmap[0] &= ~1; // zero is not explicitly stored in
                     // the bitmap; we assume that the
                     // data always contain zeroes

    for (uint16_t i = 0; i < BITMAP_SIZE; ++i)
    {
        if (bitmap[i])
        {
            if (mnnz > i) mnnz = i;
            if (mxnz < i) mxnz = i;
        }
    }
    *minNonZero = mnnz;
    *maxNonZero = mxnz;
}

static inline uint16_t
forwardLutFromBitmap (const uint8_t* bitmap, uint16_t* lut)
{
    uint16_t k = 0;

    for (uint32_t i = 0; i < USHORT_RANGE; ++i)
    {
        if ((i == 0) || (bitmap[i >> 3] & (1 << (i & 7))))
            lut[i] = k++;
        else
            lut[i] = 0;
    }

    return k - 1;
}

static inline uint16_t
reverseLutFromBitmap (const uint8_t* bitmap, uint16_t* lut)
{
    uint32_t n, k = 0;

    for (uint32_t i = 0; i < USHORT_RANGE; ++i)
    {
        if ((i == 0) || (bitmap[i >> 3] & (1 << (i & 7))))
            lut[k++] = (uint16_t) i;
    }

    n = k - 1;

    while (k < USHORT_RANGE)
        lut[k++] = 0;

    return (uint16_t) n;
}

static inline void
applyLut (const uint16_t* lut, uint16_t* data, uint64_t nData)
{
    for (uint64_t i = 0; i < nData; ++i)
        data[i] = lut[data[i]];
}

/**************************************/
//
// Wavelet basis functions without modulo arithmetic; they produce
// the best compression ratios when the wavelet-transformed data are
// Huffman-encoded, but the wavelet transform works only for 14-bit
// data (untransformed data values must be less than (1 << 14)).
//

static inline void
wenc14 (uint16_t a, uint16_t b, uint16_t* l, uint16_t* h)
{
    int16_t as = (int16_t) a;
    int16_t bs = (int16_t) b;

    int16_t ms = (as + bs) >> 1;
    int16_t ds = as - bs;

    *l = (uint16_t) ms;
    *h = (uint16_t) ds;
}

static inline void
wdec14 (uint16_t l, uint16_t h, uint16_t* a, uint16_t* b)
{
    int16_t ls = (int16_t) l;
    int16_t hs = (int16_t) h;

    int hi = hs;
    int ai = ls + (hi & 1) + (hi >> 1);

    int16_t as = (int16_t) ai;
    int16_t bs = (int16_t) (ai - hi);

    *a = (uint16_t) as;
    *b = (uint16_t) bs;
}

//
// Wavelet basis functions with modulo arithmetic; they work with full
// 16-bit data, but Huffman-encoding the wavelet-transformed data doesn't
// compress the data quite as well.
//

#define NBITS ((int) 16)
#define A_OFFSET ((int) 1 << (NBITS - 1))
#define M_OFFSET ((int) 1 << (NBITS - 1))
#define MOD_MASK ((int) (1 << NBITS) - 1)

static inline void
wenc16 (uint16_t a, uint16_t b, uint16_t* l, uint16_t* h)
{
    int ao = (((int) a) + A_OFFSET) & MOD_MASK;
    int m  = ((ao + ((int) b)) >> 1);
    int d  = ao - ((int) b);

    if (d < 0) m = (m + M_OFFSET) & MOD_MASK;

    d &= MOD_MASK;

    *l = (uint16_t) m;
    *h = (uint16_t) d;
}

static inline void
wdec16 (uint16_t l, uint16_t h, uint16_t* a, uint16_t* b)
{
    int m  = (int) l;
    int d  = (int) h;
    int bb = (m - (d >> 1)) & MOD_MASK;
    int aa = (d + bb - A_OFFSET) & MOD_MASK;
    *b     = (uint16_t) bb;
    *a     = (uint16_t) aa;
}

/**************************************/

static void
wav_2D_encode (uint16_t* in, int nx, int ox, int ny, int oy, uint16_t mx)
{
    int w14 = (mx < (1 << 14)) ? 1 : 0;
    int n   = (nx > ny) ? ny : nx;
    int p   = 1; // == 1 <<  level
    int p2  = 2; // == 1 << (level+1)

    //
    // Hierarchical loop on smaller dimension n
    //

    while (p2 <= n)
    {
        uint16_t* py  = in;
        uint16_t* ey  = in + oy * (ny - p2);
        int       oy1 = oy * p;
        int       oy2 = oy * p2;
        int       ox1 = ox * p;
        int       ox2 = ox * p2;
        uint16_t  i00, i01, i10, i11;

        //
        // Y loop
        //

        for (; py <= ey; py += oy2)
        {
            uint16_t* px = py;
            uint16_t* ex = py + ox * (nx - p2);

            //
            // X loop
            //

            for (; px <= ex; px += ox2)
            {
                uint16_t* p01 = px + ox1;
                uint16_t* p10 = px + oy1;
                uint16_t* p11 = p10 + ox1;

                //
                // 2D wavelet encoding
                //

                if (w14)
                {
                    wenc14 (*px, *p01, &i00, &i01);
                    wenc14 (*p10, *p11, &i10, &i11);
                    wenc14 (i00, i10, px, p10);
                    wenc14 (i01, i11, p01, p11);
                }
                else
                {
                    wenc16 (*px, *p01, &i00, &i01);
                    wenc16 (*p10, *p11, &i10, &i11);
                    wenc16 (i00, i10, px, p10);
                    wenc16 (i01, i11, p01, p11);
                }
            }

            //
            // Encode (1D) odd column (still in Y loop)
            //

            if (nx & p)
            {
                uint16_t* p10 = px + oy1;

                if (w14)
                    wenc14 (*px, *p10, px, p10);
                else
                    wenc16 (*px, *p10, px, p10);
            }
        }

        //
        // Encode (1D) odd line (must loop in X)
        //

        if (ny & p)
        {
            uint16_t* px = py;
            uint16_t* ex = py + ox * (nx - p2);

            for (; px <= ex; px += ox2)
            {
                uint16_t* p01 = px + ox1;

                if (w14)
                    wenc14 (*px, *p01, px, p01);
                else
                    wenc16 (*px, *p01, px, p01);
            }
        }

        //
        // Next level
        //

        p = p2;
        p2 <<= 1;
    }
}

/**************************************/

static void
wav_2D_decode (
    uint16_t* in, // io: values are transformed in place
    int       nx, // i : x size
    int       ox, // i : x offset
    int       ny, // i : y size
    int       oy, // i : y offset
    uint16_t  mx)  // i : maximum in[x][y] value
{
    int w14 = (mx < (1 << 14)) ? 1 : 0;
    int n   = (nx > ny) ? ny : nx;
    int p   = 1;
    int p2;

    //
    // Search max level
    //

    while (p <= n)
        p <<= 1;

    p >>= 1;
    p2 = p;
    p >>= 1;

    //
    // Hierarchical loop on smaller dimension n
    //

    while (p >= 1)
    {
        uint16_t* py  = in;
        uint16_t* ey  = in + oy * (ny - p2);
        int       oy1 = oy * p;
        int       oy2 = oy * p2;
        int       ox1 = ox * p;
        int       ox2 = ox * p2;
        uint16_t  i00, i01, i10, i11;

        //
        // Y loop
        //

        for (; py <= ey; py += oy2)
        {
            uint16_t* px = py;
            uint16_t* ex = py + ox * (nx - p2);

            //
            // X loop
            //

            for (; px <= ex; px += ox2)
            {
                uint16_t* p01 = px + ox1;
                uint16_t* p10 = px + oy1;
                uint16_t* p11 = p10 + ox1;

                //
                // 2D wavelet decoding
                //

                if (w14)
                {
                    wdec14 (*px, *p10, &i00, &i10);
                    wdec14 (*p01, *p11, &i01, &i11);
                    wdec14 (i00, i01, px, p01);
                    wdec14 (i10, i11, p10, p11);
                }
                else
                {
                    wdec16 (*px, *p10, &i00, &i10);
                    wdec16 (*p01, *p11, &i01, &i11);
                    wdec16 (i00, i01, px, p01);
                    wdec16 (i10, i11, p10, p11);
                }
            }

            //
            // Decode (1D) odd column (still in Y loop)
            //

            if (nx & p)
            {
                uint16_t* p10 = px + oy1;

                if (w14)
                    wdec14 (*px, *p10, &i00, p10);
                else
                    wdec16 (*px, *p10, &i00, p10);
                *px = i00;
            }
        }

        //
        // Decode (1D) odd line (must loop in X)
        //

        if (ny & p)
        {
            uint16_t* px = py;
            uint16_t* ex = py + ox * (nx - p2);

            for (; px <= ex; px += ox2)
            {
                uint16_t* p01 = px + ox1;

                if (w14)
                    wdec14 (*px, *p01, &i00, p01);
                else
                    wdec16 (*px, *p01, &i00, p01);
                *px = i00;
            }
        }

        //
        // Next level
        //

        p2 = p;
        p >>= 1;
    }
}

exr_result_t
internal_exr_apply_piz (exr_encode_pipeline_t* encode)
{
    uint8_t*       out  = encode->compressed_buffer;
    uint64_t       nOut = 0;
    uint8_t *      scratch, *tmp;
    const uint8_t* packed;
    int            nx, ny, wcount;
    uint64_t       bpl, nBytes;
    exr_result_t   rv;
    uint8_t*       bitmap;
    uint16_t*      lut;
    uint32_t*      lengthptr;
    uint8_t*       hufspare;
    size_t         hufSpareBytes = internal_exr_huf_compress_spare_bytes ();
    uint16_t       minNonZero, maxNonZero, maxValue;
    uint64_t       packedbytes = encode->packed_bytes;
    uint64_t       ndata       = packedbytes / 2;
    uint16_t*      wavbuf;

    rv = internal_encode_alloc_buffer (
        encode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(encode->scratch_buffer_1),
        &(encode->scratch_alloc_size_1),
        packedbytes);
    if (rv != EXR_ERR_SUCCESS) return rv;

    rv = internal_encode_alloc_buffer (
        encode,
        EXR_TRANSCODE_BUFFER_SCRATCH2,
        &(encode->scratch_buffer_2),
        &(encode->scratch_alloc_size_2),
        BITMAP_SIZE * sizeof (uint8_t) + USHORT_RANGE * sizeof (uint16_t) +
            hufSpareBytes);
    if (rv != EXR_ERR_SUCCESS) return rv;

    hufspare = encode->scratch_buffer_2;
    bitmap   = hufspare + hufSpareBytes;
    lut      = (uint16_t*) (bitmap + BITMAP_SIZE);

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
            scratch += nBytes;
            if (curc->y_samples > 1)
            {
                if ((cury % curc->y_samples) != 0) continue;
                tmp += ((uint64_t) (y / curc->y_samples)) * bpl;
            }
            else { tmp += ((uint64_t) y) * bpl; }

            memcpy (tmp, packed, bpl);
            priv_to_native16 (tmp, nx * (curc->bytes_per_element / 2));
            packed += bpl;
        }
    }

    bitmapFromData (
        encode->scratch_buffer_1, ndata, bitmap, &minNonZero, &maxNonZero);

    maxValue = forwardLutFromBitmap (bitmap, lut);

    applyLut (lut, encode->scratch_buffer_1, ndata);

    nOut = 0;
    unaligned_store16 (out, minNonZero);
    out += 2;
    nOut += 2;
    unaligned_store16 (out, maxNonZero);
    out += 2;
    nOut += 2;
    if (minNonZero <= maxNonZero)
    {
        bpl = (uint64_t) (maxNonZero - minNonZero + 1);
        memcpy (out, bitmap + minNonZero, bpl);
        out += bpl;
        nOut += bpl;
    }

    wavbuf = encode->scratch_buffer_1;
    for (int c = 0; c < encode->channel_count; ++c)
    {
        const exr_coding_channel_info_t* curc = encode->channels + c;

        nx     = curc->width;
        ny     = curc->height;
        wcount = (int) (curc->bytes_per_element / 2);
        for (int j = 0; j < wcount; ++j)
        {
            wav_2D_encode (wavbuf + j, nx, wcount, ny, wcount * nx, maxValue);
        }
        wavbuf += nx * ny * wcount;
    }

    nBytes    = 0;
    lengthptr = (uint32_t*) out;
    out += sizeof (uint32_t);
    nOut += sizeof (uint32_t);
    rv = internal_huf_compress (
        &nBytes,
        out,
        encode->compressed_alloc_size - nOut,
        encode->scratch_buffer_1,
        ndata,
        hufspare,
        hufSpareBytes);
    if (rv != EXR_ERR_SUCCESS)
    {
        if (rv == EXR_ERR_ARGUMENT_OUT_OF_RANGE)
        {
            memcpy (
                encode->compressed_buffer, encode->packed_buffer, packedbytes);
            nOut = packedbytes;
        }
    }
    else
    {
        nOut += nBytes;
        if (nOut < packedbytes)
        {
            unaligned_store32 (lengthptr, (uint32_t) nBytes);
        }
        else
        {
            memcpy (
                encode->compressed_buffer, encode->packed_buffer, packedbytes);
            nOut = packedbytes;
        }
    }
    encode->compressed_bytes = nOut;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
internal_exr_undo_piz (
    exr_decode_pipeline_t* decode,
    const void*            src,
    uint64_t               packsz,
    void*                  outptr,
    uint64_t               outsz)
{
    uint8_t*       out  = outptr;
    uint64_t       nOut = 0;
    uint8_t *      scratch, *tmp;
    const uint8_t* packed;
    int            nx, ny, wcount;
    uint64_t       nBytes;
    exr_result_t   rv;
    uint8_t*       bitmap;
    uint16_t*      lut;
    uint8_t*       hufspare;
    size_t         hufSpareBytes = internal_exr_huf_decompress_spare_bytes ();
    uint16_t       minNonZero, maxNonZero, maxValue;
    uint16_t*      wavbuf;
    uint32_t       hufbytes;

    rv = internal_decode_alloc_buffer (
        decode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(decode->scratch_buffer_1),
        &(decode->scratch_alloc_size_1),
        outsz);
    if (rv != EXR_ERR_SUCCESS) return rv;

    rv = internal_decode_alloc_buffer (
        decode,
        EXR_TRANSCODE_BUFFER_SCRATCH2,
        &(decode->scratch_buffer_2),
        &(decode->scratch_alloc_size_2),
        BITMAP_SIZE * sizeof (uint8_t) + USHORT_RANGE * sizeof (uint16_t) +
            hufSpareBytes);
    if (rv != EXR_ERR_SUCCESS) return rv;

    hufspare = decode->scratch_buffer_2;
    lut      = (uint16_t*) (hufspare + hufSpareBytes);
    bitmap   = (uint8_t*) (lut + USHORT_RANGE);

    //
    // Read range compression data
    //

    memset (bitmap, 0, sizeof (uint8_t) * BITMAP_SIZE);

    nBytes = 0;
    if (sizeof (uint16_t) * 2 > packsz) return EXR_ERR_CORRUPT_CHUNK;

    packed     = src;
    minNonZero = unaligned_load16 (packed + nBytes);
    nBytes += sizeof (uint16_t);
    maxNonZero = unaligned_load16 (packed + nBytes);
    nBytes += sizeof (uint16_t);

    if (maxNonZero >= BITMAP_SIZE) return EXR_ERR_CORRUPT_CHUNK;

    if (minNonZero <= maxNonZero)
    {
        uint64_t bytesToRead = maxNonZero - minNonZero + 1;
        if (nBytes + bytesToRead > packsz) return EXR_ERR_CORRUPT_CHUNK;

        memcpy (bitmap + minNonZero, packed + nBytes, bytesToRead);
        nBytes += bytesToRead;
    }

    maxValue = reverseLutFromBitmap (bitmap, lut);

    //
    // Huffman decoding
    //
    if (nBytes + sizeof (uint32_t) > packsz) return EXR_ERR_CORRUPT_CHUNK;

    hufbytes = unaligned_load32 (packed + nBytes);
    nBytes += sizeof (uint32_t);

    if (nBytes + hufbytes > packsz) return EXR_ERR_CORRUPT_CHUNK;

    wavbuf = decode->scratch_buffer_1;
    rv     = internal_huf_decompress (
        decode,
        packed + nBytes,
        hufbytes,
        wavbuf,
        outsz / 2,
        hufspare,
        hufSpareBytes);
    if (rv != EXR_ERR_SUCCESS) return rv;

    //
    // Wavelet decoding
    //

    wavbuf = decode->scratch_buffer_1;
    for (int c = 0; c < decode->channel_count; ++c)
    {
        const exr_coding_channel_info_t* curc = decode->channels + c;

        nx     = curc->width;
        ny     = curc->height;
        wcount = (int) (curc->bytes_per_element / 2);
        for (int j = 0; j < wcount; ++j)
        {
            wav_2D_decode (wavbuf + j, nx, wcount, ny, wcount * nx, maxValue);
        }
        wavbuf += nx * ny * wcount;
    }

    //
    // Expand the pixel data to their original range
    //

    wavbuf = decode->scratch_buffer_1;
    applyLut (lut, wavbuf, outsz / 2);

    //
    // Rearrange the pixel data into the format expected by the caller.
    //

    for (int y = 0; y < decode->chunk.height; ++y)
    {
        int cury = y + decode->chunk.start_y;

        scratch = decode->scratch_buffer_1;
        for (int c = 0; c < decode->channel_count; ++c)
        {
            const exr_coding_channel_info_t* curc = decode->channels + c;

            nx = curc->width;
            ny = curc->height;
            nBytes =
                ((uint64_t) curc->width) * ((uint64_t) curc->bytes_per_element);

            if (nBytes == 0) continue;

            tmp = scratch;
            scratch += ((uint64_t) ny) * nBytes;

            if (curc->y_samples > 1)
            {
                if ((cury % curc->y_samples) != 0) continue;
                tmp += ((uint64_t) (y / curc->y_samples)) * nBytes;
            }
            else
                tmp += ((uint64_t) y) * nBytes;

            memcpy (out, tmp, nBytes);
            priv_from_native16 (out, nx * (curc->bytes_per_element / 2));
            out += nBytes;
            nOut += nBytes;
        }
    }

    if (nOut != outsz) return EXR_ERR_CORRUPT_CHUNK;
    return EXR_ERR_SUCCESS;
}
