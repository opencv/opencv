//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfZip.h"
#include "Iex.h"
#include "ImfCheckedArithmetic.h"
#include "ImfNamespace.h"
#include "ImfSimd.h"
#include "ImfSystemSpecific.h"

#include <openexr_compression.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

Zip::Zip (size_t maxRawSize, int level)
    : _maxRawSize (maxRawSize), _tmpBuffer (0), _zipLevel (level)
{
    _tmpBuffer = new char[_maxRawSize];
}

Zip::Zip (size_t maxScanLineSize, size_t numScanLines, int level)
    : _maxRawSize (0), _tmpBuffer (0), _zipLevel (level)
{
    _maxRawSize = uiMult (maxScanLineSize, numScanLines);
    _tmpBuffer  = new char[_maxRawSize];
}

Zip::~Zip ()
{
    if (_tmpBuffer) delete[] _tmpBuffer;
}

size_t
Zip::maxRawSize ()
{
    return _maxRawSize;
}

size_t
Zip::maxCompressedSize ()
{
    return exr_compress_max_buffer_size (_maxRawSize);
}

int
Zip::compress (const char* raw, int rawSize, char* compressed)
{
    //
    // Reorder the pixel data.
    //

    {
        char*       t1   = _tmpBuffer;
        char*       t2   = _tmpBuffer + (rawSize + 1) / 2;
        const char* stop = raw + rawSize;

        while (true)
        {
            if (raw < stop)
                *(t1++) = *(raw++);
            else
                break;

            if (raw < stop)
                *(t2++) = *(raw++);
            else
                break;
        }
    }

    //
    // Predictor.
    //

    {
        unsigned char* t    = (unsigned char*) _tmpBuffer + 1;
        unsigned char* stop = (unsigned char*) _tmpBuffer + rawSize;
        int            p    = t[-1];

        while (t < stop)
        {
            int d = int (t[0]) - p + (128 + 256);
            p     = t[0];
            t[0]  = d;
            ++t;
        }
    }

    //
    // Compress the data using zlib
    //
    size_t outSize;
    if (EXR_ERR_SUCCESS != exr_compress_buffer (
            nullptr,
            _zipLevel,
            _tmpBuffer,
            rawSize,
            compressed,
            maxCompressedSize (),
            &outSize))
    {
        throw IEX_NAMESPACE::BaseExc ("Data compression failed.");
    }

    return outSize;
}

namespace
{

#ifdef IMF_HAVE_SSE4_1

void
reconstruct_sse41 (char* buf, size_t outSize)
{
    static const size_t bytesPerChunk = sizeof (__m128i);
    const size_t        vOutSize      = outSize / bytesPerChunk;

    const __m128i c           = _mm_set1_epi8 (-128);
    const __m128i shuffleMask = _mm_set1_epi8 (15);

    // The first element doesn't have its high bit flipped during compression,
    // so it must not be flipped here.  To make the SIMD loop nice and
    // uniform, we pre-flip the bit so that the loop will unflip it again.
    buf[0] += -128;

    __m128i* vBuf  = reinterpret_cast<__m128i*> (buf);
    __m128i  vPrev = _mm_setzero_si128 ();
    for (size_t i = 0; i < vOutSize; ++i)
    {
        __m128i d = _mm_add_epi8 (_mm_loadu_si128 (vBuf), c);

        // Compute the prefix sum of elements.
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

    unsigned char prev = _mm_extract_epi8 (vPrev, 15);
    for (size_t i = vOutSize * bytesPerChunk; i < outSize; ++i)
    {
        unsigned char d = prev + buf[i] - 128;
        buf[i]          = d;
        prev            = d;
    }
}

#endif

#ifdef IMF_HAVE_NEON_AARCH64

void
reconstruct_neon (char* buf, size_t outSize)
{
    static const size_t bytesPerChunk = sizeof (uint8x16_t);
    const size_t        vOutSize      = outSize / bytesPerChunk;

    const uint8x16_t c           = vdupq_n_u8 (-128);
    const uint8x16_t shuffleMask = vdupq_n_u8 (15);

    // The first element doesn't have its high bit flipped during compression,
    // so it must not be flipped here.  To make the SIMD loop nice and
    // uniform, we pre-flip the bit so that the loop will unflip it again.
    buf[0] += -128;

    unsigned char* vBuf  = reinterpret_cast<unsigned char*> (buf);
    uint8x16_t  vZero = vdupq_n_u8 (0);
    uint8x16_t  vPrev = vdupq_n_u8 (0);
    for (size_t i = 0; i < vOutSize; ++i)
    {
        uint8x16_t d = vaddq_u8 (vld1q_u8 (vBuf), c);

        // Compute the prefix sum of elements.
        d = vaddq_u8 (d, vextq_u8 (vZero, d, 16 - 1));
        d = vaddq_u8 (d, vextq_u8 (vZero, d, 16 - 2));
        d = vaddq_u8 (d, vextq_u8 (vZero, d, 16 - 4));
        d = vaddq_u8 (d, vextq_u8 (vZero, d, 16 - 8));
        d = vaddq_u8 (d, vPrev);

        vst1q_u8 (vBuf, d);
        vBuf += sizeof (uint8x16_t);

        // Broadcast the high byte in our result to all lanes of the prev
        // value for the next iteration.
        vPrev = vqtbl1q_u8 (d, shuffleMask);
    }

    unsigned char prev = vgetq_lane_u8 (vPrev, 15);
    for (size_t i = vOutSize * bytesPerChunk; i < outSize; ++i)
    {
        unsigned char d = prev + buf[i] - 128;
        buf[i]          = d;
        prev            = d;
    }
}

#endif


void
reconstruct_scalar (char* buf, size_t outSize)
{
    unsigned char* t    = (unsigned char*) buf + 1;
    unsigned char* stop = (unsigned char*) buf + outSize;

    while (t < stop)
    {
        int d = int (t[-1]) + int (t[0]) - 128;
        t[0]  = d;
        ++t;
    }
}

#ifdef IMF_HAVE_SSE2

void
interleave_sse2 (const char* source, size_t outSize, char* out)
{
    static const size_t bytesPerChunk = 2 * sizeof (__m128i);

    const size_t vOutSize = outSize / bytesPerChunk;

    const __m128i* v1 = reinterpret_cast<const __m128i*> (source);
    const __m128i* v2 =
        reinterpret_cast<const __m128i*> (source + (outSize + 1) / 2);
    __m128i* vOut = reinterpret_cast<__m128i*> (out);

    for (size_t i = 0; i < vOutSize; ++i)
    {
        __m128i a = _mm_loadu_si128 (v1++);
        __m128i b = _mm_loadu_si128 (v2++);

        __m128i lo = _mm_unpacklo_epi8 (a, b);
        __m128i hi = _mm_unpackhi_epi8 (a, b);

        _mm_storeu_si128 (vOut++, lo);
        _mm_storeu_si128 (vOut++, hi);
    }

    const char* t1   = reinterpret_cast<const char*> (v1);
    const char* t2   = reinterpret_cast<const char*> (v2);
    char*       sOut = reinterpret_cast<char*> (vOut);

    for (size_t i = vOutSize * bytesPerChunk; i < outSize; ++i)
    {
        *(sOut++) = (i % 2 == 0) ? *(t1++) : *(t2++);
    }
}

#endif

#ifdef IMF_HAVE_NEON_AARCH64

void
interleave_neon (const char* source, size_t outSize, char* out)
{
    static const size_t bytesPerChunk = 2 * sizeof (uint8x16_t);

    const size_t vOutSize = outSize / bytesPerChunk;

    const unsigned char* v1 = reinterpret_cast<const unsigned char*> (source);
    const unsigned char* v2 =
        reinterpret_cast<const unsigned char*> (source + (outSize + 1) / 2);
    unsigned char* vOut = reinterpret_cast<unsigned char*> (out);

    for (size_t i = 0; i < vOutSize; ++i)
    {
        uint8x16_t a = vld1q_u8 (v1); v1 += sizeof (uint8x16_t);
        uint8x16_t b = vld1q_u8 (v2); v2 += sizeof (uint8x16_t);

        uint8x16_t lo = vzip1q_u8 (a, b);
        uint8x16_t hi = vzip2q_u8 (a, b);

        vst1q_u8 (vOut, lo); vOut += sizeof (uint8x16_t);
        vst1q_u8 (vOut, hi); vOut += sizeof (uint8x16_t);
    }

    const char* t1   = reinterpret_cast<const char*> (v1);
    const char* t2   = reinterpret_cast<const char*> (v2);
    char*       sOut = reinterpret_cast<char*> (vOut);

    for (size_t i = vOutSize * bytesPerChunk; i < outSize; ++i)
    {
        *(sOut++) = (i % 2 == 0) ? *(t1++) : *(t2++);
    }
}

#endif

void
interleave_scalar (const char* source, size_t outSize, char* out)
{
    const char* t1   = source;
    const char* t2   = source + (outSize + 1) / 2;
    char*       s    = out;
    char* const stop = s + outSize;

    while (true)
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

auto reconstruct = reconstruct_scalar;
auto interleave = interleave_scalar;

} // namespace

int
Zip::uncompress (const char* compressed, int compressedSize, char* raw)
{
    size_t outSize = 0;
    if (EXR_ERR_SUCCESS != exr_uncompress_buffer (
            nullptr,
            compressed,
            (size_t)compressedSize,
            _tmpBuffer,
            _maxRawSize,
            &outSize))
    {
        throw IEX_NAMESPACE::InputExc ("Data decompression failed.");
    }

    if (outSize == 0) { return static_cast<int> (outSize); }

    //
    // Predictor.
    //
    reconstruct (_tmpBuffer, outSize);

    //
    // Reorder the pixel data.
    //
    interleave (_tmpBuffer, outSize, raw);

    return outSize;
}

void
Zip::initializeFuncs ()
{
    CpuId cpuId;

#ifdef IMF_HAVE_SSE4_1
    if (cpuId.sse4_1)
    {
        reconstruct = reconstruct_sse41;
    }
#endif

#ifdef IMF_HAVE_SSE2
    if (cpuId.sse2) 
    {
        interleave = interleave_sse2;
    }
#endif

#ifdef IMF_HAVE_NEON_AARCH64
    reconstruct = reconstruct_neon;
    interleave = interleave_neon;
#endif
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
