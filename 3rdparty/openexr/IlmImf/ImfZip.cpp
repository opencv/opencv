///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#include "ImfZip.h"
#include "ImfCheckedArithmetic.h"
#include "ImfNamespace.h"
#include "ImfSimd.h"
#include "Iex.h"

#include <math.h>
#include <zlib.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

Zip::Zip(size_t maxRawSize):
    _maxRawSize(maxRawSize),
    _tmpBuffer(0)
{
    _tmpBuffer = new char[_maxRawSize];
}

Zip::Zip(size_t maxScanLineSize, size_t numScanLines):
    _maxRawSize(0),
    _tmpBuffer(0)
{
    _maxRawSize = uiMult (maxScanLineSize, numScanLines);
    _tmpBuffer  = new char[_maxRawSize];
}

Zip::~Zip()
{
    if (_tmpBuffer) delete[] _tmpBuffer;
}

size_t
Zip::maxRawSize()
{
    return _maxRawSize;
}

size_t
Zip::maxCompressedSize()
{
    return uiAdd (uiAdd (_maxRawSize,
               size_t (ceil (_maxRawSize * 0.01))),
                  size_t (100));
}

int
Zip::compress(const char *raw, int rawSize, char *compressed)
{
    //
    // Reorder the pixel data.
    //

    {
        char *t1 = _tmpBuffer;
        char *t2 = _tmpBuffer + (rawSize + 1) / 2;
        const char *stop = raw + rawSize;

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
        unsigned char *t    = (unsigned char *) _tmpBuffer + 1;
        unsigned char *stop = (unsigned char *) _tmpBuffer + rawSize;
        int p = t[-1];

        while (t < stop)
        {
            int d = int (t[0]) - p + (128 + 256);
            p = t[0];
            t[0] = d;
            ++t;
        }
    }

    //
    // Compress the data using zlib
    //

    uLongf outSize = int(ceil(rawSize * 1.01)) + 100;

    if (Z_OK != ::compress ((Bytef *)compressed, &outSize,
                (const Bytef *) _tmpBuffer, rawSize))
    {
        throw IEX_NAMESPACE::BaseExc ("Data compression (zlib) failed.");
    }

    return outSize;
}

#ifdef IMF_HAVE_SSE4_1

static void
reconstruct_sse41(char *buf, size_t outSize)
{
    static const size_t bytesPerChunk = sizeof(__m128i);
    const size_t vOutSize = outSize / bytesPerChunk;

    const __m128i c = _mm_set1_epi8(-128);
    const __m128i shuffleMask = _mm_set1_epi8(15);

    // The first element doesn't have its high bit flipped during compression,
    // so it must not be flipped here.  To make the SIMD loop nice and
    // uniform, we pre-flip the bit so that the loop will unflip it again.
    buf[0] += -128;

    __m128i *vBuf = reinterpret_cast<__m128i *>(buf);
    __m128i vPrev = _mm_setzero_si128();
    for (size_t i=0; i<vOutSize; ++i)
    {
        __m128i d = _mm_add_epi8(_mm_loadu_si128(vBuf), c);

        // Compute the prefix sum of elements.
        d = _mm_add_epi8(d, _mm_slli_si128(d, 1));
        d = _mm_add_epi8(d, _mm_slli_si128(d, 2));
        d = _mm_add_epi8(d, _mm_slli_si128(d, 4));
        d = _mm_add_epi8(d, _mm_slli_si128(d, 8));
        d = _mm_add_epi8(d, vPrev);

        _mm_storeu_si128(vBuf++, d);

        // Broadcast the high byte in our result to all lanes of the prev
        // value for the next iteration.
        vPrev = _mm_shuffle_epi8(d, shuffleMask);
    }

    unsigned char prev = _mm_extract_epi8(vPrev, 15);
    for (size_t i=vOutSize*bytesPerChunk; i<outSize; ++i)
    {
        unsigned char d = prev + buf[i] - 128;
        buf[i] = d;
        prev = d;
    }
}

#else

static void
reconstruct_scalar(char *buf, size_t outSize)
{
    unsigned char *t    = (unsigned char *) buf + 1;
    unsigned char *stop = (unsigned char *) buf + outSize;

    while (t < stop)
    {
        int d = int (t[-1]) + int (t[0]) - 128;
        t[0] = d;
        ++t;
    }
}

#endif


#ifdef IMF_HAVE_SSE2

static void
interleave_sse2(const char *source, size_t outSize, char *out)
{
    static const size_t bytesPerChunk = 2*sizeof(__m128i);

    const size_t vOutSize = outSize / bytesPerChunk;

    const __m128i *v1 = reinterpret_cast<const __m128i *>(source);
    const __m128i *v2 = reinterpret_cast<const __m128i *>(source + (outSize + 1) / 2);
    __m128i *vOut = reinterpret_cast<__m128i *>(out);

    for (size_t i=0; i<vOutSize; ++i) {
        __m128i a = _mm_loadu_si128(v1++);
        __m128i b = _mm_loadu_si128(v2++);

        __m128i lo = _mm_unpacklo_epi8(a, b);
        __m128i hi = _mm_unpackhi_epi8(a, b);

        _mm_storeu_si128(vOut++, lo);
        _mm_storeu_si128(vOut++, hi);
    }

    const char *t1 = reinterpret_cast<const char *>(v1);
    const char *t2 = reinterpret_cast<const char *>(v2);
    char *sOut = reinterpret_cast<char *>(vOut);

    for (size_t i=vOutSize*bytesPerChunk; i<outSize; ++i)
    {
        *(sOut++) = (i%2==0) ? *(t1++) : *(t2++);
    }
}

#else

static void
interleave_scalar(const char *source, size_t outSize, char *out)
{
    const char *t1 = source;
    const char *t2 = source + (outSize + 1) / 2;
    char *s = out;
    char *const stop = s + outSize;

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

#endif

int
Zip::uncompress(const char *compressed, int compressedSize,
                char *raw)
{
    //
    // Decompress the data using zlib
    //

    uLongf outSize = _maxRawSize;

    if (Z_OK != ::uncompress ((Bytef *)_tmpBuffer, &outSize,
                     (const Bytef *) compressed, compressedSize))
    {
        throw IEX_NAMESPACE::InputExc ("Data decompression (zlib) failed.");
    }

    if (outSize == 0)
    {
        return outSize;
    }

    //
    // Predictor.
    //
#ifdef IMF_HAVE_SSE4_1
    reconstruct_sse41(_tmpBuffer, outSize);
#else
    reconstruct_scalar(_tmpBuffer, outSize);
#endif

    //
    // Reorder the pixel data.
    //
#ifdef IMF_HAVE_SSE2
    interleave_sse2(_tmpBuffer, outSize, raw);
#else
    interleave_scalar(_tmpBuffer, outSize, raw);
#endif

    return outSize;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
